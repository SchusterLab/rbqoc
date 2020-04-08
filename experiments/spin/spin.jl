#=
# foo.jl - A module to try out julia.
=#
module Foo

using CUDAnative
using CuArrays
using DiffEqSensitivity
using DifferentialEquations
using LinearAlgebra
using Polynomials
using Printf
using Zygote

mutable struct Reporter
    cost :: Float64
    costs :: Array{Float64, 1}
end

mutable struct PState
    control_count :: Int64
    control_eval_count :: Int64
    cost_multipliers :: Array{Float64, 1}
    evolution_time :: Float64
    initial_astate :: Any
    lagrange_multipliers :: Array{Float64, 1}
    
end

# We need an isomorphism from complex numbers to real numbers.
# TODO: It may be more memory efficient to store the augmented
# matrix as [mat_r ; mat_i] and define special multiplication
# operations.
"""
Promote a complex vector on the Hilbert space to
a real operator on the augmented Hilbert space.
"""
function augment_vec(vec)
    return [real(vec); imag(vec)]
end


"""
Complex conjugate an augmented vector.
"""
function conj_vec(vec)
    (len,) = size(vec)
    imag_indices = Int(len / 2) + 1:len
    vec[imag_indices] = -vec[imag_indices]
    return vec
end


"""
Promote a complex operator on the Hilbert space to a real
operator on the augmented Hilbert space.
"""
function augment_mat(mat)
    mat_r = real(mat)
    mat_i = imag(mat)
    return [mat_r -mat_i ; mat_i mat_r]
end


"""
Complex conjugate an augmented matrix.
"""
function conj_mat(mat)
    (len, len) = size(mat)
    len_by_2 = Int(len / 2)
    lo = 1:len_by_2
    hi = len_by_2 + 1:len
    mat[lo, hi] = -mat[lo, hi]
    mat[hi, lo] = -mat[hi, lo]
    return mat
end


"""
Multiply an augmented matrix by the scalar negative imaginary unit (-i).

This is faster than multiplying by a matrix but
cannot be used in the differentiable code
because Zygote does not like the indexing.
"""
function neg_i_mat(mat)
    (len, len) = size(mat)
    len_by_2 = Int(len / 2)
    lo = 1:len_by_2
    hi = len_by_2 + 1:len
    mat_r = mat[lo, lo]
    mat_i = mat[hi, lo]
    return [mat_i mat_r ; -mat_r mat_i]
end


"""
Construct an augmented matrix that represents the conjugate transpose of
a vector.
"""
function conjugate_transpose_vec(vec)
    vec_r_t = real(vec)'
    vec_i_t = imag(vec)'
    return [vec_r_t vec_i_t ; -vec_i_t vec_r_t]
end


# Define experimental constants.
OMEGA = 2 * pi * 1e-2
MAX_CONTROL_NORM_0 = 2 * pi * 3e-1
HAMILTONIAN_ARGS = [
    OMEGA,
]
MAX_CONTROL_NORMS = [
    MAX_CONTROL_NORM_0,
]

# Define the system.
SIGMA_X = augment_mat([0 1; 1  0])
SIGMA_Z = augment_mat([1 0; 0 -1])
H_S = SIGMA_Z / 2
neg_i_H_S = neg_i_mat(H_S)
H_C1 = SIGMA_X / 2

# Define the optimization.
EVOLUTION_TIME = 120.
TSPAN = (0., EVOLUTION_TIME)
COMPLEX_CONTROLS = false
CONTROL_COUNT = 1
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = Int(EVOLUTION_TIME) + 1
CONTROL_EVAL_TIMES = Array(range(0., stop=EVOLUTION_TIME, length=CONTROL_EVAL_COUNT))
INITIAL_CONTROLS = ones(CONTROL_EVAL_COUNT, CONTROL_COUNT)

# Define the problem.
INITIAL_STATE = augment_vec([1.; 0])
TARGET_STATE_DAGGER = conjugate_transpose_vec([0; 1.])
INITIAL_ASTATE = [
    INITIAL_STATE; # state
    zeros(size(INITIAL_STATE)); # dstate_dw
    zeros(size(INITIAL_STATE)); # d2state_dw2
]
(STATE_SIZE,) = size(INITIAL_STATE)
(ASTATE_SIZE,) = size(INITIAL_ASTATE)
STATE_INDICES = 1:STATE_SIZE
DSTATE_DW_INDICES = STATE_SIZE + 1:STATE_SIZE * 2
D2STATE_DW2_INDICES = STATE_SIZE * 2 + 1:STATE_SIZE * 3

HILBERT_DIM = Int(STATE_SIZE / 2)
NEG_I = [zeros(HILBERT_DIM, HILBERT_DIM) Matrix(I, HILBERT_DIM, HILBERT_DIM) ;
         -Matrix(I, HILBERT_DIM, HILBERT_DIM) zeros(HILBERT_DIM, HILBERT_DIM)]

# Define Misc
LEARNING_RATE = 1e-3
ITERATION_COUNT = 100
INITIAL_COSTS = [0., 0]
INITIAL_COST_MULTIPLIERS = [1., 1.]
INITIAL_LAGRANGE_MULTIPLIERS = [0., 0]
COST_MULTIPLIER_STEP = 5.


"""
Evaluate the polynomial using horner's method.
"""
function horner(coeffs, x)
    (len,) = size(coeffs)
    if len == 1
        val = coeffs[1]
    else
        val = coeffs[len]
        for i in len:-1:2
            val = coeffs[i - 1] + x * val
        end
    end
    return val
end


"""
Interpolate at x3 given (x1, y1), (x2, y2)
"""
function interpolate_linear_points(x1, x2, x3, y1, y2)
    return y1 + (((y2 - y1) / (x2 - x1)) * (x3 - x1))
end


"""
Interpolate the controls at a given time.
"""
function interpolate(controls, control_eval_times, time)
    (len,) = size(control_eval_times)
    if time < control_eval_times[1]
        t1 = control_eval_times[1]
        t2 = control_eval_times[2]
        c1 = controls[1]
        c2 = controls[2]
    elseif time >= control_eval_times[len]
        t1 = control_eval_times[len]
        t2 = control_eval_times[len - 1]
        c1 = controls[len]
        c2 = controls[len - 1]
    else
        hi = Int(findall(t -> t > time, control_eval_times)[1][1])
        t1 = control_eval_times[hi]
        t2 = control_eval_times[hi - 1]
        c1 = controls[hi]
        c2 = controls[hi - 1]
    end
    
    controls_ = interpolate_linear_points(t1, t2, time, c1, c2)
    return controls_
end


"""
The right-hand-side of the Schroedinger equation for the dynamics
we consider.
"""
function rhs(delta_astate, astate, controls, time)
    # Unpack augmented state.
    state = astate[STATE_INDICES]
    dstate_dw = astate[DSTATE_DW_INDICES]
    d2state_dw2 = astate[D2STATE_DW2_INDICES]

    # Compute
    controls_ = interpolate(controls, CONTROL_EVAL_TIMES, time)
    hamiltonian_ = OMEGA * H_S + controls_[1] * H_C1
    neg_i_hamiltonian = NEG_I * hamiltonian_
    
    # Pack delta augmented state.    
    delta_astate[STATE_INDICES] = neg_i_hamiltonian * state
    delta_astate[DSTATE_DW_INDICES] = neg_i_H_S * state + neg_i_hamiltonian * dstate_dw
    delta_astate[D2STATE_DW2_INDICES] = neg_i_H_S * dstate_dw + neg_i_hamiltonian * d2state_dw2
end


"""
Infidelity captures the difference between
the final state and the target final state.
"""
function infidelity(final_astate)
    final_state = final_astate[STATE_INDICES]
    inner_product = TARGET_STATE_DAGGER * final_state
    fidelity = inner_product[1]^2 + inner_product[2]^2
    infidelity_ = 1 - fidelity
    return infidelity_
end


"""
Infidelity Robustness captures the sensitivity of the difference between
the final state and the target final state to the parameter OMEGA.

Note:
d2/dw2 |<psi_t | psi_f>|^2 = 2 * |<psi_t | d2psi_f_dw2>|
"""
function infidelity_robustness(final_astate)
    d2final_state_dw2 = final_astate[D2STATE_DW2_INDICES]
    inner_product = TARGET_STATE_DAGGER * d2final_state_dw2
    d2j_dw2 = 2 * sqrt(inner_product[1]^2 + inner_product[2]^2)
    return d2j_dw2 * 1e-3
end


function augment_cost(cost, multiplier, lagrange_multiplier)
    # return cost * lagrange_multiplier + (multiplier / 2) * cost ^ 2
    return cost
end


"""
Evolve a state and compute associated costs.
"""
function evolve(controls, pstate, reporter)
    # Unpack pstate.
    tspan = (0., pstate.evolution_time)
    ode_problem = ODEProblem(rhs, pstate.initial_astate, tspan, p=controls)
    
    # Evolve the state.
    sol = Array(concrete_solve(ode_problem, Tsit5(), pstate.initial_astate, controls,
                               saveat=pstate.evolution_time, sensealg=QuadratureAdjoint(abstol=1e-8), abstol=1e-8))
    final_astate = sol[ASTATE_SIZE + 1:2 * ASTATE_SIZE]
    
    # Compute costs.
    infidelity_cost = infidelity(final_astate)
    infidelity_robustness_cost = infidelity_robustness(final_astate)
    augmented_cost = (
        augment_cost(infidelity_cost, pstate.cost_multipliers[1], pstate.lagrange_multipliers[1])
        # + augment_cost(infidelity_robustness_cost, pstate.cost_multipliers[2], pstate.lagrange_multipliers[2])
    )
    
    
    # Report.
    reporter.costs = [
        infidelity_cost
        infidelity_robustness_cost
    ]
    reporter.cost = augmented_cost

    return augmented_cost
end


function main()
    # Grab relevant information.
    control_count = CONTROL_COUNT
    control_eval_count = CONTROL_EVAL_COUNT
    controls = INITIAL_CONTROLS
    cost_multipliers = INITIAL_COST_MULTIPLIERS
    evolution_time = EVOLUTION_TIME
    control_eval_times = Array(range(0., stop=evolution_time, length=control_eval_count))
    initial_astate = INITIAL_ASTATE
    iteration_count = ITERATION_COUNT
    lagrange_multipliers = INITIAL_LAGRANGE_MULTIPLIERS
    # Construct the program state.
    pstate = PState(
        control_count, control_eval_count,
        cost_multipliers, evolution_time,
        initial_astate, lagrange_multipliers,
    )
    reporter = Reporter(0, INITIAL_COSTS)
    # Create a wrapper around the cost function. Zygote takes gradients with repsect to all arguments.
    evolve_(controls_) = evolve(controls_, pstate, reporter)
    # AL loop
    # for i = 1:5
    #     # iLQR loop
    #     println("costs")
    #     println(reporter.costs)
    #     println("cost_multipliers")
    #     println(pstate.cost_multipliers)
    #     println("lagrange_multipliers")
    #     println(pstate.lagrange_multipliers)
    learning_rate = LEARNING_RATE
    for j = 1:iteration_count
        (dcontrols,) = Zygote.gradient(evolve_, controls)
        dcontrols_norm = norm(dcontrols)
        controls = controls - dcontrols * learning_rate
        @printf("%f %f %f\n", reporter.cost, dcontrols_norm, learning_rate)
        if j == 40
            learning_rate /= 1e1
        end
    end
    #     pstate.cost_multipliers = pstate.cost_multipliers * COST_MULTIPLIER_STEP
    #     pstate.lagrange_multipliers = (
    #         pstate.lagrange_multipliers + pstate.cost_multipliers .* reporter.costs
    #     )
    # end
end

end


"""
This module is for testing.
"""
module Test

using Polynomials
using Random
using Statistics
using Zygote

function horner(coeffs, x)
    (len,) = size(coeffs)
    if len == 1
        val = coeffs[1]
    else
        val = coeffs[len]
        for i in len:2
            val = coeffs[i - 1] + x * val
        end
    end
    return val
end

function poly_zygote(x, y)
    poly_ = polyfit(x, y)
    val = horner(coeffs(poly_), mean(x))
    return val
end

"""
This function doesn't work because `polyfit` uses in-place array manipulation [0].

References:
[0] https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl
"""
function test_poly_zygote()
    x = rand(5)
    y = rand(5)
    poly_zygote_ = (x_) -> poly_zygote(x_, y)
    (dx,) = Zygote.gradient(poly_zygote_, x)
end

function main()
    test_poly_zygote()
end

end

"""
This module provides a minimum working example of obtaining the adjoint
sensitivity with respect to auxiliary parameters
of a functional defined on the final state of evolution.

References:
[0] https://docs.sciml.ai/dev/analysis/sensitivity/#concrete_solve-Examples-1
"""
module MWE

using DiffEqSensitivity
using DifferentialEquations
using Zygote

function rhs(du, u, p, t)
  du[1] = p[1] * u[1]
  du[2] = p[2] * u[2]
end

function cost(u0, p, prob, end_time)
    sol = Array(concrete_solve(prob, Tsit5(abstol=1e-10), u0, p, saveat=end_time, sensealg=QuadratureAdjoint(abstol=1e-8)))
    final_u = sol[3:4]
    cost_ = real(final_u[1])
    return cost_
end

function main()
    p = [1.0, 1.0, 3.0, 1.0]
    u0 = [1.0 ; 1.0]
    end_time = 1.0
    tspan = (0.0, end_time)
    prob = ODEProblem(rhs, u0, tspan, p=p)
    cost_(p_) = cost(u0, p_, prob, end_time)
    (dp,) = Zygote.gradient(cost_, p)
    println(dp)
end

end
