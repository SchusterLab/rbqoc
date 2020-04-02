#=
# foo.jl - A module to try out julia.
=#
module Foo

using CUDAnative
using CuArrays
using DiffEqSensitivity
using DifferentialEquations
using LinearAlgebra
using Zygote

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


# Define experimental constants.
OMEGA = 2 * pi * 1e-2

# Define the system.
SIGMA_X = augment_mat([0 1; 1  0])
SIGMA_Z = augment_mat([1 0; 0 -1])
H_S = SIGMA_Z / 2
neg_i_H_S = neg_i_mat(H_S)
H_C1 = SIGMA_X / 2

# Define the optimization.
EVOLUTION_TIME = 120.
TSPAN = (0., EVOLUTION_TIME)

# Define the problem.
INITIAL_STATE = augment_vec(map(Float64, [1; 0]))
TARGET_STATE = augment_vec(map(Float64, [0; 1]))
TARGET_STATE_DAGGER = conj_vec(TARGET_STATE)'
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

# Define the controls.
INITIAL_CONTROLS = [1.;]


"""
Interpolate the controls at a given time.
This function is a dummy. We would normally use polynomial fitting.
"""
function interpolate(controls, time)
    return controls[1]
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
    controls_ = interpolate(controls, time)
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
    fidelity = inner_product^2
    infidelity_ = 1 - fidelity
    return infidelity_
end


"""
Infidelity Robustness captures the sensitivity of the difference between
the final state and the target final state to the parameter OMEGA.
"""
function infidelity_robustness(final_astate)
    d2final_state_dw2 = final_astate[D2STATE_DW2_INDICES]
    inner_product = TARGET_STATE_DAGGER * d2final_state_dw2
    norm_ = inner_product^2
    return norm_
end


"""
Evolve a state and compute associated costs.
"""
function evolve(controls, pstate)
    (evolution_time, initial_astate, ode_problem) = pstate
    sol = Array(concrete_solve(ode_problem, Tsit5(), initial_astate, controls,
                               saveat=evolution_time, sensealg=QuadratureAdjoint()))
    final_astate = sol[ASTATE_SIZE + 1:2 * ASTATE_SIZE]
    cost_ = 0
    cost_ = cost_ + infidelity(final_astate)
    cost_ = cost_ + infidelity_robustness(final_astate)
    return cost_
end


function main()
    controls = INITIAL_CONTROLS
    tspan = (0., EVOLUTION_TIME)
    ode_problem = ODEProblem(rhs, INITIAL_ASTATE, tspan, p=controls)
    # pstate is short for program state, not to be confused with the quantum `state`.
    pstate = (EVOLUTION_TIME, INITIAL_ASTATE, ode_problem)
    # Create a wrapper around the cost function. Zygote takes gradients with repsect to all arguments.
    evolve_(controls_) = evolve(controls_, pstate)
    @time begin
        (dcontrols,) = Zygote.gradient(evolve_, controls)
    end
    println(dcontrols)
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
