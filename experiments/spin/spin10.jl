#=
# spin10.jl - A module to try out julia for the spin system.
=#
module Foo

using BenchmarkTools
using CUDAnative
using CuArrays
using DiffEqSensitivity
using DifferentialEquations
using ForwardDiff
using HDF5
using LinearAlgebra
using Polynomials
using Printf
using StaticArrays
using Zygote


EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin10"
WDIR = ENV["ROBUST_QOC_PATH"]
SAVE_PATH = joinpath(WDIR, "out/$EXPERIMENT_META/$EXPERIMENT_NAME")

function generate_save_file_path(save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$save_file_name.h5", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$save_file_name.h5"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end

mutable struct Reporter
    cost :: Float64
    costs :: Array{Float64, 1}
end

mutable struct PState
    control_count :: Int64
    control_eval_count :: Int64
    control_eval_times :: AbstractArray{Float64}
    cost_multipliers :: Array{Float64, 1}
    dt :: Float64
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
function augment_mat(mat_r, mat_i)
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
NEG_I = SA_F64[0   0  1  0 ;
               0   0  0  1 ;
               -1  0  0  0 ;
               0  -1  0  0 ;]
SIGMA_X = SA_F64[0 1 0 0;
                 1 0 0 0;
                 0 0 0 1;
                 0 0 1 0]
SIGMA_Z = SA_F64[1   0  0   0;
                 0  -1  0   0;
                 0   0  1   0;
                 0   0  0  -1]
H_S = SIGMA_Z / 2
neg_i_H_S = NEG_I * H_S
H_C1 = SIGMA_X / 2

# Define the optimization.
EVOLUTION_TIME = 120
COMPLEX_CONTROLS = false
CONTROL_COUNT = 1
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = Int(EVOLUTION_TIME) + 1
CONTROL_EVAL_TIMES = SVector{CONTROL_EVAL_COUNT, Float64}(Array(range(0., stop=EVOLUTION_TIME, length=CONTROL_EVAL_COUNT)))
DT = 1e-2
GRAB_CONTROLS = false
if GRAB_CONTROLS
    initial_controls_ = h5open(joinpath(SAVE_PATH, "00007_spin10.h5")) do save_file
        save_file["controls"][200, :, :]
    end
    initial_controls_ = SMatrix{CONTROL_EVAL_COUNT, CONTROL_COUNT, Float64}(initial_controls_)
else
    initial_controls_ = @SMatrix ones(CONTROL_EVAL_COUNT, CONTROL_COUNT)    
end
INITIAL_CONTROLS = initial_controls_


# Define the problem.
INITIAL_STATE = augment_vec(SA[1.; 0])
ZEROS_INITIAL_STATE = augment_vec(SA[0; 0])
TARGET_STATE_DAGGER = conjugate_transpose_vec(SA[0; 1.])
INITIAL_ASTATE = [
    INITIAL_STATE; # state
    ZEROS_INITIAL_STATE; # dstate_dw
    ZEROS_INITIAL_STATE; # d2state_dw2
]
(STATE_SIZE,) = size(INITIAL_STATE)
(ASTATE_SIZE,) = size(INITIAL_ASTATE)
STATE_INDICES = 1:STATE_SIZE
DSTATE_DW_INDICES = STATE_SIZE + 1:STATE_SIZE * 2
D2STATE_DW2_INDICES = STATE_SIZE * 2 + 1:STATE_SIZE * 3

HILBERT_DIM = Int(STATE_SIZE / 2)

# Define Misc
LEARNING_RATE = 1e-3
ITERATION_COUNT = 200
INITIAL_COSTS = [0., 0]
INITIAL_COST_MULTIPLIERS = [1., 1.]
INITIAL_LAGRANGE_MULTIPLIERS = [0., 0]
COST_MULTIPLIER_STEP = 5.
SAVE = false


"""
"""
function rk3_step(rhs_, x, u, t, dt, us)
    k1 = rhs_(x,             u, t       , us) * dt
    k2 = rhs_(x + k1/2,      u, t + dt/2, us) * dt
    k3 = rhs_(x - k1 + 2*k2, u, t + dt  , us) * dt
    return x + (k1 + 4 * k2 + k3) / 6
end

# Attempt at in-place was slower than static array version.
# function rk3_step(rhs_, x, u, t, dt, dx, ks)
#     rhs_(dx, x,             u, t       )
#     ks[1,:] = dx * dt
#     rhs_(dx, x + ks[1,:]/2,      u, t + dt/2)
#     ks[2,:] = dx * dt
#     rhs_(dx, x - ks[1,:] + 2*ks[2,:], u, t + dt  )
#     ks[3, :] = dx * dt
#     return x + (ks[1,:] + 4 * ks[2,:] + ks[3,:]) / 6
# end


"""
"""
function rk4_step(rhs_, x, u, t, dt, us)
    k1 = rhs_(x,        u, t       , us) * dt
	k2 = rhs_(x + k1/2, u, t + dt/2, us) * dt
	k3 = rhs_(x + k2/2, u, t + dt/2, us) * dt
	k4 = rhs_(x + k3,   u, t + dt  , us) * dt
	return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
end


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
function rhs(astate, controls, time, control_step)
    # Unpack augmented state.
    state = astate[STATE_INDICES]
    dstate_dw = astate[DSTATE_DW_INDICES]
    d2state_dw2 = astate[D2STATE_DW2_INDICES]
    # less gc time but more allocations = slower in the no gradient regime
    # state = SVector{4, Float64}(astate[STATE_INDICES])
    # dstate_dw = SVector{4, Float64}(astate[DSTATE_DW_INDICES])
    # d2state_dw2 = SVector{4, Float64}(astate[D2STATE_DW2_INDICES])

    # Compute
    # linear interpolation
    # controls_ = interpolate_linear_points(CONTROL_EVAL_TIMES[control_step],
    #                                       CONTROL_EVAL_TIMES[control_step + 1],
    #                                       time,
    #                                       controls[control_step],
    #                                       controls[control_step + 1])
    # zero-order-hold
    controls_ = controls[control_step]
    
    hamiltonian_ = OMEGA * H_S + controls_[1] * H_C1
    neg_i_hamiltonian = NEG_I * hamiltonian_
    
    delta_state = neg_i_hamiltonian * state
    delta_dstate_dw = neg_i_H_S * state + neg_i_hamiltonian * dstate_dw
    delta_d2state_dw2 = neg_i_H_S * dstate_dw + neg_i_hamiltonian * d2state_dw2
    
    delta_astate = [
        delta_state;
        delta_dstate_dw;
        delta_d2state_dw2
    ]

    return delta_astate
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
    # Evolve the state.
    t = 0
    dt = pstate.dt
    N = pstate.evolution_time / dt
    final_astate = pstate.initial_astate
    control_step = 1
    for i=1:N
        final_astate = rk3_step(rhs, final_astate, controls, t, dt, control_step)
        t += + dt
        if t > pstate.control_eval_times[control_step + 1]
            control_step += 1
        end
    end
    final_state = final_astate[STATE_INDICES]
    println(final_state)
    
    # Compute costs.
    infidelity_cost = infidelity(final_astate)
    infidelity_robustness_cost = infidelity_robustness(final_astate)
    augmented_cost = (
        augment_cost(infidelity_cost, pstate.cost_multipliers[1], pstate.lagrange_multipliers[1])
        + augment_cost(infidelity_robustness_cost, pstate.cost_multipliers[2], pstate.lagrange_multipliers[2])
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
    dt = DT
    evolution_time = EVOLUTION_TIME
    control_eval_times = Array(range(0., stop=evolution_time, length=control_eval_count))
    initial_astate = INITIAL_ASTATE
    iteration_count = ITERATION_COUNT
    lagrange_multipliers = INITIAL_LAGRANGE_MULTIPLIERS
    learning_rate = LEARNING_RATE
    save = SAVE
    # Construct the program state.
    pstate = PState(
        control_count, control_eval_count, control_eval_times,
        cost_multipliers, dt, evolution_time,
        initial_astate, lagrange_multipliers,
    )
    reporter = Reporter(0, INITIAL_COSTS)
    # Create a wrapper around the cost function. Zygote takes gradients with repsect to all arguments.
    evolve_(controls_) = evolve(controls_, pstate, reporter)
    # Generate save file path
    if save
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        println("saving this optimization to $save_file_path")
        h5write(save_file_path, "controls", zeros(iteration_count, control_eval_count, control_count))
    end
    
    # AL loop
    # for i = 1:5
    #     # iLQR loop
    #     println("costs")
    #     println(reporter.costs)
    #     println("cost_multipliers")
    #     println(pstate.cost_multipliers)
    #     println("lagrange_multipliers")
    #     println(pstate.lagrange_multipliers)

    # for iteration = 1:iteration_count
    #     (dcontrols,) = Zygote.gradient(evolve_, controls)
    #     dcontrols_norm = norm(dcontrols)
    #     controls = controls - dcontrols * learning_rate
    #     @printf("%05d %f %f\n", iteration, reporter.cost, dcontrols_norm,)
    #     if save
    #         h5open(save_file_path, "r+") do save_file
    #             save_file["controls"][iteration, :, :] = controls
    #         end
    #     end
    # end

    #     pstate.cost_multipliers = pstate.cost_multipliers * COST_MULTIPLIER_STEP
    #     pstate.lagrange_multipliers = (
    #         pstate.lagrange_multipliers + pstate.cost_multipliers .* reporter.costs
    #     )
    # end

    evolve_(controls)
    # cfg = ForwardDiff.GradientConfig(evolve_, controls)
    # out = zeros(11, control_eval_count, control_count)
    # ForwardDiff.gradient!(out[11, :, :], evolve_, controls, cfg)
    # Zygote.gradient(evolve_, controls)
    # function test()
    #     for i=1:10
    #         # evolve_(controls)
    #         # ForwardDiff.gradient!(out[i, :, :], evolve_, controls, cfg)
    #         Zygote.gradient(evolve_, controls)
    #     end
    #     return
    # end

    # @time test()
end

end
