"""
spin11.jl - use trajectory optimization for the spin system
"""

import DifferentialEquations
using HDF5
using LinearAlgebra
using TrajectoryOptimization
import Plots
using Printf
using StaticArrays

# Construct paths.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin1"
WDIR = ENV["QC_PATH"]
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Plotting configuration.
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300

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


function plot_controls(controls_file_path, save_file_path)
    (
        controls,
        evolution_time,
    ) = h5open(controls_file_path, "r+") do save_file
        controls = read(save_file, "controls")
        evolution_time = read(save_file, "evolution_time")
        return (
            controls,
            evolution_time,
        )
    end
    (control_eval_count, control_count) = size(controls)
    control_eval_times = Array(range(0., stop=evolution_time, length=control_eval_count))
    fig = Plots.plot(control_eval_times, controls[:, 1], show=false, dpi=DPI)
    Plots.savefig(fig, save_file_path)
    return
end


"""
inner_product - take the inner_product of two vectors over the complex isomorphism
"""
function inner_product(v1, v2)
    length, = size(v1)
    half_length = Int(length/2)
    r = v1' * v2
    i = (v1[1:half_length]' * v2[half_length + 1:length]
         - v1[half_length + 1:length]' * v2[1:half_length])
    return [r; i]
end


"""
fidelity - inner product squared
"""
function fidelity(v1, v2)
    ip = inner_product(v1, v2)
    return ip[1] ^2 + ip[2] ^2
end

# Define experimental constants.
OMEGA = 2 * pi * 1e-2
MAX_CONTROL_NORM_0 = 2 * pi * 3e-1

# Define the system.
NEG_I = SA_F64[0   0  1  0 ;
               0   0  0  1 ;
               -1  0  0  0 ;
               0  -1  0  0 ;]
SIGMA_X = SA_F64[0   1   0   0;
                 1   0   0   0;
                 0   0   0   1;
                 0   0   1   0]
SIGMA_Z = SA_F64[1   0   0   0;
                 0  -1   0   0;
                 0   0   1   0;
                 0   0   0  -1]
H_S = SIGMA_Z / 2
NEG_I_H_S = NEG_I * H_S
OMEGA_NEG_I_H_S = OMEGA * NEG_I_H_S
H_C1 = SIGMA_X / 2
NEG_I_H_C1 = NEG_I * H_C1

# Define the optimization.
EVOLUTION_TIME = 120.
COMPLEX_CONTROLS = false
CONTROL_COUNT = 1
DT = 1e-2
N = Int(EVOLUTION_TIME / DT) + 1
ITERATION_COUNT = Int(1e3)

# Define the problem.
INITIAL_STATE = SA[1., 0, 0, 0]
STATE_SIZE, = size(INITIAL_STATE)
INITIAL_ASTATE = [
    INITIAL_STATE;
    @SVector zeros(STATE_SIZE); # dstate_dw
    @SVector zeros(STATE_SIZE); # d2state_dw2
    # @SVector zeros(2); # d2j_dw2
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
TARGET_STATE = SA[0, 1., 0, 0]
TARGET_ASTATE = [
    TARGET_STATE;
    @SVector zeros(STATE_SIZE);
    @SVector zeros(STATE_SIZE);
    # @SVector zeros(2);
]
STATE_INDICES = 1:STATE_SIZE
DSTATE_DW_INDICES = STATE_SIZE + 1:2 * STATE_SIZE
D2STATE_DW2_INDICES = 2 * STATE_SIZE + 1:3 * STATE_SIZE
D2J_DW2_INDICES = 3 * STATE_SIZE + 1:3 * STATE_SIZE + 2

# Generate initial controls.
GRAB_CONTROLS = true
INITIAL_CONTROLS = nothing
if GRAB_CONTROLS
    controls_file_path = joinpath(SAVE_PATH, "00002_spin1.h5")
    INITIAL_CONTROLS = h5open(controls_file_path, "r") do save_file
        controls = Array(save_file["controls"])
        return [
            SVector{CONTROL_COUNT}(controls[i]) for i = 1:N-1
        ]
    end
else
    INITIAL_CONTROLS = [
        @SVector fill(MAX_CONTROL_NORM_0 * 0.25, CONTROL_COUNT) for k = 1:N-1
    ]
end

# Specify logging.
VERBOSE = true
SAVE = true

struct Model <: AbstractModel
    n :: Int
    m :: Int
end


function Base.size(model::Model)
    return model.n, model.m
end


function TrajectoryOptimization.dynamics(model::Model, astate, controls, time)
    hamiltonian = OMEGA * NEG_I_H_S + controls[1] * NEG_I_H_C1
    dhamiltonian_dw = NEG_I_H_S
    delta_state = hamiltonian * astate[STATE_INDICES]
    delta_dstate_dw = dhamiltonian_dw * astate[STATE_INDICES] + hamiltonian * astate[DSTATE_DW_INDICES]
    delta_d2state_dw2 = 2 * dhamiltonian_dw * astate[DSTATE_DW_INDICES] + hamiltonian * astate[D2STATE_DW2_INDICES]
    # delta_d2j_dw2 = (
    #     inner_product(TARGET_STATE, astate[D2STATE_DW2_INDICES])
    #     .* inner_product(astate[STATE_INDICES], TARGET_STATE)
    #     + 2 * inner_product(TARGET_STATE, astate[DSTATE_DW_INDICES])
    #     .* inner_product(astate[DSTATE_DW_INDICES], TARGET_STATE)
    #     + inner_product(TARGET_STATE, astate[STATE_INDICES])
    #     .* inner_product(astate[D2STATE_DW2_INDICES], TARGET_STATE)
    # )
    return [
        delta_state;
        delta_dstate_dw;
        delta_d2state_dw2;
        # delta_d2j_dw2;
    ]
end


function run_traj()
    dt = DT
    n = ASTATE_SIZE
    m = CONTROL_COUNT
    t0 = 0.
    tf = EVOLUTION_TIME
    x0 = INITIAL_ASTATE
    xf = TARGET_ASTATE
    u_max = SA[MAX_CONTROL_NORM_0]
    u_min = SA[-MAX_CONTROL_NORM_0]
    U0 = INITIAL_CONTROLS
    
    model = Model(n, m)

    X0 = [
        @SVector fill(NaN, n) for k = 1:N
    ]
    Z = Traj(X0, U0, dt * ones(N))


    Q = Diagonal([
        @SVector fill(1e-2, STATE_SIZE);
        @SVector zeros(STATE_SIZE);
        @SVector fill(1e-8, STATE_SIZE);
        # @SVector zeros(2)
    ])
    Qf = Q * N
    R = 1e-1 * Diagonal(@SVector ones(m))
    obj = LQRObjective(Q, R, Qf, xf, N)
    
    bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    target_state_constraint = GoalConstraint(xf, STATE_INDICES)
    
    constraints = ConstraintSet(n, m, N)
    add_constraint!(constraints, target_state_constraint, N:N)
    add_constraint!(constraints, bnd, 1:N)
    
    prob = Problem{RK4}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    opts = SolverOptions(verbose=VERBOSE)
    solver = AugmentedLagrangianSolver(prob, opts)
    solve!(solver)

    controls_raw = controls(solver)
    controls_arr = permutedims(reduce(hcat, map(Array, controls_raw)), [2, 1])
    states_raw = states(solver)
    states_arr = permutedims(reduce(hcat, map(Array, states_raw)), [2, 1])

    # Save
    if SAVE
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        @printf("Saving this optimization to %s\n", save_file_path)
        h5open(save_file_path, "cw") do save_file
            write(save_file, "controls", controls_arr)
            write(save_file, "evolution_time", tf)
            write(save_file, "states", states_arr)
        end
    end
end

