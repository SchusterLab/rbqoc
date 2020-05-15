"""
spin12.jl - sampling robustness
"""

using HDF5
using LinearAlgebra
import Plots
using Printf
using StaticArrays
using TrajectoryOptimization

# Construct paths.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin12"
WDIR = ENV["ROBUST_QOC_PATH"]
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


function plot_controls(controls_file_path, save_file_path,
                       title=nothing)
    # Grab and prep data.
    (
        controls,
        evolution_time,
        states,
    ) = h5open(controls_file_path, "r+") do save_file
        controls = read(save_file, "controls")
        evolution_time = read(save_file, "evolution_time")
        states = read(save_file, "states")
        return (
            controls,
            evolution_time,
            states
        )
    end
    (control_eval_count, control_count) = size(controls)
    control_eval_times = Array(range(0., stop=evolution_time, length=control_eval_count))
    file_name = split(basename(controls_file_path), ".h5")[1]
    if isnothing(title)
        title = file_name
    end

    # Plot.
    if false
        fig = Plots.plot(control_eval_times, controls[:, 1], show=false, dpi=DPI)
    else
        fig = Plots.plot(control_eval_times, states[1:end-1, CONTROLS_IDX], show=false, dpi=DPI,
                         label="controls", title=title)
        Plots.xlabel!("Time (ns)")
        Plots.ylabel!("Amplitude (GHz)")
    end
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
OMEGA = 2 * pi * 1.4e-2
DOMEGA = OMEGA * 5e-2
OMEGA_PLUS = OMEGA + DOMEGA
OMEGA_2PLUS = OMEGA + 2 * DOMEGA
OMEGA_MINUS = OMEGA - DOMEGA
OMEGA_2MINUS = OMEGA - 2 * DOMEGA
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
EVOLUTION_TIME = 150.
COMPLEX_CONTROLS = false
CONTROL_COUNT = 1
DT = 1e-2
N = Int(EVOLUTION_TIME / DT) + 1
ITERATION_COUNT = Int(1e3)

# Define the problem.
INITIAL_STATE = SA[1., 0, 0, 0]
STATE_SIZE, = size(INITIAL_STATE)
INITIAL_ASTATE = [
    INITIAL_STATE; # state (w)
    INITIAL_STATE; # state (w + dw)
    INITIAL_STATE; # state (w - dw)
    INITIAL_STATE; # state (w + 2dw)
    INITIAL_STATE; # state (w - 2dw)
    @SVector zeros(1); # int_control
    @SVector zeros(1); # control
    @SVector zeros(1); # dcontrol_dt
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
TARGET_STATE = SA[0, 1., 0, 0]
TARGET_ASTATE = [
    TARGET_STATE;
    TARGET_STATE;
    TARGET_STATE;
    TARGET_STATE;
    TARGET_STATE;
    @SVector zeros(1); # int_control
    @SVector zeros(1); # control
    @SVector zeros(1); # dcontrol_dt
]
STATE_IDX = 1:STATE_SIZE
STATE_PLUS_IDX = STATE_SIZE + 1:2 * STATE_SIZE
STATE_MINUS_IDX = 2 * STATE_SIZE + 1:3 * STATE_SIZE
STATE_2PLUS_IDX = 3 * STATE_SIZE + 1:4 * STATE_SIZE
STATE_2MINUS_IDX = 4 * STATE_SIZE + 1:5 * STATE_SIZE
# SE = STATE_MINUS_IDX[end]
SE = STATE_2MINUS_IDX[end]
INT_CONTROLS_IDX = SE + 1:SE + CONTROL_COUNT
CONTROLS_IDX = SE + CONTROL_COUNT + 1:SE + 2 * CONTROL_COUNT
DCONTROLS_DT_IDX = SE + 2 * CONTROL_COUNT + 1:SE + 3 * CONTROL_COUNT
CE = DCONTROLS_DT_IDX[end]


# Generate initial controls.
GRAB_CONTROLS = false
INITIAL_CONTROLS = nothing
if GRAB_CONTROLS
    controls_file_path = joinpath(SAVE_PATH, "00000_spin12.h5")
    INITIAL_CONTROLS = h5open(controls_file_path, "r") do save_file
        controls = Array(save_file["controls"])
        return [
            SVector{CONTROL_COUNT}(controls[i]) for i = 1:N-1
        ]
    end
else
    # INIITAL_CONTROLS should be small if optimizing over derivatives.
    INITIAL_CONTROLS = [
        @SVector fill(1e-4, CONTROL_COUNT) for k = 1:N-1
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


function TrajectoryOptimization.dynamics(model::Model, astate, d2controls_dt2, time)
    neg_i_control_hamiltonian = astate[CONTROLS_IDX][1] * NEG_I_H_C1
    delta_state = (OMEGA * NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_IDX]
    delta_state_plus = (OMEGA_PLUS * NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_PLUS_IDX]
    delta_state_minus = (OMEGA_MINUS * NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_MINUS_IDX]
    delta_state_2plus = (OMEGA_2PLUS * NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_2PLUS_IDX]
    delta_state_2minus = (OMEGA_2MINUS * NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_2MINUS_IDX]
    delta_int_control = astate[CONTROLS_IDX]
    delta_control = astate[DCONTROLS_DT_IDX]
    delta_dcontrol_dt = d2controls_dt2
    return [
        delta_state;
        delta_state_plus;
        delta_state_minus;
        delta_state_2plus;
        delta_state_2minus;
        delta_int_control;
        delta_control;
        delta_dcontrol_dt;
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
    # control amplitude constraint
    x_max = [
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, 1);
        @SVector fill(MAX_CONTROL_NORM_0, 1); # control
        @SVector fill(Inf, 1);
    ]
    x_min = [
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, 1);
        @SVector fill(-MAX_CONTROL_NORM_0, 1); # control
        @SVector fill(-Inf, 1);
    ]
    # controls start and end at 0
    x_max_boundary = [
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, 1);
        @SVector fill(0, 1); # control
        @SVector fill(Inf, 1);
    ]
    x_min_boundary = [
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, 1);
        @SVector fill(0, 1); # control
        @SVector fill(-Inf, 1);
    ]

    model = Model(n, m)
    U0 = INITIAL_CONTROLS
    X0 = [
        @SVector fill(NaN, n) for k = 1:N
    ]
    Z = Traj(X0, U0, dt * ones(N))

    Q = Diagonal([
        @SVector fill(1e-1, STATE_SIZE); # state (w)
        @SVector fill(5e-1, STATE_SIZE); # state (w + dw)
        @SVector fill(5e-1, STATE_SIZE); # state (w - dw)
        @SVector fill(1e-1, STATE_SIZE); # state (w + 2dw)
        @SVector fill(1e-1, STATE_SIZE); # state (w - 2dw)
        @SVector fill(1e-1, 1); # int_control
        @SVector fill(1e-1, 1); # control
        @SVector fill(1e-1, 1); # dcontrol_dt
    ])
    Qf = Q * N
    R = Diagonal(@SVector fill(1e-1, m))
    obj = LQRObjective(Q, R, Qf, xf, N)

    # must satisfy control amplitudes
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and stop at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE_IDX;INT_CONTROLS_IDX])
    
    constraints = ConstraintSet(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, 1:1)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    
    prob = Problem{RK4}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    opts = SolverOptions(verbose=VERBOSE)
    solver = ALTROSolver(prob, opts)
    solve!(solver)

    controls_raw = controls(solver)
    controls_arr = permutedims(reduce(hcat, map(Array, controls_raw)), [2, 1])
    states_raw = states(solver)
    states_arr = permutedims(reduce(hcat, map(Array, states_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    
    # Save
    if SAVE
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        @printf("Saving this optimization to %s\n", save_file_path)
        h5open(save_file_path, "cw") do save_file
            write(save_file, "controls", controls_arr)
            write(save_file, "evolution_time", tf)
            write(save_file, "states", states_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
        end
    end
end
