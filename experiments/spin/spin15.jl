"""
spin15.jl - vanilla w/ T1
"""

using HDF5
using LaTeXStrings
using LinearAlgebra
import Plots
using Printf
using StaticArrays
using Statistics
using TrajectoryOptimization


# Construct paths.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin15"
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


function plot_controls(controls_file_path, save_file_path;
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


function t1_average(controls_file_path)
    # Grab and prep data.
    (
        controls_,
        evolution_time,
        states,
    ) = h5open(controls_file_path, "r+") do save_file
        controls_ = read(save_file, "controls")
        evolution_time = read(save_file, "evolution_time")
        states = read(save_file, "states")
        return (
            controls_,
            evolution_time,
            states
        )
    end
    (control_eval_count, control_count) = size(controls_)
    control_eval_times = Array(0:1:control_eval_count) * DT
    controls = states[1:end, CONTROLS_IDX]
    t1s = map(get_t1_poly, controls / (2 * pi))
    t1_avg = mean(t1s)
    return t1_avg
end


"""
horner - compute the value of a polynomial using Horner's method

Args:
coeffs :: Array(N) - the coefficients in descending order of degree
    a_{n - 1}, a_{n - 2}, ..., a_{1}, a_{0}
val :: T - the value at which the polynomial is computed

Returns:
polyval :: T - the polynomial evaluated at val
"""
function horner(coeffs, val)
    run = coeffs[1]
    for i = 2:lastindex(coeffs)
        run = coeffs[i] + val * run
    end
    return run
end


# Define experimental constants.
# qubit frequency at flux frustration point
OMEGA = 2 * pi * 1.4e-2 #GHz
DOMEGA = OMEGA * 5e-2
OMEGA_PLUS = OMEGA + DOMEGA
OMEGA_MINUS = OMEGA - DOMEGA
MAX_CONTROL_NORM_0 = 2 * pi * 3e-1
MAX_T1 = 1e-2 #s
# E / h
EC = 0.479e9
EL = 0.132e9
EJ = 3.395e9
# Q_CAP = 1 / 8e-6
Q_CAP = 1.25e5
T_CAP = 0.042
H = 6.62607015e-34
HBAR = 1.05457148e-34
KB = 1.3806503e-23
HBAR_BY_KB = 7.63823e-12
FBFQ_A = 0.202407
FBFQ_B = 0.5
# coefficients are listed in descending order
# raw coefficients are in units of seconds
FBFQ_T1_COEFFS = [
    3276.06057; -7905.24414; 8285.24137; -4939.22432;
    1821.23488; -415.520981; 53.9684414; -3.04500484
] * 1e9

# Define the system.
function get_fbfq(amplitude)
    return -abs(amplitude) * FBFQ_A + FBFQ_B
end


function get_t1_poly(amplitude)
    fbfq = get_fbfq(amplitude)
    t1 = horner(FBFQ_T1_COEFFS, fbfq)
    return t1
end


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
EVOLUTION_TIME = 56.80
COMPLEX_CONTROLS = false
CONTROL_COUNT = 1
DT = 1e-2
DT_INV = 1e2
N = Int(EVOLUTION_TIME * DT_INV) + 1
ITERATION_COUNT = Int(1e3)

# Define the problem.
INITIAL_STATE_0 = SA[1., 0, 0, 0]
INITIAL_STATE_1 = SA[0., 1, 0, 0]
STATE_SIZE, = size(INITIAL_STATE_0)
INITIAL_ASTATE = [
    INITIAL_STATE_0; # state_0
    INITIAL_STATE_1;
    @SVector zeros(CONTROL_COUNT); # int_control
    @SVector zeros(CONTROL_COUNT); # control
    @SVector zeros(CONTROL_COUNT); # dcontrol_dt
    @SVector zeros(1); # int_gamma
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
TARGET_STATE_0 = SA[1., 0, 0, -1] / sqrt(2)
TARGET_STATE_1 = SA[0., 1, -1, 0] / sqrt(2)
TARGET_ASTATE = [
    TARGET_STATE_0;
    TARGET_STATE_1;
    @SVector zeros(CONTROL_COUNT); # int_control
    @SVector zeros(CONTROL_COUNT); # control
    @SVector zeros(CONTROL_COUNT); # dcontrol_dt
    @SVector zeros(1); # int_gamma
]
STATE_0_IDX = 1:STATE_SIZE
STATE_1_IDX = STATE_0_IDX[end] + 1:STATE_0_IDX[end] + STATE_SIZE
INT_CONTROLS_IDX = STATE_1_IDX[end] + 1:STATE_1_IDX[end] + CONTROL_COUNT
CONTROLS_IDX = INT_CONTROLS_IDX[end] + 1:INT_CONTROLS_IDX[end] + CONTROL_COUNT
DCONTROLS_DT_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
INT_GAMMA_IDX = DCONTROLS_DT_IDX[end] + 1:DCONTROLS_DT_IDX[end] + 1
    

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
    delta_state_0 = (OMEGA_NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_0_IDX]
    delta_state_1 = (OMEGA_NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_1_IDX]
    delta_int_control = astate[CONTROLS_IDX]
    delta_control = astate[DCONTROLS_DT_IDX]
    delta_dcontrol_dt = d2controls_dt2
    delta_int_gamma = get_t1_poly(astate[CONTROLS_IDX][1] / (2 * pi))^(-1)
    return [
        delta_state_0;
        delta_state_1;
        delta_int_control;
        delta_control;
        delta_dcontrol_dt;
        delta_int_gamma;
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
        @SVector fill(Inf, CONTROL_COUNT);
        @SVector fill(MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        @SVector fill(Inf, CONTROL_COUNT);
        @SVector fill(Inf, 1)
    ]
    x_min = [
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, CONTROL_COUNT);
        @SVector fill(-MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        @SVector fill(-Inf, CONTROL_COUNT);
        @SVector fill(-Inf, 1)
    ]
    # controls start and end at 0
    x_max_boundary = [
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, CONTROL_COUNT);
        @SVector fill(0, CONTROL_COUNT); # control
        @SVector fill(Inf, CONTROL_COUNT);
        @SVector fill(Inf, 1)
    ]
    x_min_boundary = [
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, CONTROL_COUNT);
        @SVector fill(0, CONTROL_COUNT); # control
        @SVector fill(-Inf, CONTROL_COUNT);
        @SVector fill(-Inf, 1)
    ]

    model = Model(n, m)
    U0 = INITIAL_CONTROLS
    X0 = [
        @SVector fill(NaN, n) for k = 1:N
    ]
    Z = Traj(X0, U0, dt * ones(N))

    Q = Diagonal([
        @SVector fill(1e-1, STATE_SIZE);
        @SVector fill(1e-1, STATE_SIZE);
        @SVector fill(1e-1, CONTROL_COUNT); # int_control
        @SVector fill(0, CONTROL_COUNT); # control
        @SVector fill(1e-1, CONTROL_COUNT); # dcontrol_dt
        @SVector fill(1e6, 1); # int_gamma
    ])
    Qf = Q * N
    R = Diagonal(@SVector fill(1e-1, m)) # d2control_dt2
    obj = LQRObjective(Q, R, Qf, xf, N)

    # must satisfy control amplitudes
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and stop at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE_0_IDX; STATE_1_IDX; INT_CONTROLS_IDX])
    
    constraints = ConstraintSet(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, 1:1)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    
    prob = Problem{RK4}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    opts = SolverOptions(verbose=VERBOSE)
    solver = AugmentedLagrangianSolver(prob, opts)
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
