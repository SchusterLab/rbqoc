"""
spin15.jl - T1 optimized pulses
"""

using HDF5
using LinearAlgebra
using StaticArrays
using TrajectoryOptimization

include("qoclib.jl")
using QOCLib
using SpinLib

# paths
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin15"
WDIR = ENV["ROBUST_QOC_PATH"]
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Define the optimization.
EVOLUTION_TIME = 20.0
CONTROL_COUNT = 1
DT_INTEGRATOR = 1
DT_INIT = DT_PREF
DT_INIT_INV = DT_PREF_INV
DT_MIN = DT_INIT / 2
DT_MAX = DT_INIT * 2
N = Int(EVOLUTION_TIME * DT_INIT_INV) + 1

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
TARGET_STATE_0 = SA[1., 0, -1, 0] / sqrt(2)
TARGET_STATE_1 = SA[0., 1, 0, 1] / sqrt(2)
TARGET_ASTATE = [
    TARGET_STATE_0;
    TARGET_STATE_1;
    @SVector zeros(CONTROL_COUNT); # int_control
    @SVector zeros(CONTROL_COUNT); # control
    @SVector zeros(CONTROL_COUNT); # dcontrol_dt
    @SVector zeros(1); # int_gamma
]
# state indices
STATE_0_IDX = 1:STATE_SIZE
STATE_1_IDX = STATE_0_IDX[end] + 1:STATE_0_IDX[end] + STATE_SIZE
INT_CONTROLS_IDX = STATE_1_IDX[end] + 1:STATE_1_IDX[end] + CONTROL_COUNT
CONTROLS_IDX = INT_CONTROLS_IDX[end] + 1:INT_CONTROLS_IDX[end] + CONTROL_COUNT
DCONTROLS_DT_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
INT_GAMMA_IDX = DCONTROLS_DT_IDX[end] + 1:DCONTROLS_DT_IDX[end] + 1


# Generate initial controls.
# INIITAL_CONTROLS should be small if optimizing over derivatives.
INITIAL_ACONTROLS = [
    SVector{CONTROL_COUNT + 1}([fill(1e-4, CONTROL_COUNT); DT_INIT]) for k = 1:N-1
]
ACONTROLS_SIZE, = size(INITIAL_ACONTROLS[1])
# control indices
D2CONTROLS_DT2_IDX = 1:CONTROL_COUNT
DT_IDX = D2CONTROLS_DT2_IDX[end] + 1:D2CONTROLS_DT2_IDX[end] + 1


# Specify logging.
VERBOSE = true
SAVE = true

struct Model <: AbstractModel
    n :: Int
    m :: Int
end

Base.size(model::Model) = (model.n, model.m)

function TrajectoryOptimization.dynamics(model::Model, astate, acontrols, time)
    neg_i_control_hamiltonian = astate[CONTROLS_IDX][1] * NEG_I_H_C1
    delta_state_0 = (OMEGA_NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_0_IDX]
    delta_state_1 = (OMEGA_NEG_I_H_S + neg_i_control_hamiltonian) * astate[STATE_1_IDX]
    delta_int_control = astate[CONTROLS_IDX]
    delta_control = astate[DCONTROLS_DT_IDX]
    delta_dcontrol_dt = acontrols[D2CONTROLS_DT2_IDX]
    delta_int_gamma = get_t1_poly(astate[CONTROLS_IDX][1] / (2 * pi))^(-1)
    return [
        delta_state_0;
        delta_state_1;
        delta_int_control;
        delta_control;
        delta_dcontrol_dt;
        delta_int_gamma;
    ] * acontrols[DT_IDX][1]
end


function run_traj(;sample=false)
    dt = DT_INTEGRATOR
    n = ASTATE_SIZE
    m = ACONTROLS_SIZE
    t0 = 0.
    tf = EVOLUTION_TIME
    x0 = INITIAL_ASTATE
    xf = TARGET_ASTATE
    # bound the control amplitude 
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
    # bound dt
    u_min = [
        @SVector fill(-Inf, CONTROL_COUNT);
        @SVector fill(DT_MIN, 1); # dt
    ]
    u_max = [
        @SVector fill(Inf, CONTROL_COUNT);
        @SVector fill(DT_MAX, 1); # dt
    ]

    # Generate initial trajectory.
    model = Model(n, m)
    U0 = INITIAL_ACONTROLS
    X0 = [
        @SVector fill(NaN, n) for k = 1:N
    ]
    Z = Traj(X0, U0, dt * ones(N))

    # Define penalties.
    Q = Diagonal(SVector{n}([
        fill(1e-1, STATE_SIZE); # state 0
        fill(1e-1, STATE_SIZE); # state 1
        fill(1e-1, CONTROL_COUNT); # int_control
        fill(0, CONTROL_COUNT); # control
        fill(1e-1, CONTROL_COUNT); # dcontrol_dt
        fill(1e-1, 1); # int_gamma
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(1e-1, CONTROL_COUNT); # d2control_dt2
        fill(1e-1, 1); # dt
    ])) 
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must statisfy conrols start and stop at 0.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must satisfy dt bound.
    dt_bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    # Must reach target state. Must have zero net flux.
    target_astate_constraint = GoalConstraint(xf, [STATE_0_IDX; STATE_1_IDX; INT_CONTROLS_IDX]);
    
    constraints = ConstraintSet(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, 1:1)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, dt_bnd, 1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);

    # Instantiate problem and solve.
    prob = Problem{TrajectoryOptimization.RK4}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    opts = SolverOptions(verbose=VERBOSE)
    solver = AugmentedLagrangianSolver(prob, opts)
    TrajectoryOptimization.solve!(solver)

    # Post-process.
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
    cidx_arr = Array(CONTROLS_IDX)
    
    # Save.
    if SAVE
        save_file_path = generate_save_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            write(save_file, "controls", controls_arr)
            write(savE_file, "controls_idx", cidx_arr)
            write(save_file, "evolution_time", tf)
            write(save_file, "states", states_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
        end
        if sample
            (controls_sample, d2controls_dt2_sample, evolution_time_sample) = sample_polynomial(save_file_path)
            h5open(save_file_path, "r+") do save_file
                write(save_file, "controls_sample", controls_sample)
                write(save_file, "d2controls_dt2_sample", d2controls_dt2_sample)
                write(save_file, "evolution_time_sample", d2controls_dt2_sample)
            end
        end
    end
end
