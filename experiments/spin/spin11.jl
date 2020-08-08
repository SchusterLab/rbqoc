"""
spin11.jl - derivative robustness
"""

using Altro
using HDF5
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization
const TO = TrajectoryOptimization

include(joinpath(ENV["ROBUST_QOC_PATH"], "rbqoc.jl"))

# paths
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin11"
WDIR = ENV["RBQOC_PATH"]
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Define the optimization.
CONTROL_COUNT = 1
DORDER = 3
DT_STATIC = DT_PREF
DT_STATIC_INV = DT_PREF_INV
CONSTRAINT_TOLERANCE = 1e-8

# Define the problem.
INITIAL_STATE_1 = SA[1., 0, 0, 0]
INITIAL_STATE_2 = SA[0., 1, 0, 0]
STATE_COUNT = 2
INITIAL_ASTATE = [
    INITIAL_STATE_1; # state1
    @SVector zeros(DORDER * STATE_SIZE_ISO); # dnstate1
    INITIAL_STATE_2; # state2
    @SVector zeros(DORDER * STATE_SIZE_ISO); # dnstate2
    @SVector zeros(3 * CONTROL_COUNT); # intcontrol, control, dcontrol
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
# state indices
STATE1_IDX = 1:STATE_SIZE_ISO
DSTATE1_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + STATE_SIZE_ISO
D2STATE1_IDX = DSTATE1_IDX[end] + 1:DSTATE1_IDX[end] + STATE_SIZE_ISO
D3STATE1_IDX = D2STATE1_IDX[end] + 1:D2STATE1_IDX[end] + STATE_SIZE_ISO
STATE2_IDX = D3STATE1_IDX[end] + 1:D3STATE1_IDX[end] + STATE_SIZE_ISO
DSTATE2_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + STATE_SIZE_ISO
D2STATE2_IDX = DSTATE2_IDX[end] + 1:DSTATE2_IDX[end] + STATE_SIZE_ISO
D3STATE2_IDX = D2STATE2_IDX[end] + 1:D2STATE2_IDX[end] + STATE_SIZE_ISO
INTCONTROLS_IDX = D3STATE2_IDX[end] + 1:D3STATE2_IDX[end] + CONTROL_COUNT
CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# control indices
D2CONTROLS_IDX = 1:CONTROL_COUNT


# Specify logging.
VERBOSE = true
SAVE = true

struct Model <: AbstractModel
    n :: Int
    m :: Int
end


Base.size(model::Model) = (model.n, model.m)


function RobotDynamics.dynamics(model::Model, astate, acontrols, time)
    negi_h = (
        FQ_NEGI_H0_ISO
        + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    )
    delta_state1 = negi_h * astate[STATE1_IDX]
    delta_dstate1 = NEGI_H0_ISO * astate[STATE1_IDX] + negi_h * astate[DSTATE1_IDX]
    delta_d2state1 = 2 * NEGI_H0_ISO * astate[DSTATE1_IDX] + negi_h * astate[D2STATE1_IDX]
    delta_d3state1 = 3 * NEGI_H0_ISO * astate[D2STATE1_IDX] + negi_h * astate[D3STATE1_IDX]
    delta_state2 = negi_h * astate[STATE2_IDX]
    delta_dstate2 = NEGI_H0_ISO * astate[STATE2_IDX] + negi_h * astate[DSTATE2_IDX]
    delta_d2state2 = 2 * NEGI_H0_ISO * astate[DSTATE2_IDX] + negi_h * astate[D2STATE2_IDX]
    delta_d3state2 = 3 * NEGI_H0_ISO * astate[D2STATE2_IDX] + negi_h * astate[D3STATE2_IDX]
    delta_intcontrol = astate[CONTROLS_IDX]
    delta_control = astate[DCONTROLS_IDX]
    delta_dcontrol = acontrols[D2CONTROLS_IDX]
    return [
        delta_state1;
        delta_dstate1;
        delta_d2state1;
        delta_d3state1;
        delta_state2;
        delta_dstate2;
        delta_d2state2;
        delta_d3state2;
        delta_intcontrol;
        delta_control;
        delta_dcontrol;
    ]
end


function run_traj(;gate_type=ypiby2, evolution_time=20.)
    dt = DT_STATIC
    N = Int(floor(evolution_time * DT_STATIC_INV)) + 1
    n = ASTATE_SIZE
    m = CONTROL_COUNT
    t0 = 0.
    tf = evolution_time
    x0 = INITIAL_ASTATE
    if gate_type == xpiby2
        target_state_1 = XPIBY2_ISO_1
        target_state_2 = XPIBY2_ISO_2
    elseif gate_type == ypiby2
        target_state_1 = YPIBY2_ISO_1
        target_state_2 = YPIBY2_ISO_2
    elseif gate_type == zpiby2
        target_state_1 = ZPIBY2_ISO_1
        target_state_2 = ZPIBY2_ISO_2
    end
    xf = [
        target_state_1;
        @SVector zeros(DORDER * STATE_SIZE_ISO);
        target_state_2;
        @SVector zeros(DORDER * STATE_SIZE_ISO);
        @SVector zeros(3 * CONTROL_COUNT);
    ]
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * (1 + DORDER) * STATE_SIZE_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # control
        fill(Inf, CONTROL_COUNT);
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * (1 + DORDER) * STATE_SIZE_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # control
        fill(-Inf, CONTROL_COUNT);
    ])
    # controls start and end at 0
    x_max_boundary = [
        fill(Inf, STATE_COUNT * (1 + DORDER) * STATE_SIZE_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(Inf, CONTROL_COUNT);
    ]
    x_min_boundary = [
        fill(-Inf, STATE_COUNT * (1 + DORDER) * STATE_SIZE_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(-Inf, CONTROL_COUNT);
    ]

    model = Model(n, m)
    U0 = [SVector{m}([
        fill(1e-4, CONTROL_COUNT)
    ]) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(n, m, dt, N)

    Qs = 1e0
    Qd1s = Qd2s = Qd3s = 5e-8
    Q = Diagonal(SVector{n}([
        fill(Qs, STATE_SIZE_ISO); # state1
        fill(Qd1s, STATE_SIZE_ISO); # dstate1
        fill(Qd2s, STATE_SIZE_ISO); # d2state1
        fill(Qd3s, STATE_SIZE_ISO); # d3state1
        fill(Qs, STATE_SIZE_ISO); # state2
        fill(Qd1s, STATE_SIZE_ISO); # dstate2
        fill(Qd2s, STATE_SIZE_ISO); # d2state2
        fill(Qd3s, STATE_SIZE_ISO); # d3state2
        fill(1e0, 1); # int_control
        fill(1e0, 1); # control
        fill(1e-1, 1); # dcontrol_dt
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(1e-1, CONTROL_COUNT);
    ]))
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must statisfy conrols start and stop at 0.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must reach target state. Must have zero net flux.
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX]);
    # Must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, 1:1)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
    add_constraint!(constraints, normalization_constraint_1, 1:N)
    add_constraint!(constraints, normalization_constraint_2, 1:N)

    # Instantiate problem and solve.
    prob = Problem{RobotDynamics.RK4}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    opts = SolverOptions(verbose=VERBOSE)
    solver = AugmentedLagrangianSolver(prob, opts)
    solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
    solver.opts.constraint_tolerance_intermediate = CONSTRAINT_TOLERANCE
    Altro.solve!(solver)

    # Post-process.
    acontrols_raw = controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TO.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    cidx_arr = Array(CONTROLS_IDX)
    d2cidx_arr = Array(D2CONTROLS_IDX)
    
    # Save.
    if SAVE
        save_file_path = generate_save_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            write(save_file, "acontrols", acontrols_arr)
            write(save_file, "controls_idx", cidx_arr)
            write(save_file, "d2controls_idx", d2cidx_arr)
            write(save_file, "evolution_time", evolution_time)
            write(save_file, "astates", astates_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
        end
    end
end

