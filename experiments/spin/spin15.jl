"""
spin15.jl - T1 optimized pulses
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
EXPERIMENT_NAME = "spin15"
WDIR = ENV["RBQOC_PATH"]
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Define the optimization.
CONTROL_COUNT = 1
DT_STATIC = DT_PREF
DT_STATIC_INV = DT_PREF_INV
DT_INIT = DT_PREF
DT_INIT_INV = DT_PREF_INV
# DT_INIT = 2e-2
# DT_INIT_INV = 5e1
DT_MIN = DT_INIT / 2
DT_MAX = DT_INIT * 2
CONSTRAINT_TOLERANCE = 1e-8
AL_KICKOUT_TOLERANCE = 1e-6
PN_STEPS = 5

# Define the problem.
INITIAL_STATE_1 = SA[1., 0, 0, 0]
INITIAL_STATE_2 = SA[0., 1, 0, 0]
STATE_SIZE, = size(INITIAL_STATE_1)
INITIAL_ASTATE = [
    INITIAL_STATE_1; # state_0
    INITIAL_STATE_2;
    @SVector zeros(CONTROL_COUNT); # int_control
    @SVector zeros(CONTROL_COUNT); # control
    @SVector zeros(CONTROL_COUNT); # dcontrol_dt
    @SVector zeros(1); # int_gamma
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
ZPIBY2_1 = SA[1., 0, -1, 0] / sqrt(2)
ZPIBY2_2 = SA[0., 1, 0, 1] / sqrt(2)
YPIBY2_1 = SA[1., 1, 0, 0] / sqrt(2)
YPIBY2_2 = SA[-1., 1, 0, 0] / sqrt(2)
XPIBY2_1 = SA[1., 0, 0, -1] / sqrt(2)
XPIBY2_2 = SA[0., 1, -1, 0] / sqrt(2)
# state indices
STATE_1_IDX = 1:STATE_SIZE
STATE_2_IDX = STATE_1_IDX[end] + 1:STATE_1_IDX[end] + STATE_SIZE
INT_CONTROLS_IDX = STATE_2_IDX[end] + 1:STATE_2_IDX[end] + CONTROL_COUNT
CONTROLS_IDX = INT_CONTROLS_IDX[end] + 1:INT_CONTROLS_IDX[end] + CONTROL_COUNT
DCONTROLS_DT_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
INT_GAMMA_IDX = DCONTROLS_DT_IDX[end] + 1:DCONTROLS_DT_IDX[end] + 1
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


function run_traj(;evolution_time=20., gate_type=zpiby2,
                  initial_save_file_path=nothing,
                  initial_save_type=jl, time_optimal=false,
                  solver_type=alilqr)
    # Choose dynamics
    if time_optimal
        expr = :(
        function RobotDynamics.dynamics(model::Model, astate, acontrols, time)
            negi_h = (
                FQ_NEGI_H0_ISO
                + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
            )
            delta_state_1 = negi_h * astate[STATE_1_IDX]
            delta_state_2 = negi_h * astate[STATE_2_IDX]
            delta_int_control = astate[CONTROLS_IDX]
            delta_control = astate[DCONTROLS_DT_IDX]
            delta_dcontrol_dt = acontrols[D2CONTROLS_DT2_IDX]
            delta_int_gamma = amp_t1_spline(astate[CONTROLS_IDX][1])^(-1)
            return [
                delta_state_1;
                delta_state_2;
                delta_int_control;
                delta_control;
                delta_dcontrol_dt;
                delta_int_gamma;
            ] .* acontrols[DT_IDX][1]^2
        end
        )
        eval(expr)
    else
        expr = :(
            function RobotDynamics.dynamics(model::Model, astate, acontrols, time)
            negi_h = (
                FQ_NEGI_H0_ISO
                + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
            )
            delta_state_1 = negi_h * astate[STATE_1_IDX]
            delta_state_2 = negi_h * astate[STATE_2_IDX]
            delta_int_control = astate[CONTROLS_IDX]
            delta_control = astate[DCONTROLS_DT_IDX]
            delta_dcontrol_dt = acontrols[D2CONTROLS_DT2_IDX]
            delta_int_gamma = amp_t1_spline(astate[CONTROLS_IDX][1])^(-1)
            return [
                delta_state_1;
                delta_state_2;
                delta_int_control;
                delta_control;
                delta_dcontrol_dt;
                delta_int_gamma;
            ]
        end
        )
        eval(expr)
    end
    # Convert to trajectory optimization language.
    n = ASTATE_SIZE
    t0 = 0.
    x0 = INITIAL_ASTATE
    if time_optimal
        m = CONTROL_COUNT + 1
    else
        m = CONTROL_COUNT
    end
    if gate_type == xpiby2
        target_state_1 = XPIBY2_1
        target_state_2 = XPIBY2_2
    elseif gate_type == ypiby2
        target_state_1 = YPIBY2_1
        target_state_2 = YPIBY2_2
    elseif gate_type == zpiby2
        target_state_1 = ZPIBY2_1
        target_state_2 = ZPIBY2_2
    end
    xf = [
        target_state_1;
        target_state_2;
        @SVector zeros(CONTROL_COUNT); # int_control
        @SVector zeros(CONTROL_COUNT); # control
        @SVector zeros(CONTROL_COUNT); # dcontrol_dt
        @SVector zeros(1); # int_gamma
    ]
    
    # Bound the control amplitude.
    x_max = SVector{n}([
        fill(Inf, STATE_SIZE);
        fill(Inf, STATE_SIZE);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, 1)
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_SIZE);
        fill(-Inf, STATE_SIZE);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, 1)
    ])
    # Controls start and end at 0.
    x_max_boundary = SVector{n}([
        fill(Inf, STATE_SIZE);
        fill(Inf, STATE_SIZE);
        fill(Inf, CONTROL_COUNT);
        fill(0, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, 1)
    ])
    x_min_boundary = SVector{n}([
        fill(-Inf, STATE_SIZE);
        fill(-Inf, STATE_SIZE);
        fill(-Inf, CONTROL_COUNT);
        fill(0, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, 1)
    ])
    # Bound dt.
    if time_optimal
        u_min = SVector{m}([
            fill(-Inf, CONTROL_COUNT);
            fill(sqrt(DT_MIN), 1); # dt
        ])
        u_max = SVector{m}([
            fill(Inf, CONTROL_COUNT);
            fill(sqrt(DT_MAX), 1); # dt
        ])
    else
        u_min = SVector{m}([
            fill(-Inf, CONTROL_COUNT);
        ])
        u_max = SVector{m}([
            fill(Inf, CONTROL_COUNT);
        ])
    end

    # Generate initial trajectory.
    model = Model(n, m)
    U0 = nothing
    if time_optimal
        # Default initial guess w/ optimization over dt.
        dt = 1
        N = Int(floor(evolution_time * DT_INIT_INV)) + 1
        U0 = [SVector{m}([
            fill(1e-4, CONTROL_COUNT);
            fill(DT_INIT, 1);
        ]) for k = 1:N - 1]
    else
        if initial_save_file_path == nothing
            # Default initial guess.
            dt = DT_STATIC
            N = Int(floor(evolution_time * DT_STATIC_INV)) + 1
            U0 = [SVector{m}(
                fill(1e-4, CONTROL_COUNT)
            ) for k = 1:N - 1]
        else
            # Initial guess pulled from initial_save_file_path.
            (d2controls_dt2, evolution_time) = h5open(initial_save_file_path, "r") do save_file
                 if initial_save_type == jl
                     d2controls_dt2_idx = read(save_file, "d2controls_dt2_idx")
                     d2controls_dt2 = read(save_file, "acontrols")[:, d2controls_dt2_idx]
                     evolution_time = read(save_file, "evolution_time")
                 elseif initial_save_type == samplejl
                     d2controls_dt2 = read(save_file, "d2controls_dt2_sample")
                     evolution_time = read(save_file, "evolution_time_sample")
                 end
                 return (d2controls_dt2, evolution_time)
            end
            # Without variable dts, evolution time will be a multiple of DT_STATIC.
            evolution_time = Int(floor(evolution_time * DT_STATIC_INV)) * DT_STATIC
            dt = DT_STATIC
            N = Int(floor(evolution_time * DT_STATIC_INV)) + 1
            U0 = [SVector{m}(d2controls_dt2[k, 1]) for k = 1:N-1]
        end
    end
    X0 = [SVector{n}(
        fill(NaN, n)
    ) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))


    # Define penalties.
    Q = Diagonal(SVector{n}([
        fill(1e2, STATE_SIZE); # state 0
        fill(1e2, STATE_SIZE); # state 1
        fill(5e1, CONTROL_COUNT); # int
        fill(5e1, CONTROL_COUNT); # control
        fill(1e-1, CONTROL_COUNT); # dcontrol_dt
        fill(1e7, 1); # int_gamma
    ]))
    Qf = Q * N
    if time_optimal
        R = Diagonal(SVector{m}([
            fill(1e-1, CONTROL_COUNT); # d2control_dt2
            fill(5e2, 1); # dt
        ]))
    else
        R = Diagonal(SVector{m}([
            fill(1e-1, CONTROL_COUNT); # d2control_dt2
        ]))
    end
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must statisfy conrols start and stop at 0.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must satisfy dt bound.
    dt_bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    # Must reach target state. Must have zero net flux.
    target_astate_constraint = GoalConstraint(xf, [STATE_1_IDX; STATE_2_IDX; INT_CONTROLS_IDX])
    # target_astate_constraint = GoalConstraint(xf, [STATE_1_IDX; STATE_2_IDX])
    # Must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE_1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE_2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
    add_constraint!(constraints, normalization_constraint_1, 2:N-1)
    add_constraint!(constraints, normalization_constraint_2, 2:N-1)
    if time_optimal
        add_constraint!(constraints, dt_bnd, 1:N-1)
    end

    # Instantiate problem and solve.
    prob = Problem{RobotDynamics.RK4}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
    solver = nothing
    opts = SolverOptions(verbose=VERBOSE)
    if solver_type == alilqr
        solver = AugmentedLagrangianSolver(prob, opts)
        solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.opts.constraint_tolerance_intermediate = CONSTRAINT_TOLERANCE
        # solver.opts.penalty_initial = AL_INITIAL_PENALTY
    elseif solver_type == altro
        solver = ALTROSolver(prob, opts)
        solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.solver_al.opts.constraint_tolerance = AL_KICKOUT_TOLERANCE
        solver.solver_al.opts.constraint_tolerance_intermediate = AL_KICKOUT_TOLERANCE
        solver.solver_pn.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.solver_pn.opts.n_steps = PN_STEPS
        # solver.solver_al.opts.penalty_initial = AL_INITIAL_PENALTY
    end
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
    d2cdt2idx_arr = Array(D2CONTROLS_DT2_IDX)
    dtidx_arr = Array(DT_IDX)
    # Square the dts.
    if time_optimal
        acontrols_arr[:, DT_IDX] = acontrols_arr[:, DT_IDX] .^2
    end
    cmax = TrajectoryOptimization.max_violation(solver)
    cmax_info = TrajectoryOptimization.findmax_violation(get_constraints(solver))
    
    # Save.
    if SAVE
        save_file_path = generate_save_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            write(save_file, "acontrols", acontrols_arr)
            write(save_file, "controls_idx", cidx_arr)
            write(save_file, "d2controls_dt2_idx", d2cdt2idx_arr)
            write(save_file, "dt_idx", dtidx_arr)
            write(save_file, "evolution_time", evolution_time)
            write(save_file, "astates", astates_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
            write(save_file, "solver_type", Integer(solver_type))
            write(save_file, "cmax", cmax)
            write(save_file, "cmax_info", cmax_info)
        end
        if time_optimal
            # Sample the important metrics.
            (controls_sample, d2controls_dt2_sample, evolution_time_sample) = sample_controls(save_file_path)
            h5open(save_file_path, "r+") do save_file
                write(save_file, "controls_sample", controls_sample)
                write(save_file, "d2controls_dt2_sample", d2controls_dt2_sample)
                write(save_file, "evolution_time_sample", evolution_time_sample)
            end
        end
    end
end
