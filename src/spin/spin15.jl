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

WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin15"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Define the optimization.
const DT_STATIC = DT_PREF
const DT_STATIC_INV = DT_PREF_INV
const DT_INIT = 5e-3
const DT_INIT_INV = 2e2
const DT_MIN = DT_INIT / 2
const DT_MAX = DT_INIT * 2
const CONSTRAINT_TOLERANCE = 1e-8
const AL_KICKOUT_TOLERANCE = 1e-7
const PN_STEPS = 5
const MAX_PENALTY = 1e10
const ILQR_DJ_TOL = 1e-4

# Define the problem.
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * STATE_SIZE_ISO + 3 * CONTROL_COUNT
const INITIAL_STATE1 = [1., 0, 0, 0]
const INITIAL_STATE2 = [0., 1, 0, 0]
# state indices
const STATE1_IDX = 1:STATE_SIZE_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + STATE_SIZE_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const INTGAMMA_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + 1
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT
const DT_IDX = D2CONTROLS_IDX[end] + 1:D2CONTROLS_IDX[end] + 1


# Specify logging.
const VERBOSE = true
const SAVE = true


# Misc. constants
const EMPTY_V = []


# Define the model and dynamics.
struct Model{DA, TO} <: AbstractModel
    Model(DA::Bool=true, TO::Bool=true) = new{DA, TO}()
end
RobotDynamics.state_dim(::Model{false, TO}) where TO = ASTATE_SIZE_NODA
RobotDynamics.state_dim(::Model{true, TO}) where TO = ASTATE_SIZE_NODA + 1
RobotDynamics.control_dim(::Model{DA, false}) where DA = CONTROL_COUNT
RobotDynamics.control_dim(::Model{DA, true}) where DA = CONTROL_COUNT + 1


# base dynamics
function RobotDynamics.dynamics(model::Model{DA, TO}, astate::StaticVector,
                                acontrols::StaticVector, time::Real) where {DA, TO}
    negi_h = (
        FQ_NEGI_H0_ISO
        + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    )
    delta_state1 = negi_h * astate[STATE1_IDX]
    delta_state2 = negi_h * astate[STATE2_IDX]
    delta_intcontrol = astate[CONTROLS_IDX]
    delta_control = astate[DCONTROLS_IDX]
    delta_dcontrol = acontrols[D2CONTROLS_IDX]
    dastate = [
        delta_state1;
        delta_state2;
        delta_intcontrol;
        delta_control;
        delta_dcontrol;
    ]
    if DA
        push!(dastate, amp_t1_reduced_spline(astate[CONTROLS_IDX][1])^(-1))
    end
    if TO
        dastate = dastate * acontrols[DT_IDX][1]^2
    end
    
    return dastate
end


function run_traj(;evolution_time=60., gate_type=zpiby2,
                  initial_save_file_path=nothing,
                  initial_save_type=jl, time_optimal=false,
                  decay_aware=false,
                  solver_type=alilqr, sqrtbp=false)
    # Convert to trajectory optimization language.
    model = Model(decay_aware, time_optimal)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    x0 = SVector{n}([
        INITIAL_STATE1; # state1
        INITIAL_STATE2; # state2
        zeros(CONTROL_COUNT); # int_control
        zeros(CONTROL_COUNT); # control
        zeros(CONTROL_COUNT); # dcontrol_dt
        eval(:($decay_aware ? zeros(1) : EMPTY_V))
    ])

    if gate_type == xpiby2
        target_state_1 = Array(XPIBY2_ISO_1)
        target_state_2 = Array(XPIBY2_ISO_2)
    elseif gate_type == ypiby2
        target_state_1 = Array(YPIBY2_ISO_1)
        target_state_2 = Array(YPIBY2_ISO_2)
    elseif gate_type == zpiby2
        target_state_1 = Array(ZPIBY2_ISO_1)
        target_state_2 = Array(ZPIBY2_ISO_2)
    end
    xf = SVector{n}([
        target_state_1;
        target_state_2;
        zeros(CONTROL_COUNT); # int_control
        zeros(CONTROL_COUNT); # control
        zeros(CONTROL_COUNT); # dcontrol_dt
        eval(:($decay_aware ? zeros(1) : EMPTY_V))
    ])
    
    # Bound the control amplitude.
    x_max = SVector{n}([
        fill(Inf, STATE_SIZE_ISO);
        fill(Inf, STATE_SIZE_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
        eval(:($decay_aware ? fill(Inf, 1) : EMPTY_V))
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_SIZE_ISO);
        fill(-Inf, STATE_SIZE_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
        eval(:($decay_aware ? fill(-Inf, 1) : EMPTY_V))
    ])

    # Controls start and end at 0.
    x_max_boundary = SVector{n}([
        fill(Inf, STATE_SIZE_ISO);
        fill(Inf, STATE_SIZE_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
        eval(:($decay_aware ? fill(Inf, 1) : EMPTY_V))
    ])
    x_min_boundary = SVector{n}([
        fill(-Inf, STATE_SIZE_ISO);
        fill(-Inf, STATE_SIZE_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
        eval(:($decay_aware ? fill(-Inf, 1) : EMPTY_V))
    ])

    # Bound dt.
    u_max = SVector{m}([
        fill(Inf, CONTROL_COUNT);
        eval(:($time_optimal ? fill(sqrt(DT_MAX), 1) : EMPTY_V)); # dt
    ])
    u_min = SVector{m}([
        fill(-Inf, CONTROL_COUNT);
        eval(:($time_optimal ? fill(sqrt(DT_MIN), 1) : EMPTY_V)); # dt
    ])

    # Generate initial trajectory.
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
        fill(1e0, STATE_SIZE_ISO); # state 0
        fill(1e0, STATE_SIZE_ISO); # state 1
        fill(1e0, CONTROL_COUNT); # int
        fill(1e0, CONTROL_COUNT); # control
        fill(1e-1, CONTROL_COUNT); # dcontrol_dt
        eval(:($decay_aware ? fill(1e-1, 1) : EMPTY_V)); # int_gamma
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(1e-1, CONTROL_COUNT); # d2control_dt2
        eval(:($time_optimal ? fill(5e3, 1) : EMPTY_V)); # dt
    ]))
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Controls must stop at zero.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must satisfy dt bound.
    dt_bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    # States must reach target. Controls must have zero net flux. 
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # States must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    # add_constraint!(constraints, normalization_constraint_1, 2:N-1)
    # add_constraint!(constraints, normalization_constraint_2, 2:N-1)
    if time_optimal
        add_constraint!(constraints, dt_bnd, 1:N-1)
    end

    # Instantiate problem and solve.
    prob = Problem{RobotDynamics.RK6}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
    opts = SolverOptions(verbose=VERBOSE)
    if solver_type == alilqr
        solver = AugmentedLagrangianSolver(prob, opts)
        solver.solver_uncon.opts.square_root = sqrtbp
        solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.opts.constraint_tolerance_intermediate = CONSTRAINT_TOLERANCE
        solver.opts.cost_tolerance_intermediate = ILQR_DJ_TOL
        solver.opts.penalty_max = MAX_PENALTY
    elseif solver_type == altro
        solver = ALTROSolver(prob, opts)
        solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.solver_al.solver_uncon.opts.square_root = sqrtbp
        solver.solver_al.opts.constraint_tolerance = AL_KICKOUT_TOLERANCE
        solver.solver_al.opts.constraint_tolerance_intermediate = AL_KICKOUT_TOLERANCE
        solver.solver_al.opts.cost_tolerance_intermediate = ILQR_DJ_TOL
        solver.solver_al.opts.penalty_max = MAX_PENALTY
        solver.solver_pn.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.solver_pn.opts.n_steps = PN_STEPS
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
    d2cidx_arr = Array(D2CONTROLS_IDX)
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
            write(save_file, "d2controls_dt2_idx", d2cidx_arr)
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
