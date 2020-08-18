"""
spin12.jl - derivative robustness
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
const EXPERIMENT_NAME = "spin12"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Define the optimization.
const DT_STATIC = DT_PREF
const DT_STATIC_INV = DT_PREF_INV
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
const S1STATE1_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + STATE_SIZE_ISO
const S1STATE2_IDX = S1STATE1_IDX[end] + 1:S1STATE1_IDX[end] + STATE_SIZE_ISO
const S2STATE1_IDX = S1STATE2_IDX[end] + 1:S1STATE2_IDX[end] + STATE_SIZE_ISO
const S2STATE2_IDX = S2STATE1_IDX[end] + 1:S2STATE1_IDX[end] + STATE_SIZE_ISO
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# Specify logging.
const VERBOSE = true
const SAVE = true

# Misc. constants
EMPTY_V = []


# Define the dynamics.
struct Model{SO} <: AbstractModel
    Model(SO::Int64=0) = new{SO}()
end
RobotDynamics.state_dim(::Model{SO}) where SO = (
    ASTATE_SIZE_BASE + SO * STATE_COUNT * STATE_SIZE_ISO
)
RobotDynamics.control_dim(::Model{SO}) where SO = CONTROL_COUNT


function RobotDynamics.dynamics(model::Model{SO}, astate::StaticVector,
                                acontrols::StaticVector, time::Real) where SO
    negi_hc = (
        astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    )
    negi_s0h = FQ_NEGI_H0_ISO + negi_hc
    delta_state1 = negi_s0h * astate[STATE1_IDX]
    delta_state2 = negi_s0h * astate[STATE2_IDX]
    delta_intcontrol = astate[CONTROLS_IDX]
    delta_control = astate[DCONTROLS_IDX]
    delta_dcontrol = acontrols[D2CONTROLS_IDX]
    delta_astate = [
        delta_state1;
        delta_state2;
        delta_intcontrol;
        delta_control;
        delta_dcontrol;
    ]
    
    if SO == 2
        negi_s1h = S1FQ_NEGI_H0_ISO + negi_hc
        negi_s2h = S2FQ_NEGI_H0_ISO + negi_hc
        delta_s1state1 = negi_s1h * astate[S1STATE1_IDX]
        delta_s1state2 = negi_s1h * astate[S1STATE2_IDX]
        delta_s2state1 = negi_s2h * astate[S2STATE1_IDX]
        delta_s2state2 = negi_s2h * astate[S2STATE2_IDX]
        push!(delta_astate, delta_s1state1)
        push!(delta_astate, delta_s1state2)
        push!(delta_astate, delta_s2state1)
        push!(delta_astate, delta_s2state2)
    end
    
    return delta_astate
end


function run_traj(;gate_type=xpiby2, evolution_time=60., solver_type=alilqr,
                  postsample=false, sqrtbp=false, sample_order=0)
    dt = DT_STATIC
    N = Int(floor(evolution_time * DT_STATIC_INV)) + 1
    model = Model(sample_order)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    tf = evolution_time
    x0 = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        repeat([INITIAL_STATE1; INITIAL_STATE2], sample_order);
    ])
    
    if gate_type == xpiby2
        target_state1 = Array(XPIBY2_ISO_1)
        target_state2 = Array(XPIBY2_ISO_2)
    elseif gate_type == ypiby2
        target_state1 = Array(YPIBY2_ISO_1)
        target_state2 = Array(YPIBY2_ISO_2)
    elseif gate_type == zpiby2
        target_state1 = Array(ZPIBY2_ISO_1)
        target_state2 = Array(ZPIBY2_ISO_2)
    end
    xf = SVector{n}([
        target_state1;
        target_state2;
        zeros(3 * CONTROL_COUNT);
        repeat([target_state1; target_state2], sample_order);
    ])
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * STATE_SIZE_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, sample_order * STATE_COUNT * STATE_SIZE_ISO);
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * STATE_SIZE_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, sample_order * STATE_COUNT * STATE_SIZE_ISO);
    ])
    # controls start and end at 0
    x_max_boundary = SVector{n}([
        fill(Inf, STATE_COUNT * STATE_SIZE_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, sample_order * STATE_COUNT * STATE_SIZE_ISO);
    ])
    x_min_boundary = SVector{n}([
        fill(-Inf, STATE_COUNT * STATE_SIZE_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, sample_order * STATE_COUNT * STATE_SIZE_ISO);
    ])

    U0 = [SVector{m}([
        fill(1e-4, CONTROL_COUNT);
    ]) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    Qs = 1e0
    Qss = 5e0
    Q = Diagonal(SVector{n}([
        fill(Qs, STATE_COUNT * STATE_SIZE_ISO); # state1, state2
        fill(1e0, 1); # int_control
        fill(1e0, 1); # control
        fill(1e-1, 1); # dcontrol_dt
        fill(Qss, sample_order * STATE_COUNT * STATE_SIZE_ISO); # snstate1, snstate2
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
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # Must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
    # add_constraint!(constraints, normalization_constraint_1, 2:N-1)
    # add_constraint!(constraints, normalization_constraint_2, 2:N-1)

    # Instantiate problem and solve.
    prob = Problem{RobotDynamics.RK6}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
    opts = SolverOptions(verbose=VERBOSE)
    solver = AugmentedLagrangianSolver(prob, opts)
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
            write(save_file, "evolution_time", evolution_time)
            write(save_file, "astates", astates_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
            write(save_file, "cmax", cmax)
            write(save_file, "cmax_info", cmax_info)
            write(save_file, "dt", dt)
        end

        if postsample
            (csample, d2csample, etsample) = sample_controls(save_file_path)
            h5open(save_file_path, "r+") do save_file
                write(save_file, "controls_sample", csample)
                write(save_file, "d2controls_dt2_sample", d2csample)
                write(save_file, "evolution_time_sample", etsample)
            end
        end
    end
end

