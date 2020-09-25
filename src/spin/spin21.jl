"""
spin21.jl - derivative robustness for the empty problem with simultaneous errors
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
const EXPERIMENT_NAME = "spin21"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Define the optimization.
const CONTROL_COUNT = 1
const PN_STEPS = 2
const MAX_COST_VALUE = 1e12
const ILQR_DJ_TOL = 1e-4

# Define the problem.
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * STATE_SIZE_ISO + 1 * CONTROL_COUNT
const INITIAL_STATE1 = [1., 0, 0, 0]
const INITIAL_STATE2 = [0., 1, 0, 0]
# state indices
const STATE1_IDX = 1:STATE_SIZE_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + STATE_SIZE_ISO
const CONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
# const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const D1STATE1_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + STATE_SIZE_ISO
const D1STATE2_IDX = D1STATE1_IDX[end] + 1:D1STATE1_IDX[end] + STATE_SIZE_ISO
const D2STATE1_IDX = D1STATE2_IDX[end] + 1:D1STATE2_IDX[end] + STATE_SIZE_ISO
const D2STATE2_IDX = D2STATE1_IDX[end] + 1:D2STATE1_IDX[end] + STATE_SIZE_ISO
# control indices
const DCONTROLS_IDX = 1:CONTROL_COUNT

# Specify logging.
const VERBOSE = true
const SAVE = true

# Misc. constants
EMPTY_V = []


# Define the dynamics.
struct Model{DO} <: AbstractModel
    Model(DO::Int64=0) = new{DO}()
end
RobotDynamics.state_dim(::Model{DO}) where DO = (
    ASTATE_SIZE_BASE + DO * STATE_COUNT * STATE_SIZE_ISO
)
RobotDynamics.control_dim(::Model{DO}) where DO = CONTROL_COUNT


function RobotDynamics.dynamics(model::Model{DO}, astate::StaticVector,
                                acontrols::StaticVector, time::Real) where DO
    negi_h = (
        astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    )
    delta_state1 = negi_h * astate[STATE1_IDX]
    delta_state2 = negi_h * astate[STATE2_IDX]
    delta_control = acontrols[DCONTROLS_IDX]
    
    if DO == 1
        delta_d11state1 = NEGI_H0_ISO * astate[STATE1_IDX] + negi_h * astate[D1STATE1_IDX]
        delta_d11state2 = NEGI_H0_ISO * astate[STATE2_IDX] + negi_h * astate[D1STATE2_IDX]
        delta_d12state1 = NEGI_H0_ISO * astate[STATE1_IDX] + negi_h * astate[D1STATE1_IDX]
        delta_d12state2 = NEGI_H0_ISO * astate[STATE2_IDX] + negi_h * astate[D1STATE2_IDX]
        delta_astate = [
            delta_state1;
            delta_state2;
            delta_control;
            delta_d11state1;
            delta_d11state2;
            delta_d12state1;
            delta_d12state2;
        ]
    elseif DO == 2
        delta_d1state1 = NEGI_H0_ISO * astate[STATE1_IDX] + negi_h * astate[D1STATE1_IDX]
        delta_d1state2 = NEGI_H0_ISO * astate[STATE2_IDX] + negi_h * astate[D1STATE2_IDX]
        delta_d2state1 = 2 * NEGI_H0_ISO * astate[D1STATE1_IDX] + negi_h * astate[D2STATE1_IDX]
        delta_d2state2 = 2 * NEGI_H0_ISO * astate[D1STATE2_IDX] + negi_h * astate[D2STATE2_IDX]
        delta_astate = [
            delta_state1;
            delta_state2;
            delta_control;
            delta_d1state1;
            delta_d1state2;
            delta_d2state1;
            delta_d2state2;
        ]
    else
        delta_astate = [
            delta_state1;
            delta_state2;
            delta_control;
        ]
    end

    return delta_astate
end


function run_traj(;gate_type=xpi, evolution_time=10., solver_type=alilqr,
                  postsample=false, initial_save_file_path=nothing,
                  initial_save_type=jl, sqrtbp=false, derivative_order=0,
                  integrator_type=rk6, max_penalty=1e11, qs=nothing,
                  smoke_test=false, dt=2.5e-3, al_kickout_tol=1e-9,
                  constraint_tol=1e-10)
    model = Model(derivative_order)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    x0 = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(1 * CONTROL_COUNT);
        zeros(derivative_order * STATE_COUNT * STATE_SIZE_ISO);
    ])
    if gate_type == xpi
        target_state1 = Array(XPI_ISO_1)
        target_state2 = Array(XPI_ISO_2)
    end
    xf = SVector{n}([
        target_state1;
        target_state2;
        zeros(1 * CONTROL_COUNT);
        zeros(derivative_order * STATE_COUNT * STATE_SIZE_ISO);
    ])
    
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * STATE_SIZE_ISO);
        fill(MAX_CONTROL_NORM_0, 1); # control
        fill(Inf, derivative_order * STATE_COUNT * STATE_SIZE_ISO)
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * STATE_SIZE_ISO);
        fill(-MAX_CONTROL_NORM_0, 1); # control
        fill(-Inf, derivative_order * STATE_COUNT * STATE_SIZE_ISO)
    ])
    
    N = Int(floor(evolution_time / dt)) + 1
    U0 = [SVector{m}(
        fill(1e-4, CONTROL_COUNT)
    ) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    if isnothing(qs)
        qs = ones(6)
    end
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * STATE_SIZE_ISO); # state1, state2
        fill(qs[2], 1); # control
        fill(qs[3], eval(:($derivative_order >= 1 ? $STATE_COUNT * $STATE_SIZE_ISO : 0))); # d1state<1,2>
        fill(qs[4], eval(:($derivative_order >= 2 ? $STATE_COUNT * $STATE_SIZE_ISO : 0))); # d2state<1,2>
        fill(qs[5], eval(:($derivative_order >= 3 ? $STATE_COUNT * $STATE_SIZE_ISO : 0))); # d3state<1,2>
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[6], CONTROL_COUNT);
    ]))
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must reach target state. Must have zero net flux.
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX])
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, target_astate_constraint, N:N);

    # Instantiate problem and solve.
    prob = Problem{IT_RDI[integrator_type]}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
    opts = SolverOptions(verbose=VERBOSE)
    solver = AugmentedLagrangianSolver(prob, opts)
    if solver_type == alilqr
        solver = AugmentedLagrangianSolver(prob, opts)
        solver.solver_uncon.opts.square_root = sqrtbp
        solver.solver_uncon.opts.max_cost_value = MAX_COST_VALUE
        solver.opts.constraint_tolerance = al_kickout_tol
        solver.opts.constraint_tolerance_intermediate = al_kickout_tol
        solver.opts.cost_tolerance_intermediate = ILQR_DJ_TOL
        solver.opts.penalty_max = max_penalty
        if smoke_test
            solver.opts.iterations = 1
            solver.solver_uncon.opts.iterations = 1
        end
    elseif solver_type == altro
        solver = ALTROSolver(prob, opts)
        solver.opts.constraint_tolerance = constraint_tol
        solver.solver_al.solver_uncon.opts.square_root = sqrtbp
        solver.solver_al.solver_uncon.opts.max_cost_value = MAX_COST_VALUE
        solver.solver_al.opts.constraint_tolerance = al_kickout_tol
        solver.solver_al.opts.constraint_tolerance_intermediate = al_kickout_tol
        solver.solver_al.opts.cost_tolerance_intermediate = ILQR_DJ_TOL
        solver.solver_al.opts.penalty_max = max_penalty
        solver.solver_pn.opts.constraint_tolerance = constraint_tol
        solver.solver_pn.opts.n_steps = PN_STEPS
        if smoke_test
            solver.solver_al.opts.iterations = 1
            solver.solver_al.solver_uncon.opts.iterations = 1
            solver.solver_pn.opts.n_steps = 1
        end
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
    cmax = TrajectoryOptimization.max_violation(solver)
    cmax_info = TrajectoryOptimization.findmax_violation(get_constraints(solver))
    iterations_ = iterations(solver)
    
    # Save.
    if SAVE
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            write(save_file, "acontrols", acontrols_arr)
            write(save_file, "controls_idx", cidx_arr)
            write(save_file, "evolution_time", evolution_time)
            write(save_file, "astates", astates_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
            write(save_file, "cmax", cmax)
            write(save_file, "cmax_info", cmax_info)
            write(save_file, "dt", dt)
            write(save_file, "derivative_order", derivative_order)
            write(save_file, "solver_type", Integer(solver_type))
            write(save_file, "sqrtbp", Integer(sqrtbp))
            write(save_file, "max_penalty", max_penalty)
            write(save_file, "constraint_tol", constraint_tol)
            write(save_file, "al_kickout_tol", al_kickout_tol)
            write(save_file, "ilqr_dj_tol", ILQR_DJ_TOL)
            write(save_file, "integrator_type", Integer(integrator_type))
            write(save_file, "gate_type", Integer(gate_type))
            write(save_file, "save_type", Integer(jl))
            write(save_file, "iterations", iterations_)
            write(save_file, "max_cost_value", MAX_COST_VALUE)
        end
    end
end
