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

WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin11"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const INITIAL_STATE1 = [1., 0, 0, 0]
const INITIAL_STATE2 = [0., 1, 0, 0]
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const DSTATE1_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
const DSTATE2_IDX = DSTATE1_IDX[end] + 1:DSTATE1_IDX[end] + HDIM_ISO
const D2STATE1_IDX = DSTATE2_IDX[end] + 1:DSTATE2_IDX[end] + HDIM_ISO
const D2STATE2_IDX = D2STATE1_IDX[end] + 1:D2STATE1_IDX[end] + HDIM_ISO
const D3STATE1_IDX = D2STATE2_IDX[end] + 1:D2STATE2_IDX[end] + HDIM_ISO
const D3STATE2_IDX = D3STATE1_IDX[end] + 1:D3STATE1_IDX[end] + HDIM_ISO
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
struct Model{DO} <: AbstractModel
    Model(DO::Int64=0) = new{DO}()
end
RobotDynamics.state_dim(::Model{DO}) where {DO} = (
    ASTATE_SIZE_BASE + DO * STATE_COUNT * HDIM_ISO
)
RobotDynamics.control_dim(::Model{DO}) where {DO} = CONTROL_COUNT

# dynamics
abstract type RobotDynamics.EM <: RobotDynamics.Explicit end

function RobotDynamics.discrete_dynamics(::Type{EM}, model::Model{DO}, astate::StaticVector,
                                         acontrols::StaticVector, time::Real, dt::Real) where {DO, MM}
    negi_h = (
        FQ_NEGI_H0_ISO
        + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    )
    negi_h_propagator = expm(negi_h * dt)
    state1 = astate[STATE1_IDX] + negi_h_propagator * astate[STATE1_IDX]
    state2 = astate[STATE2_IDX] + negi_h_propagator * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX] + astate[CONTROLS_IDX] * dt
    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] * dt
    dcontrols = astate[DCONTROLS_IDX] + acontrols[D2CONTROLS_IDX] * dt

    astate_ = [
        state1;
        state2;
        intcontrols;
        controls;
        dcontrols;
    ]
    
    # if DO == 1
    #     delta_dstate1 = NEGI_H0_ISO * astate[STATE1_IDX] + negi_h * astate[DSTATE1_IDX]
    #     delta_dstate2 = NEGI_H0_ISO * astate[STATE2_IDX] + negi_h * astate[DSTATE2_IDX]
    #     delta_astate = [
    #         delta_state1;
    #         delta_state2;
    #         delta_intcontrol;
    #         delta_control;
    #         delta_dcontrol;
    #         delta_dstate1;
    #         delta_dstate2;
    #     ]
    # elseif DO == 2
    #     delta_dstate1 = NEGI_H0_ISO * astate[STATE1_IDX] + negi_h * astate[DSTATE1_IDX]
    #     delta_dstate2 = NEGI_H0_ISO * astate[STATE2_IDX] + negi_h * astate[DSTATE2_IDX]
    #     delta_d2state1 = 2 * NEGI_H0_ISO * astate[DSTATE1_IDX] + negi_h * astate[D2STATE1_IDX]
    #     delta_d2state2 = 2 * NEGI_H0_ISO * astate[DSTATE2_IDX] + negi_h * astate[D2STATE2_IDX]
    #     delta_astate = [
    #         delta_state1;
    #         delta_state2;
    #         delta_intcontrol;
    #         delta_control;
    #         delta_dcontrol;
    #         delta_dstate1;
    #         delta_dstate2;
    #         delta_d2state1;
    #         delta_d2state2;
    #     ]
    # elseif DO == 3
    #     delta_dstate1 = NEGI_H0_ISO * astate[STATE1_IDX] + negi_h * astate[DSTATE1_IDX]
    #     delta_dstate2 = NEGI_H0_ISO * astate[STATE2_IDX] + negi_h * astate[DSTATE2_IDX]
    #     delta_d2state1 = 2 * NEGI_H0_ISO * astate[DSTATE1_IDX] + negi_h * astate[D2STATE1_IDX]
    #     delta_d2state2 = 2 * NEGI_H0_ISO * astate[DSTATE2_IDX] + negi_h * astate[D2STATE2_IDX]
    #     delta_d3state1 = 3 * NEGI_H0_ISO * astate[D2STATE1_IDX] + negi_h * astate[D3STATE1_IDX]
    #     delta_d3state2 = 3 * NEGI_H0_ISO * astate[D2STATE2_IDX] + negi_h * astate[D3STATE2_IDX]
    #     delta_astate = [
    #         delta_state1;
    #         delta_state2;
    #         delta_intcontrol;
    #         delta_control;
    #         delta_dcontrol;
    #         delta_dstate1;
    #         delta_dstate2;
    #         delta_d2state1;
    #         delta_d2state2;
    #         delta_d3state1;
    #         delta_d3state2;
    #     ]
    # else
    #     delta_astate = [
    #         delta_state1;
    #         delta_state2;
    #         delta_intcontrol;
    #         delta_control;
    #         delta_dcontrol;
    #     ]
    # end

    return astate_
    
end


function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=alilqr,
                  initial_save_file_path=nothing,
                  sqrtbp=false, derivative_order=0,
                  integrator_type=rk6, qs=nothing,
                  smoke_test=false, dt_inv=Int64(2e2), constraint_tol=1e-8, al_tol=1e-7,
                  pn_steps=2, max_penalty=1e11, ilqr_dj_tol=1e-4, verbose=true,
                  save=true)
    model = Model(derivative_order)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    x0 = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        zeros(derivative_order * STATE_COUNT * HDIM_ISO);
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
        zeros(derivative_order * STATE_COUNT * HDIM_ISO);
    ])
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, derivative_order * STATE_COUNT * HDIM_ISO)
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, derivative_order * STATE_COUNT * HDIM_ISO)
    ])
    # controls start and end at 0
    x_max_boundary = [
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, derivative_order * STATE_COUNT * HDIM_ISO)
    ]
    x_min_boundary = [
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, derivative_order * STATE_COUNT * HDIM_ISO)
    ]
    
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [SVector{m}(
        fill(1e-4, CONTROL_COUNT)
    ) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    if isnothing(qs)
        qs = fill(1, 7)
    end
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # state1, state2
        fill(qs[2], 1); # int_control
        fill(qs[3], 1); # control
        fill(qs[4], 1); # dcontrol_dt
        fill(qs[5], eval(:($derivative_order >= 1 ? $STATE_COUNT * $HDIM_ISO : 0))); # dstate<1,2>
        fill(qs[6], eval(:($derivative_order >= 2 ? $STATE_COUNT * $HDIM_ISO : 0))); # d2state<1,2>
        fill(qs[7], eval(:($derivative_order >= 3 ? $STATE_COUNT * $HDIM_ISO : 0))); # d3state<1,2>
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[8], CONTROL_COUNT);
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

    # Instantiate the problem and solve it.
    prob = Problem{IT_RDI[integrator_type]}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
    opts = SolverOptions(verbose=verbose)
    solver = AugmentedLagrangianSolver(prob, opts)
    if solver_type == alilqr
        solver = AugmentedLagrangianSolver(prob, opts)
        solver.solver_uncon.opts.square_root = sqrtbp
        solver.opts.constraint_tolerance = al_tol
        solver.opts.constraint_tolerance_intermediate = al_tol
        solver.opts.cost_tolerance_intermediate = ilqr_dj_tol
        solver.opts.penalty_max = max_penalty
        if smoke_test
            solver.opts.iterations = 1
            solver.solver_uncon.opts.iterations = 1
        end
    elseif solver_type == altro
        solver = ALTROSolver(prob, opts)
        solver.opts.constraint_tolerance = constraint_tol
        solver.solver_al.solver_uncon.opts.square_root = sqrtbp
        solver.solver_al.opts.constraint_tolerance = al_tol
        solver.solver_al.opts.constraint_tolerance_intermediate = al_tol
        solver.solver_al.opts.cost_tolerance_intermediate = ilqr_dj_tol
        solver.solver_al.opts.penalty_max = max_penalty
        solver.solver_pn.opts.constraint_tolerance = constraint_tol
        solver.solver_pn.opts.n_steps = pn_steps
        if smoke_test
            solver.solver_al.opts.iterations = 1
            solver.solver_al.solver_uncon.opts.iterations = 1
            solver.solver_pn.opts.n_steps = 1
        end
    end
    Altro.solve!(solver)

    # post-process
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
    
    # save
    if save
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
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
            write(save_file, "derivative_order", derivative_order)
            write(save_file, "solver_type", Integer(solver_type))
            write(save_file, "sqrtbp", Integer(sqrtbp))
            write(save_file, "max_penalty", max_penalty)
            write(save_file, "ctol", constraint_tol)
            write(save_file, "alko", al_tol)
            write(save_file, "ilqr_dj_tol", ilqr_dj_tol)
            write(save_file, "pn_steps", pn_steps)
            write(save_file, "integrator_type", Integer(integrator_type))
            write(save_file, "gate_type", Integer(gate_type))
            write(save_file, "save_type", Integer(jl))
        end
    end
end


function forward_pass(save_file_path; derivative_order=0, integrator_type=rk6)
    (evolution_time, d2controls, dt
     ) = h5open(save_file_path, "r+") do save_file
         save_type = SaveType(read(save_file, "save_type"))
         if save_type == jl
             d2controls_idx = read(save_file, "d2controls_dt2_idx")
             acontrols = read(save_file, "acontrols")
             d2controls = acontrols[:, d2controls_idx]
             dt = read(save_file, "dt")
             evolution_time = read(save_file, "evolution_time")
         elseif save_type == samplejl
             d2controls = read(save_file, "d2controls_dt2_sample")
             dt = DT_PREF
             ets = read(save_file, "evolution_time_sample")
             evolution_time = Integer(floor(ets / dt)) * dt
         end
         return (evolution_time, d2controls, dt)
     end
    rdi = IT_RDI[integrator_type]
    knot_count = Integer(floor(evolution_time / dt))
    
    model = Model(derivative_order)
    n = state_dim(model)
    m = control_dim(model)
    time = 0.
    astate = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        zeros(derivative_order * STATE_COUNT * HDIM_ISO);
    ])
    acontrols = [SVector{m}([d2controls[i, 1],]) for i = 1:knot_count - 1]

    for i = 1:knot_count - 1
        astate = RobotDynamics.discrete_dynamics(rdi, model, astate, acontrols[i], time, dt)
        time = time + dt
    end

    res = Dict(
        "astate" => astate
    )

    return res
end
