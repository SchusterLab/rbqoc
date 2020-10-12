"""
spin11.jl - derivative robustness for the δf problem
"""

WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
using HDF5
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin11"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const ACONTROL_SIZE = CONTROL_COUNT
const INITIAL_STATE1 = [1., 0, 0, 0]
const INITIAL_STATE2 = [0., 1, 0, 0]
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const DSTATE1_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
const D2STATE1_IDX = DSTATE1_IDX[end] + 1:DSTATE1_IDX[end] + HDIM_ISO
const D3STATE1_IDX = D2STATE1_IDX[end] + 1:D2STATE1_IDX[end] + HDIM_ISO
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
struct Model{DO} <: AbstractModel
end
@inline RD.state_dim(::Model{DO}) where {DO} = (
    ASTATE_SIZE_BASE + DO * HDIM_ISO
)
@inline RD.control_dim(::Model) = ACONTROL_SIZE

# dynamics
const NEGI2_H0_ISO = 2 * NEGI_H0_ISO
const NEGI3_H0_ISO = 3 * NEGI_H0_ISO
function RD.discrete_dynamics(::Type{RK3}, model::Model{DO}, astate::SVector,
                              acontrol::SVector, time::Real, dt::Real) where {DO}
    negi_h = (
        FQ_NEGI_H0_ISO
        + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    )
    h_prop = exp(negi_h * dt)
    state1_ = astate[STATE1_IDX]
    state2_ = astate[STATE2_IDX]
    state1 =  h_prop * state1_
    state2 = h_prop * state2_
    intcontrols = astate[INTCONTROLS_IDX] + astate[CONTROLS_IDX] * dt
    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] * dt
    dcontrols = astate[DCONTROLS_IDX] + acontrol[D2CONTROLS_IDX] * dt

    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
    ]

    if DO >= 1
        dstate1_ = astate[DSTATE1_IDX]
        dstate1 = h_prop * (dstate1_ + dt * NEGI_H0_ISO * state1_)
        append!(astate_, dstate1)
    end
    if DO >= 2
        d2state1_ = astate[D2STATE1_IDX]
        d2state1 = h_prop * (d2state1_ + dt * NEGI2_H0_ISO * dstate1_)
        append!(astate_, d2state1)
    end
    if DO >= 3
        d3state1_ = astate[D3STATE1_IDX]
        d3state1 = h_prop * (d3state1_ + dt * NEGI3_H0_ISO * d2state1_)
        append!(astate_, d3state1)
    end

    return astate_
end


# main
function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=altro,
                  sqrtbp=false, derivative_order=0, integrator_type=rk3, qs=ones(7),
                  smoke_test=false, dt_inv=Int64(1e1), constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true, max_iterations=Int64(2e5),
                  max_cost_value=1e8)
    # model configuration
    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.

    # initial state
    x0 = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        zeros(derivative_order * HDIM_ISO);
    ])

    # target state
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
        zeros(derivative_order * HDIM_ISO);
    ])
    
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, derivative_order * HDIM_ISO)
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, derivative_order * HDIM_ISO)
    ])
    # control amplitude constraint at boundary
    x_max_boundary = [
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, derivative_order * HDIM_ISO)
    ]
    x_min_boundary = [
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, derivative_order * HDIM_ISO)
    ]

    # initial trajectory
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [SVector{m}(
        fill(1e-4, CONTROL_COUNT)
    ) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    # cost function
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # state1, state2
        fill(qs[2], 1); # int_control
        fill(qs[3], 1); # control
        fill(qs[4], 1); # dcontrol_dt
        fill(qs[5], eval(:($derivative_order >= 1 ? $HDIM_ISO : 0))); # dstate<1,2>
        fill(qs[6], eval(:($derivative_order >= 2 ? $HDIM_ISO : 0))); # d2state<1,2>
        fill(qs[7], eval(:($derivative_order >= 3 ? $HDIM_ISO : 0))); # d3state<1,2>
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[8], CONTROL_COUNT);
    ]))
    objective = LQRObjective(Q, R, Qf, xf, N)

    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
    # add_constraint!(constraints, normalization_constraint_1, 2:N-1)
    # add_constraint!(constraints, normalization_constraint_2, 2:N-1)

    # solve problem
    prob = Problem{IT_RDI[integrator_type]}(model, objective, constraints, x0, xf, Z, N, t0, evolution_time)
    solver = ALTROSolver(prob)
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    projected_newton = solver_type == altro ? true : false
    constraint_tolerance = solver_type == altro ? constraint_tol : al_tol
    iterations_inner = smoke_test ? 1 : 300
    iterations_outer = smoke_test ? 1 : 30
    n_steps = smoke_test ? 1 : pn_steps
    set_options!(solver, square_root=sqrtbp, constraint_tolerance=constraint_tolerance,
                 projected_newton_tolerance=al_tol, n_steps=n_steps,
                 penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
                 projected_newton=projected_newton, iterations_inner=iterations_inner,
                 iterations_outer=iterations_outer, iterations=max_iterations,
                 max_cost_value=max_cost_value)
    Altro.solve!(solver)

    # post-process
    acontrols_raw = TO.controls(solver)
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
    cmax_info = TrajectoryOptimization.findmax_violation(TO.get_constraints(solver))
    iterations_ = Altro.iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "controls_idx" => cidx_arr,
        "d2controls_dt2_idx" => d2cidx_arr,
        "evolution_time" => evolution_time,
        "astates" => astates_arr,
        "Q" => Q_arr,
        "Qf" => Qf_arr,
        "R" => R_arr,
        "cmax" => cmax,
        "cmax_info" => cmax_info,
        "dt" => dt,
        "derivative_order" => derivative_order,
        "solver_type" => Integer(solver_type),
        "sqrtbp" => Integer(sqrtbp),
        "max_penalty" => max_penalty,
        "constraint_tol" => constraint_tol,
        "al_tol" => al_tol,
        "gate_type" => Integer(gate_type),
        "save_type" => Integer(jl),
        "integrator_type" => Integer(integrator_type),
        "iterations" => iterations_,
        "max_iterations" => max_iterations,
        "pn_steps" => pn_steps,
        "max_cost_value" => max_cost_value,
    )
    
    # save
    if save
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            for key in keys(result)
                write(save_file, key, result[key])
            end
        end
        result["save_file_path"] = save_file_path
    end

    return result
end


function forward_pass(save_file_path; derivative_order=0, integrator_type=rk3)
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
        zeros(derivative_order * HDIM_ISO);
    ])
    acontrols = [SVector{m}([d2controls[i, 1],]) for i = 1:knot_count - 1]

    for i = 1:knot_count - 1
        astate = SVector{n}(RD.discrete_dynamics(rdi, model, astate, acontrols[i], time, dt))
        time = time + dt
    end

    res = Dict(
        "astate" => astate
    )

    return res
end
