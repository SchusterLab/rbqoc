"""
spin17.jl - derivative robustness for the δa problem
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
const EXPERIMENT_NAME = "spin17"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const ACONTROL_SIZE = CONTROL_COUNT
# state indices
const STATE1_IDX = SVector{HDIM_ISO}(1:HDIM_ISO)
const STATE2_IDX = SVector{HDIM_ISO}(STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO)
const INTCONTROLS_IDX = SVector{CONTROL_COUNT}(STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT)
const CONTROLS_IDX = SVector{CONTROL_COUNT}(INTCONTROLS_IDX[end]
                                            + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT)
const DCONTROLS_IDX = SVector{CONTROL_COUNT}(CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT)
const DSTATE1_IDX = SVector{HDIM_ISO}(DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO)
const D2STATE1_IDX = SVector{HDIM_ISO}(DSTATE1_IDX[end] + 1:DSTATE1_IDX[end] + HDIM_ISO)
# const STATE3_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
# const STATE4_IDX = STATE3_IDX[end] + 1:STATE3_IDX[end] + HDIM_ISO
# const DSTATE1_IDX = STATE4_IDX[end] + 1:STATE4_IDX[end] + HDIM_ISO
# const DSTATE2_IDX = DSTATE1_IDX[end] + 1:DSTATE1_IDX[end] + HDIM_ISO
# const DSTATE3_IDX = DSTATE2_IDX[end] + 1:DSTATE2_IDX[end] + HDIM_ISO
# const DSTATE4_IDX = DSTATE3_IDX[end] + 1:DSTATE3_IDX[end] + HDIM_ISO
# const D2STATE1_IDX = DSTATE4_IDX[end] + 1:DSTATE4_IDX[end] + HDIM_ISO
# const D2STATE2_IDX = D2STATE1_IDX[end] + 1:D2STATE1_IDX[end] + HDIM_ISO
# const D2STATE3_IDX = D2STATE2_IDX[end] + 1:D2STATE2_IDX[end] + HDIM_ISO
# const D2STATE4_IDX = D2STATE3_IDX[end] + 1:D2STATE3_IDX[end] + HDIM_ISO
# control indices
const D2CONTROLS_IDX = SVector{CONTROL_COUNT}(1:CONTROL_COUNT)

# model
struct Model{DO} <: AbstractModel
end
@inline RD.state_dim(::Model{DO}) where {DO} = ASTATE_SIZE_BASE + DO * HDIM_ISO
@inline RD.control_dim(::Model) = ACONTROL_SIZE
# @inline RD.state_dim(::Model{DO}) where {DO} = (
#     ASTATE_SIZE_BASE + (SAMPLE_STATE_COUNT - STATE_COUNT) * HDIM_ISO + DO * SAMPLE_STATE_COUNT * HDIM_ISO
# )


# dynamics
const NEGI2_H1_ISO = 2 * NEGI_H1_ISO
function RD.discrete_dynamics(::Type{RK3}, model::Model{DO}, astate::SVector,
                              acontrol::SVector, time::Real, dt::Real) where {DO}
    negi_h = (
        FQ_NEGI_H0_ISO
        + astate[CONTROLS_IDX[1]] * NEGI_H1_ISO
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
        dstate1 = h_prop * (dstate1_ + dt * NEGI_H1_ISO * state1_)
        astate_ = [astate_; dstate1]
        # state3_ = astate[STATE3_IDX]
        # state3 = h_prop * state3_
        # state4_ = astate[STATE4_IDX]
        # state4 = h_prop * state4_
        # dstate1_ = astate[DSTATE1_IDX]
        # dstate1 = h_prop * (dstate1_ + dt * NEGI_H1_ISO * state1_)
        # dstate2_ = astate[DSTATE2_IDX]
        # dstate2 = h_prop * (dstate2_ + dt * NEGI_H1_ISO * state2_)
        # dstate3_ = astate[DSTATE3_IDX]
        # dstate3 = h_prop * (dstate3_ + dt * NEGI_H1_ISO * state3_)
        # dstate4_ = astate[DSTATE4_IDX]
        # dstate4 = h_prop * (dstate4_ + dt * NEGI_H1_ISO * state4_)
        # append!(astate_, [state3; state4; dstate1; dstate2; dstate3; dstate4])
    end
    if DO >= 2
        d2state1_ = astate[D2STATE1_IDX]
        d2state1 = h_prop * (d2state1_ + dt * NEGI2_H1_ISO * dstate1_)
        astate_ = [astate_; d2state1]
        # d2state1_ = astate[D2STATE1_IDX]
        # d2state1 = h_prop * (d2state1_ + dt * NEGI2_H1_ISO * dstate1_)
        # d2state2_ = astate[D2STATE2_IDX]
        # d2state2 = h_prop * (d2state2_ + dt * NEGI2_H1_ISO * dstate2_)
        # d2state3_ = astate[D2STATE3_IDX]
        # d2state3 = h_prop * (d2state3_ + dt * NEGI2_H1_ISO * dstate3_)
        # d2state4_ = astate[D2STATE4_IDX]
        # d2state4 = h_prop * (d2state4_ + dt * NEGI2_H1_ISO * dstate4_)
        # append!(astate_, [d2state1; d2state2; d2state3; d2state4])
    end

    return astate_
end


# main
function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=altro,
                  sqrtbp=false, derivative_order=0, integrator_type=rk3, qs=ones(7),
                  smoke_test=false, dt_inv=Int64(1e1), constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true, max_iterations=Int64(2e5))
    # model configuration
    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.

    # initial state, target state
    x0 = zeros(n)
    xf = zeros(n)
    gate = GT_GATE_ISO[gate_type]
    x0[STATE1_IDX] = IS1_ISO_
    x0[STATE2_IDX] = IS2_ISO_
    # x0[STATE3_IDX] = IS3_ISO_
    # x0[STATE4_IDX] = IS4_ISO_
    xf[STATE1_IDX] = gate * IS1_ISO_
    xf[STATE2_IDX] = gate * IS2_ISO_
    # xf[STATE3_IDX] = gate * IS3_ISO_
    # xf[STATE4_IDX] = gate * IS4_ISO_
    x0 = SVector{n}(x0)
    xf = SVector{n}(xf)
    
    # control amplitude constraint
    x_max = fill(Inf, n)
    x_max[CONTROLS_IDX] .= MAX_CONTROL_NORM_0
    x_max = SVector{n}(x_max)
    x_min = fill(-Inf, n)
    x_min[CONTROLS_IDX] .= -MAX_CONTROL_NORM_0
    x_min = SVector{n}(x_min)
    
    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n)
    x_max_boundary[CONTROLS_IDX] .= 0
    x_max_boundary = SVector{n}(x_max_boundary)
    x_min_boundary = fill(-Inf, n)
    x_min_boundary[CONTROLS_IDX] .= 0
    x_min_boundary = SVector{n}(x_min_boundary)

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
        fill(qs[1], STATE_COUNT * HDIM_ISO); # ψ1, ψ2
        fill(qs[2], CONTROL_COUNT); # ∫a
        fill(qs[3], CONTROL_COUNT); # a
        fill(qs[4], CONTROL_COUNT); # ∂a
        fill(qs[5], eval(:($derivative_order >=1 ? HDIM_ISO : 0))); #∂ψ1
        fill(qs[6], eval(:($derivative_order >=2 ? HDIM_ISO : 0))); #∂2ψ1
        # fill(0, eval(:($derivative_order >= 1 ?
        # ($SAMPLE_COUNT - $STATE_COUNT) * $HDIM_ISO : 0))) # ψ3, ψ4
        # fill(qs[5], eval(:($derivative_order >= 1 ? $SAMPLE_COUNT * $HDIM_ISO : 0))); # ∂ψ
        # fill(qs[6], eval(:($derivative_order >= 2 ? $SAMPLE_COUNT * $HDIM_ISO : 0))); # ∂2ψ
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[7], CONTROL_COUNT);
    ]))
    objective = LQRObjective(Q, R, Qf, xf, N)

    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX]) # ; INTCONTROLS_IDX])
    # must obey unit norm.
    norm_idxs = [STATE1_IDX, STATE2_IDX]
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idx) for idx in norm_idxs]
    
    constraints = ConstraintList(n, m, N)
    # add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
    for norm_constraint in norm_constraints
        add_constraint!(constraints, norm_constraint, 2:N-1)
    end

    # solve problem
    prob = Problem{IT_RDI[integrator_type]}(model, objective, constraints,
                                            x0, xf, Z, N, t0, evolution_time)
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
                 iterations_outer=iterations_outer, iterations=max_iterations)
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
