"""
spin13.jl - vanilla
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
const EXPERIMENT_NAME = "spin13"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const ACONTROL_SIZE = CONTROL_COUNT
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
struct Model <: AbstractModel
end
@inline RD.state_dim(::Model) = ASTATE_SIZE
@inline RD.control_dim(::Model) = ACONTROL_SIZE


# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::SVector,
                              acontrol::SVector, time::Real, dt::Real)
    h_prop = exp(dt * (FQ_NEGI_H0_ISO + astate[CONTROLS_IDX[1]] * NEGI_H1_ISO))
    state1 = h_prop * astate[STATE1_IDX]
    state2 = h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX[1]] + astate[CONTROLS_IDX[1]] * dt
    controls = astate[CONTROLS_IDX[1]] + astate[DCONTROLS_IDX[1]] * dt
    dcontrols = astate[DCONTROLS_IDX[1]] + acontrol[D2CONTROLS_IDX[1]] * dt

    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
    ]

    return astate_
end


# main
function run_traj(;gate_type=zpiby2, evolution_time=30., solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, smoke_test=false,
                  dt=1000.0/(384. * 16.), constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true, max_iterations=Int64(2e5),
                  max_cost_value=1e8, qs=[1e0, 1e0, 1e0, 1e-1, 1e-1],
                  benchmark=false, projected_newton=true)
    # model configuration
    num_steps = floor(evolution_time / dt)
    evolution_time = num_steps * dt
    model = Model()
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.

    # initial state
    x0 = zeros(n)
    x0[STATE1_IDX] = IS1_ISO_
    x0[STATE2_IDX] = IS2_ISO_
    x0 = SVector{n}(x0)

    # final state
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
    xf = zeros(n)
    xf[STATE1_IDX] = target_state1
    xf[STATE2_IDX] = target_state2
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
    N = Int(floor(evolution_time * (1. / dt))) + 1
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
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[5], CONTROL_COUNT); # ∂2a
    ]))
    objective = LQRObjective(Q, R, Qf, xf, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idx)
                        for idx in [STATE1_IDX, STATE2_IDX]]

    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
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
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end

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
#        "benchmark_result" => benchmark_result,
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

    result = benchmark ? benchmark_result : result

    return result
end
