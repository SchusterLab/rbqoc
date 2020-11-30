"""
spin15.jl - T1 optimized pulses
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
const EXPERIMENT_NAME = "spin15"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const INTGAMMA_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + 1
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT
const DT_IDX = D2CONTROLS_IDX[end] + 1:D2CONTROLS_IDX[end] + 1


# model
# DA = decay aware, TO = time optimal
struct Model{DA, TO} <: AbstractModel
    Model(DA::Bool=true, TO::Bool=true) = new{DA, TO}()
end
@inline RD.state_dim(::Model{false, TO}) where TO = ASTATE_SIZE_BASE
@inline RD.state_dim(::Model{true, TO}) where TO = ASTATE_SIZE_BASE + 1
@inline RD.control_dim(::Model{DA, false}) where DA = CONTROL_COUNT
@inline RD.control_dim(::Model{DA, true}) where DA = CONTROL_COUNT + 1


# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model{DA, TO}, astate::StaticVector,
                              acontrols::StaticVector, time::Real, dt::Real) where {DA, TO}
    if TO
        dt = acontrols[DT_IDX[1]]^2
    end
    
    h_prop = exp(dt * (FQ_NEGI_H0_ISO + astate[CONTROLS_IDX[1]] * NEGI_H1_ISO))
    state1 = h_prop * astate[STATE1_IDX]
    state2 = h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX[1]] + dt * astate[CONTROLS_IDX[1]]
    controls = astate[CONTROLS_IDX[1]] + dt * astate[DCONTROLS_IDX[1]]
    dcontrols = astate[DCONTROLS_IDX[1]] + dt * acontrols[D2CONTROLS_IDX[1]]
    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
    ]
    
    if DA
        intgamma = (astate[INTGAMMA_IDX[1]] +
                    dt * amp_t1_reduced_spline(astate[CONTROLS_IDX[1]])^(-1))
        push!(astate_, intgamma)
    end
    
    return astate_
end


# main
function run_traj(;evolution_time=21.61, gate_type=zpiby2, time_optimal=false,
                  decay_aware=false, solver_type=altro, sqrtbp=false,
                  integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e2, 1e-1, 5e2],
                  smoke_test=false,
                  al_tol=1e-4, constraint_tol=1e-8, max_penalty=1e11,
                  pn_steps=2, save=true, verbose=true, dt_inv=Int64(1e1),
                  max_cost_value=1e8, max_iterations=Int64(2e5),
                  benchmark=false,)
    # model configuration
    dt = dt_inv^(-1)
    dt_max = (dt_inv / 2)^(-1)
    dt_min = (dt_inv * 2)^(-1)
    model = Model(decay_aware, time_optimal)
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

    # dt bound
    u_max = SVector{m}([
        fill(Inf, CONTROL_COUNT);
        fill(sqrt(dt_max), eval(:($time_optimal ? 1 : 0))); #dt
    ])
    u_min = SVector{m}([
        fill(-Inf, CONTROL_COUNT);
        fill(sqrt(dt_min), eval(:($time_optimal ? 1 : 0))); #dt
    ])

    # initial trajectory
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [SVector{m}([
        fill(1e-4, CONTROL_COUNT);
        fill(sqrt(dt), eval(:($time_optimal ? 1 : 0)));
    ]) for k = 1:N - 1]
    X0 = [SVector{n}(
        fill(NaN, n)
    ) for k = 1:N]
    dt_ = time_optimal ? 1 : dt
    Z = Traj(X0, U0, dt_ * ones(N))

    # cost function
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # ψ1, ψ2
        fill(qs[2], CONTROL_COUNT); # ∫a
        fill(qs[3], CONTROL_COUNT); # a
        fill(qs[4], CONTROL_COUNT); # ∂a
        fill(qs[5], eval(:($decay_aware ? 1 : 0))); # ∫γ1
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[6], CONTROL_COUNT); # ∂2a
        fill(qs[7], eval(:($time_optimal ? 1 : 0))); # Δt
    ]))
    objective = LQRObjective(Q, R, Qf, xf, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must satisfy dt bound
    dt_bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idx)
                        for idx in [STATE1_IDX, STATE2_IDX]]
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    for norm_constraint in norm_constraints
        add_constraint!(constraints, norm_constraint, 2:N-1)
    end
    if time_optimal
        add_constraint!(constraints, dt_bnd, 1:N-1)
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
                 iterations_outer=iterations_outer, iterations=max_iterations,
                 max_cost_value=max_cost_value)
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end

    # Post-process.
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
    dtidx_arr = Array(DT_IDX)
    # Square the dts.
    if time_optimal
        acontrols_arr[:, DT_IDX] = acontrols_arr[:, DT_IDX] .^2
        # acontrols_arr[:, DT_IDX] = map(abs, acontrols_arr[:, DT_IDX])
    end
    cmax = TO.max_violation(solver)
    cmax_info = TO.findmax_violation(TO.get_constraints(solver))
    iterations_ = iterations(solver)

    result = Dict(
            "acontrols" => acontrols_arr,
            "controls_idx" => cidx_arr,
            "d2controls_dt2_idx" => d2cidx_arr,
            "dt_idx" => dtidx_arr,
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
            "integrator_type" => Integer(integrator_type),
            "gate_type" => Integer(gate_type),
            "save_type" => Integer(jl),
            "iterations" => iterations_,
            "pn_steps" => pn_steps,
            "max_iterations" => max_iterations,
            "max_cost_value" => max_cost_value,
    )

    if time_optimal
        # sample the important metrics
        (controls_sample, d2controls_dt2_sample, evolution_time_sample
         ) = sample_controls(save_file_path)
        result["controls_sample"] = controls_sample
        result["d2controls_dt2_sample"] = d2controls_dt2_sample
        result["evolution_time_sample"] = evolution_time_sample
        result["save_type"] = Integer(samplejl)
    end
    
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
