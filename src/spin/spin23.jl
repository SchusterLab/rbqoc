"""
spin23.jl - unscented transform robustness for the δfq problem
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
using Debugger
using Distributions
using HDF5
using Hyperopt
using LinearAlgebra
using Random
using RobotDynamics
using StaticArrays
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin23"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const SAMPLE_STATES = [IS1_ISO_, IS2_ISO_, IS3_ISO_, IS4_ISO_]
const SAMPLE_STATE_COUNT = 4
const SAMPLES_PER_STATE = 10
const SAMPLE_COUNT = SAMPLE_STATE_COUNT * SAMPLES_PER_STATE
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT
# state indices
const STATE1_IDX = SVector{HDIM_ISO}(1:HDIM_ISO)
const STATE2_IDX = SVector{HDIM_ISO}(STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO)
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT
# sample indices
const S1_IDX = SVector{HDIM_ISO}(HDIM_ISO * 0 + 1:HDIM_ISO * 1)
const S2_IDX = SVector{HDIM_ISO}(HDIM_ISO * 1 + 1:HDIM_ISO * 2)
const S3_IDX = SVector{HDIM_ISO}(HDIM_ISO * 2 + 1:HDIM_ISO * 3)
const S4_IDX = SVector{HDIM_ISO}(HDIM_ISO * 3 + 1:HDIM_ISO * 4)
const S5_IDX = SVector{HDIM_ISO}(HDIM_ISO * 4 + 1:HDIM_ISO * 5)
const S6_IDX = SVector{HDIM_ISO}(HDIM_ISO * 5 + 1:HDIM_ISO * 6)
const S7_IDX = SVector{HDIM_ISO}(HDIM_ISO * 6 + 1:HDIM_ISO * 7)
const S8_IDX = SVector{HDIM_ISO}(HDIM_ISO * 7 + 1:HDIM_ISO * 8)
const S9_IDX = SVector{HDIM_ISO}(HDIM_ISO * 8 + 1:HDIM_ISO * 9)
const S10_IDX = SVector{HDIM_ISO}(HDIM_ISO * 9 + 1:HDIM_ISO * 10)
const STATE_IDX = SVector{HDIM_ISO}(1:HDIM_ISO)

# model
module Data
using RobotDynamics
const RD = RobotDynamics
mutable struct Model <: RD.AbstractModel
    fq_cov::Float64
    alpha::Float64
end
end
Model = Data.Model
@inline RD.state_dim(model::Model) = ASTATE_SIZE
@inline RD.control_dim(model::Model) = ACONTROL_SIZE
@inline astate_sample_inds(sample_state_index::Int, sample_index::Int) = (
    SVector{HDIM_ISO}((
        ASTATE_SIZE_BASE + (sample_state_index - 1) * SAMPLES_PER_STATE * HDIM_ISO
        + (sample_index - 1) * HDIM_ISO + 1
    ):(
        ASTATE_SIZE_BASE + (sample_state_index - 1) * SAMPLES_PER_STATE * HDIM_ISO
        + sample_index * HDIM_ISO
    ))
)
const SAMPLE_INDICES = [astate_sample_inds(i, j)
                        for i = 1:SAMPLE_STATE_COUNT for j = 1:SAMPLES_PER_STATE]

function unscented_transform(model::Model, astate::AbstractVector,
                             negi_hc::AbstractMatrix, dt::Real, i::Int)
    # get states
    offset = ASTATE_SIZE_BASE + (i - 1) * SAMPLES_PER_STATE * HDIM_ISO
    s1 = astate[offset + S1_IDX]
    s2 = astate[offset + S2_IDX]
    s3 = astate[offset + S3_IDX]
    s4 = astate[offset + S4_IDX]
    s5 = astate[offset + S5_IDX]
    s6 = astate[offset + S6_IDX]
    s7 = astate[offset + S7_IDX]
    s8 = astate[offset + S8_IDX]
    s9 = astate[offset + S9_IDX]
    s10 = astate[offset + S10_IDX]
    # grab chol info
    fq_chol1 = fq_chol2 = fq_chol3 = fq_chol4 = 0
    fq_chol5 = sqrt(model.fq_cov)
    # propagate states
    s1 = exp(dt * ((FQ + fq_chol1) * NEGI_H0_ISO + negi_hc)) * s1
    s2 = exp(dt * ((FQ + fq_chol2) * NEGI_H0_ISO + negi_hc)) * s2
    s3 = exp(dt * ((FQ + fq_chol3) * NEGI_H0_ISO + negi_hc)) * s3
    s4 = exp(dt * ((FQ + fq_chol4) * NEGI_H0_ISO + negi_hc)) * s4
    s5 = exp(dt * ((FQ + fq_chol5) * NEGI_H0_ISO + negi_hc)) * s5
    s6 = exp(dt * ((FQ - fq_chol1) * NEGI_H0_ISO + negi_hc)) * s6
    s7 = exp(dt * ((FQ - fq_chol2) * NEGI_H0_ISO + negi_hc)) * s7
    s8 = exp(dt * ((FQ - fq_chol3) * NEGI_H0_ISO + negi_hc)) * s8
    s9 = exp(dt * ((FQ - fq_chol4) * NEGI_H0_ISO + negi_hc)) * s9
    s10 = exp(dt * ((FQ - fq_chol5) * NEGI_H0_ISO + negi_hc)) * s10
    # compute state mean
    sm = 1//SAMPLES_PER_STATE .* (
        s1 + s2 + s3 + s4 + s5
        + s6 + s7 + s8 + s9 + s10
    )
    # compute state covariance
    d1 = s1 - sm
    d2 = s2 - sm
    d3 = s3 - sm
    d4 = s4 - sm
    d5 = s5 - sm
    d6 = s6 - sm
    d7 = s7 - sm
    d8 = s8 - sm
    d9 = s9 - sm
    d10 = s10 - sm
    s_cov = 1 / (2 * model.alpha^2) .* (
        d1 * d1' + d2 * d2' + d3 * d3' + d4 * d4'
        + d5 * d5' + d6 * d6' + d7 * d7' + d8 * d8'
        + d9 * d9' + d10 * d10'
    )
    # perform cholesky decomposition on joint covariance
    cov = zeros(eltype(s_cov), HDIM_ISO + 1, HDIM_ISO + 1)
    cov[1:HDIM_ISO, 1:HDIM_ISO] .= s_cov
    cov[HDIM_ISO + 1, HDIM_ISO + 1] = model.fq_cov
    # TOOD: cholesky! requires writing zeros in upper triangle
    cov_chol = model.alpha * cholesky(Symmetric(cov)).L
    # resample states
    s_chol1 = cov_chol[STATE_IDX, 1]
    s_chol2 = cov_chol[STATE_IDX, 2]
    s_chol3 = cov_chol[STATE_IDX, 3]
    s_chol4 = cov_chol[STATE_IDX, 4]
    # s_chol5 = cov_chol[STATE_IDX, 5]
    s1 = sm + s_chol1
    s2 = sm + s_chol2
    s3 = sm + s_chol3
    s4 = sm + s_chol4
    s5 = sm # + s_chol5
    s6 = sm - s_chol1
    s7 = sm - s_chol2
    s8 = sm - s_chol3
    s9 = sm - s_chol4
    s10 = sm # - s_chol5
    # normalize
    s1 = s1 ./sqrt(s1's1)
    s2 = s2 ./sqrt(s2's2)
    s3 = s3 ./sqrt(s3's3)
    s4 = s4 ./sqrt(s4's4)
    s5 = s5 ./sqrt(s5's5)
    s6 = s6 ./sqrt(s6's6)
    s7 = s7 ./sqrt(s7's7)
    s8 = s8 ./sqrt(s8's8)
    s9 = s9 ./sqrt(s9's9)
    s10 = s10 ./sqrt(s10's10)

    samples = [s1; s2; s3; s4; s5; s6; s7; s8; s9; s10]

    return samples
end

# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::Array{T,1},
                              acontrol::Array{T,1}, time::Real, dt::Real) where {T}
    # base dynamics
    negi_hc = astate[CONTROLS_IDX[1]] * NEGI_H1_ISO
    h_prop = exp(dt * (FQ_NEGI_H0_ISO + negi_hc))
    state1 = h_prop * astate[STATE1_IDX]
    state2 = h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX[1]] + dt * astate[CONTROLS_IDX[1]]
    controls = astate[CONTROLS_IDX[1]] + dt * astate[DCONTROLS_IDX[1]]
    dcontrols = astate[DCONTROLS_IDX[1]] + dt * acontrol[D2CONTROLS_IDX[1]]

    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
    ]

    # unscented transform
    for i = 1:SAMPLE_STATE_COUNT
        sample_states = unscented_transform(model, astate, negi_hc, dt, i)
        append!(astate_, sample_states)
    end
    
    return astate_
end


function RD.discrete_dynamics(::Type{RK3}, model::Model, z::AbstractKnotPoint)
    return RD.discrete_dynamics(RK3, model, RD.state(z), RD.control(z), z.t, z.dt)
end


# This cost puts a gate error cost on
# the sample states and a LQR cost on the other terms.
# The hessian w.r.t the state and controls is constant.
struct Cost{N,M,T} <: TO.CostFunction
    Q::Diagonal{T,Array{T,1}}
    R::Diagonal{T,Array{T,1}}
    q::Array{T,1}
    c::T
    target_states::Array{T,1}
    q_sample_states::Array{T,1}
    active_samples::Array{Int,1}
end

function Cost(Q::Diagonal{T,Array{T,1}}, R::Diagonal{T,Array{T,1}},
              xf::Array{T,1}, target_states::Array{T,1}, q_sample_states::Array{T,1},
              active_samples::Array{Int,1}) where {T}
    N = ASTATE_SIZE
    M = ACONTROL_SIZE
    q = -Q * xf
    c = 0.5 * xf' * Q * xf
    return Cost{N,M,T}(Q, R, q, c, target_states, q_sample_states, active_samples)
end

@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
    cost.Q, cost.R, cost.q, cost.c, cost.target_states,
    cost.q_sample_states, cost.active_samples
)

function TO.stage_cost(cost::Cost{N,M,T}, astate::Array{T,1}) where {N,M,T}
    cost_ = 0.5 * astate' * cost.Q * astate + cost.q'astate + cost.c
    for i = 1:SAMPLE_STATE_COUNT
        targeto = (i - 1) * HDIM_ISO
        for j in cost.active_samples
            astateo = astate_sample_inds(i, j)[1] - 1
            cost_ = (
                cost_ + cost.q_sample_states[i]
                * gate_error_iso2(astate, cost.target_states; s1o=astateo, s2o=targeto)
            )
        end
    end
    return cost_
end

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::Array{T,1},
                      acontrol::Array{T,1}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T},
                      astate::Array{T,1}) where {N,M,T}
    E.q = cost.Q * astate + cost.q
    for i = 1:SAMPLE_STATE_COUNT
        targeto = (i - 1) * HDIM_ISO
        for j in cost.active_samples
            sample_idx = astate_sample_inds(i, j)
            E.q[sample_idx] = E.q[sample_idx] + (
                cost.q_sample_states[i]
                * jacobian_gate_error_iso2(astate, cost.target_states;
                                           s1o=(sample_idx[1] - 1), s2o=targeto)
            )
        end
    end
    return false
end

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::Array{T,1},
                      acontrol::Array{T,1}) where {N,M,T}
    TO.gradient!(E, cost, astate)
    E.r = cost.R * acontrol
    E.c = 0
    return false
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T},
                     astate::Array{T,1}) where {N,M,T}
    hess_astate = zeros(N, N)
    for i = 1:SAMPLE_STATE_COUNT
        targeto = (i - 1) * HDIM_ISO
        for j in cost.active_samples
            sample_idx = astate_sample_inds(i, j)
            hess_astate[sample_idx, sample_idx] = (
                -1 * cost.q_sample_states[i]
                * hessian_gate_error_iso2(cost.target_states; s2o=targeto)
            )
        end
    end
    hess_astate = hess_astate + cost.Q
    hess_astate = Symmetric(hess_astate)
    E.Q = hess_astate
    return true
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::Array{T,1},
                     acontrol::Array{T,1}) where {N,M,T}
    TO.hessian!(E, cost, astate)
    E.R = cost.R
    E.H .= 0
    return true
end

# main
function run_traj(;gate_type=xpiby2, evolution_time=60., solver_type=altro,
                  sqrtbp=false, integrator_type=rk3,
                  qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true,
                  static_bp=false,
                  fq_cov=FQ * 1e-2, max_iterations=Int64(2e5), gradient_tol_int=1,
                  dJ_counter_limit=Int(1e2), state_cov=1e-2, seed=0, alpha=1.,
                  benchmark=false)
    Random.seed!(seed)
    model = Model(fq_cov, alpha)
    n = RD.state_dim(model)
    m = RD.control_dim(model)
    t0 = 0.

    # initial state, target state
    x0 = zeros(n)
    xf = zeros(n)
    x0[STATE1_IDX] = IS1_ISO_
    x0[STATE2_IDX] = IS2_ISO_
    gate = GT_GATE_ISO[gate_type]
    xf[STATE1_IDX] = gate * IS1_ISO_
    xf[STATE2_IDX] = gate * IS2_ISO_
    state_dist = Distributions.Normal(0., state_cov)
    target_states = zeros(SAMPLE_STATE_COUNT * HDIM_ISO)
    for i = 1:SAMPLE_STATE_COUNT
        sample_state_initial = SAMPLE_STATES[i]
        target_state = gate * sample_state_initial
        indlo = (i - 1) * HDIM_ISO + 1
        indhi = indlo + HDIM_ISO - 1
        target_states[indlo:indhi] = target_state
        for j = 1:SAMPLES_PER_STATE
            sample_state = sample_state_initial .+ rand(state_dist, HDIM_ISO)
            sample_state = sample_state ./ sqrt(sample_state'sample_state)
            sample_idx = astate_sample_inds(i, j)
            x0[sample_idx] = sample_state
            xf[sample_idx] = target_state
        end
    end

    # control amplitude constraint
    x_max = fill(Inf, n)
    x_max[CONTROLS_IDX] .= MAX_CONTROL_NORM_0
    x_min = fill(-Inf, n)
    x_min[CONTROLS_IDX] .= -MAX_CONTROL_NORM_0
    
    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n)
    x_max_boundary[CONTROLS_IDX] .= 0
    x_min_boundary = fill(-Inf, n)
    x_min_boundary[CONTROLS_IDX] .= 0

    # initial trajectory
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [[
        fill(1e-4, CONTROL_COUNT);
    ] for k = 1:N-1]
    X0 = [[
        fill(NaN, n);
    ] for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    # cost function
    Q = Diagonal([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # ψ1, ψ2
        fill(qs[2], CONTROL_COUNT); # ∫a
        fill(qs[3], CONTROL_COUNT); # a
        fill(qs[4], CONTROL_COUNT); # ∂a
        fill(0, SAMPLE_COUNT * HDIM_ISO);
    ])
    Qf = Q * N
    R = Diagonal([
        fill(qs[6], CONTROL_COUNT); # ∂2a
    ])
    # objective = LQRObjective(Q, R, Qf, xf, N)
    active_samples = Array(1:SAMPLES_PER_STATE)
    q_ss = repeat(qs[5:5], SAMPLE_STATE_COUNT)
    cost_k = Cost(Q, R, xf, target_states, q_ss, active_samples)
    cost_f = Cost(Qf, R, xf, target_states, N * q_ss, active_samples)
    objective = TO.Objective(cost_k, cost_f, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    norm_idxs = copy(SAMPLE_INDICES)
    push!(norm_idxs, STATE1_IDX)
    push!(norm_idxs, STATE2_IDX)
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idxs) for idxs in norm_idxs]
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
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
    set_options!(
        solver, square_root=sqrtbp, constraint_tolerance=constraint_tolerance,
        projected_newton_tolerance=al_tol, n_steps=n_steps,
        penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
        projected_newton=projected_newton, iterations_inner=iterations_inner,
        iterations_outer=iterations_outer, iterations=max_iterations,
        gradient_tolerance_intermediate=gradient_tol_int,
        dJ_counter_limit=dJ_counter_limit, static_bp=static_bp,
    )
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
    cidx_arr = Array(CONTROLS_IDX)
    d2cidx_arr = Array(D2CONTROLS_IDX)
    cmax = TO.max_violation(solver)
    cmax_info = TO.findmax_violation(TO.get_constraints(solver))
    iterations_ = iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "controls_idx" => cidx_arr,
        "d2controls_dt2_idx" => d2cidx_arr,
        "evolution_time" => evolution_time,
        "astates" => astates_arr,
        "qs" => qs,
        "cmax" => cmax,
        "cmax_info" => cmax_info,
        "dt" => dt,
        "solver_type" => Integer(solver_type),
        "sqrtbp" => Integer(sqrtbp),
        "max_penalty" => max_penalty,
        "constraint_tol" => constraint_tol,
        "al_tol" => al_tol,
        "gradient_tolerance_intermediate" => gradient_tol_int,
        "dJ_counter_limit" => dJ_counter_limit,
        "integrator_type" => Integer(integrator_type),
        "gate_type" => Integer(gate_type),
        "save_type" => Integer(jl),
        "iterations" => iterations_,
        "seed" => seed,
        "fq_cov" => fq_cov,
        "max_iterations" => max_iterations,
        "static_bp" => static_bp,
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


function sample_diffs(saved)
    gate_type = GateType(saved["gate_type"])
    knot_count = size(saved["astates"], 1)
    fds = zeros(knot_count, SAMPLE_COUNT)
    gate_iso = GT_GATE_ISO[gate_type]
    # target_states = [gate_iso * is for is in (IS1_ISO_, IS2_ISO_, IS3_ISO_, IS4_ISO_)]
    target_states = [gate_iso * is for is in (IS3_ISO_,)]
    astates = saved["astates"]
    for i = 1:SAMPLE_STATE_COUNT
        target_state = target_states[i]
        for j = 1:SAMPLES_PER_STATE
            astate_inds = astate_sample_inds(i, j)
            store_ind = (i - 1) * SAMPLES_PER_STATE + j
            for k = 1:knot_count
                sample = astates[k, astate_inds]
                fds[k, store_ind] = fidelity_vec_iso2(sample, target_state)
            end
        end
    end

    return fds
end


function hyperopt_me(;iterations=50, save=true)
    save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    gate_type = zpiby2
    save_file_paths = fill("", iterations)
    weights = zeros(iterations)
    alphas = zeros(iterations)
    statecovs = zeros(iterations)
    gate_errors = zeros(iterations)
    
    result::Dict{String, Any} = Dict(
        "weights" => weights,
        "alphas" => alphas,
        "statecovs" => statecovs,
        "save_file_paths" => save_file_paths,
        "gate_errors" => gate_errors,
    )

    if save
        h5open(save_file_path, "cw") do save_file
            for key in keys(result)
                write(save_file, key, result[key])
            end
        end
    end
    
    weights_space = exp10.(LinRange(-5, 5, 1000))
    alpha_space = LinRange(1e-1, 10, 1000)
    statecov_space = LinRange(1e-6, 1e0, 1000)
    ho = Hyperoptimizer(iterations, GPSampler(Min); a=weights_space, b=alpha_space, c=statecov_space)
    for (i, weight, alpha, statecov) in ho
        # evaluate
        gate_error = 1
        (gate_error, save_file_path_) = try 
            res_train = run_traj(;gate_type=gate_type, qs=[1e0, 1e0, 1e0, 1e-1, weight,
                                                           weight, weight, weight, 1e-1],
                                 alpha=alpha, state_cov=statecov, verbose=true, save=true)
            save_file_path_ = res_train["save_file_path"]
            res_eval = evaluate_fqdev(;save_file_path=save_file_path_, gate_type=gate_type)
            gate_error = mean(res_eval["gate_errors"])
            (gate_error, save_file_path_)
        catch err
            gate_error = 1
            save_file_path_ = ""
            (gate_error, save_file_path_)
        end
        
        # log
        i_ = Int(i)
        weights[i_] = weight
        alphas[i_] = alpha
        statecovs[i_] = statecov
        gate_errors[i_] = gate_error
        save_file_paths[i_] = save_file_path_
        if save
            h5open(save_file_path, "cw") do save_file
                for key in keys(result)
                    o_delete(save_file, key)
                    write(save_file, key, result[key])
                end
            end
        end

        push!(ho.results, gate_error)
        push!(ho.history, [weight, alpha, statecov])
    end

    if save
        result["save_file_path"] = save_file_path
    end

    return result
end
