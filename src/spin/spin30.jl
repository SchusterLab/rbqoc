"""
spin30.jl - unscented transform robustness for the δfq problem
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
using Distributions
using HDF5
using LinearAlgebra
using Random
using RobotDynamics
using StaticArrays
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin30"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
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
# misc
const SAMPLE_STATES = [IS3_ISO]
const SAMPLE_STATE_COUNT = 1
const SAMPLES_PER_STATE = 10
const SAMPLE_COUNT = SAMPLE_STATE_COUNT * SAMPLES_PER_STATE
const STATE_IDX = SVector{HDIM_ISO}(1:HDIM_ISO)
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT

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
const SAMPLE_IDXS = [astate_sample_inds(i, j)
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
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::SVector{ASTATE_SIZE},
                              acontrol::SVector{ACONTROL_SIZE}, time::Real, dt::Real)
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

# This cost puts a gate error cost on
# the sample states and a LQR cost on the other terms.
# The hessian w.r.t the state and controls is constant.
struct Cost{N,M,T} <: TO.CostFunction
    Q::Diagonal{T, SVector{N,T}}
    R::Diagonal{T, SVector{M,T}}
    q::SVector{N, T}
    c::T
    target_states::Array{SVector{HDIM_ISO,T},1}
    q_ss::Array{T,1}
    active_samples::Array{Int,1}
end

function Cost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
              xf::SVector{N,T}, target_states::Array{SVector{HDIM_ISO,T},1},
              q_ss::Array{T,1}, active_samples::Array{Int,1}) where {N,M,T}
    q = -Q * xf
    c = 0.5 * xf' * Q * xf
    return Cost{N,M,T}(Q, R, q, c, target_states, q_ss, active_samples)
end

@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
    cost.Q, cost.R, cost.q, cost.c, cost.target_states, cost.q_ss,
    cost.active_samples
)

function TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}) where {N,M,T}
    cost_ = 0.5 * astate' * cost.Q * astate + cost.q'astate + cost.c
    for i = 1:SAMPLE_STATE_COUNT
        q_ss = cost.q_ss[i]
        target_state = cost.target_states[i]
        for j in cost.active_samples
            sample_idx = astate_sample_inds(i, j)
            cost_ = cost_ + q_ss * gate_error_iso2(astate, target_state;
                                                   s1o=sample_idx[1] - 1)
        end
    end
    return cost_
end

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}, acontrol::SVector{M}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T},
                      astate::SVector{N,T}) where {N,M,T}
    E.q = cost.Q * astate + cost.q
    for i = 1:SAMPLE_STATE_COUNT
        q_ss = cost.q_ss[i]
        target_state = cost.target_states[i]
        for j in cost.active_samples
            sample_idx = astate_sample_inds(i, j)
            E.q[sample_idx] = q_ss * jacobian_gate_error_iso2(astate, target_state;
                                                              s1o=sample_idx[1] - 1);
        end
    end
    return false
end

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T},
                      acontrol::SVector{M,T}) where {N,M,T}
    TO.gradient!(E, cost, astate)
    E.r = cost.R * acontrol
    E.c = 0
    return false
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T},
                     astate::SVector{N,T}) where {N,M,T}
    hess_astate = zeros(N, N)
    for i = 1:SAMPLE_STATE_COUNT
        q_ss = cost.q_ss[i]
        target_state = cost.target_states[i]
        hess_sample = -1 * q_ss * hessian_gate_error_iso2(target_state)
        for j in cost.active_samples
            sample_idx = astate_sample_inds(i, j)
            hess_astate[sample_idx, sample_idx] = hess_sample
        end
    end
    hess_astate = hess_astate + cost.Q
    hess_astate = Symmetric(SMatrix{N, N}(hess_astate))
    E.Q = hess_astate
    return true
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T},
                     acontrol::SVector{M,T}) where {N,M,T}
    TO.hessian!(E, cost, astate)
    E.R = cost.R
    E.H .= 0
    return true
end


# main
function run_traj(;gate_type=xpiby2, evolution_time=60., solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true,
                  fq_cov=FQ * 1e-2, max_iterations=Int64(2e5), gradient_tol_int=1,
                  dJ_counter_limit=Int(1e2), state_cov=1e-2, seed=0, alpha=1.,)
    Random.seed!(seed)
    model = Model(fq_cov, alpha)
    n = RD.state_dim(model)
    m = RD.control_dim(model)
    t0 = 0.

    # initial state, target state
    x0 = zeros(n)
    xf = zeros(n)
    target_states = Array{SVector{HDIM_ISO,Float64},1}(undef, SAMPLE_STATE_COUNT)
    gate = GT_GATE_ISO[gate_type]
    x0[STATE1_IDX] = IS1_ISO_
    x0[STATE2_IDX] = IS2_ISO_
    xf[STATE1_IDX] = gate * IS1_ISO_
    xf[STATE2_IDX]= gate * IS2_ISO_
    state_dist = Distributions.Normal(0., state_cov)
    for i = 1:SAMPLE_STATE_COUNT
        sample_state = SAMPLE_STATES[i]
        target_state = gate * sample_state
        target_states[i] = target_state
        for j = 1:SAMPLES_PER_STATE
            sample_idx = astate_sample_inds(i, j)
            sample = sample_state .+ rand(state_dist, HDIM_ISO)
            x0[sample_idx] = sample ./ sqrt(sample'sample)
            xf[sample_idx] = target_state
        end
    end
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
    U0 = [SVector{m}([
        fill(1e-4, CONTROL_COUNT);
    ]) for k = 1:N-1]
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
        fill(0, SAMPLE_COUNT * HDIM_ISO);
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[6], CONTROL_COUNT); # ∂2a
    ]))
    # objective = LQRObjective(Q, R, Qf, xf, N)
    active_samples = Array(1:SAMPLES_PER_STATE)
    # active_samples = [10]
    q_ss = repeat([qs[5]], SAMPLE_STATE_COUNT)
    cost_k = Cost(Q, R, xf, target_states, q_ss, active_samples)
    cost_f = Cost(Qf, R, xf, target_states, N * q_ss, active_samples)
    # cost_k = Cost(Q, R, xf, gate * IS1_ISO_, qs[5], active_samples)
    # cost_f = Cost(Qf, R, xf, gate * IS1_ISO_, N * qs[5], active_samples)
    objective = TO.Objective(cost_k, cost_f, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    norm_idxs = copy(SAMPLE_IDXS)
    push!(norm_idxs, STATE1_IDX)
    push!(norm_idxs, STATE2_IDX)
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idx) for idx in norm_idxs]
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
        dJ_counter_limit=dJ_counter_limit,
    )
    Altro.solve!(solver)

    # Post-process.
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
