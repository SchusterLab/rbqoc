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
const ACONTROL_SIZE = CONTROL_COUNT
const STATE_COUNT = 2
# const STATE_COUNT = 4
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const SAMPLES_PER_STATE = 10
const PENALTY_SIZE = 1
const CHUNK_SIZE = SAMPLES_PER_STATE * HDIM_ISO + PENALTY_SIZE
# state indices
const STATE1_IDX = SVector{HDIM_ISO}(1:HDIM_ISO)
const STATE2_IDX = SVector{HDIM_ISO}(STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO)
# const STATE3_IDX = SVector{HDIM_ISO}(STATE2_IDX[end] + 1:STATE2_IDX[end] + HDIM_ISO)
# const STATE4_IDX = SVector{HDIM_ISO}(STATE3_IDX[end] + 1:STATE3_IDX[end] + HDIM_ISO)
# const INTCONTROLS_IDX = SVector{CONTROL_COUNT}(STATE4_IDX[end] + 1:STATE4_IDX[end] + CONTROL_COUNT)
const INTCONTROLS_IDX = SVector{CONTROL_COUNT}(STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT)
const CONTROLS_IDX = SVector{CONTROL_COUNT}(INTCONTROLS_IDX[end] + 1:
                                            INTCONTROLS_IDX[end] + CONTROL_COUNT)
const DCONTROLS_IDX = SVector{CONTROL_COUNT}(CONTROLS_IDX[end] + 1:
                                             CONTROLS_IDX[end] + CONTROL_COUNT)
# control indices
const D2CONTROLS_IDX = SVector{CONTROL_COUNT}(1:CONTROL_COUNT)
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
struct Model <: RD.AbstractModel
    S::Diagonal{Float64,SVector{HDIM_ISO,Float64}}
    nominal_idxs::Array{SVector{HDIM_ISO,Int},1}
    fq_cov::Float64
    alpha::Float64
    sample_state_count::Int
end
@inline RD.state_dim(model::Model) = (
    ASTATE_SIZE_BASE + model.sample_state_count * CHUNK_SIZE
)
@inline RD.control_dim(model::Model) = ACONTROL_SIZE
@inline astate_sample_inds(sample_state_index::Int, sample_index::Int) = (
    SVector{HDIM_ISO}((
        ASTATE_SIZE_BASE + (sample_state_index - 1) * CHUNK_SIZE
        + (sample_index - 1) * HDIM_ISO + 1
    ):(
        ASTATE_SIZE_BASE + (sample_state_index - 1) * CHUNK_SIZE
        + sample_index * HDIM_ISO
    ))
)
@inline sample_idxs(model::Model) = [
    astate_sample_inds(i, j)
    for i = 1:model.sample_state_count
    for j = 1:SAMPLES_PER_STATE
]

function unscented_transform(model::Model, astate::AbstractVector,
                             negi_hc::AbstractMatrix, h_prop::AbstractMatrix,
                             dt::Real, i::Int)
    # get states
    offset = ASTATE_SIZE_BASE + (i - 1) * CHUNK_SIZE
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
    fq_chol5 = sqrt(model.fq_cov)
    # propagate states
    s1 = h_prop * s1
    s2 = h_prop * s2
    s3 = h_prop * s3
    s4 = h_prop * s4
    s5 = exp(dt * ((FQ + fq_chol5) * NEGI_H0_ISO + negi_hc)) * s5
    s6 = h_prop * s6
    s7 = h_prop * s7
    s8 = h_prop * s8
    s9 = h_prop * s9
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
    s1 = sm + s_chol1
    s2 = sm + s_chol2
    s3 = sm + s_chol3
    s4 = sm + s_chol4
    s5 = sm
    s6 = sm - s_chol1
    s7 = sm - s_chol2
    s8 = sm - s_chol3
    s9 = sm - s_chol4
    s10 = sm
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
    # compute penalty
    dnom = sm - astate[model.nominal_idxs[i]]
    penalty = SVector{1}(tr(s_cov * model.S) + dnom' * model.S * dnom)

    samples = [s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; penalty]

    return samples
end

# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::StaticVector,
                              acontrol::StaticVector, time::Real, dt::Real) where {n}
    # base dynamics
    negi_hc = astate[CONTROLS_IDX[1]] * NEGI_H1_ISO
    h_prop = exp(dt * (FQ_NEGI_H0_ISO + negi_hc))
    state1 = h_prop * astate[STATE1_IDX]
    state2 = h_prop * astate[STATE2_IDX]
    # state3 = h_prop * astate[STATE3_IDX]
    # state4 = h_prop * astate[STATE4_IDX]
    intcontrols = astate[INTCONTROLS_IDX] + dt * astate[CONTROLS_IDX]
    controls = astate[CONTROLS_IDX] + dt * astate[DCONTROLS_IDX]
    dcontrols = astate[DCONTROLS_IDX] + dt * acontrol[D2CONTROLS_IDX]
    
    astate_ = [
        # state1; state2; state3; state4; intcontrols; controls; dcontrols;
        state1; state2; intcontrols; controls; dcontrols;
    ]

    # unscented transform
    for i = 1:model.sample_state_count
        sample_states = unscented_transform(model, astate, negi_hc, h_prop, dt, i)
        astate_ = [astate_; sample_states]
    end
    
    return astate_
end

# main
function run_traj(;gate_type=xpiby2, evolution_time=60., solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true,
                  fq_cov=FQ * 1e-2, max_iterations=Int64(2e5), gradient_tol_int=1,
                  dJ_counter_limit=Int(1e2), state_cov=1e-2, seed=0, alpha=1.,
                  sample_states=[IS1_ISO_], nominal_idxs=[STATE1_IDX], static=true)
    Random.seed!(seed)
    (sample_state_count,) = size(sample_states)
    S = Diagonal(SVector{HDIM_ISO}(fill(qs[5], HDIM_ISO)))
    model = Model(S, nominal_idxs, fq_cov, alpha, sample_state_count)
    n = RD.state_dim(model)
    m = RD.control_dim(model)
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
    state_dist = Distributions.Normal(0., state_cov)
    for i = 1:sample_state_count
        sample_state = sample_states[i]
        for j = 1:SAMPLES_PER_STATE
            sample_idx = astate_sample_inds(i, j)
            sample = sample_state .+ rand(state_dist, HDIM_ISO)
            x0[sample_idx] = sample ./ sqrt(sample'sample)
        end
    end
    if static
        x0 = SVector{n}(x0)
        xf = SVector{n}(xf)
    end

    # control amplitude constraint
    x_max = fill(Inf, n)
    x_max[CONTROLS_IDX] .= MAX_CONTROL_NORM_0
    x_min = fill(-Inf, n)
    x_min[CONTROLS_IDX] .= -MAX_CONTROL_NORM_0
    if static
        x_max = SVector{n}(x_max)
        x_min = SVector{n}(x_min)
    end
    
    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n)
    x_max_boundary[CONTROLS_IDX] .= 0
    x_min_boundary = fill(-Inf, n)
    x_min_boundary[CONTROLS_IDX] .= 0
    if static
        x_max_boundary = SVector{n}(x_max_boundary)
        x_min_boundary = SVector{n}(x_min_boundary)
    end
    
    # initial trajectory
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    if static
        U0 = [SVector{m}(fill(1e-4, CONTROL_COUNT)) for k = 1:N-1]
        X0 = [SVector{n}(fill(NaN, n)) for k = 1:N]        
    else
        U0 = [fill(1e-4, CONTROL_COUNT) for k = 1:N-1]
        X0 = [fill(NaN, n) for k = 1:N]
    end
    Z = Traj(X0, U0, dt * ones(N))

    # cost function
    Q = zeros(n)
    Q[STATE1_IDX] = Q[STATE2_IDX] = fill(qs[1], HDIM_ISO)
    Q[INTCONTROLS_IDX] = fill(qs[2], CONTROL_COUNT)
    Q[CONTROLS_IDX] = fill(qs[3], CONTROL_COUNT)
    Q[DCONTROLS_IDX] = fill(qs[4], CONTROL_COUNT)
    # penalty has unit value, its cost can be increased through qs[5],
    # which modifies the model.S matrix
    for i = 1:sample_state_count
        Q[ASTATE_SIZE_BASE + i * CHUNK_SIZE] = 1
    end
    R = zeros(m)
    R[D2CONTROLS_IDX] = fill(qs[6], CONTROL_COUNT)
    if static
        Q = SVector{n}(Q)
        R = SVector{m}(R)
    end
    Q = Diagonal(Q)
    Qf = Q * N
    R = Diagonal(R)
    objective = LQRObjective(Q, R, Qf, xf, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    norm_idxs = sample_idxs(model)
    push!(norm_idxs, STATE1_IDX)
    push!(norm_idxs, STATE2_IDX)
    # push!(norm_idxs, STATE3_IDX)
    # push!(norm_idxs, STATE4_IDX)
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
    static_bp = static ? true : false
    set_options!(
        solver, square_root=sqrtbp, constraint_tolerance=constraint_tolerance,
        projected_newton_tolerance=al_tol, n_steps=n_steps,
        penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
        projected_newton=projected_newton, iterations_inner=iterations_inner,
        iterations_outer=iterations_outer, iterations=max_iterations,
        gradient_tolerance_intermediate=gradient_tol_int,
        dJ_counter_limit=dJ_counter_limit, static_bp=static_bp
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
