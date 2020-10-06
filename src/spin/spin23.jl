"""
spin23.jl - unscented transform robustness for the δfq problem
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
using Distributions
using HDF5
using Hyperopt
using ForwardDiff
using LinearAlgebra
using Random
using RobotDynamics
using StaticArrays
using Zygote
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin23"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const INITIAL_STATE1 = [1., 0, 0, 0]
const INITIAL_STATE2 = [0., 1, 0, 0]
const INITIAL_STATE3 = [1., 0, 0, 1] ./ sqrt(2)
const INITIAL_STATE4 = [1., -1, 0, 0] ./ sqrt(2)
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const S1_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
const S2_IDX = S1_IDX[end] + 1:S1_IDX[end] + HDIM_ISO
const S3_IDX = S2_IDX[end] + 1:S2_IDX[end] + HDIM_ISO
const S4_IDX = S3_IDX[end] + 1:S3_IDX[end] + HDIM_ISO
const S5_IDX = S4_IDX[end] + 1:S4_IDX[end] + HDIM_ISO
const S6_IDX = S5_IDX[end] + 1:S5_IDX[end] + HDIM_ISO
const S7_IDX = S6_IDX[end] + 1:S6_IDX[end] + HDIM_ISO
const S8_IDX = S7_IDX[end] + 1:S7_IDX[end] + HDIM_ISO
const S9_IDX = S8_IDX[end] + 1:S8_IDX[end] + HDIM_ISO
const S10_IDX = S9_IDX[end] + 1:S9_IDX[end] + HDIM_ISO
const SAMPLE_COUNT = 10
const SAMPLE_COUNT_INV = 1//10
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
module Data
using RobotDynamics
using StaticArrays
const RD = RobotDynamics
const HDIM_ISO = 4
mutable struct Model <: RD.AbstractModel
    fq_cov::Float64
    alpha::Float64
end
end
Model = Data.Model
@inline RD.state_dim(model::Model) = ASTATE_SIZE
@inline RD.control_dim(model::Model) = ACONTROL_SIZE

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

    # unscented transform
    s1 = SVector{HDIM_ISO}(astate[S1_IDX])
    s2 = SVector{HDIM_ISO}(astate[S2_IDX])
    s3 = SVector{HDIM_ISO}(astate[S3_IDX])
    s4 = SVector{HDIM_ISO}(astate[S4_IDX])
    s5 = SVector{HDIM_ISO}(astate[S5_IDX])
    s6 = SVector{HDIM_ISO}(astate[S6_IDX])
    s7 = SVector{HDIM_ISO}(astate[S7_IDX])
    s8 = SVector{HDIM_ISO}(astate[S8_IDX])
    s9 = SVector{HDIM_ISO}(astate[S9_IDX])
    s10 = SVector{HDIM_ISO}(astate[S10_IDX])
    # compute state mean
    sm = SAMPLE_COUNT_INV .* (
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
    cov = @MMatrix zeros(eltype(s_cov), HDIM_ISO + 1, HDIM_ISO + 1)
    cov[1:HDIM_ISO, 1:HDIM_ISO] .= s_cov
    cov[HDIM_ISO + 1, HDIM_ISO + 1] = model.fq_cov
    # TOOD: cholesky! requires writing zeros in upper triangle
    cov_chol = model.alpha * cholesky(Symmetric(cov)).L
    s_chol1 = cov_chol[1:HDIM_ISO, 1]
    s_chol2 = cov_chol[1:HDIM_ISO, 2]
    s_chol3 = cov_chol[1:HDIM_ISO, 3]
    s_chol4 = cov_chol[1:HDIM_ISO, 4]
    fq_chol1 = cov_chol[HDIM_ISO + 1, 1]
    fq_chol2 = cov_chol[HDIM_ISO + 1, 2]
    fq_chol3 = cov_chol[HDIM_ISO + 1, 3]
    fq_chol4 = cov_chol[HDIM_ISO + 1, 4]
    fq_chol5 = cov_chol[HDIM_ISO + 1, 5]
    # propagate transformed states
    s1 = exp(dt * ((FQ + fq_chol1) * NEGI_H0_ISO + negi_hc)) * (sm + s_chol1)
    s2 = exp(dt * ((FQ + fq_chol2) * NEGI_H0_ISO + negi_hc)) * (sm + s_chol2)
    s3 = exp(dt * ((FQ + fq_chol3) * NEGI_H0_ISO + negi_hc)) * (sm + s_chol3)
    s4 = exp(dt * ((FQ + fq_chol4) * NEGI_H0_ISO + negi_hc)) * (sm + s_chol4)
    s5 = exp(dt * ((FQ + fq_chol5) * NEGI_H0_ISO + negi_hc)) * sm
    s6 = exp(dt * ((FQ - fq_chol1) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol1)
    s7 = exp(dt * ((FQ - fq_chol2) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol2)
    s8 = exp(dt * ((FQ - fq_chol3) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol3)
    s9 = exp(dt * ((FQ - fq_chol4) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol4)
    s10 = exp(dt * ((FQ - fq_chol5) * NEGI_H0_ISO + negi_hc)) * sm
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
    
    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
        s1; s2; s3; s4; s5; s6; s7; s8; s9; s10
    ]
    
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
    hess_astate::Symmetric{T, SMatrix{N,N,T}}
    target_state1::SVector{HDIM_ISO, T}
    q_ss::T
end

function Cost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
              xf::SVector{N,T}, target_state1::SVector{HDIM_ISO,T}, q_ss::T) where {N,M,T}
    q = -Q * xf
    c = 0.5 * xf' * Q * xf
    hess_astate = zeros(N, N)
    # For reasons unkown to the author, throwing a -1 in front
    # of the gate error Hessian makes the cost function work.
    hess_sample = -1 * q_ss * hessian_gate_error_iso2(target_state1)
    hess_astate[S1_IDX, S1_IDX] = hess_sample
    hess_astate[S2_IDX, S2_IDX] = hess_sample
    hess_astate[S3_IDX, S3_IDX] = hess_sample
    hess_astate[S4_IDX, S4_IDX] = hess_sample
    hess_astate[S5_IDX, S5_IDX] = hess_sample
    hess_astate[S6_IDX, S6_IDX] = hess_sample
    hess_astate[S7_IDX, S7_IDX] = hess_sample
    hess_astate[S8_IDX, S8_IDX] = hess_sample
    hess_astate[S9_IDX, S9_IDX] = hess_sample
    hess_astate[S10_IDX, S10_IDX] = hess_sample
    hess_astate += Q
    hess_astate = Symmetric(SMatrix{N, N}(hess_astate))
    return Cost{N,M,T}(Q, R, q, c, hess_astate, target_state1, q_ss)
end

@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
    cost.Q, cost.R, cost.q, cost.c, cost.hess_astate,
    cost.target_state1, cost.q_ss
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}) where {N,M,T} = (
    0.5 * astate' * cost.Q * astate + cost.q'astate + cost.c
    + cost.q_ss * (
        gate_error_iso2(astate, cost.target_state1, S1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S2_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S3_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S4_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S5_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S6_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S7_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S8_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S9_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S10_IDX[1] - 1)
    )
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}, acontrol::SVector{M}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.q = (cost.Q * astate + cost.q + [
        @SVector zeros(ASTATE_SIZE_BASE);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S2_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S3_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S4_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S5_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S6_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S7_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S8_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S9_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S10_IDX[1] - 1);
    ])
    return false
end

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T},
                      acontrol::SVector{M,T}) where {N,M,T}
    TO.gradient!(E, cost, astate)
    E.r = cost.R * acontrol
    E.c = 0
    return false
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.Q = cost.hess_astate
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

    # initial state
    x0_ = [
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
    ]
    state_dist = Distributions.Normal(0., state_cov)
    sample_state = INITIAL_STATE3
    target_sample_state = GT_GATE[gate_type] * sample_state
    for i = 1:SAMPLE_COUNT
        sample = sample_state .+ rand(state_dist, HDIM_ISO)
        append!(x0_, sample ./ sqrt(sample'sample))
    end
    x0 = SVector{n}(x0_)

    # target state
    if gate_type == xpiby2
        target_state1 = XPIBY2_ISO_1
        target_state2 = XPIBY2_ISO_2
    elseif gate_type == ypiby2
        target_state1 = YPIBY2_ISO_1
        target_state2 = YPIBY2_ISO_2
    elseif gate_type == zpiby2
        target_state1 = ZPIBY2_ISO_1
        target_state2 = ZPIBY2_ISO_2
    end
    xf = SVector{n}([
        target_state1;
        target_state2;
        zeros(3 * CONTROL_COUNT);
        repeat(target_sample_state, SAMPLE_COUNT);
    ])
    
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # a
        fill(Inf, CONTROL_COUNT);
        fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # a
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    # control amplitude constraint at boundary
    x_max_boundary = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # a
        fill(Inf, CONTROL_COUNT);
        fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min_boundary = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # a
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, SAMPLE_COUNT * HDIM_ISO);
    ])

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
        # fill(qs[5], SAMPLE_COUNT * HDIM_ISO);
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[6], CONTROL_COUNT); # ∂2a
    ]))
    # objective = LQRObjective(Q, R, Qf, xf, N)
    cost_k = Cost(Q, R, xf, target_state1, qs[5])
    cost_f = Cost(Qf, R, xf, target_state1, N * qs[5])
    objective = TO.Objective(cost_k, cost_f, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idxs) for idxs in (
        STATE1_IDX, STATE2_IDX, S1_IDX, S2_IDX, S3_IDX, S4_IDX, S5_IDX, S6_IDX,
        S7_IDX, S8_IDX, S9_IDX, S10_IDX,
    )]
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    for norm_constraint in norm_constraints
        add_constraint!(constraints, norm_constraint, 2:N-1)
    end

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


function sample_diffs(saved)
    gate_type = GateType(saved["gate_type"])
    knot_count = size(saved["astates"], 1)
    diffs_ = zeros(SAMPLE_COUNT, knot_count)
    fds_ = zeros(SAMPLE_COUNT, knot_count)
    target_state = GT_GATE[gate_type] * INITIAL_STATE3
    for i = 1:knot_count
        s1 = saved["astates"][i, S1_IDX]
        s2 = saved["astates"][i, S2_IDX]
        s3 = saved["astates"][i, S3_IDX]
        s4 = saved["astates"][i, S4_IDX]
        s5 = saved["astates"][i, S5_IDX]
        s6 = saved["astates"][i, S6_IDX]
        s7 = saved["astates"][i, S7_IDX]
        s8 = saved["astates"][i, S8_IDX]
        s9 = saved["astates"][i, S9_IDX]
        s10 = saved["astates"][i, S10_IDX]
        d1 = s1 - target_state
        d2 = s2 - target_state
        d3 = s3 - target_state
        d4 = s4 - target_state
        d5 = s5 - target_state
        d6 = s6 - target_state
        d7 = s7 - target_state
        d8 = s8 - target_state
        d9 = s9 - target_state
        d10 = s10 - target_state
        diffs_[1, i] = d1'd1
        diffs_[2, i] = d2'd2
        diffs_[3, i] = d3'd3
        diffs_[4, i] = d4'd4
        diffs_[5, i] = d5'd5
        diffs_[6, i] = d6'd6
        diffs_[7, i] = d7'd7
        diffs_[8, i] = d8'd8
        diffs_[9, i] = d9'd9
        diffs_[10, i] = d10'd10
        fds_[1, i] = fidelity_vec_iso2(s1, target_state)
        fds_[2, i] = fidelity_vec_iso2(s2, target_state)
        fds_[3, i] = fidelity_vec_iso2(s3, target_state)
        fds_[4, i] = fidelity_vec_iso2(s4, target_state)
        fds_[5, i] = fidelity_vec_iso2(s5, target_state)
        fds_[6, i] = fidelity_vec_iso2(s6, target_state)
        fds_[7, i] = fidelity_vec_iso2(s7, target_state)
        fds_[8, i] = fidelity_vec_iso2(s8, target_state)
        fds_[8, i] = fidelity_vec_iso2(s9, target_state)
        fds_[8, i] = fidelity_vec_iso2(s10, target_state)
    end
    return (fds_, diffs_)
end


function hyperopt_me(;iterations=50, save=true)
    save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    gate_type = zpiby2
    save_file_paths = fill("", iterations)
    weights = zeros(iterations)
    alphas = zeros(iterations)
    gate_errors = zeros(iterations)
    
    result::Dict{String, Any} = Dict(
        "weights" => weights,
        "alphas" => alphas,
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
    ho = Hyperoptimizer(iterations, GPSampler(Min); a=weights_space, b=alpha_space)
    for (i, weight, alpha) in ho
        # evaluate
        res_train = run_traj(;gate_type=gate_type, qs=[1e0, 1e0, 1e0, 1e-1, weight, 1e-1],
                             alpha=alpha, verbose=true, save=true)
        save_file_path_ = res_train["save_file_path"]
        res_eval = evaluate_fqdev(;save_file_path=save_file_path_, gate_type=gate_type)
        gate_error = mean(res_eval["gate_errors"])
        
        # log
        i_ = Int(i)
        weights[i_] = weight
        alphas[i_] = alpha
        gate_errors[i_] = gate_error
        save_file_paths[i_] = save_file_path
        if save
            h5open(save_file_path, "cw") do save_file
                for key in keys(result)
                    o_delete(save_file, key)
                    write(save_file, key, result[key])
                end
            end
        end

        push!(ho.results, gate_error)
        push!(ho.history, [weight, alpha])
    end

    if save
        result["save_file_path"] = save_file_path
    end

    return result
end
