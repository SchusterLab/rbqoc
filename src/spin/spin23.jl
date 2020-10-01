"""
spin23.jl - unscented transform robustness for the δfq problem
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
using Distributions
using HDF5
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
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const S1STATE1_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
const S2STATE1_IDX = S1STATE1_IDX[end] + 1:S1STATE1_IDX[end] + HDIM_ISO
const S3STATE1_IDX = S2STATE1_IDX[end] + 1:S2STATE1_IDX[end] + HDIM_ISO
const S4STATE1_IDX = S3STATE1_IDX[end] + 1:S3STATE1_IDX[end] + HDIM_ISO
const S5STATE1_IDX = S4STATE1_IDX[end] + 1:S4STATE1_IDX[end] + HDIM_ISO
const S6STATE1_IDX = S5STATE1_IDX[end] + 1:S5STATE1_IDX[end] + HDIM_ISO
const S7STATE1_IDX = S6STATE1_IDX[end] + 1:S6STATE1_IDX[end] + HDIM_ISO
const S8STATE1_IDX = S7STATE1_IDX[end] + 1:S7STATE1_IDX[end] + HDIM_ISO
const S9STATE1_IDX = S8STATE1_IDX[end] + 1:S8STATE1_IDX[end] + HDIM_ISO
const SAMPLE_COUNT = 9
const SAMPLE_COUNT_INV = 1//9
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
    cov::MMatrix{HDIM_ISO + 1, HDIM_ISO + 1}
    negi_hp::SMatrix{HDIM_ISO, HDIM_ISO}
    negi_hn::SMatrix{HDIM_ISO, HDIM_ISO}
    fq_cov::Float64
    alpha::Float64
end
end
Model = Data.Model
@inline RD.state_dim(model::Model) = ASTATE_SIZE
@inline RD.control_dim(model::Model) = ACONTROL_SIZE

# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::StaticVector{ASTATE_SIZE},
                              acontrol::StaticVector{ACONTROL_SIZE}, time::Real, dt::Real)
    # base dynamics
    negi_hc = astate[CONTROLS_IDX[1]] * NEGI_H1_ISO
    h_prop = exp(dt * (FQ_NEGI_H0_ISO + negi_hc))
    state1 = h_prop * astate[STATE1_IDX]
    state2 = h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX[1]] + dt * astate[CONTROLS_IDX[1]]
    controls = astate[CONTROLS_IDX[1]] + dt * astate[DCONTROLS_IDX[1]]
    dcontrols = astate[DCONTROLS_IDX[1]] + dt * acontrol[D2CONTROLS_IDX[1]]

    # unscented transform
    s1 = SVector{HDIM_ISO}(astate[S1STATE1_IDX])
    s2 = SVector{HDIM_ISO}(astate[S2STATE1_IDX])
    s3 = SVector{HDIM_ISO}(astate[S3STATE1_IDX])
    s4 = SVector{HDIM_ISO}(astate[S4STATE1_IDX])
    s5 = SVector{HDIM_ISO}(astate[S5STATE1_IDX])
    s6 = SVector{HDIM_ISO}(astate[S6STATE1_IDX])
    s7 = SVector{HDIM_ISO}(astate[S7STATE1_IDX])
    s8 = SVector{HDIM_ISO}(astate[S8STATE1_IDX])
    s9 = SVector{HDIM_ISO}(astate[S9STATE1_IDX])
    # compute state mean
    sm = SAMPLE_COUNT_INV .* (
        s1 + s2 + s3 + s4
        + s5 + s6 + s7 + s8
        + s9
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
    s_cov = 0.5 .* (
        d1 * d1' + d2 * d2' + d3 * d3' + d4 * d4'
        + d5 * d5' + d6 * d6' + d7 * d7' + d8 * d8'
        + d9 * d9'
    )
    # perform cholesky decomposition on joint covariance
    model.cov[1:HDIM_ISO, 1:HDIM_ISO] = s_cov
    model.cov[HDIM_ISO + 1, HDIM_ISO + 1] = model.fq_cov
    cov_chol = cholesky(Symmetric(model.cov)).L
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
    s5 = exp(dt * ((FQ - fq_chol1) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol1)
    s6 = exp(dt * ((FQ - fq_chol2) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol2)
    s7 = exp(dt * ((FQ - fq_chol3) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol3)
    s8 = exp(dt * ((FQ - fq_chol4) * NEGI_H0_ISO + negi_hc)) * (sm - s_chol4)
    s9 = exp(dt * ((FQ + fq_chol5) * NEGI_H0_ISO + negi_hc)) * sm
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
    
    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
        s1; s2; s3; s4; s5; s6; s7; s8; s9;
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
    hess_astate[S1STATE1_IDX, S1STATE1_IDX] = hess_sample
    hess_astate[S2STATE1_IDX, S2STATE1_IDX] = hess_sample
    hess_astate[S3STATE1_IDX, S3STATE1_IDX] = hess_sample
    hess_astate[S4STATE1_IDX, S4STATE1_IDX] = hess_sample
    hess_astate[S5STATE1_IDX, S5STATE1_IDX] = hess_sample
    hess_astate[S6STATE1_IDX, S6STATE1_IDX] = hess_sample
    hess_astate[S7STATE1_IDX, S7STATE1_IDX] = hess_sample
    hess_astate[S8STATE1_IDX, S8STATE1_IDX] = hess_sample
    hess_astate[S9STATE1_IDX, S9STATE1_IDX] = hess_sample
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
        gate_error_iso2(astate, cost.target_state1, S1STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S2STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S3STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S4STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S5STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S6STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S7STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S8STATE1_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state1, S9STATE1_IDX[1] - 1)
    )
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}, acontrol::SVector{M}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.q = (cost.Q * astate + cost.q + [
        @SVector zeros(ASTATE_SIZE_BASE);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S1STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S2STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S3STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S4STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S5STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S6STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S7STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S8STATE1_IDX[1] - 1);
        cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S9STATE1_IDX[1] - 1);
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
    cov = MMatrix{HDIM_ISO + 1, HDIM_ISO + 1}(zeros(HDIM_ISO + 1, HDIM_ISO + 1))
    negi_hp = (FQ + fq_cov) * NEGI_H0_ISO
    negi_hn = (FQ - fq_cov) * NEGI_H0_ISO
    model = Model(cov, negi_hp, negi_hn, fq_cov, alpha)
    n = RD.state_dim(model)
    m = RD.control_dim(model)
    t0 = 0.

    # initial state
    state_dist = Distributions.Normal(0., state_cov)
    x0_ = [
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
    ]
    for i = 1:SAMPLE_COUNT
        sample = INITIAL_STATE1 .+ rand(state_dist, HDIM_ISO)
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
        repeat(target_state1, SAMPLE_COUNT);
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
        # fill(0, SAMPLE_COUNT * HDIM_ISO);
        fill(qs[5], SAMPLE_COUNT * HDIM_ISO);
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[6], CONTROL_COUNT); # ∂2a
    ]))
    objective = LQRObjective(Q, R, Qf, xf, N)
    # cost_k = Cost(Q, R, xf, target_state1, qs[5])
    # cost_f = Cost(Qf, R, xf, target_state1, qs[5])
    # objective = TO.Objective(cost_k, cost_f, N)

    # constraints
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
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
    knot_count = size(saved["astates"], 1)
    diffs_ = zeros(SAMPLE_COUNT, knot_count)
    fds_ = zeros(SAMPLE_COUNT, knot_count)
    for i = 1:knot_count
        x11 = saved["astates"][i, S1STATE1_IDX]
        x21 = saved["astates"][i, S2STATE1_IDX]
        x31 = saved["astates"][i, S3STATE1_IDX]
        x41 = saved["astates"][i, S4STATE1_IDX]
        x51 = saved["astates"][i, S5STATE1_IDX]
        x61 = saved["astates"][i, S6STATE1_IDX]
        x71 = saved["astates"][i, S7STATE1_IDX]
        x81 = saved["astates"][i, S8STATE1_IDX]
        d11 = x11 - XPIBY2_ISO_1
        d21 = x21 - XPIBY2_ISO_1
        d31 = x31 - XPIBY2_ISO_1
        d41 = x41 - XPIBY2_ISO_1
        d51 = x51 - XPIBY2_ISO_1
        d61 = x61 - XPIBY2_ISO_1
        d71 = x71 - XPIBY2_ISO_1
        d81 = x81 - XPIBY2_ISO_1
        diffs_[1, i] = d11'd11
        diffs_[2, i] = d21'd21
        diffs_[3, i] = d31'd31
        diffs_[4, i] = d41'd41
        diffs_[5, i] = d51'd51
        diffs_[6, i] = d61'd61
        diffs_[7, i] = d71'd71
        diffs_[8, i] = d81'd81
        fds_[1, i] = fidelity_vec_iso2(x11, XPIBY2_ISO_1)
        fds_[2, i] = fidelity_vec_iso2(x21, XPIBY2_ISO_1)
        fds_[3, i] = fidelity_vec_iso2(x31, XPIBY2_ISO_1)
        fds_[4, i] = fidelity_vec_iso2(x41, XPIBY2_ISO_1)
        fds_[5, i] = fidelity_vec_iso2(x51, XPIBY2_ISO_1)
        fds_[6, i] = fidelity_vec_iso2(x61, XPIBY2_ISO_1)
        fds_[7, i] = fidelity_vec_iso2(x71, XPIBY2_ISO_1)
        fds_[8, i] = fidelity_vec_iso2(x81, XPIBY2_ISO_1)
    end
    return (diffs_, fds_)
end
