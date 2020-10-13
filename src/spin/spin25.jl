"""
spin25.jl - unscented transform robustness for the δa problem
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
const EXPERIMENT_NAME = "spin25"
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
const PFIRX_IDX = S10_IDX[end] + 1:S10_IDX[end] + 3
const PFIRY_IDX = PFIRX_IDX[end] + 1:PFIRX_IDX[end] + 3
const SAMPLE_COUNT = 10
const SAMPLE_COUNT_INV = 1//10
const PFIR_SIZE = 6
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO + PFIR_SIZE
const ACONTROL_SIZE = CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
module Data
using Distributions
using RobotDynamics
using StaticArrays
const RD = RobotDynamics
const HDIM_ISO = 4
mutable struct Model <: RD.AbstractModel
    namp::Float64
    wnamp_cov::Float64
    alpha::Float64
end
end
Model = Data.Model
@inline RD.state_dim(model::Model) = ASTATE_SIZE
@inline RD.control_dim(model::Model) = ACONTROL_SIZE


# finite impulse response pink noise filter
function pink_filter(xk, xprev::Array{Float64, 1}, yprev::Array{Float64, 1},
                     time::Real, dt::Real)

end


# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::SVector{ASTATE_SIZE},
                              acontrol::SVector{ACONTROL_SIZE}, time::Real, dt::Real)
    # base dynamics
    camp = astate[CONTROLS_IDX[1]]
    h_prop = exp(dt * (FQ_NEGI_H0_ISO + camp * NEGI_H1_ISO))
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
    cov[HDIM_ISO + 1, HDIM_ISO + 1] = model.wnamp_cov
    # TOOD: cholesky! requires writing zeros in upper triangle
    cov_chol = model.alpha * cholesky(Symmetric(cov)).L
    s_chol1 = cov_chol[1:HDIM_ISO, 1]
    s_chol2 = cov_chol[1:HDIM_ISO, 2]
    s_chol3 = cov_chol[1:HDIM_ISO, 3]
    s_chol4 = cov_chol[1:HDIM_ISO, 4]
    amp_chol1 = cov_chol[HDIM_ISO + 1, 1]
    amp_chol2 = cov_chol[HDIM_ISO + 1, 2]
    amp_chol3 = cov_chol[HDIM_ISO + 1, 3]
    amp_chol4 = cov_chol[HDIM_ISO + 1, 4]
    if amp_chol1 != 0 || amp_chol2 != 0 || amp_chol3 != 0 || amp_chol4 != 0
        show_nice(cov_chol[HDIM_ISO + 1, :])
    end
    xk = cov_chol[HDIM_ISO + 1, 5]
    # filter white noise
    knot_point = Int(div(time, dt))
    xp1 = astate[PFIRX_IDX[1]]
    xp2 = astate[PFIRX_IDX[2]]
    xp3 = astate[PFIRX_IDX[3]]
    yp1 = astate[PFIRY_IDX[1]]
    yp2 = astate[PFIRY_IDX[2]]
    yp3 = astate[PFIRY_IDX[3]]
    if knot_point == 1
        yk = PFIR_B1 * xk
    elseif knot_point == 2
        yk = PFIR_B1 * xk + PFIR_B2 * xp1 - PFIR_A2 * yp1
    elseif knot_point == 3
        yk = (PFIR_B1 * xk + PFIR_B2 * xp1 + PFIR_B3 * xp2
              - PFIR_A2 * yp1 - PFIR_A3 * yp2)
    else
        yk = (PFIR_B1 * xk + PFIR_B2 * xp1 + PFIR_B3 * xp2 + PFIR_B4 * xp3
              - PFIR_A2 * yp1 - PFIR_A3 * yp2 - PFIR_A4 * yp3)
    end
    xp3 = xp2
    xp2 = xp1
    xp1 = xk
    yp3 = yp2
    yp2 = yp1
    yp1 = yk
    namp5 = model.namp * yk
    # propagate transformed states
    # s1 = exp(dt * (FQ_NEGI_H0_ISO + (camp + namp1) * NEGI_H1_ISO)) * (sm + s_chol1)
    # s2 = exp(dt * (FQ_NEGI_H0_ISO + (camp + namp2) * NEGI_H1_ISO)) * (sm + s_chol2)
    # s3 = exp(dt * (FQ_NEGI_H0_ISO + (camp + namp3) * NEGI_H1_ISO)) * (sm + s_chol3)
    # s4 = exp(dt * (FQ_NEGI_H0_ISO + (camp + namp4) * NEGI_H1_ISO)) * (sm + s_chol4)
    # s6 = exp(dt * (FQ_NEGI_H0_ISO + (camp - namp1) * NEGI_H1_ISO)) * (sm - s_chol1)
    # s7 = exp(dt * (FQ_NEGI_H0_ISO + (camp - namp2) * NEGI_H1_ISO)) * (sm - s_chol2)
    # s8 = exp(dt * (FQ_NEGI_H0_ISO + (camp - namp3) * NEGI_H1_ISO)) * (sm - s_chol3)
    # s9 = exp(dt * (FQ_NEGI_H0_ISO + (camp - namp4) * NEGI_H1_ISO)) * (sm - s_chol4)
    s1 = h_prop * (sm + s_chol1)
    s2 = h_prop * (sm + s_chol2)
    s3 = h_prop * (sm + s_chol3)
    s4 = h_prop * (sm + s_chol4)
    s5 = exp(dt * (FQ_NEGI_H0_ISO + (camp + namp5) * NEGI_H1_ISO)) * sm
    s6 = h_prop * (sm - s_chol1)
    s7 = h_prop * (sm - s_chol2)
    s8 = h_prop * (sm - s_chol3)
    s9 = h_prop * (sm - s_chol4)
    s10 = exp(dt * (FQ_NEGI_H0_ISO + (camp - namp5) * NEGI_H1_ISO)) * sm
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
        s1; s2; s3; s4; s5; s6; s7; s8; s9; s10;
        xp1; xp2; xp3; yp1; yp2; yp3;
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
        @SVector zeros(PFIR_SIZE);
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
                  namp=NAMP_PREFACTOR, wnamp_cov=1.,
                  max_iterations=Int64(2e5), gradient_tol_int=1,
                  dJ_counter_limit=Int(1e2), state_cov=1e-2, seed=0, alpha=1.,)
    Random.seed!(seed)
    namp = namp * dt_inv
    model = Model(namp, wnamp_cov, alpha)
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
    append!(x0_, zeros(PFIR_SIZE))
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
        zeros(PFIR_SIZE);
    ])
    
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
        # fill(qs[5], SAMPLE_COUNT * HDIM_ISO);
        fill(0, PFIR_SIZE);
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
        "wnamp_cov" => wnamp_cov,
        "namp" => namp,
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
