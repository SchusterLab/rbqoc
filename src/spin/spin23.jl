"""
spin23.jl - unscented transform robustness for the δf_q problem
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
# const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const ASTATE_SIZE_BASE = HDIM_ISO
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
const SAMPLE_COUNT = 8
const SAMPLE_COUNT_INV = 1//8
const ASTATE_SIZE = ASTATE_SIZE_BASE # + SAMPLE_COUNT * HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
mutable struct Model <: RD.AbstractModel
    s1_samples::Array{SVector{HDIM_ISO}, 1}
    fq_samples::MVector{SAMPLE_COUNT}
    fq_dist::Distributions.Sampleable
    alpha::Float64
end
@inline RD.state_dim(model::Model) = ASTATE_SIZE
@inline RD.control_dim(model::Model) = ACONTROL_SIZE


function unscented_transform!(model::Model, z::AbstractKnotPoint{T,N,M}) where {T,N,M}
    s11 = z.z[S1STATE1_IDX]
    s21 = z.z[S2STATE1_IDX]
    s31 = z.z[S3STATE1_IDX]
    s41 = z.z[S4STATE1_IDX]
    s51 = z.z[S5STATE1_IDX]
    s61 = z.z[S6STATE1_IDX]
    s71 = z.z[S7STATE1_IDX]
    s81 = z.z[S8STATE1_IDX]
    
    s1m = SAMPLE_COUNT_INV .* (
        s11 + s21 + s31 + s41
        + s51 + s61 + s71 + s81
    )
    d1 = s11 - s1m
    d2 = s21 - s1m
    d3 = s31 - s1m
    d4 = s41 - s1m
    d5 = s51 - s1m
    d6 = s61 - s1m
    d7 = s71 - s1m
    d8 = s81 - s1m
    cov = 0.5 .* (
        d1 * d1' + d2 * d2' + d3 * d3' + d4 * d4'
        + d5 * d5' + d6 * d6' + d7 * d7' + d8 * d8'
    )
    cov_chol = model.alpha * cholesky(cov, check=false).L

    s11 = s1m + cov_chol[1:HDIM_ISO, 1]
    s21 = s1m - cov_chol[1:HDIM_ISO, 1]
    s31 = s1m + cov_chol[1:HDIM_ISO, 2]
    s41 = s1m - cov_chol[1:HDIM_ISO, 2]
    s51 = s1m + cov_chol[1:HDIM_ISO, 3]
    s61 = s1m - cov_chol[1:HDIM_ISO, 3]
    s71 = s1m + cov_chol[1:HDIM_ISO, 4]
    s81 = s1m - cov_chol[1:HDIM_ISO, 4]
    model.s1_samples[1] = s11 ./ sqrt(s11's11)
    model.s1_samples[2] = s21 ./ sqrt(s21's21)
    model.s1_samples[3] = s31 ./ sqrt(s31's31)
    model.s1_samples[4] = s41 ./ sqrt(s41's41)
    model.s1_samples[5] = s51 ./ sqrt(s51's51)
    model.s1_samples[6] = s61 ./ sqrt(s61's61)
    model.s1_samples[7] = s71 ./ sqrt(s71's71)
    model.s1_samples[8] = s81 ./ sqrt(s81's81)
    rand!(model.fq_dist, model.fq_samples)
    model.fq_samples .+= FQ

    return nothing
end


# dynamics
function discrete_dynamics_(model::Model, astate::StaticVector{ASTATE_SIZE},
                            acontrol::StaticVector{ACONTROL_SIZE}, time::Real, dt::Real)
    # negi_hc = astate[CONTROLS_IDX[1]] * NEGI_H1_ISO
    negi_hc = acontrol[D2CONTROLS_IDX[1]] * NEGI_H1_ISO
    negi_s0h = FQ_NEGI_H0_ISO + negi_hc
    negi_s0h_prop = exp(negi_s0h * dt)
    state1 = negi_s0h_prop * astate[STATE1_IDX]
    # state2 = negi_s0h_prop * astate[STATE2_IDX]
    # intcontrols = astate[INTCONTROLS_IDX[1]] + dt * astate[CONTROLS_IDX[1]]
    # controls = astate[CONTROLS_IDX[1]] + dt * astate[DCONTROLS_IDX[1]]
    # dcontrols = astate[DCONTROLS_IDX[1]] + dt * acontrol[D2CONTROLS_IDX[1]]

    # s11 = exp(dt * (model.fq_samples[1] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[1]
    # s21 = exp(dt * (model.fq_samples[2] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[2]
    # s31 = exp(dt * (model.fq_samples[3] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[3]
    # s41 = exp(dt * (model.fq_samples[4] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[4]
    # s51 = exp(dt * (model.fq_samples[5] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[5]
    # s61 = exp(dt * (model.fq_samples[6] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[6]
    # s71 = exp(dt * (model.fq_samples[7] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[7]
    # s81 = exp(dt * (model.fq_samples[8] * NEGI_H0_ISO + negi_hc)) * model.s1_samples[8]
    
    astate_ = [
        state1;
        # state2;
        # intcontrols; controls; dcontrols;
        # s11; s21; s31; s41; s51; s61; s71; s81;
    ]
    
    return astate_
end


# Note that TO.rollout! uses RK3.
function RD.discrete_dynamics(::Type{RK3}, model::Model, z::AbstractKnotPoint{T,N,M}) where {T,N,M}
    # unscented_transform!(model, z)
    return discrete_dynamics_(model, RD.state(z), RD.control(z), z.t, z.dt)
end


function RD.discrete_jacobian!(::Type{RK3}, ∇f, model::Model, z::AbstractKnotPoint{T,N,M}) where {T,N,M}
    ix,iu,idt = z._x, z._u, N+M+1
    t = z.t
    # unscented_transform!(model, z)
    fd_aug(s) = discrete_dynamics_(model, s[ix], s[iu], t, z.dt)
    ∇f .= ForwardDiff.jacobian(fd_aug, SVector{N+M}(z.z))
    return nothing
end


# cost function
struct Cost{N,M,T} <: TO.CostFunction
    hess_astate::Symmetric{T, SMatrix{N,N,T}}
    R::Diagonal{T, SVector{M,T}}
    target_state1::SVector{HDIM_ISO, T}
    target_state2::SVector{HDIM_ISO, T}
    q_s::SVector{1, T}
    q_inta::SVector{1, T}
    q_a::SVector{1, T}
    q_da::SVector{1, T}
    q_ss::SVector{1, T}
end


function Cost(target_state1::SVector{HDIM_ISO, T}, target_state2::SVector{HDIM_ISO, T},
              qs::Array{T, 1}) where {T}
    N = ASTATE_SIZE
    M = ACONTROL_SIZE
    # hessian is independent of astate_k and acontrol_k (huzzah!)
    hess_astate = zeros(N, N)
    hess_astate[STATE1_IDX, STATE1_IDX] = qs[1] * hessian_gate_error_iso2(target_state1)
    # hess_astate[STATE2_IDX, STATE2_IDX] = qs[1] * hessian_gate_error_iso2(target_state2)
    # hess_astate[INTCONTROLS_IDX, INTCONTROLS_IDX] .= qs[2]
    # hess_astate[CONTROLS_IDX, CONTROLS_IDX] .= qs[3]
    # hess_astate[DCONTROLS_IDX, DCONTROLS_IDX] .= qs[4]
    # hess_sample = qs[5] * hessian_gate_error_iso2(target_state1)
    # hess_astate[S1STATE1_IDX, S1STATE1_IDX] = hess_sample
    # hess_astate[S2STATE1_IDX, S2STATE1_IDX] = hess_sample
    # hess_astate[S3STATE1_IDX, S3STATE1_IDX] = hess_sample
    # hess_astate[S4STATE1_IDX, S4STATE1_IDX] = hess_sample
    # hess_astate[S5STATE1_IDX, S5STATE1_IDX] = hess_sample
    # hess_astate[S6STATE1_IDX, S6STATE1_IDX] = hess_sample
    # hess_astate[S7STATE1_IDX, S7STATE1_IDX] = hess_sample
    # hess_astate[S8STATE1_IDX, S8STATE1_IDX] = hess_sample
    hess_astate = Symmetric(SMatrix{N, N}(hess_astate))
    R = Diagonal(SVector{M}(qs[6]))
    return Cost{N,M,T}(
        hess_astate, R, target_state1, target_state2, SVector{1}(qs[1]),
        SVector{1}(qs[2]), SVector{1}(qs[3]), SVector{1}(qs[4]), SVector{1}(qs[5])
    )
end


@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
        cost.hess_astate, cost.R, cost.target_state1, cost.target_state2, cost.q_s,
        cost.q_inta, cost.q_a, cost.q_da, cost.q_ss
)


function TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}) where {N,M,T}
    q_s = cost.q_s[1] * (
        gate_error_iso2(astate, cost.target_state1, STATE1_IDX[1] - 1)
        # + gate_error_iso2(astate, cost.target_state2, STATE2_IDX[1] - 1)
    )
    # cost_as = (
    #     + 0.5 * cost.q_inta[1] * astate[INTCONTROLS_IDX[1]]^2
    #     + 0.5 * cost.q_a[1] * astate[CONTROLS_IDX[1]]^2
    #     + 0.5 * cost.q_da[1] * astate[DCONTROLS_IDX[1]]^2
    # )
    # if cost_as < 0
    #     println("cost_as: $(cost_as)")
    # end
    # + cost.q_ss * (
    #     gate_error_iso2(astate, cost.target_state1, S1STATE1_IDX[1] - 1)
    #     + gate_error_iso2(astate, cost.target_state1, S2STATE1_IDX[1] - 1)
    #     + gate_error_iso2(astate, cost.target_state1, S3STATE1_IDX[1] - 1)
    #     + gate_error_iso2(astate, cost.target_state1, S4STATE1_IDX[1] - 1)
    #     + gate_error_iso2(astate, cost.target_state1, S5STATE1_IDX[1] - 1)
    #     + gate_error_iso2(astate, cost.target_state1, S6STATE1_IDX[1] - 1)
    #     + gate_error_iso2(astate, cost.target_state1, S7STATE1_IDX[1] - 1)
    #     + gate_error_iso2(astate, cost.target_state1, S8STATE1_IDX[1] - 1)
    # )
    @assert q_s > 0
    return q_s
end


# TO.stage_cost(cost::Cost{N,M,T}, astate::StaticVector{N,T}, acontrol::StaticVector{M,T}) where {N,M,T} = (
function TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}, acontrol::SVector{M}) where {N,M,T}
    ucost_ = 0.5 * acontrol' * cost.R * acontrol
    @assert ucost_ > 0
    cost_ = TO.stage_cost(cost, astate) + ucost_
    return cost_
end


function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.q .= [
        cost.q_s[1] * jacobian_gate_error_iso2(astate, cost.target_state1, STATE1_IDX[1] - 1);
        # cost.q_s[1] * jacobian_gate_error_iso2(astate, cost.target_state2, STATE2_IDX[1] - 1);
        # cost.q_inta * astate[INTCONTROLS_IDX[1]];
        # cost.q_a * astate[CONTROLS_IDX[1]];
        # cost.q_da * astate[DCONTROLS_IDX[1]];
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S1STATE1_IDX[1] - 1);
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S2STATE1_IDX[1] - 1);
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S3STATE1_IDX[1] - 1);
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S4STATE1_IDX[1] - 1);
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S5STATE1_IDX[1] - 1);
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S6STATE1_IDX[1] - 1);
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S7STATE1_IDX[1] - 1);
        # cost.q_ss * jacobian_gate_error_iso2(astate, cost.target_state1, S8STATE1_IDX[1] - 1);
    ]
    return false
end


function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T},
                      acontrol::SVector{M,T}) where {N,M,T}
    TO.gradient!(E, cost, astate)
    E.r .= cost.R * acontrol
    E.c = 0
    # f(x, u) = TO.stage_cost(cost, x, u)
    # (jac_x, jac_u) = Zygote.gradient(f, astate, acontrol)
    # @assert isapprox(E.q, jac_x)
    # @assert isapprox(E.r, jac_u)
    return false
end


function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.Q .= cost.hess_astate
    return true
end


function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T},
                     acontrol::SVector{M,T}) where {N,M,T}
    TO.hessian!(E, cost, astate)
    E.R .= cost.R
    E.H .= 0
    # fx(x) = TO.stage_cost(cost, x, acontrol)    
    # fu(u) = TO.stage_cost(cost, astate, u)
    # hess_x = Zygote.hessian(fx, astate)
    # hess_u = Zygote.hessian(fu, acontrol)
    # @assert isapprox(E.Q, hess_x)
    # @assert isapprox(E.R, hess_u)
    return true
end


"""
Ensure gate error -> 0
"""
struct GateErrorConstraint{P,T} <: TO.StateConstraint
    target_state::SVector{P,T}
    inds::UnitRange{Int}
end
function GateErrorConstraint(target_state::SVector{P,T}, inds::UnitRange{Int}) where {P,T}
    return GateErrorConstraint{P,T}(target_state, inds)
end
@inline Base.copy(con::GateErrorConstraint{P,T}) where {P,T} = GateErrorConstraint{P,T}(con.target_state, con.inds)
@inline Base.length(con::GateErrorConstraint{P,T}) where {P,T} = P
@inline TO.sense(con::GateErrorConstraint) = TO.Equality()
@inline TO.state_dim(con::GateErrorConstraint{P,T}) where {P,T} = P
@inline TO.widths(con::GateErrorConstraint{P,T}) where {P,T} = (P,)
@inline TO.is_bound(con::GateErrorConstraint) = true
@inline TO.get_inds(con::GateErrorConstraint) = con.inds
@inline TO.evaluate(con::GateErrorConstraint, astate::SVector) = gate_error_iso2(astate, con.target_state,
                                                                              con.inds[1] - 1)


function TO.jacobian!(con_grad, con::GateErrorConstraint, z::AbstractKnotPoint)
    con_grad .= jacobian_gate_error_iso2(z.z, con.target_state, con.inds[1] - 1);
end


function TO.∇jacobian!(hess, con::GateErrorConstraint, z::AbstractKnotPoint{T,N,M}, λ::AbstractVector) where {T,N,M}
    hess_ = λ[1] * hessian_gate_error_iso2(con.target_state)
    hess .+= hess_
    return true
end


# main
function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true,
                  fq_cov=FQ * 1e-2, max_iterations=Int64(2e5), gradient_tol_int=1,
                  dJ_counter_limit=Int(1e2), astate_cov=1e-2, seed=0, alpha=1.,
                  iterations_linesearch=20, line_search_lower_bound=1e-8,
                  line_search_upper_bound=10.)
    Random.seed!(seed)
    astate_dist = Distributions.Normal(0., astate_cov)
    s1_samples = [SVector{HDIM_ISO}(zeros(HDIM_ISO)) for i = 1:SAMPLE_COUNT]
    fq_dist = Distributions.Normal(0., fq_cov)
    fq_samples = MVector{SAMPLE_COUNT}(zeros(SAMPLE_COUNT))
    model = Model(s1_samples, fq_samples, fq_dist, alpha)
    n = RD.state_dim(model)
    m = RD.control_dim(model)
    t0 = 0.

    # initial state
    x0_ = [
        INITIAL_STATE1;
        # INITIAL_STATE2;
        # zeros(3 * CONTROL_COUNT);
    ]
    # for i = 1:SAMPLE_COUNT
    #     sample = INITIAL_STATE1 .+ rand(astate_dist, HDIM_ISO)
    #     append!(x0_, sample ./ sqrt(sample'sample))
    # end
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
        # target_state2;
        # zeros(3 * CONTROL_COUNT);
        # repeat(target_state1, SAMPLE_COUNT);
    ])
    
    # control amplitude constraint
    x_max = SVector{n}([
        # fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, HDIM_ISO);
        # fill(Inf, CONTROL_COUNT);
        # fill(MAX_CONTROL_NORM_0, 1); # control
        # fill(Inf, CONTROL_COUNT);
        # fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min = SVector{n}([
        # fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, HDIM_ISO);
        # fill(-Inf, CONTROL_COUNT);
        # fill(-MAX_CONTROL_NORM_0, 1); # control
        # fill(-Inf, CONTROL_COUNT);
        # fill(-Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    # controls start and end at 0
    x_max_boundary = SVector{n}([
        # fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, HDIM_ISO);
        # fill(Inf, CONTROL_COUNT);
        # fill(0, 1); # control
        # fill(Inf, CONTROL_COUNT);
        # fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min_boundary = SVector{n}([
        # fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, HDIM_ISO);
        # fill(-Inf, CONTROL_COUNT);
        # fill(0, 1); # control
        # fill(-Inf, CONTROL_COUNT);
        # fill(-Inf, SAMPLE_COUNT * HDIM_ISO);
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

    cost_k = Cost(target_state1, target_state2, qs)
    cost_f = Cost(target_state1, target_state2, [qs[1:5] * N; qs[6]])
    obj = TO.Objective(cost_k, cost_f, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must statisfy conrols start and stop at 0.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must reach target state. Must have zero net flux.
    target_state1_constraint = GateErrorConstraint(target_state1, STATE1_IDX)
    # target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # Must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    # add_constraint!(constraints, control_bnd, 2:N-2)
    # add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    # add_constraint!(constraints, target_astate_constraint, N:N);
    add_constraint!(constraints, target_state1_constraint, N:N)
    # add_constraint!(constraints, normalization_constraint_1, 2:N-1)
    # add_constraint!(constraints, normalization_constraint_2, 2:N-1)

    # Instantiate problem and solve.
    prob = Problem{IT_RDI[integrator_type]}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
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
        dJ_counter_limit=dJ_counter_limit, iterations_linesearch=iterations_linesearch,
        line_search_lower_bound=line_search_lower_bound,
        line_search_upper_bound=line_search_upper_bound,
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
