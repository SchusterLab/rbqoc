"""
spin24.jl - multiple sampling robustness for the δ
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
const EXPERIMENT_NAME = "spin24"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const ASTATE_SIZE = HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT
const INITIAL_STATE1 = [1., 0, 0, 0]
# state indices
const STATE1_IDX = 1:HDIM_ISO
# control indices
const CONTROL_IDX = 1:CONTROL_COUNT

# model
mutable struct Model <: RD.AbstractModel
end
@inline RD.state_dim(model::Model) = ASTATE_SIZE
@inline RD.control_dim(model::Model) = ACONTROL_SIZE


# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::StaticVector{ASTATE_SIZE},
                              acontrol::StaticVector{ACONTROL_SIZE}, time::Real, dt::Real)
    negi_hc = acontrol[CONTROL_IDX[1]] * NEGI_H1_ISO
    state1 = exp(dt * (FQ_NEGI_H0_ISO + negi_hc)) * astate
    return state1
end


# cost function
struct Cost{N,M,T} <: TO.CostFunction
    hess_astate::Symmetric{T, SMatrix{N,N,T}}
    target_state1::SVector{HDIM_ISO, T}
    R::Diagonal{T, SVector{M,T}}
    q::T
end

function Cost(target_state1::SVector{HDIM_ISO, T}, q::T, r::T) where {T}
    N = ASTATE_SIZE
    M = ACONTROL_SIZE
    # For reasons unkown to the author, this cost function works properly
    # when a -1 is thrown infront of the hessian.
    hess_astate = -1 * q * hessian_gate_error_iso2(target_state1)
    hess_astate = Symmetric(SMatrix{N, N}(hess_astate))
    R = Diagonal(SVector{M}(r))
    return Cost{N,M,T}(hess_astate, target_state1, R, q)
end
@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
    cost.hess_astate, cost.target_state1, cost.R, cost.q
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}) where {N,M,T} = (
    cost.q * gate_error_iso2(astate, cost.target_state1)
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}, acontrol::SVector{M}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.q = cost.q * jacobian_gate_error_iso2(astate, cost.target_state1)
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
function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, q=1e0, r=1e0,
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e8, verbose=true, save=true,
                  max_iterations=Int64(2e5), line_search_lower_bound=1e-8,
                  line_search_upper_bound=10, dJ_counter_limit=Int(1e3))
    model = Model()
    n = RD.state_dim(model)
    m = RD.control_dim(model)
    t0 = 0.

    # initial state
    x0 = SVector{n}([
        INITIAL_STATE1;
    ])

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
    ])

    # initial trajectory
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [SVector{m}([
        fill(1e-2, CONTROL_COUNT);
    ]) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    # cost function
    ck = Cost(target_state1, q, r)
    cf = Cost(target_state1, q * N, r)
    objective = Objective(ck, cf, N)

    # constraints
    constraints = ConstraintList(n, m, N)

    # solve
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
        line_search_lower_bound=line_search_lower_bound,
        line_search_upper_bound=line_search_upper_bound,
        dJ_counter_limit=dJ_counter_limit,
    )
    Altro.solve!(solver)

    # post-process
    acontrols_raw = TO.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TO.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    cmax = TO.max_violation(solver)
    cmax_info = TO.findmax_violation(TO.get_constraints(solver))
    iterations_ = Altro.iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "evolution_time" => evolution_time,
        "astates" => astates_arr,
        "q" => q,
        "r" => r,
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
        "max_iterations" => max_iterations,
    )

    return result
end
