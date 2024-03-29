"""
spin18.jl - sampling robustness
"""

WDIR = joinpath(@__DIR__, "../../")
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
const EXPERIMENT_NAME = "spin18"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const SAMPLE_COUNT = 8
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT
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
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
struct Model <: AbstractModel
    namp::Float64
end
@inline RD.state_dim(::Model) = ASTATE_SIZE
@inline RD.control_dim(::Model) = ACONTROL_SIZE


# This cost puts a gate error cost on
# the sample states and a LQR cost on the other terms.
# The hessian w.r.t the state and controls is constant.
struct Cost{N,M,T} <: TO.CostFunction
    Q::Diagonal{T, SVector{N,T}}
    R::Diagonal{T, SVector{M,T}}
    q::SVector{N, T}
    c::T
    hess_astate::Symmetric{T, SMatrix{N,N,T}}
    target_states::Array{SVector{HDIM_ISO, T}, 1}
    q_ss1::T
    q_ss2::T
    q_ss3::T
    q_ss4::T
end

function Cost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
              xf::SVector{N,T}, target_states::Array{SVector{HDIM_ISO}, 1},
              q_ss1::T, q_ss2::T, q_ss3::T, q_ss4::T) where {N,M,T}
    q = -Q * xf
    c = 0.5 * xf' * Q * xf
    hess_astate = zeros(N, N)
    # For reasons unknown to the author, throwing a -1 in front
    # of the gate error Hessian makes the cost function work.
    # This is strange, because the gate error Hessian has been
    # checked against autodiff.
    hess_state1 = -1 * q_ss1 * hessian_gate_error_iso2(target_states[1])
    hess_state2 = -1 * q_ss2 * hessian_gate_error_iso2(target_states[2])
    hess_state3 = -1 * q_ss3 * hessian_gate_error_iso2(target_states[3])
    hess_state4 = -1 * q_ss4 * hessian_gate_error_iso2(target_states[4])
    hess_astate[S1_IDX, S1_IDX] = hess_state1
    hess_astate[S2_IDX, S2_IDX] = hess_state2
    hess_astate[S3_IDX, S3_IDX] = hess_state3
    hess_astate[S4_IDX, S4_IDX] = hess_state4
    hess_astate[S5_IDX, S5_IDX] = hess_state1
    hess_astate[S6_IDX, S6_IDX] = hess_state2
    hess_astate[S7_IDX, S7_IDX] = hess_state3
    hess_astate[S8_IDX, S8_IDX] = hess_state4
    hess_astate += Q
    hess_astate = Symmetric(SMatrix{N, N}(hess_astate))
    return Cost{N,M,T}(Q, R, q, c, hess_astate, target_states, q_ss1, q_ss2, q_ss3, q_ss4)
end

@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
    cost.Q, cost.R, cost.q, cost.c, cost.hess_astate,
    cost.target_states, cost.q_ss1, cost.q_ss2, cost.q_ss3, cost.q_ss4
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}) where {N,M,T} = (
    0.5 * astate' * cost.Q * astate + cost.q'astate + cost.c
    + cost.q_ss1 * gate_error_iso2(astate, cost.target_states[1], S1_IDX[1] - 1)
    + cost.q_ss2 * gate_error_iso2(astate, cost.target_states[2], S2_IDX[1] - 1)
    + cost.q_ss3 * gate_error_iso2(astate, cost.target_states[3], S3_IDX[1] - 1)
    + cost.q_ss4 * gate_error_iso2(astate, cost.target_states[4], S4_IDX[1] - 1)
    + cost.q_ss1 * gate_error_iso2(astate, cost.target_states[1], S5_IDX[1] - 1)
    + cost.q_ss2 * gate_error_iso2(astate, cost.target_states[2], S6_IDX[1] - 1)
    + cost.q_ss3 * gate_error_iso2(astate, cost.target_states[3], S7_IDX[1] - 1)
    + cost.q_ss4 * gate_error_iso2(astate, cost.target_states[4], S8_IDX[1] - 1)
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}, acontrol::SVector{M}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.q = (cost.Q * astate + cost.q + [
        @SVector zeros(ASTATE_SIZE_BASE);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_states[1], S1_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_states[2], S2_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_states[3], S3_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_states[4], S4_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_states[1], S5_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_states[2], S6_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_states[3], S7_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_states[4], S8_IDX[1] - 1);
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


# dynamics
function RD.discrete_dynamics(::Type{RD.RK3}, model::Model, astate::StaticVector,
                              acontrols::StaticVector, time::Real, dt::Real) where {SC}
    camp = astate[CONTROLS_IDX[1]]
    negi_hc = camp * NEGI_H1_ISO
    h_prop = exp((FQ_NEGI_H0_ISO + negi_hc) * dt)
    state1 = h_prop * astate[STATE1_IDX]
    state2 = h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX[1]] + dt * astate[CONTROLS_IDX[1]]
    controls = astate[CONTROLS_IDX[1]] + dt * astate[DCONTROLS_IDX[1]]
    dcontrols = astate[DCONTROLS_IDX[1]] + dt * acontrols[D2CONTROLS_IDX[1]]

    hp_prop = exp((FQ_NEGI_H0_ISO + (camp + model.namp) * NEGI_H1_ISO) * dt)
    hn_prop = exp((FQ_NEGI_H0_ISO + (camp - model.namp) * NEGI_H1_ISO) * dt)
    s1 = hp_prop * astate[S1_IDX]
    s2 = hp_prop * astate[S2_IDX]
    s3 = hp_prop * astate[S3_IDX]
    s4 = hp_prop * astate[S4_IDX]
    s5 = hn_prop * astate[S5_IDX]
    s6 = hn_prop * astate[S6_IDX]
    s7 = hn_prop * astate[S7_IDX]
    s8 = hn_prop * astate[S8_IDX]

    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
        s1; s2; s3; s4; s5; s6; s7; s8;
    ]

    return astate_
end


# main
function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true, max_iterations=Int64(2e5),
                  namp=NAMP_PREFACTOR)
    # model configuration
    model = Model(namp)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    tf = evolution_time
    
    # initial state
    x0 = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        repeat([INITIAL_STATE1; INITIAL_STATE2; INITIAL_STATE3; INITIAL_STATE4], 2);
    ])
    # target state
    gate_unitary = GT_GATE[gate_type]
    target_states = Array{SVector{HDIM_ISO}, 1}(undef, 4)
    target_states[1] = gate_unitary * INITIAL_STATE1
    target_states[2] = gate_unitary * INITIAL_STATE2
    target_states[3] = gate_unitary * INITIAL_STATE3
    target_states[4] = gate_unitary * INITIAL_STATE4
    xf = SVector{n}([
        target_states[1];
        target_states[2];
        zeros(3 * CONTROL_COUNT);
        repeat([target_states[1]; target_states[2];
                target_states[3]; target_states[4]], 2);
    ])
    
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    # control amplitude constraint at boundary
    x_max_boundary = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min_boundary = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # control
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
        fill(qs[2], 1); # ∫a
        fill(qs[3], 1); # a
        fill(qs[4], 1); # ∂a
        fill(0, SAMPLE_COUNT * HDIM_ISO); 
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[9], CONTROL_COUNT);
    ]))
    cost_k = Cost(Q, R, xf, target_states, qs[5], qs[6], qs[7], qs[8])
    cost_f = Cost(Qf, R, xf, target_states, N * qs[5], N * qs[6], N * qs[7], N * qs[8])
    objective = TO.Objective(cost_k, cost_f, N)
    
    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idxs) for idxs in (
        STATE1_IDX, STATE2_IDX, S1_IDX, S2_IDX, S3_IDX, S4_IDX, S5_IDX, S6_IDX, S7_IDX, S8_IDX,
    )]
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
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
    set_options!(solver, square_root=sqrtbp, constraint_tolerance=constraint_tolerance,
                 projected_newton_tolerance=al_tol, n_steps=n_steps,
                 penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
                 projected_newton=projected_newton, iterations_inner=iterations_inner,
                 iterations_outer=iterations_outer, iterations=max_iterations)
    Altro.solve!(solver)

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
    cmax = TO.max_violation(solver)
    cmax_info = TO.findmax_violation(TO.get_constraints(solver))
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
        "sample_count" => SAMPLE_COUNT,
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
