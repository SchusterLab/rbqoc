"""
spin12.jl - sampling robustness
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
const EXPERIMENT_NAME = "spin12"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const SAMPLE_COUNT = 2
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT
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
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
struct Model <: AbstractModel
    h0_samples::Vector{SMatrix{HDIM_ISO, HDIM_ISO}}
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
    target_state1::SVector{HDIM_ISO, T}
    q_ss::T
end

function Cost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
              xf::SVector{N,T}, target_state1::SVector{HDIM_ISO,T}, q_ss::T) where {N,M,T}
    q = -Q * xf
    c = 0.5 * xf' * Q * xf
    hess_astate = zeros(N, N)
    # For reasons unknown to the author, throwing a -1 in front
    # of the gate error Hessian makes the cost function work.
    # This is strange, because the gate error Hessian has been
    # checked against autodiff.
    hess_sample = -1 * q_ss * hessian_gate_error_iso2(target_state1)
    hess_astate[S1STATE1_IDX, S1STATE1_IDX] = hess_sample
    hess_astate[S2STATE1_IDX, S2STATE1_IDX] = hess_sample
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
    negi_hc = astate[CONTROLS_IDX[1]] * NEGI_H1_ISO
    s0h_prop = exp((FQ_NEGI_H0_ISO + negi_hc) * dt)
    state1 = s0h_prop * astate[STATE1_IDX]
    state2 = s0h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX[1]] + dt * astate[CONTROLS_IDX[1]]
    controls = astate[CONTROLS_IDX[1]] + dt * astate[DCONTROLS_IDX[1]]
    dcontrols = astate[DCONTROLS_IDX[1]] + dt * acontrols[D2CONTROLS_IDX[1]]

    s1state1 = exp((model.h0_samples[1] + negi_hc) * dt) * astate[S1STATE1_IDX]
    s2state1 = exp((model.h0_samples[2] + negi_hc) * dt) * astate[S2STATE1_IDX]

    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
        s1state1; s2state1;
    ]

    return astate_
end


# main
function run_traj(;gate_type=zpiby2, evolution_time=17.5, solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true, max_iterations=Int64(2e5),
                  fq_cov=FQ * 1e-2)
    # model configuration
    h0_samples = Array{SMatrix{HDIM_ISO, HDIM_ISO}}(undef, SAMPLE_COUNT)
    h0_samples[1] = (FQ + fq_cov) * NEGI_H0_ISO
    h0_samples[2] = (FQ - fq_cov) * NEGI_H0_ISO
    model = Model(h0_samples)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    tf = evolution_time

    # initial state
    x0 = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        repeat(INITIAL_STATE1, SAMPLE_COUNT);
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
        target_state2;
        zeros(3 * CONTROL_COUNT);
        repeat(target_state1, SAMPLE_COUNT);
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
        fill(qs[6], CONTROL_COUNT);
    ]))
    cost_k = Cost(Q, R, xf, target_state1, qs[5])
    cost_f = Cost(Qf, R, xf, target_state1, qs[5])
    objective = TO.Objective(cost_k, cost_f, N)

    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
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


function forward_pass(save_file_path; integrator_type=rk6, gate_type=xpiby2)
    (evolution_time, d2controls, dt
     ) = h5open(save_file_path, "r+") do save_file
         save_type = SaveType(read(save_file, "save_type"))
         if save_type == jl
             d2controls_idx = read(save_file, "d2controls_dt2_idx")
             acontrols = read(save_file, "acontrols")
             d2controls = acontrols[:, d2controls_idx]
             dt = read(save_file, "dt")
             evolution_time = read(save_file, "evolution_time")
         elseif save_type == samplejl
             d2controls = read(save_file, "d2controls_dt2_sample")
             dt = DT_PREF
             ets = read(save_file, "evolution_time_sample")
             evolution_time = Integer(floor(ets / dt)) * dt
         end
         return (evolution_time, d2controls, dt)
     end
    rdi = IT_RDI[integrator_type]
    knot_count = Integer(floor(evolution_time / dt))

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
    
    model = Model(sample_count)
    n = state_dim(model)
    m = control_dim(model)
    time = 0.
    astate = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        repeat([INITIAL_STATE1; INITIAL_STATE2], sample_count);
    ])
    acontrols = [SVector{m}([d2controls[i, 1],]) for i = 1:knot_count - 1]

    for i = 1:knot_count - 1
        astate = RD.discrete_dynamics(rdi, model, astate, acontrols[i], time, dt)
        time = time + dt
    end

    res = Dict(
        "astate" => astate,
        "target_state1" => target_state1,
        "target_state2" => target_state2,
    )

    return res
end
