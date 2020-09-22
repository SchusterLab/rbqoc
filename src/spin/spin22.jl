"""
spin22.jl - dynamic sampling robustness for the δf_q problem
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
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
const EXPERIMENT_NAME = "spin22"
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
const S1STATE2_IDX = S1STATE1_IDX[end] + 1:S1STATE1_IDX[end] + HDIM_ISO
# const S2STATE1_IDX = S1STATE2_IDX[end] + 1:S1STATE2_IDX[end] + HDIM_ISO
# const S2STATE2_IDX = S2STATE1_IDX[end] + 1:S2STATE1_IDX[end] + HDIM_ISO
# const S3STATE1_IDX = S2STATE2_IDX[end] + 1:S2STATE2_IDX[end] + HDIM_ISO
# const S3STATE2_IDX = S3STATE1_IDX[end] + 1:S3STATE1_IDX[end] + HDIM_ISO
# const S4STATE1_IDX = S3STATE2_IDX[end] + 1:S3STATE2_IDX[end] + HDIM_ISO
# const S4STATE2_IDX = S4STATE1_IDX[end] + 1:S4STATE1_IDX[end] + HDIM_ISO
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
struct Model{SC} <: AbstractModel
    edist::Distributions.Sampleable
    eamp::Float64
end
RD.state_dim(::Model{SC}) where SC = (
    ASTATE_SIZE_BASE + SC * STATE_COUNT * HDIM_ISO
)
RD.control_dim(::Model{SC}) where SC = CONTROL_COUNT


# dynamics
# Note that TO.rollout! uses RK3.
function RD.discrete_dynamics(::Type{RD.RK3}, model::Model{SC}, astate::StaticVector,
                              acontrols::StaticVector, time::Real, dt::Real) where {SC}

    negi_hc = astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    negi_s0h = FQ_NEGI_H0_ISO + negi_hc
    negi_s0h_prop = exp(negi_s0h * dt)
    state1 = negi_s0h_prop * astate[STATE1_IDX]
    state2 = negi_s0h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX] + dt * astate[CONTROLS_IDX]
    controls = astate[CONTROLS_IDX] + dt * astate[DCONTROLS_IDX]
    dcontrols = astate[DCONTROLS_IDX] + dt * acontrols[D2CONTROLS_IDX]
    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
    ]

    if SC >= 1
        negi_s1h = (FQ + model.eamp * rand(model.edist)) * NEGI_H0_ISO + negi_hc
        negi_s1h_prop = exp(negi_s1h * dt)
        s1state1 = negi_s1h_prop * astate[S1STATE1_IDX]
        s1state2 = negi_s1h_prop * astate[S1STATE2_IDX]
        append!(astate_, [
            s1state1; s1state2;
        ])
    end

    return astate_
end


function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=altro,
                  sqrtbp=false, sample_count=0,
                  integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true, edist=STD_NORMAL,
                  esigma=1e-2, seed=0, max_iterations=Int64(2e5), gradient_tol_int=1,
                  dJ_counter_limit=Int(1e2),)
    eamp = esigma * FQ
    model = Model{sample_count}(edist, eamp)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    x0 = SVector{n}([
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        repeat([INITIAL_STATE1; INITIAL_STATE2], sample_count);
    ])
    
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
    xf = SVector{n}([
        target_state1;
        target_state2;
        zeros(3 * CONTROL_COUNT);
        repeat([target_state1; target_state2], sample_count);
    ])
    # control amplitude constraint
    x_max = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, sample_count * STATE_COUNT * HDIM_ISO);
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, sample_count * STATE_COUNT * HDIM_ISO);
    ])
    # controls start and end at 0
    x_max_boundary = SVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, sample_count * STATE_COUNT * HDIM_ISO);
    ])
    x_min_boundary = SVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, sample_count * STATE_COUNT * HDIM_ISO);
    ])
    
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [SVector{m}([
        fill(1e-4, CONTROL_COUNT);
    ]) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # state1, state2
        fill(qs[2], 1); # intcontrol
        fill(qs[3], 1); # control
        fill(qs[4], 1); # dcontrol
        fill(qs[5], eval(:($sample_count >= 1 ? $STATE_COUNT * $HDIM_ISO : 0))); # s1state1, s1state2
        fill(qs[6], eval(:($sample_count >= 2 ? $STATE_COUNT * $HDIM_ISO : 0))); # s1state1, s1state2
    ]))
    Qf = Q * N
    R = SVector{m}([
        fill(qs[7], CONTROL_COUNT);
    ])
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must statisfy conrols start and stop at 0.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must reach target state. Must have zero net flux.
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INTCONTROLS_IDX])
    # Must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
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
        dJ_counter_limit=dJ_counter_limit
    )
    Random.seed!(seed)
    Altro.solve!(solver)

    # Post-process.
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
    iterations_ = iterations(solver)

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
        "sample_count" => sample_count,
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
        "esigma" => esigma,
        "edist" => string(edist),
        "seed" => seed,
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
