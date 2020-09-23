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
const INITIAL_STATE1_SV = SVector{HDIM}(INITIAL_STATE1)
const INITIAL_STATE2 = [0., 1, 0, 0]
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# const S1STATE1_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
# const S2STATE1_IDX = S1STATE1_IDX[end] + 1:S1STATE1_IDX[end] + HDIM_ISO
# const S3STATE1_IDX = S2STATE1_IDX[end] + 1:S2STATE1_IDX[end] + HDIM_ISO
# const S4STATE1_IDX = S3STATE1_IDX[end] + 1:S3STATE1_IDX[end] + HDIM_ISO
# const S5STATE1_IDX = S4STATE1_IDX[end] + 1:S4STATE1_IDX[end] + HDIM_ISO
# const S6STATE1_IDX = S5STATE1_IDX[end] + 1:S5STATE1_IDX[end] + HDIM_ISO
# const S7STATE1_IDX = S6STATE1_IDX[end] + 1:S6STATE1_IDX[end] + HDIM_ISO
# const S8STATE1_IDX = S7STATE1_IDX[end] + 1:S7STATE1_IDX[end] + HDIM_ISO
# const S9STATE1_IDX = S8STATE1_IDX[end] + 1:S8STATE1_IDX[end] + HDIM_ISO
const SAMPLE_COUNT = 8
const SAMPLE_COUNT_INV = 1//8
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
module Data
using Distributions
using RobotDynamics
using StaticArrays
const HDIM_ISO = 4
const SAMPLE_COUNT = 8
mutable struct Model <: RobotDynamics.AbstractModel
    state1_samples::Array{SVector{HDIM_ISO}, 1}
    fq_samples::MVector{SAMPLE_COUNT}
    fq_dist::Distributions.Sampleable
    alpha::Float64
end
end
RD.state_dim(::Data.Model) = ASTATE_SIZE
RD.control_dim(::Data.Model) = CONTROL_COUNT


function unscented_transform!(model::Data.Model, )
    s11 = astate[S1STATE1_IDX]
    s21 = astate[S2STATE1_IDX]
    s31 = astate[S3STATE1_IDX]
    s41 = astate[S4STATE1_IDX]
    s51 = astate[S5STATE1_IDX]
    s61 = astate[S6STATE1_IDX]
    s71 = astate[S7STATE1_IDX]
    s81 = astate[S8STATE1_IDX]
    
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
    scov = 0.5 .* (
        d1 * d1' + d2 * d2' + d3 * d3' + d4 * d4'
        + d5 * d5' + d6 * d6' + d7 * d7' + d8 * d8'
    )
    # cov = @SMatrix [
    #     scov[1, 1] scov[1, 2] scov[1, 3] scov[1, 4] 0;
    #     scov[2, 1] scov[2, 2] scov[2, 3] scov[2, 4] 0;
    #     scov[3, 1] scov[3, 2] scov[3, 3] scov[3, 4] 0;
    #     scov[4, 1] scov[4, 2] scov[4, 3] scov[4, 4] 0;
    #     0          0          0          0          model.fq_cov;
    # ]
    cov_chol = model.alpha * cholesky(scov, check=false).L

    s11 = s1m + cov_chol[1:HDIM_ISO, 1]
    s21 = s1m - cov_chol[1:HDIM_ISO, 1]
    s31 = s1m + cov_chol[1:HDIM_ISO, 2]
    s41 = s1m - cov_chol[1:HDIM_ISO, 2]
    s51 = s1m + cov_chol[1:HDIM_ISO, 3]
    s61 = s1m - cov_chol[1:HDIM_ISO, 3]
    s71 = s1m + cov_chol[1:HDIM_ISO, 4]
    s81 = s1m - cov_chol[1:HDIM_ISO, 4]
    model.state1_samples[1] = s11 ./ sqrt(s11's11)
    model.state1_samples[2] = s21 ./ sqrt(s21's21)
    model.state1_samples[3] = s31 ./ sqrt(s31's31)
    model.state1_samples[4] = s41 ./ sqrt(s41's41)
    model.state1_samples[5] = s51 ./ sqrt(s51's51)
    model.state1_samples[6] = s61 ./ sqrt(s61's61)
    model.state1_samples[7] = s71 ./ sqrt(s71's71)
    model.state1_samples[8] = s81 ./ sqrt(s81's81)
    rand!(model.fq_dist, model.fq_samples)
    model.fq_samples .+= FQ
    # model.fq_samples[1] = FQ + cov_chol[HDIM_ISO + 1, 1]
    # model.fq_samples[5] = FQ - cov_chol[HDIM_ISO + 1, 1]
    # model.fq_samples[2] = FQ + cov_chol[HDIM_ISO + 1, 2]
    # model.fq_samples[6] = FQ - cov_chol[HDIM_ISO + 1, 2]
    # model.fq_samples[3] = FQ + cov_chol[HDIM_ISO + 1, 3]
    # model.fq_samples[7] = FQ - cov_chol[HDIM_ISO + 1, 3]
    # model.fq_samples[4] = FQ + cov_chol[HDIM_ISO + 1, 4]
    # model.fq_samples[8] = FQ - cov_chol[HDIM_ISO + 1, 4]
    return nothing
end


# dynamics
function discrete_dynamics_(model::Data.Model, astate::StaticVector{ASTATE_SIZE},
                            acontrols::StaticVector{CONTROL_COUNT}, time::Real, dt::Real)
    negi_hc = astate[CONTROLS_IDX][1] * NEGI_H1_ISO
    negi_s0h = FQ_NEGI_H0_ISO + negi_hc
    negi_s0h_prop = exp(negi_s0h * dt)
    state1 = negi_s0h_prop * astate[STATE1_IDX]
    state2 = negi_s0h_prop * astate[STATE2_IDX]
    intcontrols = astate[INTCONTROLS_IDX] + dt * astate[CONTROLS_IDX]
    controls = astate[CONTROLS_IDX] + dt * astate[DCONTROLS_IDX]
    dcontrols = astate[DCONTROLS_IDX] + dt * acontrols[D2CONTROLS_IDX]

    s1state1 = exp(dt*(model.fq_samples[1] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[1]
    s2state1 = exp(dt*(model.fq_samples[2] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[2]
    s3state1 = exp(dt*(model.fq_samples[3] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[3]
    s4state1 = exp(dt*(model.fq_samples[4] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[4]
    s5state1 = exp(dt*(model.fq_samples[5] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[5]
    s6state1 = exp(dt*(model.fq_samples[6] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[6]
    s7state1 = exp(dt*(model.fq_samples[7] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[7]
    s8state1 = exp(dt*(model.fq_samples[8] * NEGI_H0_ISO + negi_hc)) * model.state1_samples[8]
    
    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
        s1state1; s2state1; s3state1; s4state1; s5state1; s6state1;
        s7state1; s8state1;
    ]
    
    return astate_
end


# Note that TO.rollout! uses RK3.
function RD.discrete_dynamics(::Type{RK3}, model::Data.Model, astate::StaticVector{ASTATE_SIZE},
                              acontrols::StaticVector{CONTROL_COUNT}, time::Real, dt::Real)
    unscented_transform!(model, astate)
    return discrete_dynamics_(model, astate, acontrols, time, dt)
end


function RD.discrete_jacobian!(::Type{RK3}, ∇f, model::Data.Model, z::AbstractKnotPoint{T,N,M}) where {T,N,M}
    ix,iu,idt = z._x, z._u, N+M+1
    t = z.t
    unscented_transform!(model, RD.state(z))
    fd_aug(s) = discrete_dynamics_(model, s[ix], s[iu], t, z.dt)
    ∇f .= ForwardDiff.jacobian(fd_aug, SVector{N+M}(z.z))
    return nothing
end


function run_traj(;gate_type=xpiby2, evolution_time=56.8, solver_type=altro,
                  sqrtbp=false, integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true,
                  fq_cov=FQ * 1e-2, max_iterations=Int64(2e5), gradient_tol_int=1,
                  dJ_counter_limit=Int(1e2), astate_cov=1e-2, seed=0, alpha=1.)
    Random.seed!(seed)
    astate_dist = Distributions.Normal(0., astate_cov)
    state1_samples = [SVector{HDIM_ISO}(zeros(HDIM_ISO)) for i = 1:SAMPLE_COUNT]
    fq_dist = Distributions.Normal(0., fq_cov)
    fq_samples = MVector{SAMPLE_COUNT}([0. for i = 1:SAMPLE_COUNT])
    model = Data.Model(state1_samples, fq_samples, fq_dist, alpha)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.

    # initial state
    x0_ = [
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        repeat(INITIAL_STATE1, SAMPLE_COUNT);
    ]
    # for i = 1:SAMPLE_COUNT
    #     sample = INITIAL_STATE1 .+ rand(astate_dist, HDIM_ISO)
    #     append!(x0_, sample ./ sqrt(sample'sample))
    # end
    x0 = SVector{n}(x0_)

    # target state
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
    # controls start and end at 0
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

    # penalties
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # state1, state2
        fill(qs[2], 1); # intcontrol
        fill(qs[3], 1); # control
        fill(qs[4], 1); # dcontrol
        fill(qs[5], SAMPLE_COUNT * HDIM_ISO); # s1state1, s1state2
    ]))
    Qf = Q * N
    R = SVector{m}([
        fill(qs[6], CONTROL_COUNT);
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
