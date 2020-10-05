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
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

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
const S11_IDX = S10_IDX[end] + 1:S10_IDX[end] + HDIM_ISO
const S12_IDX = S11_IDX[end] + 1:S11_IDX[end] + HDIM_ISO
const S13_IDX = S12_IDX[end] + 1:S12_IDX[end] + HDIM_ISO
const S14_IDX = S13_IDX[end] + 1:S13_IDX[end] + HDIM_ISO
const S15_IDX = S14_IDX[end] + 1:S14_IDX[end] + HDIM_ISO
const S16_IDX = S15_IDX[end] + 1:S15_IDX[end] + HDIM_ISO
const S17_IDX = S16_IDX[end] + 1:S16_IDX[end] + HDIM_ISO
const S18_IDX = S17_IDX[end] + 1:S17_IDX[end] + HDIM_ISO
const S19_IDX = S18_IDX[end] + 1:S18_IDX[end] + HDIM_ISO
const S20_IDX = S19_IDX[end] + 1:S19_IDX[end] + HDIM_ISO
const S21_IDX = S20_IDX[end] + 1:S20_IDX[end] + HDIM_ISO
const S22_IDX = S21_IDX[end] + 1:S21_IDX[end] + HDIM_ISO
const S23_IDX = S22_IDX[end] + 1:S22_IDX[end] + HDIM_ISO
const S24_IDX = S23_IDX[end] + 1:S23_IDX[end] + HDIM_ISO
const S25_IDX = S24_IDX[end] + 1:S24_IDX[end] + HDIM_ISO
const S26_IDX = S25_IDX[end] + 1:S25_IDX[end] + HDIM_ISO
const S27_IDX = S26_IDX[end] + 1:S26_IDX[end] + HDIM_ISO
const S28_IDX = S27_IDX[end] + 1:S27_IDX[end] + HDIM_ISO
const S29_IDX = S28_IDX[end] + 1:S28_IDX[end] + HDIM_ISO
const S30_IDX = S29_IDX[end] + 1:S29_IDX[end] + HDIM_ISO
const S31_IDX = S30_IDX[end] + 1:S30_IDX[end] + HDIM_ISO
const S32_IDX = S31_IDX[end] + 1:S31_IDX[end] + HDIM_ISO
const S33_IDX = S32_IDX[end] + 1:S32_IDX[end] + HDIM_ISO
const S34_IDX = S33_IDX[end] + 1:S33_IDX[end] + HDIM_ISO
const S35_IDX = S34_IDX[end] + 1:S34_IDX[end] + HDIM_ISO
const S36_IDX = S35_IDX[end] + 1:S35_IDX[end] + HDIM_ISO
const S37_IDX = S36_IDX[end] + 1:S36_IDX[end] + HDIM_ISO
const S38_IDX = S37_IDX[end] + 1:S37_IDX[end] + HDIM_ISO
const S39_IDX = S38_IDX[end] + 1:S38_IDX[end] + HDIM_ISO
const S40_IDX = S39_IDX[end] + 1:S39_IDX[end] + HDIM_ISO
const S1INDS = [S1_IDX, S2_IDX, S3_IDX, S4_IDX, S5_IDX, S6_IDX, S7_IDX, S8_IDX, S9_IDX, S10_IDX]
const S2INDS = [S11_IDX, S12_IDX, S13_IDX, S14_IDX, S15_IDX, S16_IDX, S17_IDX, S18_IDX, S19_IDX, S20_IDX]
const S3INDS = [S21_IDX, S22_IDX, S23_IDX, S24_IDX, S25_IDX, S26_IDX, S27_IDX, S28_IDX, S29_IDX, S30_IDX]
const S4INDS = [S31_IDX, S32_IDX, S33_IDX, S34_IDX, S35_IDX, S36_IDX, S37_IDX, S38_IDX, S39_IDX, S40_IDX]
const BATCH_SAMPLE_COUNT = 10
const SAMPLE_COUNT = 40
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


# ukf
function unscented_transform(model::Model, astate::SVector{ASTATE_SIZE}, negi_hc::SMatrix{HDIM_ISO, HDIM_ISO},
                             inds::Array{UnitRange{Int64}, 1}, dt::Real)
    # get states
    s1 = SVector{HDIM_ISO}(astate[inds[1]])
    s2 = SVector{HDIM_ISO}(astate[inds[2]])
    s3 = SVector{HDIM_ISO}(astate[inds[3]])
    s4 = SVector{HDIM_ISO}(astate[inds[4]])
    s5 = SVector{HDIM_ISO}(astate[inds[5]])
    s6 = SVector{HDIM_ISO}(astate[inds[6]])
    s7 = SVector{HDIM_ISO}(astate[inds[7]])
    s8 = SVector{HDIM_ISO}(astate[inds[8]])
    s9 = SVector{HDIM_ISO}(astate[inds[9]])
    s10 = SVector{HDIM_ISO}(astate[inds[10]])
    # compute state mean
    sm = 1//10 .* (
        s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10
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
        d1 * d1' + d2 * d2' + d3 * d3' + d4 * d4' + d5 * d5' +
        d6 * d6' + d7 * d7' + d8 * d8' + d9 * d9' + d10 * d10'
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

    return (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
end


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

    s1s = unscented_transform(model, astate, negi_hc, S1INDS, dt)
    s2s = unscented_transform(model, astate, negi_hc, S2INDS, dt)
    s3s = unscented_transform(model, astate, negi_hc, S3INDS, dt)
    s4s = unscented_transform(model, astate, negi_hc, S4INDS, dt)
    
    astate_ = [
        state1; state2; intcontrols; controls; dcontrols;
        s1s[1]; s1s[2]; s1s[3]; s1s[4]; s1s[5]; s1s[6]; s1s[7]; s1s[8]; s1s[9]; s1s[10];
        s2s[1]; s2s[2]; s2s[3]; s2s[4]; s2s[5]; s2s[6]; s2s[7]; s2s[8]; s2s[9]; s2s[10];
        s3s[1]; s3s[2]; s3s[3]; s3s[4]; s3s[5]; s3s[6]; s3s[7]; s3s[8]; s3s[9]; s3s[10];
        s4s[1]; s4s[2]; s4s[3]; s4s[4]; s4s[5]; s4s[6]; s4s[7]; s4s[8]; s4s[9]; s4s[10];
    ]

    return astate_
end


# This cost puts a gate error cost on
# the sample states and a LQR cost on the other terms.
# The hessian w.r.t the state and controls is constant.
module Data2
using LinearAlgebra
using StaticArrays
using TrajectoryOptimization
const TO = TrajectoryOptimization
const HDIM_ISO = 4
struct Cost{N,M,T} <: TO.CostFunction
    Q::Diagonal{T,SizedArray{Tuple{N},T,1,1}}
    R::Diagonal{T,SizedArray{Tuple{M},T,1,1}}
    q::SizedVector{N}
    c::T
    hess_astate::Symmetric{T,SizedArray{Tuple{N,N},T,2,2}}
    target_states::Array{SVector{HDIM_ISO, T},1}
    q_ss1::T
    q_ss2::T
    q_ss3::T
    q_ss4::T
end
end

function Cost(Q::Diagonal{T,SizedArray{Tuple{N},T,1,1}},
              R::Diagonal{T,SizedArray{Tuple{M},T,1,1}},
              xf::SizedVector{N,T}, target_states::Array{SVector{HDIM_ISO}, 1},
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
    hess_astate[S2_IDX, S2_IDX] = hess_state1
    hess_astate[S3_IDX, S3_IDX] = hess_state1
    hess_astate[S4_IDX, S4_IDX] = hess_state1
    hess_astate[S5_IDX, S5_IDX] = hess_state1
    hess_astate[S6_IDX, S6_IDX] = hess_state1
    hess_astate[S7_IDX, S7_IDX] = hess_state1
    hess_astate[S8_IDX, S8_IDX] = hess_state1
    hess_astate[S9_IDX, S9_IDX] = hess_state1
    hess_astate[S10_IDX, S10_IDX] = hess_state1
    hess_astate[S11_IDX, S11_IDX] = hess_state2
    hess_astate[S12_IDX, S12_IDX] = hess_state2
    hess_astate[S13_IDX, S13_IDX] = hess_state2
    hess_astate[S14_IDX, S14_IDX] = hess_state2
    hess_astate[S15_IDX, S15_IDX] = hess_state2
    hess_astate[S16_IDX, S16_IDX] = hess_state2
    hess_astate[S17_IDX, S17_IDX] = hess_state2
    hess_astate[S18_IDX, S18_IDX] = hess_state2
    hess_astate[S19_IDX, S19_IDX] = hess_state2
    hess_astate[S20_IDX, S20_IDX] = hess_state2
    hess_astate[S21_IDX, S21_IDX] = hess_state3
    hess_astate[S22_IDX, S22_IDX] = hess_state3
    hess_astate[S23_IDX, S23_IDX] = hess_state3
    hess_astate[S24_IDX, S24_IDX] = hess_state3
    hess_astate[S25_IDX, S25_IDX] = hess_state3
    hess_astate[S26_IDX, S26_IDX] = hess_state3
    hess_astate[S27_IDX, S27_IDX] = hess_state3
    hess_astate[S28_IDX, S28_IDX] = hess_state3
    hess_astate[S29_IDX, S29_IDX] = hess_state3
    hess_astate[S30_IDX, S30_IDX] = hess_state3
    hess_astate[S31_IDX, S31_IDX] = hess_state4
    hess_astate[S32_IDX, S32_IDX] = hess_state4
    hess_astate[S33_IDX, S33_IDX] = hess_state4
    hess_astate[S34_IDX, S34_IDX] = hess_state4
    hess_astate[S35_IDX, S35_IDX] = hess_state4
    hess_astate[S36_IDX, S36_IDX] = hess_state4
    hess_astate[S37_IDX, S37_IDX] = hess_state4
    hess_astate[S38_IDX, S38_IDX] = hess_state4
    hess_astate[S39_IDX, S39_IDX] = hess_state4
    hess_astate[S40_IDX, S40_IDX] = hess_state4
    hess_astate += Q
    hess_astate = Symmetric(SizedMatrix{N, N}(hess_astate))
    return Data2.Cost{N,M,T}(Q, R, q, c, hess_astate, target_states, q_ss1, q_ss2, q_ss3, q_ss4)
end

@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
    cost.Q, cost.R, cost.q, cost.c, cost.hess_astate,
    cost.target_states, cost.q_ss1, cost.q_ss2, cost.q_ss3, cost.q_ss4
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SizedVector{N}) where {N,M,T} = (
    0.5 * astate' * cost.Q * astate + cost.q'astate + cost.c
    + cost.q_ss1 * (
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
    + cost.q_ss2 * (
        gate_error_iso2(astate, cost.target_state2, S11_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S12_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S13_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S14_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S15_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S16_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S17_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S18_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S19_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state2, S20_IDX[1] - 1)
        )
    + cost.q_ss3 * (
        gate_error_iso2(astate, cost.target_state3, S21_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S22_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S23_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S24_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S25_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S26_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S27_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S28_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S29_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state3, S30_IDX[1] - 1)
        )
    + cost.q_ss4 * (
        gate_error_iso2(astate, cost.target_state4, S31_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S32_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S33_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S34_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S35_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S36_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S37_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S38_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S39_IDX[1] - 1)
        + gate_error_iso2(astate, cost.target_state4, S40_IDX[1] - 1)
    )
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SizedVector{N}, acontrol::SizedVector{M}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SizedVector{N,T}) where {N,M,T}
    E.q = (cost.Q * astate + cost.q + [
        @SVector zeros(ASTATE_SIZE_BASE);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S1_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S2_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S3_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S4_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S5_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S6_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S7_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S8_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S9_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_state1, S10_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S11_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S12_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S13_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S14_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S15_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S16_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S17_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S18_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S19_IDX[1] - 1);
        cost.q_ss2 * jacobian_gate_error_iso2(astate, cost.target_state2, S20_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S21_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S22_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S23_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S24_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S25_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S26_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S27_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S28_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S29_IDX[1] - 1);
        cost.q_ss3 * jacobian_gate_error_iso2(astate, cost.target_state3, S30_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S31_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S32_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S33_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S34_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S35_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S36_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S37_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S38_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S39_IDX[1] - 1);
        cost.q_ss4 * jacobian_gate_error_iso2(astate, cost.target_state4, S40_IDX[1] - 1);
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
                  sqrtbp=false, integrator_type=rk3, qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e-1],
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
    state_dist = Distributions.Normal(0., state_cov)
    x0_ = [
        INITIAL_STATE1;
        INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
    ]
    for initial_state in (INITIAL_STATE1, INITIAL_STATE2, INITIAL_STATE3, INITIAL_STATE4)
        for j = 1:BATCH_SAMPLE_COUNT
            sample = initial_state .+ rand(state_dist, HDIM_ISO)
            sample = sample ./ sqrt(sample'sample)
            append!(x0_, sample)
        end
    end
    x0 = SizedVector{n}(x0_)
    target_states = Array{SVector{HDIM_ISO}, 1}(undef, 4)
    target_states[1] = GT_GATE[gate_type] * INITIAL_STATE1
    target_states[2] = GT_GATE[gate_type] * INITIAL_STATE2
    target_states[3] = GT_GATE[gate_type] * INITIAL_STATE3
    target_states[4] = GT_GATE[gate_type] * INITIAL_STATE4
    xf = SizedVector{n}([
        GT_GATE[gate_type] * INITIAL_STATE1;
        GT_GATE[gate_type] * INITIAL_STATE2;
        zeros(3 * CONTROL_COUNT);
        repeat(target_states[1], BATCH_SAMPLE_COUNT);
        repeat(target_states[2], BATCH_SAMPLE_COUNT);
        repeat(target_states[3], BATCH_SAMPLE_COUNT);
        repeat(target_states[4], BATCH_SAMPLE_COUNT);
    ])
    
    # control amplitude constraint
    x_max = SizedVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, 1); # a
        fill(Inf, CONTROL_COUNT);
        fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min = SizedVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, 1); # a
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    # control amplitude constraint at boundary
    x_max_boundary = SizedVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO);
        fill(Inf, CONTROL_COUNT);
        fill(0, 1); # a
        fill(Inf, CONTROL_COUNT);
        fill(Inf, SAMPLE_COUNT * HDIM_ISO);
    ])
    x_min_boundary = SizedVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO);
        fill(-Inf, CONTROL_COUNT);
        fill(0, 1); # a
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, SAMPLE_COUNT * HDIM_ISO);
    ])

    # initial trajectory
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [SizedVector{m}([
        fill(1e-4, CONTROL_COUNT);
    ]) for k = 1:N-1]
    X0 = [SizedVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    # cost function
    Q = Diagonal(SizedVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # ψ1, ψ2
        fill(qs[2], CONTROL_COUNT); # ∫a
        fill(qs[3], CONTROL_COUNT); # a
        fill(qs[4], CONTROL_COUNT); # ∂a
        fill(0, SAMPLE_COUNT * HDIM_ISO);
        # fill(qs[5], SAMPLE_COUNT * HDIM_ISO);
    ]))
    Qf = Q * N
    R = Diagonal(SizedVector{m}([
        fill(qs[9], CONTROL_COUNT); # ∂2a
    ]))
    # objective = LQRObjective(Q, R, Qf, xf, N)
    cost_k = Cost(Q, R, xf, target_states, qs[5], qs[6], qs[7], qs[8])
    cost_f = Cost(Qf, R, xf, target_states, N * qs[5], N * qs[6], N * qs[7], N * qs[8])
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
        STATE1_IDX, STATE2_IDX, S1_IDX, S2_IDX, S3_IDX, S4_IDX, S5_IDX, S6_IDX, S7_IDX, S8_IDX,
    )]    
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
    static_bp = false
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


function sample_diffs(saved; gate_type=zpiby2)
    target_state = GT_GATE[gate_type] * INITIAL_STATE3
    knot_count = size(saved["astates"], 1)
    diffs_ = zeros(SAMPLE_COUNT, knot_count)
    fds_ = zeros(SAMPLE_COUNT, knot_count)
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
        fds_[9, i] = fidelity_vec_iso2(s9, target_state)
        fds_[10, i] = fidelity_vec_iso2(s10, target_state)
    end
    return (diffs_, fds_)
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
