"""
spin.jl - common definitions for the spin directory
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "rbqoc.jl"))

using Dates
using Dierckx
using DifferentialEquations
using Distributions
using FFTW
using HDF5
using Random
using StaticArrays
using Statistics
using Zygote

# paths
const SPIN_OUT_PATH = abspath(joinpath(WDIR, "out", "spin"))
const FBFQ_DFQ_DATA_FILE_PATH = joinpath(SPIN_OUT_PATH, "figures", "misc", "dfq.h5")

# simulation constants
const DT_PREF = 1e-2
const DT_PREF_INV = 1e2
const DT_NOISE_INV = 1e1

# types
@enum GateType begin
    zpiby2 = 1
    ypiby2 = 2
    xpiby2 = 3
    xpi = 4
end

@enum DynamicsType begin
    schroed = 1
    lindbladnodis = 2
    lindbladt1 = 3
    ypiby2nodis = 4
    ypiby2t1 = 5
    xpiby2nodis = 6
    xpiby2t1 = 7
    zpiby2nodis = 9
    zpiby2t1 = 10
    schroeddf = 11
    xpiby2da = 13
    xpiby2t2 = 14
    lindbladt2 = 15
    lindbladdf = 16
    zpiby2da = 17
    schroedda = 18
    empty = 19
    ypiby2da = 20
    xpiby2corpserwa = 21
    xpiby2corpse = 22
    xpicorpse = 23
end

@enum StateType begin
    st_state = 1
    st_density = 2
end

const DT_STR = Dict(
    schroed => "Schroedinger",
    lindbladnodis => "Lindblad No Dissipation",
    lindbladt1 => "Lindblad T_1 Dissipation",
    lindbladt1 => "Lindblad T_2 Dissipation",
)

const DT_ST = Dict(
    schroed => st_state,
    schroedda => st_state,
    lindbladnodis => st_density,
    lindbladt1 => st_density,
    lindbladt2 => st_density,
    ypiby2nodis => st_state,
    ypiby2t1 => st_density,
    xpiby2nodis => st_state,
    xpiby2t1 => st_density,
    zpiby2nodis => st_state,
    zpiby2t1 => st_density,
    xpiby2da => st_state,
    xpiby2t2 => st_density,
    zpiby2da => st_state,
    empty => st_state,
    ypiby2da => st_state,
    xpiby2corpserwa => st_state,
    xpiby2corpse => st_state,
    xpicorpse => st_state,
)

const GT_STR = Dict(
    zpiby2 => "Z/2",
    ypiby2 => "Y/2",
    xpiby2 => "X/2",
)

const HDIM = 2
const HDIM_ISO = HDIM * 2
struct SimParams
    controls :: Array{Float64, 2}
    control_knot_count :: Int64
    controls_dt_inv :: Int64
    negi_h0 :: StaticMatrix{HDIM_ISO, HDIM_ISO}
    noise_offsets :: Array{Float64, 1}
    noise_dt_inv :: Int64
    sim_dt_inv :: Int64
end

# other constants
const DEQJL_MAXITERS = 1e10


# Define experimental constants.
# E / h e-9
const EC = 0.479
const EL = 0.132
const EJ = 3.395
const E_PLASMA = sqrt(8 * EL * EC)
const PHI_OSC = (8 * EC / EL)^(1//4)
# qubit frequency at flux frustration point
const FQ = 1.4e-2 #GHz
const WQ = 2 * pi * FQ
const SP1FQ = FQ + FQ * 1e-2
const SN1FQ = FQ - FQ * 1e-2
const SP2FQ = FQ + FQ * 2e-2
const SN2FQ = FQ - FQ * 2e-2
const SP3FQ = FQ + FQ * 3e-2
const SN3FQ = FQ - FQ * 3e-2
const MAX_CONTROL_NORM_0 = 5e-1 #GHz
const FBFQ_A = 0.202407
const FBFQ_B = 0.5
const AYPIBY2 = 1.25e-1 #GHz
const FBFQ_NAMP = 5.21e-6 # flux noise amplitude
const STD_NORMAL = Normal(0., 1.)
const GAMMAC = 1 / 3e5 #GHz(T_c = 300 us)
const SQRTLNIR = 4
const NAMP_PREFACTOR = FBFQ_NAMP / FBFQ_A
const GAMMAF_PREFACTOR = FBFQ_NAMP * SQRTLNIR * 2 * pi
# coefficients are listed in descending order
# raw coefficients are in units of seconds
const FBFQ_T1_COEFFS = [
    3276.06057; -7905.24414; 8285.24137; -4939.22432;
    1821.23488; -415.520981; 53.9684414; -3.04500484
] * 1e9
# raw T1 times are in units of microseconds, yielding times in units of nanoseconds
const T1_ARRAY = [
    1597.923, 1627.93, 301.86, 269.03, 476.33, 1783.19, 2131.76, 2634.50, 
    4364.68, 2587.82, 1661.915, 1794.468, 2173.88, 1188.83, 
    1576.493, 965.183, 560.251, 310.88
] * 1e3
const T1_ARRAY_ERR = [
    78.034, 70.57, 20.71, 20.93, 12.92, 66.93, 137.06, 319.19, 
    194.84, 146.87, 170.584, 374.582, 322.533, 125.10,
    105.987, 58.612, 22.295, 11.29
] * 1e3
# raw T1 times are in units of microseconds
const T1_ARRAY_REDUCED = [
    1597.923, 1627.93, 301.86, 269.03, 476.33, 1783.19, 2131.76, 2634.50, 
    4364.68, 2587.82, 1661.915, 1794.468, 2173.88, 1188.83, 
    1576.493, 965.183, 560.251, 310.88
]
const FBFQ_ARRAY = [
    0.26, 0.28, 0.32, 0.34, 0.36, 0.38, 0.4,
    0.42, 0.44, 0.46, 0.465, 0.47, 0.475,
    0.48, 0.484, 0.488, 0.492, 0.5
]
const FBFQ_T1_SPLINE_DIERCKX = Spline1D(FBFQ_ARRAY, T1_ARRAY)
const FBFQ_T1_SPLINE_ITP = extrapolate(interpolate(
    (FBFQ_ARRAY,), T1_ARRAY, Gridded(Linear())), Flat())
const FBFQ_T1_REDUCED_SPLINE_ITP = extrapolate(interpolate(
    (FBFQ_ARRAY,), T1_ARRAY_REDUCED, Gridded(Linear())), Flat())

# (FBFQ_DFQ_FBFQS, FBFQ_DFQ_DFQS) = h5open(FBFQ_DFQ_DATA_FILE_PATH, "r") do data_file
#     fbfqs = read(data_file, "fbfqs")
#     dfqs = read(data_file, "dfqs")
#     return(fbfqs, dfqs)
# end
# const FBFQ_DFQ_SPLINE_DIERCKX = Spline1D(FBFQ_DFQ_FBFQS, FBFQ_DFQ_DFQS)


# Define the system.
# ISO indicates the object is defined in the complex to real isomorphism.
# NEGI is the negative complex unit.
const NEGI = SA_F64[0   0  1  0 ;
                    0   0  0  1 ;
                    -1  0  0  0 ;
                    0  -1  0  0 ;]
# SIGMAX, SIGMAZ are the X and Z pauli matrices
const SIGMAX = [0 1;
                1 0]
const SIGMAY = [0   -1im;
                1im 0]
const SIGMAZ = [1 0;
                0 -1]
const SIGMAX_ISO = SMatrix{HDIM_ISO, HDIM_ISO, Int64}(get_mat_iso(SIGMAX))
const SIGMAY_ISO = SMatrix{HDIM_ISO, HDIM_ISO, Int64}(get_mat_iso(SIGMAY))
const SIGMAZ_ISO = SMatrix{HDIM_ISO, HDIM_ISO, Int64}(get_mat_iso(SIGMAZ))
const NEGI_H0_ISO = pi * NEGI * SIGMAZ_ISO
const NEGI_H0_ISO_BIG = pi * NEGI * SMatrix{HDIM_ISO, HDIM_ISO, BigFloat}(SIGMAZ_ISO)
const NEGI_H1_ISO = pi * NEGI * SIGMAX_ISO
const NEGI_H1_ISO_BIG = pi * NEGI * SMatrix{HDIM_ISO, HDIM_ISO, BigFloat}(SIGMAX_ISO)
const FQ_NEGI_H0_ISO = FQ * NEGI_H0_ISO
const AYPIBY2_NEGI_H1_ISO = AYPIBY2 * NEGI_H1_ISO
# relaxation dissipation ops
# L_{0} = |g> <e|
# L_{0}^{\dagger} = |e> <g|
# L_{0}^{\dagger} L_{0} = |e> <e|
# L_{1} = L_{0}^{\dagger} = |e> <g|
# L_{1}^{\dagger} = L_{0} = |g> <e|
# L_{1}^{\dagger} L_{1} = |g> <g|
const G_E = SA_F64[0 1 0 0;
             0 0 0 0;
             0 0 0 1;
             0 0 0 0;]
const E_G = SA_F64[0 0 0 0;
             1 0 0 0;
             0 0 0 0;
             0 0 1 0;]
const NEG_G_G_BY2 = SA_F64[1 0 0 0;
                           0 0 0 0;
                           0 0 1 0;
                           0 0 0 0] * -0.5
const NEG_E_E_BY2 = SA_F64[0 0 0 0;
                           0 1 0 0;
                           0 0 0 0;
                           0 0 0 1;] * -0.5
# dephasing dissipation ops
const NEG_DOP_ISO = -SA_F64[0 1 0 1;
                            1 0 1 0;
                            0 1 0 1;
                            1 0 1 0]
const NEG_GAMMAC_DOP_ISO = GAMMAC * NEG_DOP_ISO

# gates
const ZPIBY2 = [1-1im 0;
                0 1+1im] / sqrt(2)
const ZPIBY2_ISO = SMatrix{HDIM_ISO, HDIM_ISO}(get_mat_iso(ZPIBY2))
const ZPIBY2_ISO_1 = SVector{HDIM_ISO}(get_vec_iso(ZPIBY2[:,1]))
const ZPIBY2_ISO_2 = SVector{HDIM_ISO}(get_vec_iso(ZPIBY2[:,2]))
const YPIBY2 = [1 -1;
                1  1] / sqrt(2)
const YPIBY2_ISO = SMatrix{HDIM_ISO, HDIM_ISO}(get_mat_iso(YPIBY2))
const YPIBY2_ISO_1 = SVector{HDIM_ISO}(get_vec_iso(YPIBY2[:,1]))
const YPIBY2_ISO_2 = SVector{HDIM_ISO}(get_vec_iso(YPIBY2[:,2]))
const XPIBY2 = [1 -1im;
                -1im 1] / sqrt(2)
const XPIBY2_ISO = SMatrix{HDIM_ISO, HDIM_ISO}(get_mat_iso(XPIBY2))
const XPIBY2_ISO_1 = SVector{HDIM_ISO}(get_vec_iso(XPIBY2[:,1]))
const XPIBY2_ISO_2 = SVector{HDIM_ISO}(get_vec_iso(XPIBY2[:,2]))
const XPI = [0 -1im;
             -1im 0]
const XPI_ISO = SMatrix{HDIM_ISO, HDIM_ISO}(get_mat_iso(XPI))
const XPI_ISO_1 = SVector{HDIM_ISO}(get_vec_iso(XPI[:,1]))
const XPI_ISO_2 = SVector{HDIM_ISO}(get_vec_iso(XPI[:,2]))

const GT_GATE = Dict(
    xpiby2 => XPIBY2_ISO,
    ypiby2 => YPIBY2_ISO,
    zpiby2 => ZPIBY2_ISO,
    xpi => XPI_ISO,
)


# methods

"""
amp_fbfq - Compute flux by flux quantum from flux
drive amplitude.
"""
@inline amp_fbfq(amplitude) = amplitude * FBFQ_A + FBFQ_B


"""
amp_fbfq_lo - Compute flux by flux quantum
from flux drive quantum. Flux by flux
quantum will always be under the flux frustration
point (0.5).

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
@inline amp_fbfq_lo(amplitude) = -abs(amplitude) * FBFQ_A + FBFQ_B


"""
fbfq_amp - Compute the amplitude from the flux by
flux quantum. Reflects over the flux frustration point.
"""
@inline fbfq_amp(fbfq) = (fbfq - FBFQ_B) / FBFQ_A


"""
amp_t1_poly - Compute the t1 time for the given amplitude in units
of nanoseconds.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
@inline amp_t1_poly(amplitude) = horner(FBFQ_T1_COEFFS, amp_fbfq_lo(amplitude))


"""
amp_t1_spline - Compute the t1 time in nanoseconds
for the given amplitude.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
@inline amp_t1_spline(amplitude) = FBFQ_T1_SPLINE_ITP(amp_fbfq_lo(amplitude))


@inline amp_t1_reduced_spline(amplitude) = FBFQ_T1_REDUCED_SPLINE_ITP(amp_fbfq_lo(amplitude))


@inline amp_t1_spline_cubic(amplitude) = FBFQ_T1_SPLINE_DIERCKX(amp_fbfq_lo(amplitude))


"""
amp_dfq - Compute the derivative of the qubit frequency
with respect to the flux by flux quantum from the flux drive
amplitude.
"""
# amp_dfq(amplitude) = FBFQ_DFQ_SPLINE_DIERCKX(amp_fbfq(amplitude))


"""
drift hamiltonian subject to flux noise
"""
@inline fqp_negi_h0(dfq, namp_dist) = (FQ + dfq * FBFQ_NAMP * rand(namp_dist)) * NEGI_H0_ISO


"""
control hamiltonian subject to flux noise
"""
@inline ap_negi_h1(amp, namp_dist) = fbfq_amp(amp_fbfq(amp) + FBFQ_NAMP * rand(namp_dist)) * NEGI_H1_ISO


# simulation dynamics
"""
Schroedinger dynamics.
"""
function dynamics_schroed_deqjl(state::StaticVector, params::SimParams, time::Float64)
    controls_knot_point = (Int(floor(time * params.controls_dt_inv)) % params.control_knot_count) + 1
    negi_h = (
        params.negi_h0
        + params.controls[controls_knot_point][1] * NEGI_H1_ISO
    )
    return (
        negi_h * state
    )
end


function dynamics_schroedda_deqjl(state::StaticVector, params::SimParams, time::Float64)
    control_knot_point = (Int(floor(time * params.controls_dt_inv)) % params.control_knot_count) + 1
    noise_knot_point = Int(floor(time * params.noise_dt_inv)) + 1
    delta_a = params.noise_offsets[noise_knot_point]
    negi_h = (
        FQ_NEGI_H0_ISO
        + (params.controls[control_knot_point][1] + delta_a) * NEGI_H1_ISO
    )
    return (
        negi_h * state
    )
end


const maxh1 = MAX_CONTROL_NORM_0 * NEGI_H1_ISO
@inline dynamics_empty_deqjl(state::StaticVector, params::SimParams, time::Float64) = (
    (params.negi_h0 + maxh1) * state
)


function dynamics_lindbladnodis_deqjl(state::StaticMatrix, params::SimParams, time::Float64)
    knot_point = (Int(floor(time * params.dt_inv)) % params.control_knot_count) + 1
    negi_h = (
        params.negi_h0
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        negi_h * density - density * negi_h
    )
end


function dynamics_lindbladt1_deqjl(density::StaticMatrix, params::SimParams, time::Float64)
    controls_knot_point = (Int(floor(time * params.controls_dt_inv)) % params.control_knot_count) + 1
    control1 = params.controls[controls_knot_point][1]
    gamma_1 = (amp_t1_spline(control1))^(-1)
    negi_h = (
        FQ_NEGI_H0_ISO
        + control1 * NEGI_H1_ISO
    )
    return (
        negi_h * density - density * negi_h
        + gamma_1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2)
        + gamma_1 * (E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


function dynamics_lindbladt2_deqjl(state, (controls, control_knot_count, dt_inv, negi_h0, namp_dist), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    negi_h = (
        FQ_NEGI_H0_ISO
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    # gammaf = GAMMAF_PREFACTOR * abs(amp_dfq(controls[knot_point][1]))
    gammaf = GAMMAF_PREFACTOR
    return (
        negi_h * state - state * negi_h
        + (GAMMAC + 2 * gammaf^2 * t) * NEG_DOP_ISO .* state
    )
end


const TTOT_ZPIBY2 = 17.857142857142858
const GAMMA_ZPIBY2 = amp_t1_spline(0)^(-1)
@inline dynamics_zpiby2nodis_deqjl(state::StaticVector, params::SimParams, time::Float64) = (
    params.negi_h0 * state
)


@inline dynamics_zpiby2t1_deqjl(density::StaticMatrix, params::SimParams, time::Float64) = (
    FQ_NEGI_H0_ISO * density - density * FQ_NEGI_H0_ISO
    + GAMMA_ZPIBY2 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2
                      + E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)    
)


@inline dynamics_zpiby2da_deqjl(state::StaticVector, params::SimParams, time::Float64) = (
    (FQ_NEGI_H0_ISO + params.noise_offsets[Int(floor(time * params.noise_dt_inv)) + 1] * NEGI_H1_ISO) * state
)


# standard
const H11_YPIBY2 = AYPIBY2_NEGI_H1_ISO
const H13_YPIBY2 = -AYPIBY2_NEGI_H1_ISO
const H1_YPIBY2 = H11_YPIBY2 + FQ_NEGI_H0_ISO
const H2_YPIBY2 = FQ_NEGI_H0_ISO
const H3_YPIBY2 = H13_YPIBY2 + FQ_NEGI_H0_ISO
const TX_YPIBY2 = 2.1656249366575766
const TZ_YPIBY2 = 15.1423305995572655
const TTOT_YPIBY2 = 19.4735804728724204
const T1_YPIBY2 = TX_YPIBY2
const T2_YPIBY2 = T1_YPIBY2 + TZ_YPIBY2
# t1 noise
const GAMMA11_YPIBY2 = amp_t1_spline(AYPIBY2)^(-1)
const GAMMA12_YPIBY2 = amp_t1_spline(0)^(-1)
function dynamics_ypiby2nodis_deqjl(state::StaticVector, params::SimParams, time::Float64)
    time = rem(time, TTOT_YPIBY2)
    if time <= T1_YPIBY2
        negi_h = FQ_NEGI_H0_ISO + H11_YPIBY2
    elseif time <= T2_YPIBY2
        negi_h = FQ_NEGI_H0_ISO
    else
        negi_h = FQ_NEGI_H0_ISO + H13_YPIBY2
    end
    return(
        negi_h * state
    )
end


function dynamics_ypiby2t1_deqjl(density::StaticMatrix, params::SimParams, time::Float64)
    time = rem(time, TTOT_YPIBY2)
    if time <= T1_YPIBY2
        negi_h = FQ_NEGI_H0_ISO + H11_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif time <= T2_YPIBY2
        negi_h = FQ_NEGI_H0_ISO
        gamma1 = GAMMA12_YPIBY2
    else
        negi_h = FQ_NEGI_H0_ISO + H13_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
        + gamma1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2
                    + E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


const TTOT_XPIBY2 = 4 * TX_YPIBY2 + 2 * TZ_YPIBY2 + TTOT_ZPIBY2
const T1_XPIBY2 = TX_YPIBY2
const T2_XPIBY2 = T1_YPIBY2 + TZ_YPIBY2
const T3_XPIBY2 = T2_XPIBY2 + TX_YPIBY2
const T4_XPIBY2 = T3_XPIBY2 + TTOT_ZPIBY2
const T5_XPIBY2 = T4_XPIBY2 + TX_YPIBY2
const T6_XPIBY2 = T5_XPIBY2 + TZ_YPIBY2
function dynamics_xpiby2nodis_deqjl(state::StaticVector, params::SimParams, time::Float64)
    time = rem(time, TTOT_XPIBY2)
    if time <= T1_XPIBY2
        negi_h = params.negi_h0 + H13_YPIBY2
    elseif time <= T2_XPIBY2
        negi_h = params.negi_h0
    elseif time <= T3_XPIBY2
        negi_h = params.negi_h0 + H11_YPIBY2
    elseif time <= T4_XPIBY2
        negi_h = params.negi_h0
    elseif time <= T5_XPIBY2
        negi_h = params.negi_h0 + H11_YPIBY2
    elseif time <= T6_XPIBY2
        negi_h = params.negi_h0
    else
        negi_h = params.negi_h0 + H13_YPIBY2
    end
    return(
        negi_h * state
    )
end


"""
t1 dissipation via lindblad
"""
function dynamics_xpiby2t1_deqjl(density::StaticMatrix, params::SimParams, time::Float64)
    time = rem(time, TTOT_XPIBY2)
    if time <= T1_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + H13_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif time <= T2_XPIBY2
        negi_h = FQ_NEGI_H0_ISO
        gamma1 = GAMMA12_YPIBY2
    elseif time <= T3_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + H11_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif time <= T4_XPIBY2
        negi_h = FQ_NEGI_H0_ISO
        gamma1 = GAMMA12_YPIBY2
    elseif time <= T5_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + H11_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif time <= T6_XPIBY2
        negi_h = FQ_NEGI_H0_ISO
        gamma1 = GAMMA12_YPIBY2
    else
        negi_h = FQ_NEGI_H0_ISO + H13_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
        + gamma1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2
                    + E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


function dynamics_ypiby2da_deqjl(state::StaticVector, params::SimParams, time::Float64)
    time = rem(time, TTOT_YPIBY2)
    noise_knot_point = Int(floor(time * params.noise_dt_inv)) + 1
    delta_a = params.noise_offsets[noise_knot_point]
    time = rem(time, TTOT_YPIBY2)
    if time <= T1_YPIBY2
        negi_h = FQ_NEGI_H0_ISO + (AYPIBY2 + delta_a) * NEGI_H1_ISO
    elseif time <= T2_YPIBY2
        negi_h = FQ_NEGI_H0_ISO + (delta_a) * NEGI_H1_ISO
    else
        negi_h = FQ_NEGI_H0_ISO + (-AYPIBY2 + delta_a) * NEGI_H1_ISO
    end
    return(
        negi_h * state
    )
end


"""
flux noise via amplitude fluctuation
"""
function dynamics_xpiby2da_deqjl(state::StaticVector, params::SimParams, time::Float64)
    time = rem(time, TTOT_XPIBY2)
    noise_knot_point = Int(floor(time * params.noise_dt_inv)) + 1
    delta_a = params.noise_offsets[noise_knot_point]
    if time <= T1_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (-AYPIBY2 + delta_a) * NEGI_H1_ISO
    elseif time <= T2_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + delta_a * NEGI_H1_ISO
    elseif time <= T3_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (AYPIBY2 + delta_a) * NEGI_H1_ISO
    elseif time <= T4_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + delta_a * NEGI_H1_ISO
    elseif time <= T5_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (AYPIBY2 + delta_a) * NEGI_H1_ISO
    elseif time <= T6_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + delta_a * NEGI_H1_ISO
    else
        negi_h = FQ_NEGI_H0_ISO + (-AYPIBY2 + delta_a) * NEGI_H1_ISO
    end
    return(
        negi_h * state
    )
end


"""
this function assumes noise_dt_inv = 1e1
"""
function integrate_prop_xpiby2da!(gate_count::Int, states::Array{T, 2}, state::SVector, params::SimParams) where {T}
    dt = 1e-1
    states[1, :] = state
    for i = 1:gate_count
        for j = 1:21
            amp = -AYPIBY2 + params.noise_offsets[j]
            prop = exp(dt * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
            state = prop * state
        end
        amp = -AYPIBY2 + params.noise_offsets[22]
        prop = exp((T1_XPIBY2 - 2.1) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        amp = params.noise_offsets[22]
        prop = exp((2.2 - T1_XPIBY2) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        for j = 23:173
            amp = params.noise_offsets[j]
            prop = exp(dt * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
            state = prop * state
        end
        amp = params.noise_offsets[174]
        prop = exp((T2_XPIBY2 - 17.3) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        amp = AYPIBY2 + params.noise_offsets[174]
        prop = exp((17.4 - T2_XPIBY2) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        for j = 175:194
            amp = AYPIBY2 + params.noise_offsets[j]
            prop = exp(dt * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
            state = prop * state
        end
        amp = AYPIBY2 + params.noise_offsets[195]
        prop = exp((T3_XPIBY2 - 19.4) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        amp = params.noise_offsets[195]
        prop = exp((19.5 - T3_XPIBY2) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        for j = 196:373
            amp = params.noise_offsets[j]
            prop = exp(dt * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
            state = prop * state
        end
        amp = params.noise_offsets[374]
        prop = exp((T4_XPIBY2 - 37.3) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        amp = AYPIBY2 + params.noise_offsets[374]
        prop = exp((37.4 - T4_XPIBY2) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        for j = 375:394
            amp = AYPIBY2 + params.noise_offsets[j]
            prop = exp(dt * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
            state = prop * state
        end
        amp = AYPIBY2 + params.noise_offsets[395]
        prop = exp((T5_XPIBY2 - 39.4) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        amp = params.noise_offsets[395]
        prop = exp((39.5 - T5_XPIBY2) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        for j = 396:546
            amp = params.noise_offsets[j]
            prop = exp(dt * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
            state = prop * state
        end
        amp = params.noise_offsets[547]
        prop = exp((T6_XPIBY2 - 54.6) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        amp = -AYPIBY2 + params.noise_offsets[547]
        prop = exp((54.7 - T6_XPIBY2) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        for j = 548:568
            amp = -AYPIBY2 + params.noise_offsets[j]
            prop = exp(dt * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
            state = prop * state
        end
        amp = -AYPIBY2 + params.noise_offsets[569]
        prop = exp((TTOT_XPIBY2 - 56.8) * (FQ_NEGI_H0_ISO + amp * NEGI_H1_ISO))
        state = prop * state
        
        # save state
        states[i + 1, :] = state
    end
end


const TTOT_XPIBY2C = 56.8
const A_XPIBY2C = 0.03556243915206662
const ARWA_XPIBY2C = 2 * A_XPIBY2C
const T1_XPIBY2C = 30.017250262129778
const T2_XPIBY2C = T1_XPIBY2C + 24.8850554973045
const H11_XPIBY2C = A_XPIBY2C * NEGI_H1_ISO
const H12_XPIBY2C = -A_XPIBY2C * NEGI_H1_ISO
function dynamics_xpiby2corpse_deqjl(state::StaticVector, params::SimParams, time::Float64)
    time_ = rem(time, TTOT_XPIBY2C)
    if time_ <= T1_XPIBY2C
        negi_h = params.negi_h0 + H11_XPIBY2C
    elseif time_ <= T2_XPIBY2C
        negi_h = params.negi_h0 + H12_XPIBY2C
    else
        negi_h = params.negi_h0 + H11_XPIBY2C
    end
    return (
        negi_h * state
    )
end


function dynamics_xpiby2corpserwa_deqjl(state::StaticVector, params::SimParams, time::Float64)
    time_ = rem(time, TTOT_XPIBY2C)
    if time_ <= T1_XPIBY2C
        negi_h = params.negi_h0 + (ARWA_XPIBY2C * cos(WQ * time)) * NEGI_H1_ISO
    elseif time_ <= T2_XPIBY2C
        negi_h = params.negi_h0 + (-ARWA_XPIBY2C * cos(WQ * time)) * NEGI_H1_ISO
    else
        negi_h = params.negi_h0 + (ARWA_XPIBY2C * cos(WQ * time)) * NEGI_H1_ISO
    end
    return (
        negi_h * state
    )
end


const TTOT_XPIC = 10
const A_XPIC = 13//60
const T1_XPIC = 420//78
const T2_XPIC = 720//78
const H11_XPIC = A_XPIC * NEGI_H1_ISO_BIG
const H12_XPIC = -A_XPIC * NEGI_H1_ISO_BIG
function dynamics_xpicorpse_deqjl(state::StaticVector, params::SimParams, time)
    time_ = rem(time, TTOT_XPIC)
    if time_ <= T1_XPIC
        negi_h = H11_XPIC # params.negi_h0 + H11_XPIC
    elseif time_ <= T2_XPIC
        negi_h = H12_XPIC # params.negi_h0 + 
    else
        negi_h = H11_XPIC # params.negi_h0 + 
    end
    return (
        negi_h * state
    )
end


# dynamics lookup
const DT_DYN = Dict(
    schroed => dynamics_schroed_deqjl,
    schroedda => dynamics_schroedda_deqjl,
    lindbladnodis => dynamics_lindbladnodis_deqjl,
    lindbladt1 => dynamics_lindbladt1_deqjl,
    lindbladt2 => dynamics_lindbladt2_deqjl,
    ypiby2nodis => dynamics_ypiby2nodis_deqjl,
    ypiby2t1 => dynamics_ypiby2t1_deqjl,
    xpiby2nodis => dynamics_xpiby2nodis_deqjl,
    xpiby2t1 => dynamics_xpiby2t1_deqjl,
    zpiby2nodis => dynamics_zpiby2nodis_deqjl,
    zpiby2t1 => dynamics_zpiby2t1_deqjl,
    xpiby2da => dynamics_xpiby2da_deqjl,
    zpiby2da => dynamics_zpiby2da_deqjl,
    empty => dynamics_empty_deqjl,
    ypiby2da => dynamics_ypiby2da_deqjl,
    xpiby2corpserwa => dynamics_xpiby2corpserwa_deqjl,
    xpiby2corpse => dynamics_xpiby2corpse_deqjl,
    xpicorpse => dynamics_xpicorpse_deqjl,
)

# integartor lookup
const DT_INT = Dict(
    xpiby2da => integrate_prop_xpiby2da!,
)


# gate time lookup
const DT_GTM = Dict(
    zpiby2nodis => TTOT_ZPIBY2,
    zpiby2t1 => TTOT_ZPIBY2,
    zpiby2da => TTOT_ZPIBY2,
    ypiby2nodis => TTOT_YPIBY2,
    ypiby2t1 => TTOT_YPIBY2,
    xpiby2nodis => TTOT_XPIBY2,
    xpiby2t1 => TTOT_XPIBY2,
    xpiby2da => TTOT_XPIBY2,
    empty => 160.,
    ypiby2da => TTOT_YPIBY2,
    xpiby2corpserwa => TTOT_XPIBY2C,
    xpiby2corpse => TTOT_XPIBY2C,
    xpicorpse => TTOT_XPIC,
)


# save file path lookup
const DT_EN = Dict(
    zpiby2nodis => "spin14",
    zpiby2t1 => "spin14",
    zpiby2da => "spin14",
    ypiby2nodis => "spin14",
    ypiby2t1 => "spin14",
    xpiby2nodis => "spin14",
    xpiby2t1 => "spin14",
    xpiby2da => "spin14",
    ypiby2da => "spin14",
    xpiby2corpserwa => "spin14",
    xpiby2corpse => "spin14",
    xpicorpse => "spin14",
)


# cost functions
@inline fidelity_vec_iso2(s1, s2) = (
    (s1's2)^2 + (s1[1] * s2[3] + s1[2] * s2[4] - s1[3] * s2[1] - s1[4] * s2[2])^2
)


@inline gate_error_iso2a(s1::SVector, s2::SVector{HDIM_ISO}, s1o::Int64) = (
    s1[1 + s1o] * s2[1] + s1[2 + s1o] * s2[2] + s1[3 + s1o] * s2[3] + s1[4 + s1o] * s2[4]
)


@inline gate_error_iso2b(s1::SVector, s2::SVector{HDIM_ISO}, s1o::Int64) = (
    -s1[3 + s1o] * s2[1] - s1[4 + s1o] * s2[2] + s1[1 + s1o] * s2[3] + s1[2 + s1o] * s2[4]
)


@inline gate_error_iso2(s1::SVector, s2::SVector{HDIM_ISO}, s1o::Int64=0) = (
    1 - gate_error_iso2a(s1, s2, s1o)^2 - gate_error_iso2b(s1, s2, s1o)^2
)


function jacobian_gate_error_iso2(s1::SVector, s2::SVector{HDIM_ISO},
                                  s1o::Int64=0) :: SVector{HDIM_ISO}
    a = 2 * gate_error_iso2a(s1, s2, s1o)
    b = 2 * gate_error_iso2b(s1, s2, s1o)
    jac = @SVector [
        -a * s2[1] - b * s2[3],
        -a * s2[2] - b * s2[4],
        -a * s2[3] + b * s2[1],
        -a * s2[4] + b * s2[2],
    ]
    return jac
end


function hessian_gate_error_iso2(s2::SVector{HDIM_ISO}) :: SMatrix{4, 4}
    d11 = -2 * s2[1]^2 - 2 * s2[3]^2
    d12 = -2 * s2[1] * s2[2] -2 * s2[3] * s2[4]
    d13 = 0
    d14 = 2 * s2[2] * s2[3] - 2 * s2[1] * s2[4]
    d22 = -2 * s2[2]^2 - 2 * s2[4]^2
    d23 = -2 * s2[2] * s2[3] + 2 * s2[1] * s2[4]
    d24 = 0
    d33 = -2 * s2[1]^2 - 2 * s2[3]^2
    d34 = -2 * s2[1] * s2[2] - 2 * s2[3] * s2[4]
    d44 = -2 * s2[2]^2 - 2 * s2[4]^2
    hes = @SMatrix [
        d11 d12 d13 d14;
        d12 d22 d23 d24;
        d13 d23 d33 d34;
        d14 d24 d34 d44;
    ]
    return hes
end


"""
See e.q. 9.71 in [0]

[0] Nielsen, M. A., & Chuang, I. (2002).
    Quantum computation and quantum information.
"""
function fidelity_mat_iso(m1_, m2_)
    n = size(m1_)[1]
    nby2 = Integer(n/2)
    i1 = 1:nby2
    i2 = (nby2 + 1):n
    m1 = m1_[i1, i1] + 1im * m1_[i2, i1]
    m2 = m2_[i1, i1] + 1im * m2_[i2, i1]
    sqrt_m1 = sqrt(Hermitian(m1))
    sqrt_m2 = sqrt(Hermitian(m2))
    return tr(sqrt_m1 * sqrt_m2)^2
end


function gen_rand_state_iso(;seed=0)
    if seed == 0
        state = [1, 1] / sqrt(2)
    else
        Random.seed!(seed)
        state = rand(HDIM) + 1im * rand(HDIM)
        state = state / sqrt(real(state'state))
    end
    return SVector{HDIM_ISO, BigFloat}(
        [real(state); imag(state)]
    )
end


function gen_rand_density_iso(;seed=0)
    if seed == 0
        state = [1, 1] / sqrt(2)
    else
        Random.seed!(seed)
        state = rand(HDIM) + 1im * rand(HDIM)
        state = state / sqrt(real(state'state))
    end
    density = state * state'
    density_r = real(density)
    density_i = imag(density)
    density_iso = SMatrix{HDIM_ISO, HDIM_ISO, BigFloat}([
        density_r -density_i;
        density_i density_r;
    ])
    return density_iso
end


"""
Generate noise with spectral density
Sxx(f) = |x̂(f)|^2 = f^(1/2)

References:
[0] https://dsp.stackexchange.com/questions/322/pink-1-f-pseudo-random-noise-generation
[1] https://www.mathworks.com/help/matlab/ref/filter.html;jsessionid=6d7d2599d97bf87d3ac199b5eaef
[2] https://ccrma.stanford.edu/~jos/filters/
"""
function gen_pink_noise(count, dt_inv, ndist; seed=0)
    b1 = 0.049922035
    b2 = -0.095993537
    b3 = 0.050612699
    b4 = -0.004408786
    a1 = 1
    a2 = -2.494956002
    a3 = 2.017265875
    a4 = -0.522189400
    Random.seed!(seed)
    xs = rand(ndist, count)
    ys = zeros(count)
    ys[1] = b1 * xs[1]
    ys[2] = b1 * xs[2] + b2 * xs[1] - a2 * ys[1]
    ys[3] = b1 * xs[3] + b2 * xs[2] + b3 * xs[1] - a2 * ys[2] - a3 * ys[1]
    for i = 4:count
        ys[i] = (b1 * xs[i] + b2 * xs[i - 1] + b3 * xs[i - 2] + b4 * xs[i - 3]
                 - a2 * ys[i - 1] - a3 * ys[i - 2] - a4 * ys[i - 3])
    end
    ys = ys .* dt_inv
    
    return ys
end


"""
compute_fidelities
"""
function compute_fidelities(gate_count, gate_type, states)
    # Compute the fidelities.
    # All of the gates we consider are 4-cyclic up to phase.
    state_type = length(size(states)) == 2 ? st_state : st_density
    initial_state = state_type == st_state ? states[1, :] : states[1, :, :]
    fidelities = zeros(BigFloat, gate_count + 1)
    g1 = GT_GATE[gate_type]
    g2 = g1^2
    g3 = g1^3
    id0 = initial_state
    if state_type == st_state
        id1 = g1 * id0
        id2 = g2 * id0
        id3 = g3 * id0
    elseif state_type == st_density
        id1 = g1 * id0 * g1'
        id2 = g2 * id0 * g2'
        id3 = g3 * id0 * g3'
    end
    # Compute the fidelity after each gate.
    for i = 1:gate_count + 1
        # 1-indexing means we are 1 ahead for modulo arithmetic.
        i_eff = i - 1
        if i_eff % 4 == 0
            target = id0
        elseif i_eff % 4 == 1
            target = id1
        elseif i_eff % 4 == 2
            target = id2
        elseif i_eff % 4 == 3
            target = id3
        end
        if state_type == st_state
            fidelities[i] = fidelity_vec_iso2(states[i, :], target)
        elseif state_type == st_density
            fidelities[i] = fidelity_mat_iso(states[i, :, :], target)
        end
    end

    return fidelities
end


"""
run_sim_deqjl - Apply a gate multiple times and measure the fidelity
after each application. Save the output.

Arguments:
save_file_path :: String - The file path to grab the controls from

Returns:
result :: Union{String, Dict} - string to save file path if save is true,
otherwise returns a dictionary of the result
"""
function run_sim_deqjl(
    gate_count, gate_type;
    save_file_path=nothing,
    adaptive=true, dynamics_type=schroed,
    dt_inv=DT_PREF_INV, save=true, seed=0,
    solver=DifferentialEquations.Vern9, print_seq=false, print_final=false,
    negi_h0=FQ_NEGI_H0_ISO, namp=NAMP_PREFACTOR, ndist=STD_NORMAL,
    noise_dt_inv=DT_NOISE_INV, state_seed=nothing, reltol=1e-12, abstol=1e-12)
    dt = BigFloat(dt_inv^(-1))
    start_time = Dates.now()
    if isnothing(state_seed)
        state_seed = seed
    end
    # grab
    analytic = false
    if isnothing(save_file_path)
        controls = Array{Float64, 2}([0 0])
        controls_dt_inv = 0
        control_knot_count = 0
        gate_time = DT_GTM[dynamics_type]
        analytic = true
    else
        (controls, controls_dt_inv, gate_time) = grab_controls(save_file_path)
        control_knot_count = Int(floor(gate_time * controls_dt_inv))
    end
    save_times = Array(0:1:gate_count) * gate_time
    evolution_time = gate_time * gate_count
    knot_count = Int(ceil(evolution_time * dt_inv))

    # get noise offsets
    noise_knot_count = Int(ceil(evolution_time * noise_dt_inv)) + 1
    noise_offsets = namp * gen_pink_noise(noise_knot_count, noise_dt_inv, ndist; seed=seed)
    
    # integrate
    dynamics = DT_DYN[dynamics_type]
    state_type = DT_ST[dynamics_type]
    if state_type == st_state
        initial_state = gen_rand_state_iso(;seed=state_seed)
    elseif state_type == st_density
        initial_state =  gen_rand_density_iso(;seed=state_seed)
    end
    tspan = (0., evolution_time)
    params = SimParams(controls, control_knot_count, controls_dt_inv, negi_h0,
                       noise_offsets, noise_dt_inv, dt_inv)
    prob = ODEProblem(dynamics, initial_state, tspan, params)
    result_deqjl = solve(prob, solver(), dt=dt, saveat=save_times,
                         maxiters=DEQJL_MAXITERS, adaptive=adaptive,
                         reltol=reltol, abstol=abstol)
    
    if state_type == st_state
        states = zeros(gate_count + 1, HDIM_ISO)
        states[1, :] = initial_state
        for i = 2:gate_count + 1
            states[i, :] = Array(result_deqjl.u[i])
        end
    else
        states = zeros(gate_count + 1, HDIM_ISO, HDIM_ISO)
        states[1, :, :] = initial_state
        for i = 2:gate_count + 1
            states[i, :, :] = Array(result.u[i])
        end
    end
    fidelities = compute_fidelities(gate_count, gate_type, states)
    end_time = Dates.now()
    run_time = end_time - start_time

    # Generate the result.
    result = Dict(
        "dynamics_type" => Integer(dynamics_type),
        "gate_count" => gate_count,
        "gate_time" => gate_time,
        "gate_type" => Integer(gate_type),
        "save_file_path" => isnothing(save_file_path) ? "" : save_file_path,
        "seed" => seed,
        "states" => states,
        "fidelities" => fidelities,
        "run_time" => string(run_time),
        "dt" => dt,
        "negi_h0" => Array(negi_h0),
        "namp" => namp,
        "ndist" => string(ndist),
        "noise_dt_inv" => noise_dt_inv,
    )
    
    # Save the data.
    if save
        experiment_name = save_path = nothing
        if isnothing(save_file_path)
            experiment_name = DT_EN[dynamics_type]
            save_path = joinpath(SPIN_OUT_PATH, experiment_name)
        else
            experiment_name = split(save_file_path, "/")[end - 1]
            save_path = dirname(save_file_path)
        end
        data_file_path = generate_file_path("h5", experiment_name, save_path)
        h5open(data_file_path, "w") do data_file
            for key in keys(result)
                write(data_file, key, result[key])
            end
        end
        result["data_file_path"] = data_file_path
        println("Saved simulation to $(data_file_path)")
    end
    
    return result
end


"""
run_sim_fine_deqjl - Run a simulation, save points.

Arguments:
save_file_path :: String - The file path to grab the controls from
"""
function run_sim_fine_deqjl(
    ;gate_count = 1,
    save_file_path=nothing,
    adaptive=DEQJL_ADAPTIVE, dynamics_type=schroed,
    dt=DT_PREF, save=true, save_type=jl, seed=0,
    solver=DifferentialEquations.Vern9, print_seq=false, print_final=false,
    negi_h0=FQ_NEGI_H0_ISO, namp=NAMP_PREFACTOR, ndist=STD_NORMAL,
    noise_dt_inv=DT_NOISE_INV, save_step=1e-1)
    start_time = Dates.now()
    # grab
    if isnothing(save_file_path)
        controls = Array{Float64, 2}([0 0])
        controls_dt_inv = 0
        control_knot_count = 0
        gate_time = DT_GTM[dynamics_type]
    else
        (controls, controls_dt_inv, gate_time) = grab_controls(save_file_path; save_type=save_type)
        control_knot_count = Int(floor(gate_time * controls_dt_inv))
    end
    dt_inv = dt^(-1)
    evolution_time = gate_time * gate_count
    save_step_inv = save_step^(-1)
    save_count = Int(ceil(evolution_time * save_step_inv))
    save_times = Array(0:save_count) * save_step

    # get noise offsets
    noise_knot_count = Int(ceil(evolution_time * noise_dt_inv)) + 1
    noise_offsets = namp .* gen_pink_noise(noise_knot_count, noise_dt_inv, ndist; seed=seed)
    
    # integrate
    dynamics = DT_DYN[dynamics_type]
    state_type = DT_ST[dynamics_type]
    if state_type == st_state
        initial_state = gen_rand_state_iso(;seed=seed)
    elseif state_type == st_density
        initial_state =  gen_rand_density_iso(;seed=seed)
    end
    tspan = (0., evolution_time)
    params = SimParams(controls, control_knot_count, controls_dt_inv, negi_h0,
                       noise_offsets, noise_dt_inv, dt_inv)
    prob = ODEProblem(dynamics, initial_state, tspan, params)

    result = solve(prob, solver(), dt=dt, saveat=save_times,
                   maxiters=DEQJL_MAXITERS, adaptive=adaptive)

    if state_type == st_state
        states_ = zeros(save_count + 1, HDIM_ISO)
        for i = 1:save_count
            states_[i, :] = Array(result.u[i])
        end
    elseif state_type == st_density
        states_ = zeros(save_count + 1, HDIM_ISO, HDIM_ISO)
        for i = 1:save_count
            states_[i, :, :] = Array(result.u[i])
        end
    end

    end_time = Dates.now()
    run_time = end_time - start_time

    # Save the data.
    experiment_name = save_path = nothing
    if isnothing(save_file_path)
        experiment_name = DT_EN[dynamics_type]
        save_path = joinpath(SPIN_OUT_PATH, experiment_name)
    else
        experiment_name = split(save_file_path, "/")[end - 1]
        save_path = dirname(save_file_path)
    end
    data_file_path = nothing
    if save
        data_file_path = generate_file_path("h5", experiment_name, save_path)
        h5open(data_file_path, "w") do data_file
            write(data_file, "dynamics_type", Integer(dynamics_type))
            write(data_file, "gate_count", gate_count)
            write(data_file, "gate_time", gate_time)
            write(data_file, "save_file_path", isnothing(save_file_path) ? "" : save_file_path)
            write(data_file, "seed", seed)
            write(data_file, "states", states_)
            write(data_file, "run_time", string(run_time))
            write(data_file, "dt", dt)
            write(data_file, "negi_h0", Array(negi_h0))
            write(data_file, "namp", namp)
            write(data_file, "ndist", string(ndist))
            write(data_file, "noise_dt_inv", noise_dt_inv)
        end
        println("Saved simulation to $(data_file_path)")
    end
    return data_file_path
end


"""
sample_controls - Sample controls and d2controls_dt2
on the preferred time axis using a spline.
"""
function sample_controls(save_file_path; dt=DT_PREF, dt_inv=DT_PREF_INV,
                         plot=false, plot_file_path=nothing)
    # Grab data to sample from.
    save = read_save(save_file_path)
    controls = save["astates"][1:end - 1, (save["controls_idx"])]
    d2controls_dt2 = save["acontrols"][1:end, save["d2controls_dt2_idx"]]
    (control_knot_count, control_count) = size(controls)
    if "dt_idx" in keys(save)
        dts = save["acontrols"][1:end, save["dt_idx"]]
    elseif "dt" in keys(save)
        dts = save["dt"] * ones(control_knot_count)
    end
    time_axis = [0; cumsum(dts, dims=1)[1:end - 1]]

    # Construct time axis to sample over.
    final_time_sample = sum(dts)
    knot_count_sample = Int(floor(final_time_sample * dt_inv))
    # The last control should be DT_PREF before final_time_sample.
    time_axis_sample = Array(0:1:knot_count_sample - 1) * dt

    # Sample time_axis_sample via spline.
    controls_sample = zeros(knot_count_sample, control_count)
    d2controls_dt2_sample = zeros(knot_count_sample, control_count)
    for i = 1:control_count
        controls_spline = Spline1D(time_axis, controls[:, i])
        controls_sample[:, i] = map(controls_spline, time_axis_sample)
        d2controls_dt2_spline = Spline1D(time_axis, d2controls_dt2[:, i])
        d2controls_dt2_sample[:, i] = map(d2controls_dt2_spline, time_axis_sample)
    end

    # Plot.
    if plot
        fig = Plots.plot(dpi=DPI)
        Plots.scatter!(time_axis, controls[:, 1], label="controls data", markersize=MS_SMALL, alpha=ALPHA)
        Plots.scatter!(time_axis_sample, controls_sample[:, 1], label="controls fit",
                       markersize=MS_SMALL, alpha=ALPHA)
        Plots.scatter!(time_axis, d2controls_dt2[:, 1], label="d2_controls_dt2 data")
        Plots.scatter!(time_axis_sample, d2controls_dt2_sample[:, 1], label="d2_controls_dt2 fit")
        Plots.xlabel!("Time (ns)")
        Plots.ylabel!("Amplitude (GHz)")
        Plots.savefig(fig, plot_file_path)
        println("Plotted to $(plot_file_path)")
    end
    return (controls_sample, d2controls_dt2_sample, final_time_sample)
end


"""
t1_average - Compute the average t1 time for a control pulse.
"""
function t1_average(save_file_path; save_type=jl)
    # Grab and prep data.
    (controls, controls_dt_inv, evolution_time) = grab_controls(save_file_path; save_type=save_type)
    (control_knot_count, control_count) = size(controls)
    t1_avgs = zeros(control_count)
    for i = 1:control_count
        t1s = map(amp_t1_spline, controls[:, i] / (2 * pi))
        t1_avgs[i] = mean(t1s)
    end
    
    return t1_avgs
end


"""
integrate the schroedinger equation using unitary propagators
"""
function run_sim_prop(
    gate_count, gate_type; save_file_path=nothing,
    negi_h0=FQ_NEGI_H0_ISO, namp=NAMP_PREFACTOR, ndist=STD_NORMAL,
    noise_dt_inv=DT_NOISE_INV, seed=0, state_seed=nothing, save=false,
    dynamics_type=schroed)
    if isnothing(save_file_path)
        gate_time = DT_GTM[dynamics_type]
        controls = zeros(1, 1)
        control_knot_count = dt_inv = 0
        integrator = DT_INT[dynamics_type]
    else
        (controls, dt_inv, gate_time) = grab_controls(save_file_path)
        dt = dt_inv^(-1)
        control_knot_count = Int(floor(gate_time * dt_inv))
        integrator = nothing
    end
    save_times = Array(0:1:gate_count) * gate_time
    evolution_time = gate_time * gate_count
    noise_knot_count = Int(ceil(evolution_time * noise_dt_inv)) + 1
    noise_offsets = namp * gen_pink_noise(noise_knot_count, noise_dt_inv, ndist; seed=seed)
    params = SimParams(controls, control_knot_count, dt_inv, negi_h0,
                       noise_offsets, noise_dt_inv, dt_inv)

    if isnothing(state_seed)
        state_seed = seed
    end
    initial_state = state = gen_rand_state_iso(;seed=state_seed)
    states = zeros(BigFloat, gate_count + 1, HDIM_ISO)
    states[1, :] = Array(state)
    time = 0.
    
    if !isnothing(integrator)
        integrator(gate_count, states, state, params)
    elseif dynamics_type == schroed
        for i = 1:gate_count
            for j = 1:control_knot_count
                hamiltonian = negi_h0 + controls[j, 1] * NEGI_H1_ISO
                unitary = exp(hamiltonian * dt)
                state = unitary * state
            end
            states[i + 1, :] = state
        end
    elseif dynamics_type == schroedda
        for i = 1:gate_count
            for j = 1:control_knot_count
                delta_a = params.noise_offsets[Int(floor(time * params.noise_dt_inv)) + 1]
                hamiltonian = negi_h0 + (controls[j, 1] + delta_a) * NEGI_H1_ISO
                unitary = exp(hamiltonian * dt)
                state = unitary * state
                time = time + dt
            end
            states[i + 1, :] = state
        end
    end
    

    fidelities = compute_fidelities(gate_count, gate_type, states)

    result = Dict(
        "gate_count" => gate_count,
        "gate_type" => Integer(gate_type),
        "seed" => seed,
        "state_seed" => state_seed,
        "states" => states,
        "fidelities" => fidelities,
        "negi_h0" => Array(negi_h0),
        "namp" => namp,
        "ndist" => string(ndist),
        "noise_dt_inv" => noise_dt_inv,
        "save_file_path" => save_file_path
    )

    return result
end


### TESTS ###
function gen_rand_state_()
    s = SVector{2}(rand(2) + 1im * rand(2))
    return s / sqrt(s's)
end


function test_ge_(;seed=0, do_ge=true, do_gej=true, do_geh=true)
    Random.seed!(seed)
    big_ge = Int(1e4)
    big_gej = Int(1e4)
    big_geh = Int(1e4)
    
    if do_ge
        for i = 1:big_ge
            x1 = gen_rand_state_()
            x2 = gen_rand_state_()
            x1_iso = get_vec_iso(x1)
            x2_iso = get_vec_iso(x2)
            ge = 1 - abs(x1'x2)^2
            ge_iso = gate_error_iso2(x1_iso, x2_iso)
            @assert isapprox(ge, ge_iso)
        end
    end

    if do_gej
        for i = 1:big_gej
            x1 = gen_rand_state_()
            x2 = gen_rand_state_()
            x1_iso = get_vec_iso(x1)
            x2_iso = get_vec_iso(x2)
            ge_aug(x1_iso_) = gate_error_iso2(x1_iso_, x2_iso)
            (gej_z,) = Zygote.gradient(ge_aug, x1_iso)
            gej_m = jacobian_gate_error_iso2(x1_iso, x2_iso)
            @assert isapprox(gej_z, gej_m)
        end
    end

    if do_geh
        for i = 1:big_geh
            x1 = gen_rand_state_()
            x2 = gen_rand_state_()
            x1_iso = get_vec_iso(x1)
            x2_iso = get_vec_iso(x2)
            ge_aug(x1_iso_) = gate_error_iso2(x1_iso_, x2_iso)
            geh_z = Zygote.hessian(ge_aug, x1_iso)
            geh_m = hessian_gate_error_iso2(x2_iso)
            @assert isapprox(geh_z, geh_m)
        end
    end
end


const NEGI2_H0_ISO = 2 * NEGI_H0_ISO
function dynamics_dfstate_deqjl(astate::SVector, params::SimParams, time::Float64)
    state_idx = 1:4
    dstate_idx = 5:8
    d2state_idx = 9:12
    controls_knot_point = (Int(floor(time * params.controls_dt_inv)) % params.control_knot_count) + 1
    negi_h = (
        params.negi_h0
        + params.controls[controls_knot_point, 1] * NEGI_H1_ISO
    )
    
    dstate = negi_h * astate[state_idx]
    ddstate = negi_h * astate[dstate_idx] + NEGI_H0_ISO * astate[state_idx]
    dd2state = negi_h * astate[d2state_idx] + NEGI2_H0_ISO * astate[dstate_idx]
    dastate = [dstate; ddstate; dd2state]
    
    return dastate
end


function test_expdeqjl(;seed=1, negi_h0=FQ_NEGI_H0_ISO, reltol=1e-12, abstol=1e-12,
                       adaptive=true, solver=DifferentialEquations.Vern9)
    save_file_path = joinpath(SPIN_OUT_PATH, "spin12/00627_spin12.h5")
    (controls, controls_dt_inv, evolution_time) = grab_controls(save_file_path)
    dt = controls_dt_inv^(-1)
    control_knot_count = Int(floor(evolution_time * controls_dt_inv))
    initial_state = gen_rand_state_iso(;seed=seed)
    initial_astate = [initial_state; @SVector zeros(2 * HDIM_ISO)]

    # deqjl
    # tspan = (0., evolution_time)
    # save_times = [0., evolution_time]
    # params = SimParams(controls, control_knot_count, controls_dt_inv, negi_h0,
    #                    zeros(1), 0, 0)
    # prob = ODEProblem(dynamics_dfstate_deqjl, initial_astate, tspan, params)
    # result_deqjl = solve(prob, solver(), saveat=save_times, maxiters=DEQJL_MAXITERS,
    #                      adaptive=adaptive, reltol=reltol, abstol=abstol)
    # final_astate_deqjl = result_deqjl.u[end]
    final_astate_deqjl = [
        0.3297502016083930496857710107507070282439170198459780516957163426947026553848819;
        0.8911153066503424150898098622796038253240145733901272546764101366400421501465792;
        -0.04570436686707923806850534875355927461876076593027880863091868412801804526696696;
        -0.3083657335729272441538341062792230805961115399686374631442072705867296065360932;
        -1.762081244798284307668549658483409907816552049440236848997044142716682669755047;
        -6.774290676829364446826621882987452576489399582278228041439766274786186033794044;
       -94.29777221085506261481004895271138678686498582042056237068641281709503443245519;
        -7.48429715886366019870079175966160929452897909028678916403918021630619572993541;
     -2782.797334522913580692186177346094111441609178846508985051472623722131186875704;
     -7576.02782757433488986108912497316375333435530772993752967348280813288799442085;
     -2641.4045650586007199458644455694431946142558061399391205939738927683334249393;
        4699.16633090618516756463787460347438315674579889839606170982251549847103430348;
    ]

    # exp
    knot_count = Int(floor(evolution_time * controls_dt_inv))
    state = initial_state
    dstate = @SVector zeros(HDIM_ISO)
    d2state = @SVector zeros(HDIM_ISO)
    for i = 1:knot_count
        # assume negi_h and NEGI_H0_ISO commute
        # negi_h = negi_h0 + controls[i, 1] * NEGI_H1_ISO
        # ez = exp(dt * negi_h)
        # d2state = ez * d2state + dt * NEGI2_H0_ISO * dstate
        # dstate = ez * dstate + dt * NEGI_H0_ISO * state
        # state = ez * state
        
        # lawson-euler
        # negi_h = negi_h0 + controls[i, 1] * NEGI_H1_ISO
        # ez = exp(dt * negi_h)
        # d2state = ez * d2state + dt * ez * NEGI2_H0_ISO * dstate
        # dstate = ez * dstate + dt * ez * NEGI_H0_ISO * state
        # state = ez * state

        # lawson4
        # negi_h = negi_h0 + controls[i, 1] * NEGI_H1_ISO
        # ez = exp(dt * negi_h)
        # ezby2 = exp(dt * 1//2 * negi_h)
        # k1d = NEGI_H0_ISO * state
        # k2d = NEGI_H0_ISO * (state + 1//2 * ezby2 * k1d)
        # k3d = NEGI_H0_ISO * (state + 1//2 * k2d)
        # k4d = NEGI_H0_ISO * (state + ezby2 * k3d)
        # dstate = (ez * dstate + dt * 1//6 * (
        #         ez * k1d + 2 * ezby2 * k2d + 2 * ezby2 * k3d + k4d
        # ))
        # state = ez * state
    end
    final_astate_exp = [state; dstate; d2state]

    println("final_astate_deqjl")
    show_nice(final_astate_deqjl)
    println("\nfinal_astate_exp")
    show_nice(final_astate_exp)
    println("")
end
