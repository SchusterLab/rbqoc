"""
spin.jl - common definitions for the spin directory
"""

using Dates
using Dierckx
using DifferentialEquations
using Distributions
using FFTW
using HDF5
using Random
using StaticArrays
using Statistics

WDIR = get(ENV, "RBQOC_PATH", "../../")
include(joinpath(WDIR, "src", "rbqoc.jl"))

# paths
const SPIN_OUT_PATH = joinpath(WDIR, "out", "spin")
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
    ypiby2nodis => st_density,
    ypiby2t1 => st_density,
    xpiby2nodis => st_state,
    xpiby2t1 => st_density,
    zpiby2nodis => st_density,
    zpiby2t1 => st_density,
    xpiby2da => st_state,
    xpiby2t2 => st_density,
    zpiby2da => st_state,
    empty => st_state,
)

const GT_STR = Dict(
    zpiby2 => "Z/2",
    ypiby2 => "Y/2",
    xpiby2 => "X/2",
)

struct SimParams
    controls :: Array{Float64, 2}
    control_knot_count :: Int64
    controls_dt_inv :: Int64
    negi_h0 :: StaticMatrix
    noise_offsets :: Array{Float64, 1}
    noise_dt_inv :: Int64
    sim_dt_inv :: Int64
end

# other constants
const DEQJL_MAXITERS = 1e10
const DEQJL_ADAPTIVE = false


# Define experimental constants.
# E / h e-9
const EC = 0.479
const EL = 0.132
const EJ = 3.395
const E_PLASMA = sqrt(8 * EL * EC)
const PHI_OSC = (8 * EC / EL)^(1//4)
# qubit frequency at flux frustration point
const FQ = 1.4e-2 #GHz
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
# TODO: idk if hdf5 is inter-process safe
(FBFQ_DFQ_FBFQS, FBFQ_DFQ_DFQS) = h5open(FBFQ_DFQ_DATA_FILE_PATH, "r") do data_file
    fbfqs = read(data_file, "fbfqs")
    dfqs = read(data_file, "dfqs")
    return(fbfqs, dfqs)
end
const FBFQ_DFQ_SPLINE_DIERCKX = Spline1D(FBFQ_DFQ_FBFQS, FBFQ_DFQ_DFQS)
const STATE_SIZE_NOISO = 2
const STATE_SIZE_ISO = 2 * STATE_SIZE_NOISO
const ZPIBY2_GATE_TIME = 17.86

# Define the system.
# ISO indicates the object is defined in the complex to real isomorphism.
# NEGI is the negative complex unit.
const NEGI = SA_F64[0   0  1  0 ;
                    0   0  0  1 ;
                    -1  0  0  0 ;
                    0  -1  0  0 ;]
# SIGMAX, SIGMAZ are the X and Z pauli matrices
const SIGMAX_ISO = SA_F64[0   1   0   0;
                          1   0   0   0;
                          0   0   0   1;
                          0   0   1   0]
const SIGMAZ_ISO = SA_F64[1   0   0   0;
                          0  -1   0   0;
                          0   0   1   0;
                          0   0   0  -1]
const NEGI_H0_ISO = pi * NEGI * SIGMAZ_ISO
const NEGI_H1_ISO = pi * NEGI * SIGMAX_ISO
const FQ_NEGI_H0_ISO = FQ * NEGI_H0_ISO
const SP1FQ_NEGI_H0_ISO = SP1FQ * NEGI_H0_ISO
const SN1FQ_NEGI_H0_ISO = SN1FQ * NEGI_H0_ISO
const SP2FQ_NEGI_H0_ISO = SP2FQ * NEGI_H0_ISO
const SN2FQ_NEGI_H0_ISO = SN2FQ * NEGI_H0_ISO
const SP3FQ_NEGI_H0_ISO = SP3FQ * NEGI_H0_ISO
const SN3FQ_NEGI_H0_ISO = SN3FQ * NEGI_H0_ISO
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
const ZPIBY2_ISO = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}(get_mat_iso(ZPIBY2))
const ZPIBY2_ISO_1 = SVector{STATE_SIZE_ISO}(get_vec_iso(ZPIBY2[:,1]))
const ZPIBY2_ISO_2 = SVector{STATE_SIZE_ISO}(get_vec_iso(ZPIBY2[:,2]))
const YPIBY2 = [1 -1;
                1  1] / sqrt(2)
const YPIBY2_ISO = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}(get_mat_iso(YPIBY2))
const YPIBY2_ISO_1 = SVector{STATE_SIZE_ISO}(get_vec_iso(YPIBY2[:,1]))
const YPIBY2_ISO_2 = SVector{STATE_SIZE_ISO}(get_vec_iso(YPIBY2[:,2]))
const XPIBY2 = [1 -1im;
                -1im 1] / sqrt(2)
const XPIBY2_ISO = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}(get_mat_iso(XPIBY2))
const XPIBY2_ISO_1 = SVector{STATE_SIZE_ISO}(get_vec_iso(XPIBY2[:,1]))
const XPIBY2_ISO_2 = SVector{STATE_SIZE_ISO}(get_vec_iso(XPIBY2[:,2]))
const XPI = [0 -1im;
             -1im 0]
XPI_ISO = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}(get_mat_iso(XPI))

const GT_GATE = Dict(
    xpiby2 => XPIBY2_ISO,
    ypiby2 => YPIBY2_ISO,
    zpiby2 => ZPIBY2_ISO,
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
amp_dfq - Compute the derivative of the qubit frequency
with respect to the flux by flux quantum from the flux drive
amplitude.
"""
@inline amp_dfq(amplitude) = FBFQ_DFQ_SPLINE_DIERCKX(amp_fbfq(amplitude))


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
    knot_point = (Int(floor(time * dt_inv)) % control_knot_count) + 1
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
    gammaf = GAMMAF_PREFACTOR * abs(amp_dfq(controls[knot_point][1]))
    return (
        negi_h * state - state * negi_h
        + (GAMMAC + 2 * gammaf^2 * t) * NEG_DOP_ISO .* state
    )
end


const TTOT_ZPIBY2 = 17.857142857142858
const GAMMA_ZPIBY2 = amp_t1_spline(0)^(-1)
@inline dynamics_zpiby2nodis_deqjl(
    density, (_, _, _, negi_h0, _), _,
) = (
    negi_h0 * density - density * negi_h0
)


@inline dynamics_zpiby2t1_deqjl(
    density, (_, _, _, _, _), _,
) = (
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
function dynamics_ypiby2nodis_deqjl(density, (controls, control_knot_count, dt_inv, negi_h0, namp_dist), t)
    t = t - Int(floor(t / TTOT_YPIBY2)) * TTOT_YPIBY2
    if t <= T1_YPIBY2
        negi_h = negi_h0 + H11_YPIBY2
    elseif t <= T2_YPIBY2
        negi_h = negi_h0
    else
        negi_h = negi_h0 + H13_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
    )
end


function dynamics_ypiby2t1_deqjl(density, (controls, control_knot_count, dt_inv, negi_h0, namp_dist), t)
    t = t - Int(floor(t / TTOT_YPIBY2)) * TTOT_YPIBY2
    if t <= T1_YPIBY2
        negi_h = negi_h0 + H11_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T2_YPIBY2
        negi_h = negi_h0
        gamma1 = GAMMA12_YPIBY2
    else
        negi_h = negi_h0 + H13_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
        + gamma1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2
                    + E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


const TTOT_ZPIBY2 = 17.857142857142858
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
        negi_h = params.negi_h0 + H11_YPIBY2
    elseif time <= T2_XPIBY2
        negi_h = params.negi_h0
    elseif time <= T3_XPIBY2
        negi_h = params.negi_h0 + H13_YPIBY2
    elseif time <= T4_XPIBY2
        negi_h = params.negi_h0
    elseif time <= T5_XPIBY2
        negi_h = params.negi_h0 + H13_YPIBY2
    elseif time <= T6_XPIBY2
        negi_h = params.negi_h0
    else
        negi_h = params.negi_h0 + H11_YPIBY2
    end
    return(
        negi_h * state
    )
end


"""
t1 dissipation via lindblad
"""
function dynamics_xpiby2t1_deqjl(density, (controls, control_knot_count, dt_inv, negi_h0, namp_dist), t)
    t = t - Int(floor(t / TTOT_XPIBY2)) * TTOT_XPIBY2
    if t <= T1_XPIBY2
        negi_h = negi_h0 + H11_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T2_XPIBY2
        negi_h = negi_h0
        gamma1 = GAMMA12_YPIBY2
    elseif t <= T3_XPIBY2
        negi_h = negi_h0 + H13_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T4_XPIBY2
        negi_h = negi_h0
        gamma1 = GAMMA12_YPIBY2
    elseif t <= T5_XPIBY2
        negi_h = negi_h0 + H13_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T6_XPIBY2
        negi_h = negi_h0
        gamma1 = GAMMA12_YPIBY2
    else
        negi_h = negi_h0 + H11_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
        + gamma1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2
                    + E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


"""
flux noise via amplitude fluctuation
"""
function dynamics_xpiby2da_deqjl(state :: StaticVector, params :: SimParams, time :: Float64)
    controls_time = time - Int(floor(time / TTOT_XPIBY2)) * TTOT_XPIBY2
    noise_knot_point = Int(floor(time * params.noise_dt_inv)) + 1
    delta_a = params.noise_offsets[noise_knot_point]
    if controls_time <= T1_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (AYPIBY2 + delta_a) * NEGI_H1_ISO
    elseif controls_time <= T2_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (delta_a) * NEGI_H1_ISO
    elseif controls_time <= T3_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (-AYPIBY2 + delta_a) * NEGI_H1_ISO
    elseif controls_time <= T4_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (delta_a) * NEGI_H1_ISO
    elseif controls_time <= T5_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (-AYPIBY2 + delta_a) * NEGI_H1_ISO
    elseif controls_time <= T6_XPIBY2
        negi_h = FQ_NEGI_H0_ISO + (delta_a) * NEGI_H1_ISO
    else
        negi_h = FQ_NEGI_H0_ISO + (AYPIBY2 + delta_a) * NEGI_H1_ISO
    end
    return(
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
)


@inline fidelity_vec_iso2(s1, s2) = (
    (s1's2)^2 + (s1[1] * s2[3] + s1[2] * s2[4] - s1[3] * s2[1] - s1[4] * s2[2])^2
)


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
        state = rand(STATE_SIZE_NOISO) + 1im * rand(STATE_SIZE_NOISO)
        state = state / sqrt(Real(state'state))
    end
    return SVector{STATE_SIZE_ISO}(
        [real(state); imag(state)]
    )
end


function gen_rand_density_iso(;seed=0)
    if seed == 0
        state = [1, 1] / sqrt(2)
    else
        Random.seed!(seed)
        state = rand(STATE_SIZE_NOISO) + 1im * rand(STATE_SIZE_NOISO)
        state = state / sqrt(Real(state'state))
    end
    density = state * state'
    density_r = real(density)
    density_i = imag(density)
    density_iso = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}([
        density_r -density_i;
        density_i density_r;
    ])
    return density_iso
end


"""
Generate the modulus of the noise with the rough
spectral density Sxx(f) = |x̂(f)|^2 = 1 / |f|
"""
function pink_noise_from_white(count, dt_inv, ndist; seed=0, plot=false)
    Random.seed!(seed)
    freqs = fftfreq(count, dt_inv)
    pink_noise_ = Array{Complex{Float64}, 1}(rand(ndist, count))
    # transform white noise to frequency domain
    fft!(pink_noise_)
    # square root of the spectral density is the
    # fourier transform of the noise
    # normalize by count
    for i in 2:length(pink_noise_)
        pink_noise_[i] = pink_noise_[i] / (sqrt(abs(freqs[i])) * count)
    end
    # normalize to dt_inv, this is the fft value at f=0
    pink_noise_[1] = dt_inv / count
    # transform to time domain
    ifft!(pink_noise_)
    # take modulus, normalize to count
    for i = 1:length(pink_noise_)
        pink_noise_[i] = abs(pink_noise_[i]) * count
    end
    pink_noise_ = Array{Float64, 1}(pink_noise_)

    if plot
        plot_file_path = generate_file_path("png", "figures", joinpath(SPIN_OUT_PATH, "figures"))
        taxis = Array(0:1:size(pink_noise_)[1] - 1) / dt_inv
        fig = Plots.plot(taxis, pink_noise_, dpi=DPI)
        Plots.savefig(fig, plot_file_path)
        println("Plotted noise to $(plot_file_path)")
    end

    return pink_noise_
end


"""
Generate the modulus of the noise with the exact
spectral density Sxx(f) = |x̂(f)|^2 = 1 / |f|
"""
function pink_noise_from_spectrum(count, dt_inv)
    time = count / dt_inv
    pink_noise_ = Array{Complex{Float64}, 1}(fftfreq(count, dt_inv))
    # square root of the spectral density is the
    # fourier transform of the noise
    for i in 2:length(pink_noise_)
        pink_noise_[i] = 1 / sqrt(abs(pink_noise_[i]))
    end
    # normalize to dt_inv, this is the fft value at f=0
    pink_noise_[1] = dt_inv
    # transform to time domain
    ifft!(pink_noise_)
    # take modulus, normalize by count
    for i = 1:length(pink_noise_)
        pink_noise_[i] = abs(pink_noise_[i]) / count
    end
    pink_noise_ = Array{Float64, 1}(pink_noise_)

    return pink_noise_
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
    adaptive=DEQJL_ADAPTIVE, dynamics_type=schroed,
    dt=DT_PREF, save=true, seed=0,
    solver=DifferentialEquations.Vern9, print_seq=false, print_final=false,
    negi_h0=FQ_NEGI_H0_ISO, namp=NAMP_PREFACTOR, ndist=STD_NORMAL,
    noise_dt_inv=DT_NOISE_INV, seed_state=true,)
    start_time = Dates.now()
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
    dt_inv = dt^(-1)
    save_times = Array(0:1:gate_count) * gate_time
    evolution_time = gate_time * gate_count
    knot_count = Int(ceil(evolution_time * dt_inv))

    # get noise offsets
    noise_knot_count = Int(ceil(evolution_time * noise_dt_inv)) + 1
    noise_offsets = (namp) * pink_noise_from_white(noise_knot_count, noise_dt_inv, ndist; seed=seed)
    
    # integrate
    dynamics = DT_DYN[dynamics_type]
    state_type = DT_ST[dynamics_type]
    state_seed = seed_state ? seed : 0
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
                   maxiters=DEQJL_MAXITERS, adaptive=adaptive)

    # Compute the fidelities.
    # All of the gates we consider are 4-cyclic up to phase.
    fidelities = zeros(gate_count + 1)
    g1 = GT_GATE[gate_type]
    g2 = g1^2
    g3 = g1^3
    id0 = initial_state
    if state_type == st_state
        states_ = zeros(gate_count + 1, STATE_SIZE_ISO)
        id1 = g1 * id0
        id2 = g2 * id0
        id3 = g3 * id0
    elseif state_type == st_density
        states_ = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
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
            if analytic
                target = XPI_ISO * target
            end
        elseif i_eff % 4 == 2
            target = id2
        elseif i_eff % 4 == 3
            target = id3
            if analytic
                target = XPI_ISO * target
            end
        end
        if state_type == st_state
            states_[i, :] = state = result_deqjl.u[i]
            fidelities[i] = fidelity_vec_iso2(state, target)
        elseif state_type == st_density
            states_[i, :, :] = state = result_deqjl.u[i]
            fidelities[i] = abs(fidelity_mat_iso(state, target))
        end

        if print_seq || (print_final && i == gate_count + 1)
            println("fidelities[$(i)]\n$(fidelities[i])")
            println("state")
            show_nice(state)
            println("")
            println("target")
            show_nice(target)
            println("")
        end
    end
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
        "states" => states_,
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
        result = data_file_path
        println("Saved simulation to $(data_file_path)")
    end
    
    return result
end


"""
1 gate, many h0s
"""
function run_sim_h0sweep_deqjl(
    gate_type, negi_h0s;
    save_file_path=nothing,
    adaptive=DEQJL_ADAPTIVE, dynamics_type=schroed,
    dt=DT_PREF, save=true, seed=0,
    solver=DifferentialEquations.Vern9, print_seq=false,
    seed_state=true)
    start_time = Dates.now()
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
    dt_inv = 1 / dt
    knot_count = Int(ceil(gate_time * dt_inv))
    save_times = [0., gate_time]

    # set up integration
    dynamics = DT_DYN[dynamics_type]
    state_type = DT_ST[dynamics_type]
    state_seed = seed_state ? seed : 0
    if state_type == st_state
        initial_state = gen_rand_state_iso(;seed=state_seed)
    elseif state_type == st_density
        initial_state =  gen_rand_density_iso(;seed=state_seed)
    end
    tspan = (0., gate_time)

    # integrate and compute fidelity
    sample_count = size(negi_h0s)[1]
    fidelities = zeros(sample_count)
    gate = GT_GATE[gate_type]
    if state_type == st_state
        target_state = gate * initial_state
        if analytic
            target_state = XPI_ISO * target_state
        end
    elseif state_type == st_density
        target_state = gate * initial_state * gate'
        if analytic
            target_state = XPI_ISO * target_state * XPI_ISO'
        end
    end
    for i = 1:sample_count
        params = SimParams(controls, control_knot_count, controls_dt_inv, negi_h0s[i],
                           [0.], 0., dt_inv)
        prob = ODEProblem(dynamics, initial_state, tspan, params)
        result = solve(prob, solver(), dt=dt, saveat=save_times,
                       maxiters=DEQJL_MAXITERS, adaptive=adaptive)
        final_state = result.u[end]
        if state_type == st_state
            fidelities[i] = fidelity_vec_iso2(final_state, target_state)
        elseif state_type == st_density
            fidelities[i] = fidelity_mat_iso(final_state, target_state)
        end
        if print_seq
            println("fidelities[$(i)] = $(fidelities[i])")
        end
    end
    end_time = Dates.now()
    run_time = end_time - start_time

    # Generate the result.
    result = Dict(
        "dynamics_type" => Integer(dynamics_type),
        "gate_time" => gate_time,
        "gate_type" => Integer(gate_type),
        "save_file_path" => isnothing(save_file_path) ? "" : save_file_path,
        "seed" => seed,
        "fidelities" => fidelities,
        "run_time" => string(run_time),
        "dt" => dt,
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
                write(data_file, key, result[keys])
            end
        end
        result = data_file_path
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
    noise_dt_inv=DT_NOISE_INV, seed_state=true, save_step=1e-1)
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
    noise_offsets = (namp) * pink_noise_from_white(noise_knot_count, noise_dt_inv, ndist; seed=seed)
    
    # integrate
    dynamics = DT_DYN[dynamics_type]
    state_type = DT_ST[dynamics_type]
    state_seed = seed_state ? seed : 0
    if state_type == st_state
        initial_state = gen_rand_state_iso(;seed=state_seed)
    elseif state_type == st_density
        initial_state =  gen_rand_density_iso(;seed=state_seed)
    end
    tspan = (0., evolution_time)
    params = SimParams(controls, control_knot_count, controls_dt_inv, negi_h0,
                       noise_offsets, noise_dt_inv, dt_inv)
    prob = ODEProblem(dynamics, initial_state, tspan, params)
    result = solve(prob, solver(), dt=dt, saveat=save_times,
                   maxiters=DEQJL_MAXITERS, adaptive=adaptive)

    if state_type == st_state
        states_ = zeros(save_count + 1, STATE_SIZE_ISO)
        for i = 1:save_count
            states_[i, :] = Array(result.u[i])
        end
    elseif state_type == st_density
        states_ = zeros(save_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
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
