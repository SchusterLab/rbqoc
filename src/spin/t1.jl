"""
t1.jl - do some calculations for fluxonium t1
"""

using Dierckx
using HDF5
using Interpolations
using LaTeXStrings
using LinearAlgebra
using TrajectoryOptimization
import Plots
using Printf
using StaticArrays

# Construct paths.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin15"
WDIR = ENV["ROBUST_QOC_PATH"]
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)
T1_SAVE_FILE_PATH = joinpath(SAVE_PATH, "t1.h5")
T1CMP_PLOT_FILE_PATH = joinpath(SAVE_PATH, "t1splinecmp.png")

# Plotting configuration.
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300

# misc. constants
SAMPLE_SIZE = Int(1e3)

# Define experimental constants.
# E / h
EC = 0.479e9
EL = 0.132e9
EJ = 3.395e9
# Q_CAP = 1 / 8e-6
Q_CAP = 1.25e5
T_CAP = 0.042
H = 6.62607015e-34
HBAR = 1.05457148e-34
KB = 1.3806503e-23
# HBAR_BY_KB = HBAR / KB
HBAR_BY_KB = 7.638225390210369e-12
E_CHARGE = 1.60217646e-19
# FLUX_QUANTUM = H / (2 * E_CHARGE)
# INVERSE_FLUX_QUANTUM = 2 * E_CHARGE / H
INVERSE_FLUX_QUANTUM = 4.835977958971655e14
# FLUX_FRUSTRATION = FLUX_QUANTUM / 2
FLUX_FRUSTRATION = 1.033917036516689e-15
# raw T1s reported in units of microseconds
T1_ARRAY = [
    1597.923, 1627.93, 301.86, 269.03, 476.33, 1783.19, 2131.76, 2634.50, 
    4364.68, 2587.82, 1661.915, 1794.468, 2173.88, 1188.83, 
    1576.493, 965.183, 560.251, 310.88
] * 1e3
FBFQ_ARRAY = [
    0.26, 0.28, 0.32, 0.34, 0.36, 0.38, 0.4,
    0.42, 0.44, 0.46, 0.465, 0.47, 0.475,
    0.48, 0.484, 0.488, 0.492, 0.5
]

# Define the system.
# FLUXONIUM_STATE_COUNT is the state count used in the T1 calculations
FLUXONIUM_STATE_COUNT = 110
FLUXONIUM_LEVELS = Array(range(0., stop=FLUXONIUM_STATE_COUNT - 1,
                               length=FLUXONIUM_STATE_COUNT))
SQRT_FLUXONIUM_LEVELS_TRUNC = map(sqrt, FLUXONIUM_LEVELS[2:FLUXONIUM_STATE_COUNT])
ANNIHILATE = diagm(1 => SQRT_FLUXONIUM_LEVELS_TRUNC)
CREATE = diagm(-1 => SQRT_FLUXONIUM_LEVELS_TRUNC)
GAMMA_CAP_PREFACTOR = (16 * pi * EC * Q_CAP)^(-1)
E_PLASMA = (8 * EL * EC)^(0.5)
PHI_OSC = (8 * EC / EL)^(0.25)
PHI_OP = PHI_OSC * 2^(-0.5) * (CREATE + ANNIHILATE)
H_EXP_RAW = exp(1im * PHI_OP)
H_LC = diagm(E_PLASMA * FLUXONIUM_LEVELS)

### ANALYTICAL CALCULATIONS ###
function get_hamiltonian_oscillator_basis(flux :: Float64)
    reduced_flux = 2 * pi * flux * INVERSE_FLUX_QUANTUM
    h_exp = H_EXP_RAW * exp(1im * reduced_flux)
    h_cos = 0.5 * (h_exp + h_exp')
    h_fluxonium = real(H_LC - EJ * h_cos)
    return h_fluxonium
end


function get_t1(flux :: Float64)
    # cap
    h_fluxonium = get_hamiltonian_oscillator_basis(flux)
    evals, evecs = eigen(Hermitian(h_fluxonium))
    g0_phi_g1 = evecs[:, 1]' * PHI_OP * evecs[:, 2]
    omega = (evals[2] - evals[1]) * (2 * pi)
    gamma_cap = (GAMMA_CAP_PREFACTOR * omega^2 * coth(HBAR_BY_KB * omega / (2 * T_CAP))
                 * g0_phi_g1 * conj(g0_phi_g1))

    gamma = gamma_cap
    t1 = gamma^(-1)
    
    return t1
end


### EXPERIMENTAL DATA FIT ###
function compare_splines()
    # fit
    t1_spline_dierckx = Spline1D(FBFQ_ARRAY, T1_ARRAY)
    t1_spline_itp = extrapolate(interpolate((FBFQ_ARRAY,), T1_ARRAY, Gridded(Linear())), Flat())
    # fbfq_axis = range(minimum(FBFQ_ARRAY), stop=maximum(FBFQ_ARRAY), length=SAMPLE_SIZE)
    fbfq_axis = range(0, stop=2 * maximum(FBFQ_ARRAY), length=SAMPLE_SIZE)

    # plot
    fig = Plots.plot(dpi=DPI)
    Plots.plot!(fbfq_axis, map(t1_spline_dierckx, fbfq_axis), label="dierckx")
    Plots.plot!(fbfq_axis, map(t1_spline_itp, fbfq_axis), label="itp")
    Plots.scatter!(FBFQ_ARRAY, T1_ARRAY, label="data")
    Plots.xlabel!(L"\Phi / \Phi_{0}")
    Plots.ylabel!(L"T_{1}")
    Plots.savefig(fig, T1CMP_PLOT_FILE_PATH)
end
