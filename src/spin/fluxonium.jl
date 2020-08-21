"""
fluxonium.jl - do some calculations for fluxonium
"""

using Dierckx
using HDF5
using Interpolations
using LaTeXStrings
using LinearAlgebra
using TrajectoryOptimization
using Polynomials
import Plots
using Printf
using StaticArrays
using Zygote

# Construct paths.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "figures"
WDIR = ENV["ROBUST_QOC_PATH"]
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)
DFQ_PLOT_FILE_PATH = joinpath(SAVE_PATH, "dfq.png")
DFQ_DATA_FILE_PATH = joinpath(SAVE_PATH, "dfq.h5")

# Plotting configuration.
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300
MS = 2
ALPHA = 0.2

# Define experimental constants.
# E / h e-9
EC = 0.479
EL = 0.132
EJ = 3.395

## SYSTEM DEFINITION ##
# FLUXONIUM_STATE_COUNT is the state count used in the T1 calculations
FLUXONIUM_STATE_COUNT = 110
FLUXONIUM_LEVELS = Array(range(0., stop=FLUXONIUM_STATE_COUNT - 1,
                               length=FLUXONIUM_STATE_COUNT))
SQRT_FLUXONIUM_LEVELS_TRUNC = map(sqrt, FLUXONIUM_LEVELS[2:FLUXONIUM_STATE_COUNT])
ANNIHILATE = diagm(1 => SQRT_FLUXONIUM_LEVELS_TRUNC)
CREATE = diagm(-1 => SQRT_FLUXONIUM_LEVELS_TRUNC)
E_PLASMA = sqrt(8 * EL * EC)
PHI_OSC = (8 * EC / EL)^(1//4)
PHI_OP = PHI_OSC * 2^(-1//2) * (CREATE + ANNIHILATE)
H_EXP_RAW = exp(1im * PHI_OP)
H_LC = diagm(E_PLASMA * FLUXONIUM_LEVELS)

## DOMEGA FIT ##
FBFQ_MIN = 0.3987965
FBFQ_MAX = 1 - FBFQ_MIN
FBFQ_SAMPLE_COUNT = Integer(1e3)
FBFQ_SAMPLES = Array(range(FBFQ_MIN, stop=FBFQ_MAX, length=FBFQ_SAMPLE_COUNT))
FBFQ_DFQ_POLYDEG = 10

function fbfq_hamiltonian(fbfq)
    reduced_flux = 2 * pi * fbfq
    h_exp = H_EXP_RAW * exp(1im * reduced_flux)
    h_cos = 0.5 * (h_exp + h_exp')
    h_fluxonium = real(H_LC - EJ * h_cos)
    return h_fluxonium
end


function fbfq_fq(fbfq)
    hamiltonian = fbfq_hamiltonian(fbfq)
    eigvals_ = eigvals(Hermitian(hamiltonian))
    fq = (eigvals_[2] - eigvals_[1])
    return fq
end


function fbfq_dfq_helin(fbfq)
    dfbfq = fbfq - 0.5
    delta = 0.014 * 2 * pi / 2
    A = 0.15 * 2 * pi
    B = 0.06
    dfq = (2 * A^2 * dfbfq / B / sqrt(A^2 * dfbfq^2 + B^2 * delta^2))
    return dfq / (2 * pi)
end

function fit_dfq(;plot=false, save=false, pull=false)
    if pull
        (fqs, dfqs) = h5open(DFQ_DATA_FILE_PATH, "r") do data_file
            fqs = read(data_file, "fqs")
            dfqs = read(data_file, "dfqs")
            return(fqs, dfqs)
        end
    else
        fqs = zeros(size(FBFQ_SAMPLES)[1])
        dfqs = zeros(size(FBFQ_SAMPLES)[1])
        dfqhs = zeros(size(FBFQ_SAMPLES)[1])
        for (i, fbfq) in enumerate(FBFQ_SAMPLES)
            fqs[i] = fbfq_fq(fbfq)
            (dfq,) = Zygote.gradient(fbfq_fq, fbfq)
            dfqs[i] = real(dfq)
            dfqhs[i] = fbfq_dfq_helin(fbfq)
        end
    end

    # fbfq_dfq_poly = Polynomials.fit(FBFQ_SAMPLES, dfqs, FBFQ_DFQ_POLYDEG)
    # dfqs_poly = map(fbfq_dfq_poly, FBFQ_SAMPLES)

    fbfq_dfq_dierckx = Dierckx.Spline1D(FBFQ_SAMPLES, dfqs)
    dfqs_dierckx = map(fbfq_dfq_dierckx, FBFQ_SAMPLES)

    if save
        h5open(DFQ_DATA_FILE_PATH, "cw") do data_file
            write(data_file, "fbfqs", FBFQ_SAMPLES)
            write(data_file, "fqs", fqs)
            write(data_file, "dfqs", dfqs)
        end
    end

    if plot
        fig = Plots.plot(dpi=DPI, legend=:bottomright)
        Plots.scatter!(fig, FBFQ_SAMPLES, fqs, label="fq", alpha=ALPHA, markersize=MS)
        Plots.scatter!(fig, FBFQ_SAMPLES, dfqs, label="dfq", alpha=ALPHA, markersize=MS)
        # Plots.plot!(fig, FBFQ_SAMPLES, dfqs_poly, label="dfq poly")
        Plots.plot!(fig, FBFQ_SAMPLES, dfqs_dierckx, label="dfq dierckx")
        Plots.plot!(fig, FBFQ_SAMPLES, dfqhs, label="dfqh")
        Plots.xlabel!(L"$\Phi / \Phi_{0}$")
        Plots.ylabel!("Amp (GHz)")
        Plots.savefig(fig, DFQ_PLOT_FILE_PATH)
    end
end
