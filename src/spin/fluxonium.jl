"""
fluxonium.jl - do some calculations for fluxonium
"""

using Dierckx
using Distributions
using FFTW
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

WDIR = get(ENV, "RBQOC_PATH", "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

# Construct paths.
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "figures"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)
const DFQ_PLOT_FILE_PATH = joinpath(SAVE_PATH, "dfq.png")
const DFQ_DATA_FILE_PATH = joinpath(SAVE_PATH, "dfq.h5")
const DELTAFQ_PLOT_FILE_PATH = joinpath(SAVE_PATH, "deltafq.png")

## SYSTEM DEFINITION ##
# FLUXONIUM_STATE_COUNT is the state count used in the T1 calculations
const FLUXONIUM_STATE_COUNT = 110
const FLUXONIUM_LEVELS = Array(range(0., stop=FLUXONIUM_STATE_COUNT - 1,
                               length=FLUXONIUM_STATE_COUNT))
const SQRT_FLUXONIUM_LEVELS_TRUNC = map(sqrt, FLUXONIUM_LEVELS[2:FLUXONIUM_STATE_COUNT])
const ANNIHILATE = diagm(1 => SQRT_FLUXONIUM_LEVELS_TRUNC)
const CREATE = diagm(-1 => SQRT_FLUXONIUM_LEVELS_TRUNC)
const PHI_OP = PHI_OSC * 2^(-1//2) * (CREATE + ANNIHILATE)
const H_EXP_RAW = exp(1im * PHI_OP)
const H_LC = diagm(E_PLASMA * FLUXONIUM_LEVELS)

## DOMEGA FIT ##
const FBFQ_MIN = 0.3987965
const FBFQ_MAX = 1 - FBFQ_MIN
const FBFQ_SAMPLE_COUNT = Integer(1e3)
const FBFQ_SAMPLES = Array(range(FBFQ_MIN, stop=FBFQ_MAX, length=FBFQ_SAMPLE_COUNT))
const FBFQ_DFQ_POLYDEG = 10

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


function generate_flux_noise(;plot=false, dtinv=1e2, dt=1e-2)
    h_ff = fbfq_hamiltonian(0.5)
    evecs = eigvecs(Hermitian(h_ff))
    gphie_ff = evecs[:,1]' * PHI_OP * evecs[:, 2]
    dfq_dfbfq = 4 * pi * gphie_ff * EL
    Random.seed!(0)
    delta_fqs = FBFQ_NAMP * dfq_dfbfq * rand(FBFQ_NDIST, FBFQ_SAMPLE_COUNT)
    # delta_fqs = FBFQ_NAMP * dfq_dfbfq * rand(Uniform(0, 1), FBFQ_SAMPLE_COUNT)
    delta_fqs_fft = fft(delta_fqs)
    freqs = fftfreq(FBFQ_SAMPLE_COUNT, dtinv)
    delta_fqs_fft_pink = delta_fqs_fft[2:end] ./ freqs[2:end]
    delta_fqs_pink = ifft(delta_fqs_fft_pink)
    ts = range(0, stop=FBFQ_SAMPLE_COUNT - 1, length=FBFQ_SAMPLE_COUNT) * dt
    if plot
        subfig1 = Plots.plot()
        Plots.plot!(subfig1, freqs, map(abs, delta_fqs_fft), label="white")
        Plots.plot!(subfig1, freqs[2:end], map(abs, delta_fqs_fft_pink), label="pink")
        Plots.ylabel!(L"\textrm{(a.u.)}")
        Plots.xlabel!(L"f \; \textrm{(GHz)}")
        
        subfig2 = Plots.plot()
        Plots.plot!(subfig2, ts, delta_fqs, label="white")
        Plots.plot!(subfig2, ts[2:end], map(abs, delta_fqs_pink), label="pink")
        Plots.ylabel!(L"\Delta f_{q} \; \textrm{(GHz)}")
        Plots.xlabel!(L"t \; \textrm{(ns)}")

        layout = @layout [a; b]
        fig = Plots.plot(subfig1, subfig2, dpi=DPI, layout=layout)
        Plots.savefig(fig, DELTAFQ_PLOT_FILE_PATH)
        println("plotted to $(DELTAFQ_PLOT_FILE_PATH)")
    end
    return freqs
end
