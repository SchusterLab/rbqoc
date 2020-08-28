"""
fluxonium.jl - do some calculations for fluxonium
"""

using Dierckx
using FFTW
using HDF5
using LaTeXStrings
using LinearAlgebra
import Plots
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
const PINK_PLOT_FILE_PATH = joinpath(SAVE_PATH, "pink.png")
const PINKFW_PLOT_FILE_PATH = joinpath(SAVE_PATH, "pinkfw.png")

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


function plot_pink_noise_from_white(count; ndist=FBFQ_NDIST, dt_inv=1e2, plot=false, seed=0,
                                    namp=NAMP_PREFACTOR)
    Random.seed!(seed)
    freqs = fftfreq(count, dt_inv)
    times = (0:count-1) ./ dt_inv
    time = count / dt_inv
    
    white_noise = rand(ndist, count)
    pink_noise_ = Array{Complex{Float64}, 1}(white_noise)
    # transform white noise to frequency domain
    fft!(pink_noise_)
    white_fft = Array{Float64, 1}(map(abs, pink_noise_)) / count
    # square root of the spectral density is the
    # fourier transform of the noise
    # normalize by count
    for i in 2:length(pink_noise_)
        pink_noise_[i] = pink_noise_[i] / (sqrt(abs(freqs[i])) * count)
    end
    # normalize to dt_inv, this is the fft value at f=0
    pink_noise_[1] = dt_inv / count
    pink_fft = Array{Float64, 1}(map(abs, pink_noise_))
    # transform to time domain
    ifft!(pink_noise_)
    # take modulus, normalize by count
    for i = 1:length(pink_noise_)
        pink_noise_[i] = abs(pink_noise_[i]) * count
    end
    pink_noise_ = Array{Float64, 1}(pink_noise_)
    
    # get delta_fq for plotting
    fnoise = pink_noise_ * namp

    if plot
        # inds = 1:length(pink_fft)
        inds = 1:Int(1e4):length(pink_fft)
        subfig1 = Plots.plot()
        Plots.plot!(subfig1, freqs[inds], pink_fft[inds], label="pink")
        Plots.plot!(subfig1, freqs[inds], white_fft[inds], label="white")
        Plots.xlabel!(L"f \; \textrm{(GHz)}")
        Plots.ylabel!(L"|\hat{x}(f)| \; \textrm{(a.u.)}")
        subfig2 = Plots.plot()
        Plots.plot!(subfig2, times[inds], pink_noise_[inds], label="pink")
        Plots.plot!(subfig2, times[inds], white_noise[inds], label="white")
        Plots.xlabel!(L"t \; \textrm{(ns)}")
        Plots.ylabel!(L"x(t) \; \textrm{(a.u.)}")
        subfig3 = Plots.plot()
        Plots.plot!(subfig3, times[inds], fnoise[inds], label=:none)
        Plots.xlabel!(L"t \; \textrm{(ns)}")
        Plots.ylabel!(L"\Delta f_{q} \; \textrm{(GHz)}")
        layout = @layout [a; b; c]
        fig = Plots.plot(subfig1, subfig2, subfig3, dpi=DPI, layout=layout)
        Plots.savefig(fig, PINKFW_PLOT_FILE_PATH)
    end

    return fnoise
end


function plot_pink_noise_from_density(count; dt_inv=1e2, namp=NAMP_PREFACTOR, plot=false, seed=0,
                                      ndist=FBFQ_NDIST)
    Random.seed!(seed)
    times = Array(0:1:count-1) / dt_inv
    time = count / dt_inv
    freqs = fftfreq(count, dt_inv)
    
    white_noise = rand(ndist, count)
    pink_noise_ = Array{Complex{Float64}, 1}(freqs)

    pink_noise_[1] = dt_inv
    for i = 2:length(pink_noise_)
        pink_noise_[i] = 1 / sqrt(abs(pink_noise_[i]))
    end
    pink_fft = Array{Float64, 1}(pink_noise_)
    
    ifft!(pink_noise_)
    for i = 1:length(pink_noise_)
        pink_noise_[i] = abs(pink_noise_[i]) / count
    end
    pink_noise_ = Array{Float64, 1}(pink_noise_)


    fnoise = pink_noise * namp
    
    if plot
        subfig1 = Plots.plot()
        Plots.plot!(subfig1, freqs, pink_fft, label="pink")
        Plots.xlabel!(L"f \; \textrm{(GHz)}")
        Plots.ylabel!(L"|\hat{x}(f)| \; \textrm{(a.u.)}")
        subfig2 = Plots.plot()
        Plots.plot!(subfig2, times, pink_noise, label="pink")
        Plots.xlabel!(L"t \; \textrm{(ns)}")
        Plots.ylabel!(L"x(t) \; \textrm{(a.u.)}")
        subfig3 = Plots.plot()
        Plots.plot!(subfig3, times, fnoise, label=:none)
        Plots.xlabel!(L"t \; \textrm{(ns)}")
        Plots.ylabel!(L"\Delta f_{q} \; \textrm{(GHz)}")
        layout = @layout [a; b; c]
        fig = Plots.plot(subfig1, subfig2, subfig3, dpi=DPI, layout=layout)
        Plots.savefig(fig, PINK_PLOT_FILE_PATH)
    end

    return fnoise
end
    
