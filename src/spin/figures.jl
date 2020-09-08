"""
figures.jl
"""

using LaTeXStrings
using Printf
import Plots
using Statistics

WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

# Configure paths.
const EXPERIMENT_NAME = "figures"
const SAVE_PATH = joinpath(SPIN_OUT_PATH, EXPERIMENT_NAME)
const F3C_DATA_FILE_PATH = joinpath(SAVE_PATH, "f3c.h5")
const F3D_DATA_FILE_PATH = joinpath(SAVE_PATH, "f3d.h5")

# Configure plotting.
ENV["GKSwstype"] = "nul"
Plots.gr()

# types
@enum PulseType begin
    analytic = 1
    qoc = 2
    s2 = 3
    s4 = 4
    d2 = 5
    d3 = 6
    s2b = 7
    s4b = 8
    d2b = 9
    d3b = 10
end

const GT_LIST = [zpiby2, ypiby2, xpiby2]

# common dict keys
const SAVEFP_KEY = 1
const SAVET_KEY = 2
const DATAFP_KEY = 3
const COLOR_KEY = 4
const ACORDS_KEY = 5
const MARKER_KEY = 6
const LCORDS_KEY = 7
const DATA2FP_KEY = 8
const INVAL = 99999

### FIGURE 1 ###

const F1_GATE_COUNT = Integer(1.6e4)
const F1_DATA = Dict(
    zpiby2 => Dict(
        qoc => Dict(
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00209_spin15.h5"),
        ),
        analytic => Dict(
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00000_spin14.h5"),
        ),
    ),
    ypiby2 => Dict(
        qoc => Dict(
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00205_spin15.h5"),
        ),
        analytic => Dict(
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00003_spin14.h5"),
        )
    ),
    xpiby2 => Dict(
        qoc => Dict(
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00174_spin15.h5"),
        ),
        analytic => Dict(
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
        )
    ),
)

const F1A_DATA_FILE_PATH = joinpath(SAVE_PATH, "f1a.h5")
const F1A_PT_LIST = [analytic, qoc]
function gen_1a()
    gate_types = [Integer(gate_type) for gate_type in GT_LIST]
    pulse_types = [Integer(pulse_type) for pulse_type in F1A_PT_LIST]
    save_file_paths = Array{String, 1}([])
    for (i, gate_type) in enumerate(GT_LIST)
        for (j, pulse_type) in enumerate(F1A_PT_LIST)
            data = F1_DATA[gate_type][pulse_type]
            push!(save_file_paths, data[SAVEFP_KEY])
        end
    end

    h5open(F1A_DATA_FILE_PATH, "w") do data_file
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "gate_types", gate_types)
        write(data_file, "pulse_types", pulse_types)
    end
        
end


const F1B_DATA_FILE_PATH = joinpath(SAVE_PATH, "f1b.h5")
function gen_1b()
    max_amp = MAX_CONTROL_NORM_0
    amps_fit = Array(range(0, stop=max_amp, length=F1C_S2_LEN))
    t1s_fit =  map(amp_t1_spline_cubic, amps_fit)
    amps_data = -1 .* map(fbfq_amp, FBFQ_ARRAY)
    t1s_data = T1_ARRAY
    t1s_data_err = T1_ARRAY_ERR
    fig = Plots.plot(dpi=DPI_FINAL, legend=:bottomright, yscale=:log10,
                     tickfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
                     legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND)
    Plots.plot!(amps_fit, t1s_fit, label=nothing, color=:mediumaquamarine)
    Plots.scatter!(amps_data, t1s_data, yerror=t1s_data_err, label=nothing, marker=(:circle, MS_DATA),
                   color=:mediumorchid)
    if save
        h5open(F1B_DATA_FILE_PATH, "w") do save_file
            write(save_file, "amps_fit", amps_fit)
            write(save_file, "t1s_fit", t1s_fit)
            write(save_file, "amps_data", amps_data)
            write(save_file, "t1s_data", t1s_data)
            write(save_file, "t1s_data_err", t1s_data_err)
        end
    end    
end


function gen_1c()
    return
end


### FIGURE 2 ###
F2_DATA = Dict(
    analytic => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
    s2 => joinpath(SPIN_OUT_PATH, "spin12/00496_spin12.h5"),
    s4 => joinpath(SPIN_OUT_PATH, "spin12/00498_spin12.h5"),
    d2 => joinpath(SPIN_OUT_PATH, "spin11/00428_spin11.h5"),
    d3 => joinpath(SPIN_OUT_PATH, "spin11/00429_spin11.h5"),
)

function gen_2a()
    return
end


const F2B_TRIAL_COUNT = Integer(1e2)
const F2B_FQ_DEV = 3e-2
const F2B_PT_LIST = [analytic, s2, s4, d2, d3]
const F2B_AVG_COUNT = 10
function gen_2b()
    @assert iseven(F2B_TRIAL_COUNT)
    gate_type = xpiby2
    pulse_types_integer = [Integer(pt) for pt in F2B_PT_LIST]
    pulse_type_count = size(F2B_PT_LIST)[1]
    trial_count_by2 = Integer(F2B_TRIAL_COUNT / 2)
    lo_idx = 1:trial_count_by2
    hi0_idx = (trial_count_by2 + 1):(F2B_TRIAL_COUNT + 1)
    hi_idx = (trial_count_by2 + 2):(F2B_TRIAL_COUNT + 1)
    fq_devs = Array(range(-F2B_FQ_DEV, stop=F2B_FQ_DEV, length=F2B_TRIAL_COUNT))
    insert!(fq_devs, trial_count_by2 + 1, 0)
    fq_devs_abs = map(abs, fq_devs[hi0_idx])
    fqs = (fq_devs .* FQ) .+ FQ
    negi_h0s = [NEGI_H0_ISO * fq for fq in fqs]
    gate_errorss = ones(pulse_type_count, trial_count_by2 + 1)
    for (i, pulse_type) in enumerate(F2B_PT_LIST)
        println("pt[$(i)]: $(pulse_type)")
        gate_errors = zeros(F2B_TRIAL_COUNT + 1)
        for j = 1:F2B_AVG_COUNT
            println("avg[$(j)]")
            if pulse_type == analytic
                res = run_sim_h0sweep_deqjl(
                    gate_type, negi_h0s; dynamics_type=xpiby2nodis, dt=1e-3, save=false,
                    seed=j
                )
            else
                save_file_path = F2_DATA[pulse_type]
                res = run_sim_h0sweep_deqjl(
                    gate_type, negi_h0s; save_file_path=save_file_path,
                    dynamics_type=schroed, dt=1e-3, save=false, seed=j
                )
            end
            gate_errors = gate_errors .+ (1 .- res["fidelities"])
        end
        gate_errors = [
            gate_errors[trial_count_by2 + 1]; # dfq = 0
            (reverse(gate_errors[lo_idx]) + gate_errors[hi_idx]) ./ (2 * F2B_AVG_COUNT) # everything else
        ]
        gate_errorss[i, :] = gate_errors
    end

    data_file_path = generate_file_path("h5", "f2b", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "fq_devs", fq_devs_abs)
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "gate_errorss", gate_errorss)
    end
    println("Saved f2b data to $(data_file_path)")
end


const F2C_GATE_TIMES = [50., 56.8, 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160.]
const F2C_DATA = Dict(
    analytic => [joinpath(SPIN_OUT_PATH, "spin14/$(lpad(index, 5, '0'))_spin14.h5") for index in [
        INVAL, 4, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL
    ]],
    s2 => [joinpath(SPIN_OUT_PATH, "spin12/$(lpad(index, 5, '0'))_spin12.h5") for index in [
        336, 496, 301, 302, 304, 303, 305, 306, 340, 399, 400, INVAL, INVAL
    ]],
    s4 => [joinpath(SPIN_OUT_PATH, "spin12/$(lpad(index, 5, '0'))_spin12.h5") for index in [
        INVAL, 498, 307, 308, 310, 309, 335, 337, 339, INVAL, INVAL, INVAL, INVAL
    ]],
    d2 => [joinpath(SPIN_OUT_PATH, "spin11/$(lpad(index, 5, '0'))_spin11.h5") for index in [
        105, 428, 98, 100, 99, 229, 231, 230, 232, 291, 289, 290, 292
    ]],
    d3 => [joinpath(SPIN_OUT_PATH, "spin11/$(lpad(index, 5, '0'))_spin11.h5") for index in [
        141, 429, 114, 115, 113, 235, 236, 234, 233, 295, 293, 294, 362
    ]],
)
const F2C_PT_LIST = [analytic, s2, s4, d2, d3]
const F2C_AVG_COUNT = 10
function gen_2c()
    gate_type = xpiby2
    pulse_types = F2C_PT_LIST
    pulse_types_integer = [Integer(pulse_type) for pulse_type in F2C_PT_LIST]
    gate_times = F2C_GATE_TIMES
    ptcount = size(pulse_types)[1]
    gtcount = size(gate_times)[1]
    gate_errors = ones(ptcount, gtcount)
    for (i, pulse_type) in enumerate(pulse_types)
        println("pt[$(i)]: $(pulse_type)")
        for (j, gate_time) in enumerate(gate_times)
            println("gt[$(j)]: $(gate_time)")
            save_file_path = F2C_DATA[pulse_type][j]
            if !isnothing(findfirst("$(INVAL)", save_file_path))
                continue
            else
                # temp
                save_type = pulse_type == analytic ? py : jl
                h5open(save_file_path, "r+") do save_file
                    if !("save_type" in names(save_file))
                        write(save_file, "save_type", Integer(save_type))
                    end
                end                
                fidelity = 0
                for k = 1:F2C_AVG_COUNT
                    res1 = run_sim_deqjl(
                        1, gate_type; save_file_path=save_file_path,
                        dynamics_type=schroed, dt=1e-3, negi_h0=SP1FQ_NEGI_H0_ISO,
                        save=false, seed=k,
                    )
                    fidelity1 = res1["fidelities"][end]
                    res2 = run_sim_deqjl(
                        1, gate_type; save_file_path=save_file_path,
                        dynamics_type=schroed, dt=1e-3, negi_h0=SN1FQ_NEGI_H0_ISO,
                        save=false, seed=k,
                    )
                    fidelity2 = res2["fidelities"][end]
                    fidelity = fidelity + fidelity1 + fidelity2
                end
                fidelity = fidelity / (2 * F2C_AVG_COUNT)
                gate_error = 1 - fidelity
                gate_errors[i, j] = gate_error
            end
        end
    end

    data_file_path = generate_file_path("h5", "f2c", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "gate_errors", gate_errors)
        write(data_file, "gate_times", gate_times)
        write(data_file, "pulse_types", pulse_types_integer)
    end
    print("Saved f2c data to $(data_file_path)")
end


### FIGURE 3 ###
F3_DATA = Dict(
    s2 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin18/00003_spin18.h5"),
        SAVET_KEY => jl,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin18/00053_spin18.h5")
    ),
    s4 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin18/00005_spin18.h5"),
        SAVET_KEY => jl,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin18/00054_spin18.h5")
    ),
    d2 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin17/00003_spin17.h5"),
        SAVET_KEY => jl,
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin17/00051_spin17.h5"),
    ),
    d3 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin17/00005_spin17.h5"),
        SAVET_KEY => jl,
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin17/00052_spin17.h5"),
    ),
    analytic => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
        SAVET_KEY => py,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00101_spin14.h5"),
    ),
)

F3A_GATE_COUNT = Integer(1700)
F3A_DT = 1e-3
function make_figure3a()
    # compute
    gate_type = ypiby2
    dynamics_type = lindbladcfn
    for pulse_type in keys(F3_DATA)
        pulse_data = F3_DATA[pulse_type]
        if isnothing(pulse_data[DATAFP_KEY])
            save_file_path = pulse_data[SAVEFP_KEY]
            save_type = pulse_data[SAVET_KEY]
            data_file_path = run_sim_deqjl(
                F3A_GATE_COUNT, gate_type; save_file_path=save_file_path,
                save_type=save_type, dynamics_type=dynamics_type, dt=F3A_DT
            )
            pulse_data[DATAFP_KEY] = data_file_path
        end
    end
    
    # plot
    colors = []; fidelitiess = []; labels = []
    for pulse_type in keys(F3_DATA)
        (fidelities,) = h5open(F3_DATA[pulse_type][SAVEFP_KEY], "r") do save_file
            fidelities = read(save_file, "fidelities")
            return (fidelities,)
        end
        color = F3_DATA[pulse_type][COLOR_KEY]
        label = "$(PT_STR[pulse_type])"
        push!(colors, color)
        push!(fideltiess, fidelities)
        push!(labels, label)
    end
    plot_file_path = plot_fidelity_by_gate_count(fidelitiess; labels=labels, colors=colors)
    println("Plotted Figure3a to $(plot_file_path)")
end


"""
gen_3cdata - generate an average of the flux noise used in fig3b,
see rbqoc/src/spin/spin.jl/run_sim_deqjl for noise construction
"""
function gen_3cdata()
    noise_dt_inv = DT_NOISE_INV
    noise_dist = STD_NORMAL
    noise_amp = NAMP_PREFACTOR
    evolution_time = 56.8 * 5e3
    noise_count = Int(ceil(evolution_time * noise_dt_inv)) + 1

    # generate noise
    noise = pink_noise_from_white(noise_count, noise_dt_inv, noise_dist; seed=0)
    for seed = 1:9
        noise .+= pink_noise_from_white(noise_count, noise_dt_inv, noise_dist; seed=seed)
    end
    # average
    noise ./= 10

    # take fft
    noise_fft = Array{Complex{Float64}, 1}(noise)
    fft!(noise_fft)
    # normalize and take absolute value
    for i = 1:size(noise_fft)[1]
        noise_fft[i] = abs(noise_fft[i]) / noise_count
    end
    noise_fft = Array{Float64, 1}(noise_fft)

    # heh, linearity
    noise .*= NAMP_PREFACTOR
    noise_fft .*= NAMP_PREFACTOR

    times = Array(0:(noise_count - 1)) / noise_dt_inv
    freqs = Array(fftfreq(noise_count, noise_dt_inv))

    h5open(F3C_DATA_FILE_PATH, "w") do data_file
        write(data_file, "noise", noise)
        write(data_file, "times", times)
        write(data_file, "noise_fft", noise_fft)
        write(data_file, "freqs", freqs)
    end
end


const SIGMAX = [0 1;
                1 0]
const SIGMAY = [0   -1im;
                1im 0]
const SIGMAZ = [1 0;
                0 -1]
const F3D_SAVE_STEP = 1e-1
const F3D_PTS = [analytic, s2, s4, d2, d3]
function gen_3d()
    pulse_types_int = [Integer(pulse_type) for pulse_type in F3D_PTS]
    point_count = Integer(ceil(56.80 * F3D_SAVE_STEP^(-1)))
    pointss = zeros(size(F3D_PTS)[1], point_count, 3)
    for (i, pulse_type) in enumerate(F3D_PTS)
        data = F3_DATA[pulse_type]
        if !(DATAFP_KEY in keys(data))
            if pulse_type == analytic
                data_file_path = run_sim_fine_deqjl(
                    dynamics_type=xpiby2nodis, dt=1e-3,
                    save_step=F3D_SAVE_STEP)
            else
                data_file_path = run_sim_fine_deqjl(
                    save_file_path=data[SAVEFP_KEY],
                    save_type=data[SAVET_KEY],
                    dt=1e-3, save_step=F3D_SAVE_STEP,
                )
            end
        else
            data_file_path = data[DATAFP_KEY]
        end

        (states_) = h5open(data_file_path, "r") do data_file
            states_ = read(data_file, "states")
            return (states_)
        end
        for j = 1:point_count
            state = get_vec_uniso(states_[j, :])
            state_dag = state'
            pointss[i, j, 1] = Real(state_dag * SIGMAX * state)
            pointss[i, j, 2] = Real(state_dag * SIGMAY * state)
            pointss[i, j, 3] = Real(state_dag * SIGMAZ * state)
        end
    end
    h5open(F3D_DATA_FILE_PATH, "w") do data_file
        write(data_file, "pointss", pointss)
        write(data_file, "pulse_types", pulse_types_int)
    end
end


function make_figure1()
    make_figure1a()
    make_figure1b()
    make_figure1c()
end
