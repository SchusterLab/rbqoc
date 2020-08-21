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
META_SAVE_PATH = joinpath(ENV["RBQOC_PATH"], "out", "spin")
EXPERIMENT_NAME = "figures"
SAVE_PATH = joinpath(META_SAVE_PATH, EXPERIMENT_NAME)

# Configure plotting.
ENV["GKSwstype"] = "nul"
Plots.gr()

# types
@enum PulseType begin
    qoc = 1
    analytic = 2
    derivative = 3
    sample = 4
    derivative2 = 5
    sample2 = 6
end

const PT_STR = Dict(
    qoc => "QOC",
    analytic => "Anl.",
    sample => "S-2",
    derivative => "D-2",
    derivative2 => "D-3",
    sample2 => "S-4"
)

const PT_MARKER = Dict(
    sample => :circle,
    sample2 => :square,
    derivative2 => :utriangle,
    derivative => :diamond,
)

const PT_COLOR = Dict(
    analytic => :lightskyblue,
    qoc => :coral,
    sample => :limegreen,
    sample2 => :darkgreen,
    derivative => :crimson,
    derivative2 => :firebrick,
)

const PT_LINESTYLE = Dict(
    analytic => :solid,
    qoc => :solid,
    sample => :solid,
    sample2 => :dash,
    derivative => :solid,
    derivative2 => :dash,
)

const GT_LIST = [zpiby2, ypiby2, xpiby2]

# plotting constants
const ALPHA_POINT = 0.4
const MS_DATA = 4
const MS_POINT = 8
const FS_AXIS_LABELS = 12
const FS_AXIS_TICKS = 10
const FS_ANNOTATE = 10
const FS_LEGEND = 10
const FG_COLOR_LEGEND = nothing
const DPI_FINAL = Integer(1e3)

# common dict keys
const SAVE_FILE_PATH_KEY = 1
const SAVE_TYPE_KEY = 2
const DATA_FILE_PATH_KEY = 3
const COLOR_KEY = 4
const ACORDS_KEY = 5
const MARKER_KEY = 6
const LCORDS_KEY = 7
const DATA2_FILE_PATH_KEY = 8


### ALL ###
function plot_fidelity_by_gate_count(fidelitiess; inds=nothing, title="", ylims=(0, 1),
                                     yticks=(0:0.1:1), legend=nothing, yscale=:none,
                                     labels=nothing, colors=nothing, linestyles=nothing,
                                     xlims=nothing)
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    fig = Plots.plot(dpi=DPI_FINAL, ylims=ylims, yticks=yticks, title=title,
                     legend=legend, yscale=yscale, xlims=xlims,
                     tickfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
                     legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND)
    gate_count = size(fidelitiess[1])[1] - 1
    gate_count_axis = Array(0:1:gate_count)
    if isnothing(inds)
        inds = 1:gate_count + 1
    end
    for (i, fidelities) in enumerate(fidelitiess)
        color = isnothing(colors) ? :auto : colors[i]
        label = isnothing(labels) ? nothing : labels[i]
        linestyle = isnothing(linestyles) ? :auto : linestyles[i]
        Plots.plot!(fig, gate_count_axis[inds], 1 .- fidelities[inds], label=label,
                    color=color, linestyle=linestyle)
    end
    Plots.ylabel!("Gate Error")
    Plots.xlabel!("Gate Count")
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end


### FIGURE 1 ###
const F1_GATE_COUNT = Integer(1.5e4)
F1_PULSE_DATA = Dict(
    zpiby2 => Dict(
        qoc => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00196_spin15.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00194_spin15.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        analytic => Dict(
            ACORDS_KEY => (0, 0.25),
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00013_spin14.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00000_spin14.h5"),
            SAVE_TYPE_KEY => py,
        ),
    ),
    ypiby2 => Dict(
        qoc => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00188_spin15.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00185_spin15.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        analytic => Dict(
            ACORDS_KEY => (0, 0.4),
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00018_spin14.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00003_spin14.h5"),
            SAVE_TYPE_KEY => py,
        )
    ),
    xpiby2 => Dict(
        qoc => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00176_spin15.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00174_spin15.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        analytic => Dict(
            ACORDS_KEY => (0, 0.5),
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00022_spin14.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00004_spin14.h5"),
            SAVE_TYPE_KEY => py,
        )
    ),
)

const F1A_PT_LIST = [analytic, qoc]
function make_figure1a()
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    save_file_paths = []; save_types = []; labels = []; colors = [];
    subfigs = []
    for (i, gate_type) in enumerate(instances(GateType))
        subfig = Plots.plot()
        if i == 2
            Plots.ylabel!(subfig, latexstring("\$a \\ \\textrm{(GHz)}\$"))
        elseif i == 3
            Plots.xlabel!(subfig, latexstring("\$t \\ \\textrm{(ns)}\$"))
        end
        text_ = GT_STR[gate_type]
        (ax, ay) = F1_PULSE_DATA[gate_type][analytic][ACORDS_KEY]
        Plots.annotate!(subfig, ax, ay, text(text_, FS_ANNOTATE))
        for pulse_type in F1A_PT_LIST
            if pulse_type == analytic
                linestyle = :solid
            elseif pulse_type == qoc
                linestyle = :solid
            end
            data = F1_PULSE_DATA[gate_type][pulse_type]
            color = PT_COLOR[pulse_type]
            label = "$(GT_STR[gate_type]) $(PT_STR[pulse_type])"
            save_file_path = data[SAVE_FILE_PATH_KEY]
            save_type = data[SAVE_TYPE_KEY]
            (controls, evolution_time) = grab_controls(save_file_path; save_type=save_type)
            (control_eval_count, control_count) = size(controls)
            control_eval_times = Array(1:1:control_eval_count) * DT_PREF
            Plots.plot!(subfig, control_eval_times, controls[:,1], color=color, label=nothing,
                        linestyle=linestyle)
        end
        push!(subfigs, subfig)
    end
    layout = @layout [a; b; c]
    fig = Plots.plot(subfigs[1], subfigs[2], subfigs[3], layout=layout, dpi=DPI_FINAL,
                     ticksfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
                     legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND)
    Plots.savefig(fig, plot_file_path)
    println("Saved Figure1a to $(plot_file_path)")
end


const GT_LS_1B = Dict(
    zpiby2 => :solid,
    ypiby2 => :dash,
    xpiby2 => :dashdot,
)

function make_figure1b()
    # TODO: get data
    for gate_type in keys(F1_PULSE_DATA)
        for pulse_type in keys(F1_PULSE_DATA[gate_type])
            pulse_data = F1_PULSE_DATA[gate_type][pulse_type]
            if !(DATA_FILE_PATH_KEY in keys(pulse_data))
                # get data_file_path and write it to pulse data here
                data_file_path = nothing
            end
        end
    end
    
    # plot
    fidelitiess = []; labels = []; colors = []; linestyles = []
    for gate_type in GT_LIST
        for pulse_type in keys(F1_PULSE_DATA[gate_type])
            pulse_data = F1_PULSE_DATA[gate_type][pulse_type]
            (fidelities,) = h5open(pulse_data[DATA_FILE_PATH_KEY], "r") do data_file
                fidelities = read(data_file, "fidelities")
                return (fidelities,)
            end
            color = PT_COLOR[pulse_type]
            label = "$(GT_STR[gate_type]) $(PT_STR[pulse_type])"
            linestyle = GT_LS_1B[gate_type]
            push!(fidelitiess, fidelities)
            push!(labels, label)
            push!(colors, color)
            push!(linestyles, linestyle)
        end
    end
    plot_file_path = plot_fidelity_by_gate_count(
        fidelitiess; ylims=(0, 0.05), yticks=0:0.01:0.05, legend=:topleft,
        labels=labels, colors=colors, linestyles=linestyles,
        xlims=(0, 1700)
    )
    println("Plotted Figure1b to $(plot_file_path)")
end


const F1C_SAMPLE_LEN = Integer(1e4)
const GT_MK_1C = Dict(
    zpiby2 => :diamond,
    ypiby2 => :square,
    xpiby2 => :utriangle,
)
const MS_F1C = 6
const ALPHA_F1C = 1.

function make_figure1c()
    # Collect data and plot.
    max_amp = MAX_CONTROL_NORM_0
    amps_fit = Array(range(0, stop=max_amp, length=F1C_SAMPLE_LEN))
    t1s_fit =  map(amp_t1_spline_cubic, amps_fit)
    amps_data = -1 .* map(fbfq_amp_lo, FBFQ_ARRAY)
    t1s_data = T1_ARRAY
    t1s_data_err = T1_ARRAY_ERR
    fig = Plots.plot(dpi=DPI_FINAL, legend=:bottomright, yscale=:log10,
                     tickfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
                     legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND)
    Plots.plot!(amps_fit, t1s_fit, label=nothing, color=:mediumaquamarine)
    Plots.scatter!(amps_data, t1s_data, yerror=t1s_data_err, label=nothing, marker=(:circle, MS_DATA),
                   color=:mediumorchid)
    for gate_type in GT_LIST
        for pulse_type in keys(F1_PULSE_DATA[gate_type])
            pulse_data = F1_PULSE_DATA[gate_type][pulse_type]
            (controls, _) = grab_controls(
                pulse_data[SAVE_FILE_PATH_KEY];
                save_type=pulse_data[SAVE_TYPE_KEY]
            )
            avg_amp = mean(map(abs, controls[:,1]))
            avg_t1 = amp_t1_spline_cubic(avg_amp)
            avg_label = "$(GT_STR[gate_type]) $(PT_STR[pulse_type])"
            avg_color = PT_COLOR[pulse_type]
            marker = GT_MK_1C[gate_type]
            Plots.plot!([avg_amp], [avg_t1], label=avg_label,
                        marker=(marker, MS_F1C), color=avg_color, alpha=ALPHA_F1C)
        end
    end
    Plots.xlabel!(latexstring("\$ {<a>}_{t} \\textrm{(GHz)} \$"))
    Plots.ylabel!(latexstring("\$T_1 \\ \\textrm{(ns)}\$"))
    Plots.xlims!((-0.02, max_amp))
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    println("Plotted Figure1c to $(plot_file_path)")
end


### FIGURE 2 ###
F2_PULSE_DATA = Dict(
    derivative => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00091_spin11.h5"),
        SAVE_TYPE_KEY => jl,
        DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00092_spin11.h5"),
        LCORDS_KEY => (45, 0.2),
    ),
    derivative2 => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00110_spin11.h5"),
        SAVE_TYPE_KEY => jl,
        DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00111_spin11.h5"),
        LCORDS_KEY => (45, 0.2),
    ),
    sample => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00132_spin12.h5"),
        SAVE_TYPE_KEY => jl,
        DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00200_spin12.h5"),
        LCORDS_KEY => (30, 0.2),
    ),
    analytic => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00004_spin14.h5"),
        SAVE_TYPE_KEY => py,
        DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00028_spin14.h5"),
        LCORDS_KEY => (20, 0.05),
    ),
    sample2 => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00229_spin12.h5"),
        SAVE_TYPE_KEY => jl,
        DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00231_spin12.h5"),
        LCORDS_KEY => (30, 0.2),
    ),
)

F2A_PT_LIST = [[analytic], [sample, sample2,], [derivative, derivative2]]
"""
Show the pulses.
"""
function make_figure2a()
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    save_file_paths = []; save_types = []; labels = []; colors = [];
    subfigs = []
    for (i, pulse_types) in enumerate(F2A_PT_LIST)
        subfig = Plots.plot()
        if i == 2
            Plots.ylabel!(subfig, latexstring("\$a \\ \\textrm{(GHz)}\$"))
        elseif i == 3
            Plots.xlabel!(subfig, latexstring("\$ t \\ \\textrm{(ns)}\$"))
        end
        for (j, pulse_type) in enumerate(pulse_types)
            data = F2_PULSE_DATA[pulse_type]
            color = PT_COLOR[pulse_type]
            label = "$(PT_STR[pulse_type])"
            linestyle = PT_LINESTYLE[pulse_type]
            save_file_path = data[SAVE_FILE_PATH_KEY]
            save_type = data[SAVE_TYPE_KEY]
            (controls, evolution_time) = grab_controls(save_file_path; save_type=save_type)
            (control_eval_count, control_count) = size(controls)
            control_eval_times = Array(1:1:control_eval_count) * DT_PREF
            Plots.plot!(subfig, control_eval_times, controls[:,1], color=color, label=label,
                        linestyle=linestyle, legend=:none)
        end
        push!(subfigs, subfig)
    end
    layout = @layout [a; b; c]
    fig = Plots.plot(
        subfigs[1], subfigs[2], subfigs[3], layout=layout, dpi=DPI_FINAL,
        ticksfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
        legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND,
        legend=:outerbottomright
    )
    Plots.savefig(fig, plot_file_path)
    println("Saved Figure1a to $(plot_file_path)")
end


const F2B_TRIAL_COUNT = Integer(1e3)
const F2B_FQ_DEV = 1e-1
# TODO: add derivative2
const PT_LIST_F2B = [analytic, sample, sample2, derivative, derivative2]
"""
Show gate error vs. detuning
"""
function make_figure2b()
    # sample detunings
    fq_devs = Array(range(-F2B_FQ_DEV, stop=F2B_FQ_DEV, length=F2B_TRIAL_COUNT))
    fqs = (fq_devs .* FQ) .+ FQ
    negi_h0s = [NEGI_H0_ISO * fq for fq in fqs]

    # sweep
    gate_type = xpiby2
    for pulse_type in keys(F2_PULSE_DATA)
        if !(DATA_FILE_PATH_KEY in keys(F2_PULSE_DATA[pulse_type]))
            save_file_path = F2_PULSE_DATA[pulse_type][SAVE_FILE_PATH_KEY]
            save_type = F2_PULSE_DATA[pulse_type][SAVE_TYPE_KEY]
            data_file_path = run_sim_h0sweep_deqjl(gate_type, negi_h0s; save_file_path=save_file_path,
                                                   save_type=save_type)
            h5open(data_file_path, "r+") do data_file
                write(data_file, "fqs", fqs)
                write(data_file, "pulse_type", Integer(pulse_type))
            end
            F2_PULSE_DATA[pulse_type][DATA_FILE_PATH_KEY] = data_file_path
        end
    end

    # plot
    fig = Plots.plot(dpi=DPI_FINAL, yticks=[0, 0.0025, 0.005, 0.0075], ylim=(0., 0.0075),
                     xlim=(minimum(fq_devs), maximum(fq_devs)), xticks=[-0.1, -0.05, 0, 0.05, 0.1],
                     legend=:none,
                     tickfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
                     legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND)
    for pulse_type in PT_LIST_F2B
        pulse_data = F2_PULSE_DATA[pulse_type]
        label = "$(PT_STR[pulse_type])"
        linestyle = PT_LINESTYLE[pulse_type]
        color = PT_COLOR[pulse_type]
        data_file_path = pulse_data[DATA_FILE_PATH_KEY]
        (fidelities,) = h5open(data_file_path, "r") do data_file
            fidelities = read(data_file, "fidelities")
            return (fidelities,)
        end
        Plots.plot!(fig, fq_devs, 1 .- fidelities, label=label, color=color,
                    linestyle=linestyle)
                    
    end
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    Plots.xlabel!(L"$\delta \omega_{q} / \omega_{q}$")
    Plots.ylabel!("Gate Error")
    Plots.savefig(fig, plot_file_path)
    println("Plotted Figure2b to $(plot_file_path)")
end

F2C_GATE_TIMES = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
F2C_PULSE_DATA = Dict(
    sample => Dict(
        F2C_GATE_TIMES[1] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00246_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00247_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00198_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[2] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00248_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00249_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00138_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[3] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00250_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00251_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00145_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[4] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00252_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00253_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00147_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[5] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00254_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00255_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00148_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[6] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00256_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00257_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00150_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[7] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00258_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00259_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00151_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[8] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00260_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00261_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00153_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[9] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00262_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00263_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00197_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[10] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00264_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00265_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00201_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[11] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00266_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00267_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00202_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[12] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00268_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00269_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00203_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
    ),
    sample2 => Dict(
        F2C_GATE_TIMES[1] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00270_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00271_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00236_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[2] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00272_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00273_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00233_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[3] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00274_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00275_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00234_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[4] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00276_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00277_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00235_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[5] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00278_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00279_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00239_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[6] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00280_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00281_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00241_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[7] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00282_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00283_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00240_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[8] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00284_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00285_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00238_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[9] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00286_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00287_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00243_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[10] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00288_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00289_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00242_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[11] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00290_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00291_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00245_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[12] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00292_spin12.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00293_spin12.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00244_spin12.h5"),
            SAVE_TYPE_KEY => jl,
        ),
    ),
    derivative => Dict(
        F2C_GATE_TIMES[1] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00117_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00118_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00105_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[2] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00119_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00120_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00098_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[3] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00121_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00122_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00100_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[4] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00123_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00124_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00099_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[5] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00125_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00126_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00101_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[6] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00127_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00128_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00102_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[7] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00129_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00130_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00103_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[8] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00131_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00132_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00107_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[9] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00133_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00134_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00104_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[10] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00135_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00136_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00106_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[11] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00137_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00138_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00108_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[12] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00139_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00140_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00109_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
    ),
    derivative2 => Dict(
        F2C_GATE_TIMES[1] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00148_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00149_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00141_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[2] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00150_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00151_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00114_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[3] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00152_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00153_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00115_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[4] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00154_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00155_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00113_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[5] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00156_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00157_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00143_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[6] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00158_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00159_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00145_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[7] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00160_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00161_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00112_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[8] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00162_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00163_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00116_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[9] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00164_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00165_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00147_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[10] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00166_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00167_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00142_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[11] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00168_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00169_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00144_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
        F2C_GATE_TIMES[12] => Dict(
            DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00170_spin11.h5"),
            DATA2_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00171_spin11.h5"),
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00146_spin11.h5"),
            SAVE_TYPE_KEY => jl,
        ),
    ),
)
const F2C_PT_LIST = [sample, sample2, derivative, derivative2]
function make_figure2c()
    gate_type = xpiby2
    # get data and plot
    fig = Plots.plot(
        dpi=DPI_FINAL, legend=:bottomleft,
        ticksfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
        legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND,
        ylims=[0, 0.005], yticks=[0., 0.001, 0.002, 0.003, 0.004, 0.005],
        xticks=F2C_GATE_TIMES
    )
    Plots.xlabel!(fig, latexstring("\$t_{N} \\textrm{(ns)}\$"))
    Plots.ylabel!(fig, latexstring("\$ \\textrm{Avg. Gate Error at} \\ \\omega_{q} "
                                   * "\\pm 0.05 \\omega_{q}\$"))
    for pulse_type in F2C_PT_LIST
        label = "$(PT_STR[pulse_type])"
        color = PT_COLOR[pulse_type]
        marker = PT_MARKER[pulse_type]
        for (i, gate_time) in enumerate(F2C_GATE_TIMES)
            # compute
            data = F2C_PULSE_DATA[pulse_type][gate_time]
            save_file_path = data[SAVE_FILE_PATH_KEY]
            save_type = data[SAVE_TYPE_KEY]
            if !(DATA_FILE_PATH_KEY in keys(data))
                data_file_path1 = run_sim_deqjl(
                    1, gate_type; save_file_path=save_file_path,
                    save_type=save_type, dynamics_type=schroed, dt=1e-3,
                    negi_h0=S1FQ_NEGI_H0_ISO,
                )
            else
                data_file_path1 = data[DATA_FILE_PATH_KEY]
            end
            if !(DATA2_FILE_PATH_KEY in keys(data))
                data_file_path2 = run_sim_deqjl(
                    1, gate_type; save_file_path=save_file_path,
                    save_type=save_type, dynamics_type=schroed, dt=1e-3,
                    negi_h0=S2FQ_NEGI_H0_ISO,
                )
            else
                data_file_path2 = data[DATA2_FILE_PATH_KEY]
            end
            (fidelity1,) = h5open(data_file_path1, "r") do data_file1
                fidelity1 = read(data_file1, "fidelities")[end]
                return (fidelity1,)
            end
            (fidelity2,) = h5open(data_file_path2, "r") do data_file2
                fidelity2 = read(data_file2, "fidelities")[end]
                return (fidelity2,)
            end
            gate_error = 1 - mean([fidelity1, fidelity2])

            # plot
            label = i == 1 ? label : nothing
            Plots.scatter!(fig, [gate_time], [gate_error], label=label, color=color,
                           marker=(marker, MS_DATA))
        end
    end
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    println("Plotted Figure2c to $(plot_file_path)")
end


### FIGURE 3 ###
F3_PULSE_DATA = Dict(
    derivative => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, ""),
        SAVE_TYPE_KEY => jl,
        COLOR_KEY => :red,
        DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "")
    ),
    sample => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, ""),
        SAVE_TYPE_KEY => jl,
        COLOR_KEY => :green
    ),
    analytic => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, ""),
        SAVE_TYPE_KEY => py,
        COLOR_KEY => :lightskyblue,
        DATA_FILE_PATH_KEY => joinpath(META_SAVE_PATH, ""),
    ),
)

F3A_GATE_COUNT = Integer(1700)
F3A_DT = 1e-3
function make_figure3a()
    # compute
    gate_type = ypiby2
    dynamics_type = lindbladcfn
    for pulse_type in keys(F3_PULSE_DATA)
        pulse_data = F3_PULSE_DATA[pulse_type]
        if isnothing(pulse_data[DATA_FILE_PATH_KEY])
            save_file_path = pulse_data[SAVE_FILE_PATH_KEY]
            save_type = pulse_data[SAVE_TYPE_KEY]
            data_file_path = run_sim_deqjl(
                F3A_GATE_COUNT, gate_type; save_file_path=save_file_path,
                save_type=save_type, dynamics_type=dynamics_type, dt=F3A_DT
            )
            pulse_data[DATA_FILE_PATH_KEY] = data_file_path
        end
    end
    
    # plot
    colors = []; fidelitiess = []; labels = []
    for pulse_type in keys(F3_PULSE_DATA)
        (fidelities,) = h5open(F3_PULSE_DATA[pulse_type][SAVE_FILE_PATH_KEY], "r") do save_file
            fidelities = read(save_file, "fidelities")
            return (fidelities,)
        end
        color = F3_PULSE_DATA[pulse_type][COLOR_KEY]
        label = "$(PT_STR[pulse_type])"
        push!(colors, color)
        push!(fideltiess, fidelities)
        push!(labels, label)
    end
    plot_file_path = plot_fidelity_by_gate_count(fidelitiess; labels=labels, colors=colors)
    println("Plotted Figure3a to $(plot_file_path)")
end


function make_figure1()
    make_figure1a()
    make_figure1b()
    make_figure1c()
end
