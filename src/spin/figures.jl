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
const F1B_DATA_FILE_PATH = joinpath(SAVE_PATH, "f1b.h5")
const F2C_DATA_FILE_PATH = joinpath(SAVE_PATH, "f2c.h5")
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
end

const PT_STR = Dict(
    qoc => "QOC",
    analytic => "Anl.",
    s2 => "S-2",
    d2 => "D-2",
    d3 => "D-3",
    s4 => "S-4"
)

const PT_MARKER = Dict(
    s2 => :circle,
    s4 => :square,
    d3 => :utriangle,
    d2 => :diamond,
)

const PT_COLOR = Dict(
    analytic => :lightskyblue,
    qoc => :coral,
    s2 => :limegreen,
    s4 => :darkgreen,
    d2 => :crimson,
    d3 => :firebrick,
)

const PT_LINESTYLE = Dict(
    analytic => :solid,
    qoc => :solid,
    s2 => :solid,
    s4 => :dash,
    d2 => :solid,
    d3 => :dash,
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
const SAVEFP_KEY = 1
const SAVET_KEY = 2
const DATAFP_KEY = 3
const COLOR_KEY = 4
const ACORDS_KEY = 5
const MARKER_KEY = 6
const LCORDS_KEY = 7
const DATA2FP_KEY = 8


### ALL ###
function plot_fidelity_by_gate_count(fidelitiess; inds=nothing, title="", ylims=(0, 1),
                                     yticks=(0:0.1:1), legend=:best, yscale=:auto,
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
        label = isnothing(labels) ? "" : labels[i]
        linestyle = isnothing(linestyles) ? :solid : linestyles[i]
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
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00201_spin15.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00194_spin15.h5"),
            SAVET_KEY => jl,
        ),
        analytic => Dict(
            ACORDS_KEY => (0, 0.25),
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00043_spin14.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00000_spin14.h5"),
            SAVET_KEY => py,
        ),
    ),
    ypiby2 => Dict(
        qoc => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00200_spin15.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00185_spin15.h5"),
            SAVET_KEY => jl,
        ),
        analytic => Dict(
            ACORDS_KEY => (0, 0.4),
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00041_spin14.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00003_spin14.h5"),
            SAVET_KEY => py,
        )
    ),
    xpiby2 => Dict(
        qoc => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00202_spin15.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin15/00174_spin15.h5"),
            SAVET_KEY => jl,
        ),
        analytic => Dict(
            ACORDS_KEY => (0, 0.5),
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00042_spin14.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
            SAVET_KEY => py,
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
            save_file_path = data[SAVEFP_KEY]
            save_type = data[SAVET_KEY]
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
            if !(DATAFP_KEY in keys(pulse_data))
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
            (fidelities,) = h5open(pulse_data[DATAFP_KEY], "r") do data_file
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


const F1C_S2_LEN = Integer(1e4)
const GT_MK_1C = Dict(
    zpiby2 => :diamond,
    ypiby2 => :square,
    xpiby2 => :utriangle,
)
const MS_F1C = 6
const ALPHA_F1C = 1.

function make_figure1c(;save=false)
    # Collect data and plot.
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
    for gate_type in GT_LIST
        for pulse_type in keys(F1_PULSE_DATA[gate_type])
            pulse_data = F1_PULSE_DATA[gate_type][pulse_type]
            (controls, _) = grab_controls(
                pulse_data[SAVEFP_KEY];
                save_type=pulse_data[SAVET_KEY]
            )
            avg_amp = mean(map(abs, controls[:,1]))
            avg_t1 = amp_t1_spline_cubic(avg_amp)
            avg_label = "$(GT_STR[gate_type]) $(PT_STR[pulse_type])"
            println("amp: $(avg_amp), t1: $(avg_t1), label: $(avg_label)")
            avg_color = PT_COLOR[pulse_type]
            marker = GT_MK_1C[gate_type]
            Plots.plot!([avg_amp], [avg_t1], label=avg_label,
                        marker=(marker, MS_F1C), color=avg_color, alpha=ALPHA_F1C)
        end
    end
    Plots.xlabel!(latexstring("\$ {<a>}_{t} \\textrm{(GHz)} \$"))
    Plots.ylabel!(latexstring("\$T_1 \\ \\textrm{(ns)}\$"))
    Plots.xlims!((-0.02, max_amp))
    # plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    # Plots.savefig(fig, plot_file_path)
    # println("Plotted Figure1c to $(plot_file_path)")
end


### FIGURE 2 ###
F2_DATA = Dict(
    analytic => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
        SAVET_KEY => py,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00074_spin14.h5"),
        LCORDS_KEY => (20, 0.05),
    ),
    s2 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00132_spin12.h5"),
        SAVET_KEY => jl,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00200_spin12.h5"),
        LCORDS_KEY => (30, 0.2),
    ),
    s4 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00229_spin12.h5"),
        SAVET_KEY => jl,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00231_spin12.h5"),
        LCORDS_KEY => (30, 0.2),
    ),
    d2 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00091_spin11.h5"),
        SAVET_KEY => jl,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00092_spin11.h5"),
        LCORDS_KEY => (45, 0.2),
    ),
    d3 => Dict(
        SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00110_spin11.h5"),
        SAVET_KEY => jl,
        DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00111_spin11.h5"),
        LCORDS_KEY => (45, 0.2),
    ),
)

F2A_PT_LIST = [[analytic], [s2, s4,], [d2, d3]]
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
            save_file_path = data[SAVEFP_KEY]
            save_type = data[SAVET_KEY]
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


const F2B_TRIAL_COUNT = Integer(3e2)
const F2B_FQ_DEV = 3e-2
const F2B_PT_LIST = [analytic, s2, s4, d2, d3]
"""
Show gate error vs. detuning
"""
function make_figure2b()
    # s2 detunings
    fq_devs = Array(range(-F2B_FQ_DEV, stop=F2B_FQ_DEV, length=F2B_TRIAL_COUNT))
    fqs = (fq_devs .* FQ) .+ FQ
    negi_h0s = [NEGI_H0_ISO * fq for fq in fqs]

    # sweep
    gate_type = xpiby2
    for pulse_type in F2B_PT_LIST
        # if !(DATAFP_KEY in keys(F2_PULSE_DATA[pulse_type]))
        if true
            if pulse_type == analytic
                data_file_path = run_sim_h0sweep_deqjl(
                    gate_type, negi_h0s; dynamics_type=xpiby2nodis, dt=1e-3
                )
            else
                save_file_path = F2_PULSE_DATA[pulse_type][SAVEFP_KEY]
                save_type = F2_PULSE_DATA[pulse_type][SAVET_KEY]
                data_file_path = run_sim_h0sweep_deqjl(
                    gate_type, negi_h0s; save_file_path=save_file_path,
                    dynamics_type=schroed, save_type=save_type, dt=1e-3
                )
            end
            h5open(data_file_path, "r+") do data_file
                write(data_file, "fqs", fqs)
                write(data_file, "pulse_type", Integer(pulse_type))
            end
            F2_PULSE_DATA[pulse_type][DATAFP_KEY] = data_file_path
            println("datafp: $(data_file_path), gt: $(GT_STR[gate_type]), pt: $(PT_STR[pulse_type])")
            return
        end
    end

    # plot
    fig = Plots.plot(dpi=DPI_FINAL, yticks=[0, 0.0025, 0.005, 0.0075], ylim=(0., 0.0075),
                     xlim=(minimum(fq_devs), maximum(fq_devs)), xticks=[-0.1, -0.05, 0, 0.05, 0.1],
                     legend=:none,
                     tickfontsize=FS_AXIS_TICKS, guidefontsize=FS_AXIS_LABELS,
                     legendfontsize=FS_LEGEND, foreground_color_legend=FG_COLOR_LEGEND)
    for pulse_type in F2B_PT_LIST
        pulse_data = F2_PULSE_DATA[pulse_type]
        label = "$(PT_STR[pulse_type])"
        linestyle = PT_LINESTYLE[pulse_type]
        color = PT_COLOR[pulse_type]
        data_file_path = pulse_data[DATAFP_KEY]
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


function f2b_sweep(;save_file_path=nothing, save_type=jl)
    fq_devs = Array(range(-F2B_FQ_DEV, stop=F2B_FQ_DEV, length=F2B_TRIAL_COUNT))
    fqs = (fq_devs .* FQ) .+ FQ
    negi_h0s = [NEGI_H0_ISO * fq for fq in fqs]
    gate_type=xpiby2
    data_file_path = run_sim_h0sweep_deqjl(
        gate_type, negi_h0s; save_file_path=save_file_path,
        dynamics_type=schroed, save_type=save_type, dt=1e-3
    )
    h5open(data_file_path, "r+") do data_file
        write(data_file, "fqs", fqs)
        write(data_file, "fq_devs", fq_devs)
    end
    println("datafp: $(data_file_path)")
end


F2C_GATE_TIMES = [50., 56.8, 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160.]
F2C_DATA = Dict(
    analytic => Dict(
        1 => Dict(),
        2 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00095_spin14.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00096_spin14.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
            SAVET_KEY => py,
        ),
        3 => Dict(),
        4 => Dict(),
        5 => Dict(),
        6 => Dict(),
        7 => Dict(),
        8 => Dict(),
        9 => Dict(),
        10 => Dict(),
        11 => Dict(),
        12 => Dict(),
        13 => Dict(),
    ),
    s2 => Dict(
        1 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00365_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00366_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00336_spin12.h5"),
            SAVET_KEY => jl,
        ),
        2 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00367_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00368_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00295_spin12.h5"),
            SAVET_KEY => jl,
        ),
        3 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00369_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00370_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00301_spin12.h5"),
            SAVET_KEY => jl,
        ),
        4 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00371_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00372_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00302_spin12.h5"),
            SAVET_KEY => jl,
        ),
        5 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00373_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00374_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00304_spin12.h5"),
            SAVET_KEY => jl,
        ),
        6 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00375_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00376_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00303_spin12.h5"),
            SAVET_KEY => jl,
        ),
        7 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00492_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00493_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00305_spin12.h5"),
            SAVET_KEY => jl,
        ),
        8 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00379_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00380_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00306_spin12.h5"),
            SAVET_KEY => jl,
        ),
        9 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00381_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00382_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00340_spin12.h5"),
            SAVET_KEY => jl,
        ),
        10 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00402_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00403_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00399_spin12.h5"),
            SAVET_KEY => jl,
                ),
        11 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00404_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00405_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00400_spin12.h5"),
            SAVET_KEY => jl,
                ),
        12 => Dict(
            # DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00381_spin12.h5"),
            # DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00382_spin12.h5"),
            # SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00340_spin12.h5"),
            SAVET_KEY => jl,
                ),
        13 => Dict(
            # DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00381_spin12.h5"),
            # DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00382_spin12.h5"),
            # SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00340_spin12.h5"),
            SAVET_KEY => jl,
        ),
    ),
    s4 => Dict(
        1 => Dict(
            # SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/_spin12.h5"), #TODO
            SAVET_KEY => jl,
        ),
        2 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00383_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00384_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00298_spin12.h5"),
            SAVET_KEY => jl,
        ),
        3 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00385_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00386_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00307_spin12.h5"),
            SAVET_KEY => jl,
        ),
        4 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00387_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00388_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00308_spin12.h5"),
            SAVET_KEY => jl,
        ),
        5 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00389_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00390_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00310_spin12.h5"),
            SAVET_KEY => jl,
        ),
        6 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00391_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00392_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00309_spin12.h5"),
            SAVET_KEY => jl,
        ),
        7 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00393_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00394_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00335_spin12.h5"),
            SAVET_KEY => jl,
        ),
        8 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00395_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00396_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00337_spin12.h5"),
            SAVET_KEY => jl,
        ),
        9 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00397_spin12.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00398_spin12.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00339_spin12.h5"),
            SAVET_KEY => jl,
        ),
        10 => Dict(
            # DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00391_spin12.h5"),
            # DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00392_spin12.h5"),
            # SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00309_spin12.h5"),
            SAVET_KEY => jl,
        ),
        11 => Dict(
            # DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00393_spin12.h5"),
            # DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00394_spin12.h5"),
            # SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00335_spin12.h5"),
            SAVET_KEY => jl,
        ),
        12 => Dict(
            # DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00395_spin12.h5"),
            # DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00396_spin12.h5"),
            # SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00337_spin12.h5"),
            SAVET_KEY => jl,
        ),
        13 => Dict(
            # DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00397_spin12.h5"),
            # DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00398_spin12.h5"),
            # SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin12/00339_spin12.h5"),
            SAVET_KEY => jl,
        ),
    ),
    d2 => Dict(
        1 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00176_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00177_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00105_spin11.h5"),
            SAVET_KEY => jl,
        ),
        2 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00239_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00240_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00091_spin11.h5"),
            SAVET_KEY => jl,
        ),
        3 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00178_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00179_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00098_spin11.h5"),
            SAVET_KEY => jl,
        ),
        4 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00180_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00181_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00100_spin11.h5"),
            SAVET_KEY => jl,
        ),
        5 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00182_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00183_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00099_spin11.h5"),
            SAVET_KEY => jl,
        ),
        6 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00273_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00274_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00229_spin11.h5"),
            SAVET_KEY => jl,
        ),
        7 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00275_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00276_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00231_spin11.h5"),
            SAVET_KEY => jl,
        ),
        8 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00277_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00278_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00230_spin11.h5"),
            SAVET_KEY => jl,
        ),
        9 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00279_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00280_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00232_spin11.h5"),
            SAVET_KEY => jl,
        ),
        10 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00296_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00297_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00291_spin11.h5"),
            SAVET_KEY => jl,
        ),
        11 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00298_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00299_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00289_spin11.h5"),
            SAVET_KEY => jl,
        ),
        12 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00300_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00301_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00290_spin11.h5"),
            SAVET_KEY => jl,
        ),
        13 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00302_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00303_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00292_spin11.h5"),
            SAVET_KEY => jl,
        ),
    ),
    d3 => Dict(
        1 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00200_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00201_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00141_spin11.h5"),
            SAVET_KEY => jl,
        ),
        2 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00257_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00258_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00110_spin11.h5"),
            SAVET_KEY => jl,
        ),
        3 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00202_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00203_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00114_spin11.h5"),
            SAVET_KEY => jl,
        ),
        4 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00204_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00205_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00115_spin11.h5"),
            SAVET_KEY => jl,
        ),
        5 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00206_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00207_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00113_spin11.h5"),
            SAVET_KEY => jl,
        ),
        6 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00281_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00282_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00235_spin11.h5"),
            SAVET_KEY => jl,
        ),
        7 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00283_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00284_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00236_spin11.h5"),
            SAVET_KEY => jl,
        ),
        8 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00285_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00286_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00234_spin11.h5"),
            SAVET_KEY => jl,
        ),
        9 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00287_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00288_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00233_spin11.h5"),
            SAVET_KEY => jl,
        ),
        10 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00304_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00305_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00295_spin11.h5"),
            SAVET_KEY => jl,
        ),
        11 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00306_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00307_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00293_spin11.h5"),
            SAVET_KEY => jl,
        ),
        12 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00308_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00309_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00294_spin11.h5"),
            SAVET_KEY => jl,
        ),
        13 => Dict(
            DATAFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00423_spin11.h5"),
            DATA2FP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00424_spin11.h5"),
            SAVEFP_KEY => joinpath(SPIN_OUT_PATH, "spin11/00362_spin11.h5"),
            SAVET_KEY => jl,
        ),
    ),
)
const F2C_PT_LIST = [s2, s4, d2, d3]
function gen_2c()
    gate_type=xpiby2
    gtcount = size(F2C_GATE_TIMES)[1]
    ptcount = size(F2C_PT_LIST)[1]
    gate_errors = ones(ptcount, gtcount)
    for (i, pulse_type) in enumerate(F2C_PT_LIST)
        println("pt: $(pulse_type)")
        for (j, gate_time) in enumerate(F2C_GATE_TIMES)
            data = F2C_DATA[pulse_type][j]
            keys_ = keys(data)
            if !(SAVEFP_KEY in keys_)
                continue
            elseif DATAFP_KEY in keys_ && DATA2FP_KEY in keys_
                data_file_path1 = data[DATAFP_KEY]
                data_file_path2 = data[DATAFP_KEY]
            else
                save_file_path = data[SAVEFP_KEY]
                save_type = data[SAVET_KEY]
                println("gt[$(j)]: $(gate_time)")
                data_file_path1 = run_sim_deqjl(
                    1, gate_type; save_file_path=save_file_path,
                    save_type=save_type, dynamics_type=schroed, dt=1e-3,
                    negi_h0=SP1FQ_NEGI_H0_ISO,
                )
                data_file_path2 = run_sim_deqjl(
                    1, gate_type; save_file_path=save_file_path,
                    save_type=save_type, dynamics_type=schroed, dt=1e-3,
                    negi_h0=SN1FQ_NEGI_H0_ISO,
                )
            end
            (fidelity1,) = h5open(data_file_path1, "r") do df1
                fidelity1 = read(df1, "fidelities")[end]
                return (fidelity1,)
            end
            (fidelity2,) = h5open(data_file_path2, "r") do df2
                fidelity2 = read(df2, "fidelities")[end]
                return (fidelity2,)
            end
            fidelity = mean([fidelity1, fidelity2])
            gate_errors[i, j] = 1 - fidelity
        end
    end
    h5open(F2C_DATA_FILE_PATH, "w") do data_file
        write(data_file, "gate_errors", gate_errors)
        write(data_file, "gate_times", F2C_GATE_TIMES)
        write(data_file, "pulse_types", [Integer(pulse_type) for pulse_type in F2C_PT_LIST])
    end
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
