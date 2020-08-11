"""
figures.jl
"""

using LaTeXStrings
using Printf
import Plots
using Statistics

include(joinpath(ENV["RBQOC_PATH"], "rbqoc.jl"))

# Configure paths.
META_SAVE_PATH = joinpath(ENV["RBQOC_PATH"], "out", "spin")
EXPERIMENT_NAME = "figures"
SAVE_PATH = joinpath(META_SAVE_PATH, EXPERIMENT_NAME)

# Plotting configuration.
ENV["GKSwstype"] = "nul"
Plots.gr()

# Constants
SAMPLE_LEN = Integer(1e4)


# Figure 1
@enum PulseType begin
    qoc = 1
    analytic = 2
    derivative = 3
    sample = 4
end

PT_LIST = [analytic, qoc]

PT_STR = Dict(
    qoc => "QOC",
    analytic => "Analytic",
    sample => "Sample",
    derivative => "Derivative"
)

MS_DATA = 4
MS_POINT = 8

SAVE_FILE_PATH_KEY = 1
SAVE_TYPE_KEY = 2
DATA_FILE_PATH_KEY = 3
COLOR_KEY = 4
ACORDS_KEY = 5


### FIGURE 1 ###
F1_GATE_COUNT = Integer(1.5e4)
F1_PULSE_DATA = Dict(
    zpiby2 => Dict(
        qoc => Dict(
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00099_spin15.h5"),
            SAVE_TYPE_KEY => jl,
            COLOR_KEY => :coral,
        ),
        analytic => Dict(
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00000_spin14.h5"),
            SAVE_TYPE_KEY => py,
            COLOR_KEY => :lightskyblue,
            ACORDS_KEY => (0, 0.13),
        ),
    ),
    ypiby2 => Dict(
        qoc => Dict(
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00125_spin15.h5"),
            SAVE_TYPE_KEY => jl,
            COLOR_KEY => :coral,
        ),
        analytic => Dict(
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00003_spin14.h5"),
            SAVE_TYPE_KEY => py,
            COLOR_KEY => :lightskyblue,
            ACORDS_KEY => (0, 0.5),
        )
    ),
    xpiby2 => Dict(
        qoc => Dict(
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00126_spin15.h5"),
            SAVE_TYPE_KEY => samplejl,
            COLOR_KEY => :coral,
        ),
        analytic => Dict(
            SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00004_spin14.h5"),
            SAVE_TYPE_KEY => py,
            COLOR_KEY => :lightskyblue,
            ACORDS_KEY => (0, 0.6),
        )
    ),
)


function make_figure1a()
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    save_file_paths = []; save_types = []; labels = []; colors = [];
    subfigs = []
    for (i, gate_type) in enumerate(instances(GateType))
        subfig = Plots.plot()
        if i == 2
            Plots.ylabel!(subfig, "Amplitude (GHz)")
        elseif i == 3
            Plots.xlabel!(subfig, "Time (ns)")
        end
        text_ = GT_STR[gate_type]
        (ax, ay) = F1_PULSE_DATA[gate_type][analytic][ACORDS_KEY]
        Plots.annotate!(subfig, ax, ay, text(text_, 10))
        for pulse_type in PT_LIST
            if pulse_type == analytic
                linestyle = :solid
            elseif pulse_type == qoc
                linestyle = :solid
            end
            data = F1_PULSE_DATA[gate_type][pulse_type]
            color = data[COLOR_KEY]
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
    fig = Plots.plot(subfigs[1], subfigs[2], subfigs[3], layout=layout, dpi=DPI)
    Plots.savefig(fig, plot_file_path)
    println("Saved Figure1a to $(plot_file_path)")
end


function plot_fidelity_by_gate_count_single(fig, path; inds=nothing)
    (fidelities, gate_type,
     pulse_type) = h5open(path, "r") do data_file
         fidelities = read(data_file, "fidelities")
         pulse_type = PulseType(read(data_file, "pulse_type"))
         gate_type = GateType(read(data_file, "gate_type"))
         return (fidelities, gate_type, pulse_type)
    end
    gate_count = size(fidelities)[1] - 1
    gate_count_axis = Array(0:1:gate_count)
    label = "$(GT_STR[gate_type]) $(PT_STR[pulse_type])"
    if isnothing(inds)
        inds = 1:gate_count + 1
    end
    # inds = 1:4:gate_count + 1
    Plots.plot!(fig, gate_count_axis[inds], fidelities[inds], label=label, legend=:bottomleft)
end


function plot_fidelity_by_gate_count(paths; inds=nothing, title="", ylims=(0, 1),
                                     yticks=(0:0.1:1), legend=nothing, yscale=:none)
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    fig = Plots.plot(dpi=DPI_FINAL, ylims=ylims, yticks=yticks, title=title,
                     legend=legend, yscale=yscale)
    for path in paths
        plot_fidelity_by_gate_count_single(fig, path; inds=inds)
    end
    Plots.ylabel!("Fidelity")
    Plots.xlabel!("Gate Count")
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end


function make_figure1b()
    # Simulate dissipation and plot.
    dynamics_type = lindbladdis
    for gate_type in instances(GateType)
        if gate_type == xpiby2 || gate_type == ypiby2
            continue
        end
        # # Simulate dissipation.
        # paths = []
        # for pulse_type in instances(PulseType)
        #     path = F1_PULSE_DATA[gate_type][pulse_type][DATA_FILE_PATH_KEY] = (
        #         run_sim_deqjl(
        #             F1_GATE_COUNT, gate_type,
        #             F1_PULSE_DATA[gate_type][pulse_type][SAVE_FILE_PATH_KEY];
        #             dynamics_type=dynamics_type,
        #             save_type=F1_PULSE_DATA[gate_type][pulse_type][SAVE_TYPE_KEY]
        #         )
        #     )
        #     h5open(path, "r+") do save_file
        #         write(save_file, "pulse_type", Integer(pulse_type))
        #     end
        #     push!(paths, path)
        # end
        paths = [
            joinpath(META_SAVE_PATH, "spin14/00013_spin14.h5"), # z/2 analytic
            joinpath(META_SAVE_PATH, "spin15/00105_spin15.h5"), # z/2 qoc
            joinpath(META_SAVE_PATH, "spin14/00018_spin14.h5"), # y/2 analytic
            joinpath(META_SAVE_PATH, "spin14/00022_spin14.h5"), # x/2 analytic
        ]
        # Plot.
        plot_file_path = plot_fidelity_by_gate_count(
            paths; ylims=(0.95, 1), yticks=0.95:0.01:1, legend=:bottomleft,
        )
        println("Plotted Figure1b to $(plot_file_path)")
    end
end


function make_figure1c()
    # Collect data.
    max_amp = MAX_CONTROL_NORM_0 / (2 * pi)
    amps_fit = Array(range(0, stop=max_amp, length=SAMPLE_LEN))
    t1s_fit =  map(amp_t1_spline, amps_fit)
    amps_data = -1 .* map(fbfq_amp, FBFQ_ARRAY)
    t1s_data = T1_ARRAY
    avg_amps = []; avg_t1s = []; avg_labels = []; avg_colors = []
    for gate_type in instances(GateType)
        if gate_type == ypiby2 || gate_type == xpiby2
            continue
        end
        for pulse_type in PT_LIST
            (controls, _) = grab_controls(
                F1_PULSE_DATA[gate_type][pulse_type][SAVE_FILE_PATH_KEY];
                save_type=F1_PULSE_DATA[gate_type][pulse_type][SAVE_TYPE_KEY]
            )
            controls = controls ./ (2 * pi)
            avg_amp = mean(map(abs, controls[:,1]))
            avg_t1 = amp_t1_spline(avg_amp)
            avg_label = "$(GT_STR[gate_type]) $(PT_STR[pulse_type])"
            avg_color = F1_PULSE_DATA[gate_type][pulse_type][COLOR_KEY]
            push!(avg_amps, avg_amp); push!(avg_t1s, avg_t1)
            push!(avg_labels, avg_label); push!(avg_colors, avg_color)
        end
    end
    

    # Plot.
    fig = Plots.plot(dpi=DPI_FINAL, legend=:bottomright, yscale=:log10)
    Plots.plot!(amps_fit, t1s_fit, label="Fit", color=:mediumaquamarine)
    Plots.scatter!(amps_data, t1s_data, label="Data", marker=(:circle, MS_DATA),
                   color=:mediumorchid)
    for i = 1:length(avg_amps)
        Plots.plot!([avg_amps[i]], [avg_t1s[i]], label=avg_labels[i],
                    marker=(:star, MS_POINT), color=avg_colors[i])
    end
    Plots.xlabel!("Avg. Amplitude (GHz)")
    Plots.ylabel!(latexstring("\$T_1 \\ \\textrm{(ns)}\$"))
    Plots.xlims!((-0.02, max_amp))
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    println("Plotted Figure1c to $(plot_file_path)")
end


### FIGURE 2 ###
F2_PULSE_DATA = Dict(
    derivative => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin11/00047_spin11.h5"),
        SAVE_TYPE_KEY => jl,
        COLOR_KEY => :red
    ),
    # sample => Dict(
    #     SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin12/00040_spin12.h5"),
    #     SAVE_TYPE_KEY => jl,
    #     COLOR_KEY => :green
    # ),
    analytic => Dict(
        SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00003_spin14.h5"),
        SAVE_TYPE_KEY => py,
        COLOR_KEY => :lightskyblue
    ),
)
F2B_TRIAL_COUNT = Integer(1e3)
FQ_DEV = 1e-1



"""
Show the pulses.
"""
function make_figure2a()
    return
end


"""
Show gate error vs. detuning
"""
function make_figure2b()
    # sample detunings
    fq_devs = Array(range(-FQ_DEV, stop=FQ_DEV, length=F2B_TRIAL_COUNT))
    fqs = (fq_devs .* FQ) .+ FQ
    negi_h0s = [NEGI_H0_ISO * fq for fq in fqs]

    # sweep
    # data_file_paths = []
    # gate_type = ypiby2
    # for pulse_type in keys(F2_PULSE_DATA)
    #     save_file_path = F2_PULSE_DATA[pulse_type][SAVE_FILE_PATH_KEY]
    #     save_type = F2_PULSE_DATA[pulse_type][SAVE_TYPE_KEY]
    #     data_file_path = run_sim_h0sweep_deqjl(ypiby2, negi_h0s; save_file_path=save_file_path,
    #                                            save_type=save_type)
    #     h5open(data_file_path, "r+") do data_file
    #         write(data_file, "fqs", fqs)
    #         write(data_file, "pulse_type", Integer(pulse_type))
    #     end
    #     push!(data_file_paths, data_file_path)
    # end
    data_file_paths = [
        # joinpath(META_SAVE_PATH, "spin12/00052_spin12.h5"),
        joinpath(META_SAVE_PATH, "spin11/00049_spin11.h5"),
        joinpath(META_SAVE_PATH, "spin14/00026_spin14.h5"),
    ]

    # plot
    fig = Plots.plot(dpi=DPI)
    for (i, pulse_type) in enumerate(keys(F2_PULSE_DATA))
        pulse_data = F2_PULSE_DATA[pulse_type]
        label = "$(PT_STR[pulse_type])"
        color = pulse_data[COLOR_KEY]
        data_file_path = data_file_paths[i]
        (fidelities,) = h5open(data_file_path, "r") do data_file
            fidelities = read(data_file, "fidelities")
            return (fidelities,)
        end
        Plots.scatter!(fig, fq_devs, fidelities, label=label, color=color)
    end
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    Plots.xlabel!(L"$\delta \omega_{q} / \omega_{q}$")
    Plots.ylabel!("Fidelity")
    Plots.savefig(fig, plot_file_path)
    println("Plotted Figure2b to $(plot_file_path)")
end
