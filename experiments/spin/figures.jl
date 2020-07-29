"""
figures.jl
"""

using LaTeXStrings
using Printf
import Plots

include(joinpath(ENV["RBQOC_PATH"], "rbqoc.jl"))

# Configure paths.
META_SAVE_PATH = joinpath(ENV["RBQOC_PATH"], "out", "spin")
EXPERIMENT_NAME = "figures"
SAVE_PATH = joinpath(META_SAVE_PATH, EXPERIMENT_NAME)

# Configure plotting.
ENV["GKSwstype"] = "nul"
Plots.gr()

# constants
SAVE_FILE_PATH_KEY = 1
SAVE_TYPE_KEY = 2
DATA_FILE_PATH_KEY = 3
F1_GATE_COUNT = Integer(1.5e4)

@enum PulseType begin
    qoc = 1
    analytic = 2
end

PT_TO_STR = Dict(
    qoc => "QOC",
    analytic => "Analytic",
)

# Figure 1
function plot_fidelity_by_gate_count_single(fig, path; inds=nothing)
    (fidelities,
     pulse_type) = h5open(path, "r") do data_file
         fidelities = read(data_file, "fidelities")
         pulse_type = PulseType(read(data_file, "pulse_type"))
         return (fidelities, pulse_type)
    end
    gate_count = size(fidelities)[1] - 1
    gate_count_axis = Array(0:1:gate_count)
    label = "$(PT_TO_STR[pulse_type])"
    if isnothing(inds)
        inds = 1:gate_count + 1
    end
    # inds = 1:4:gate_count + 1
    Plots.plot!(fig, gate_count_axis[inds], fidelities[inds], label=label, legend=:bottomleft)
end


function plot_fidelity_by_gate_count(paths; inds=nothing, title=nothing, ylims=(0, 1), yticks=(0:0.1:1))
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    fig = Plots.plot(dpi=DPI, ylims=ylims, yticks=yticks, title=title)
    for path in paths
        plot_fidelity_by_gate_count_single(fig, path; inds=inds)
    end
    Plots.ylabel!("Fidelity")
    Plots.xlabel!("Gate Count")
    Plots.savefig(fig, plot_file_path)
    println("Saved plot to $(plot_file_path)")
end


function make_figure1()
    # Gather pulses to benchmark.
    pulse_data = Dict(
        zpiby2 => Dict(
            qoc => Dict(
                SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00066_spin15.h5"),
                SAVE_TYPE_KEY => samplejl,
            ),
            analytic => Dict(
                SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00000_spin14.h5"),
                SAVE_TYPE_KEY => py,
            ),
        ),
        ypiby2 => Dict(
            qoc => Dict(
                SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00050.h5"),
                SAVE_TYPE_KEY => samplejl,
            ),
            analytic => Dict(
                SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00001_spin14.h5"),
                SAVE_TYPE_KEY => py,
            )
        ),
        xpiby2 => Dict(
            qoc => Dict(
                SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin15/00051.h5"),
                SAVE_TYPE_KEY => samplejl,
            ),
            analytic => Dict(
                SAVE_FILE_PATH_KEY => joinpath(META_SAVE_PATH, "spin14/00002_spin14.h5"),
                SAVE_TYPE_KEY => py,
            )
        ),
    )

    # Simulate dissipation and plot.
    dynamics_type = lindbladdis
    for gate_type in instances(GateType)
        if gate_type == xpiby2 || gate_type == ypiby2
            continue
        end
        # # Simulate dissipation.
        # paths = []
        # for pulse_type in instances(PulseType)
        #     path = pulse_data[gate_type][pulse_type][DATA_FILE_PATH_KEY] = (
        #         run_sim_deqjl(
        #             F1_GATE_COUNT, gate_type,
        #             pulse_data[gate_type][pulse_type][SAVE_FILE_PATH_KEY];
        #             dynamics_type=dynamics_type,
        #             save_type=pulse_data[gate_type][pulse_type][SAVE_TYPE_KEY]
        #         )
        #     )
        #     h5open(path, "r+") do save_file
        #         write(save_file, "pulse_type", Integer(pulse_type))
        #     end
        #     push!(paths, path)
        # end
        paths = [
            joinpath(META_SAVE_PATH, "spin15/00076_spin15.h5")
            joinpath(META_SAVE_PATH, "spin15/00078_spin15.h5");
            joinpath(META_SAVE_PATH, "spin14/00013_spin14.h5")
        ]
        # Plot.
        plot_fidelity_by_gate_count(
            paths; title=latexstring("$(GT_TO_STR[gate_type]) \$T_{1}\$ Dissipation"),
            ylims=(0.95, 1), yticks=0.9:0.01:1
        )
    end
end
