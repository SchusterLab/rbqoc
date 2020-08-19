"""
rbqoc.jl - common definitions for the rbqoc repo
"""

# imports
using HDF5
using Interpolations
using LinearAlgebra
import Pkg
using Plots
using Printf
using RobotDynamics

# paths / venv
WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
Pkg.activate(joinpath(WDIR))

# plotting configuration and constants
ENV["GKSwstype"] = "nul"
Plots.gr()
const DPI = 300
const MS_SMALL = 2
const MS_MED = 6
const ALPHA = 0.2

# types
@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end


@enum SolverType begin
    ilqr = 1
    alilqr = 2
    altro = 3
end


@enum IntegratorType begin
    rk2 = 1
    rk3 = 2
    rk4 = 3
    rk6 = 4
end


const IT_RDI = Dict(
    rk2 => RobotDynamics.RK2,
    rk3 => RobotDynamics.RK3,
    rk4 => RobotDynamics.RK4,
    rk6 => RobotDynamics.RK6,
)


# methods
function generate_save_file_path(extension, save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$(save_file_name).$(extension)", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$(save_file_name).$(extension)"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


"""
grab_controls - do some extraction of relevant controls
data for common h5 save formats
"""
function grab_controls(save_file_path; save_type=jl)
    data = h5open(save_file_path, "r") do save_file
        if save_type == jl
            cidx = read(save_file, "controls_idx")
            controls = read(save_file, "astates")[:, cidx]
            evolution_time = read(save_file, "evolution_time")
        elseif save_type == samplejl
            controls = read(save_file, "controls_sample")
            evolution_time = read(save_file, "evolution_time_sample")
        elseif save_type == py
            controls = permutedims(read(save_file, "controls"), (2, 1))
            evolution_time = read(save_file, "evolution_time")
        end
        return (controls, evolution_time)
    end

    return data
end


"""
read_save - Read all data from an h5 file into memory.
"""
function read_save(save_file_path)
    dict = h5open(save_file_path, "r") do save_file
        dict = Dict()
        for key in names(save_file)
            dict[key] = read(save_file, key)
        end
        return dict
    end

    return dict
end


"""
horner - compute the value of a polynomial using Horner's method
Args:
coeffs :: Array(N) - the coefficients in descending order of degree
    a_{n - 1}, a_{n - 2}, ..., a_{1}, a_{0}
val :: T - the value at which the polynomial is computed
Returns:
polyval :: T - the polynomial evaluated at val
"""
function horner(coeffs, val)
    run = coeffs[1]
    for i = 2:lastindex(coeffs)
        run = coeffs[i] + val * run
    end
    return run
end


function plot_controls(save_file_paths, plot_file_path;
                       save_types=[jl,], labels=nothing,
                       title="", colors=nothing, print_out=true,
                       legend=nothing)
    fig = Plots.plot(dpi=DPI, title=title, legend=legend)
    for (i, save_file_path) in enumerate(save_file_paths)
        # Grab and prep data.
        (controls, evolution_time) = grab_controls(save_file_path; save_type=save_types[i])
        (control_eval_count, control_count) = size(controls)
        control_eval_times = Array(1:1:control_eval_count) * DT_PREF
        
        # Plot.
        for j = 1:control_count
            label = isnothing(labels) ? nothing : labels[i][j]
            color = isnothing(colors) ? :auto : colors[i][j]
            Plots.plot!(control_eval_times, controls[:, j],
                        label=label, color=color)
        end
    end
    Plots.xlabel!("Time (ns)")
    Plots.ylabel!("Amplitude (GHz)")
    Plots.savefig(fig, plot_file_path)
    if print_out
        println("Plotted to $(plot_file_path)")
    end
    return
end


@inline show_nice(x) = show(IOContext(stdout), "text/plain", x)


@inline get_vec_iso(vec) = vcat(real(vec), imag(vec))


function get_mat_iso(mat)
    len = size(mat)[1]
    mat_r = real(mat)
    mat_i = imag(mat)
    return vcat(hcat(mat_r, -mat_i),
                hcat(mat_i,  mat_r))
end
