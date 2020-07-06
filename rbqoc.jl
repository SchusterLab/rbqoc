"""
rbqoc.jl - common definitions for the rbqoc repo
"""

module RBQOC

# imports
using HDF5
using Plots
using HDF5
using Polynomials
using Printf
using StaticArrays
using Statistics


### COMMON ###

# simulation constants
DT_PREF = 1e-2
DT_PREF_INV = 1e2

# plotting constants
DPI = 300
MS = 2
ALPHA = 0.2

# types
@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end

# methods
function grab_controls(controls_file_path; mode="r", save_type=jl)
    data = h5open(controls_file_path, mode) do save_file
        if savetype == jl
            cidx = read(save_file, "controls_idx")
            controls = read(save_file, "states")[:, cidx]
            evolution_time = read(save_file, "evolution_time")
        end
        if savetype == samplejl
            controls = read(save_file, "controls_sample")
            evolution_time = read(save_file, "evolution_time_sample")
        end
        return (controls, evolution_time)
    end

    return data
end


function generate_save_file_path(extension, save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$save_file_name.$(extension)", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$save_file_name.$(extension)"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


function plot_controls(controls_file_path, save_file_path;
                       savetype=jl, title=nothing)
    # Grab and prep data.
    (controls, evolution_time) = grab_controls(controls_file_path; savetype=savetype)
    controls = controls ./ (2 * pi)
    (control_eval_count, control_count) = size(controls)
    control_eval_times = Array(1:1:control_eval_count) * DT_PREF
    file_name = split(basename(controls_file_path), ".h5")[1]
    if isnothing(title)
        title = file_name
    end

    # Plot.
    fig = Plots.plot(control_eval_times, controls, dpi=DPI,
                     label="controls", title=title)
    Plots.xlabel!("Time (ns)")
    Plots.ylabel!("Amplitude (GHz)")
    Plots.savefig(fig, save_file_path)
    return
end


### SPIN ###

# Define experimental constants.
# qubit frequency at flux frustration point
OMEGA = 2 * pi * 1.4e-2 #2 pi GHz
DOMEGA = OMEGA * 5e-2
OMEGA_PLUS = OMEGA + DOMEGA
OMEGA_MINUS = OMEGA - DOMEGA
MAX_CONTROL_NORM_0 = 2 * pi * 3e-1
FBFQ_A = 0.202407
FBFQ_B = 0.5
# coefficients are listed in descending order
# raw coefficients are in units of seconds
FBFQ_T1_COEFFS = [
    3276.06057; -7905.24414; 8285.24137; -4939.22432;
    1821.23488; -415.520981; 53.9684414; -3.04500484
] * 1e9

# Define the system.
# These matrices are defined in the complex to real isomorphism.
# NEG_I is the negative complex unit times the identity matrix
NEG_I = SA_F64[0   0  1  0 ;
               0   0  0  1 ;
               -1  0  0  0 ;
               0  -1  0  0 ;]
# SIGMA_X, SIGMA_Z are the X and Z pauli matrices
SIGMA_X = SA_F64[0   1   0   0;
                 1   0   0   0;
                 0   0   0   1;
                 0   0   1   0]
SIGMA_Z = SA_F64[1   0   0   0;
                 0  -1   0   0;
                 0   0   1   0;
                 0   0   0  -1]
H_S = SIGMA_Z / 2
NEG_I_H_S = NEG_I * H_S
OMEGA_NEG_I_H_S = OMEGA * NEG_I_H_S
H_C1 = SIGMA_X / 2
NEG_I_H_C1 = NEG_I * H_C1


"""
get_fbfq - Compute flux by flux quantum.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
get_fbfq(amplitude) = -abs(amplitude) * FBFQ_A + FBFQ_B


"""
get_t1_poly - Compute the t1 time for the given amplitude in units
of nanoseconds.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
get_t1_poly(amplitude) = horner(FBFQ_T1_COEFFS, get_fbfq(amplitude))


"""
t1_average - Compute the average t1 time for a control pulse.
"""
function t1_average(controls_file_path; save_type=qocjl)
    # Grab and prep data.
    (controls, evolution_time) = grab_controls(controls_file_path; save_type=save_type)
    t1s = map(get_t1_poly, controls / (2 * pi))
    t1_avg = mean(t1s)
    return t1_avg
end


"""
sample_polynomial - Fit a polynomial to arbitrary data
for a variable dt optimization. Sample
the polynomial at a given time step dt.
"""
function sample_polynomial(controls_file_path; plot=false)
    # Grab data.
    (
        acontrols,
        evolution_time,
        astates,
    ) = h5open(controls_file_path, "r+") do save_file
        acontrols = read(save_file, "controls")
        evolution_time = read(save_file, "evolution_time")
        astates = read(save_file, "states")
        return (
            acontrols,
            evolution_time,
            astates
        )
    end
    dts = acontrols[1:end, DT_IDX]
    knot_count = size(dts)[1]
    time_axis = [0; cumsum(dts, dims=1)[1:end-1]]
    final_time_sample = sum(dts)
    knot_count_sample = Int(floor(final_time_sample * DT_PREF_INV))
    # The last control should be DT_PREF before final_time_sample.
    time_axis_sample = Array(0:1:knot_count_sample - 1) * DT_PREF

    # Fit a quadratic polynomial to each knot point. Ends use their inner-neighbor's polynomial
    # astates has one more knot point than acontrols.
    controls = astates[1:end - 1, CONTROLS_IDX[1]]
    controls_polynomials = []
    d2controls_dt2 = acontrols[1:end, D2CONTROLS_DT2_IDX[1]]
    d2controls_dt2_polynomials = []
    deg = 2
    append!(controls_polynomials, fit(time_axis[1:3], controls[1:3], deg))
    append!(d2controls_dt2_polynomials, fit(time_axis[1:3], d2controls_dt2[1:3], deg))
    for i = 2:knot_count - 1
        inds = i - 1:i + 1
        ts = time_axis[inds]
        append!(controls_polynomials, fit(ts, controls[inds], deg))
        append!(d2controls_dt2_polynomials, fit(ts, d2controls_dt2[inds], deg))
    end
    append!(controls_polynomials, controls_polynomials[end])
    append!(d2controls_dt2_polynomials, d2controls_dt2_polynomials[end])
    
    # Sample the new time axis using the polynomials.
    controls_sample = zeros(knot_count_sample)
    d2controls_dt2_sample = zeros(knot_count_sample)
    time_axis_index = 1
    for i = 1:knot_count_sample
        now = time_axis_sample[i]
        # Advance the time axis to the nearest point.
        while ((time_axis_index != knot_count)
               && (abs(time_axis[time_axis_index] - now)
                   >= abs(time_axis[time_axis_index + 1] - now)))
            time_axis_index = time_axis_index + 1
        end
        controls_sample[i] = controls_polynomials[time_axis_index](now)
        d2controls_dt2_sample[i] = d2controls_dt2_polynomials[time_axis_index](now)
        # println("tas[$(i)]: $(now), ta[$(time_axis_index)]: $(time_axis[time_axis_index]) "
        #         * "sval: $(controls_sample[i]), val: $(controls[time_axis_index])")
    end

    # Plot.
    if plot
        plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
        fig = Plots.plot(dpi=DPI)
        Plots.scatter!(time_axis, controls, label="controls data", markersize=MS, alpha=ALPHA)
        Plots.scatter!(time_axis_sample, controls_sample, label="controls fit", markersize=MS, alpha=ALPHA)
        Plots.scatter!(time_axis, d2controls_dt2, label="d2_controls_dt2 data")
        Plots.scatter!(time_axis_sample, d2controls_dt2_sample, label="d2_controls_dt2 fit")
        Plots.xlabel!("Time (ns)")
        Plots.ylabel!("Amplitude (2 pi GHz)")
        Plots.savefig(fig, plot_file_path)
        println("Plotted to $(plot_file_path)")
    end
    return (controls_sample, d2controls_dt2_sample, final_time_sample)
end

end
