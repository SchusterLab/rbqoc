"""
rbqoc.jl - common definitions for the rbqoc repo
"""

# imports
using Dierckx
using DifferentialEquations
using HDF5
using Plots
using HDF5
using Polynomials
using Printf
using Random
using StaticArrays
using Statistics


### COMMON ###

# simulation constants
DT_PREF = 1e-2
DT_PREF_INV = 1e2

# plotting configuration and constants
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300
MS = 2
ALPHA = 0.2

# other constants
DEQJL_MAXITERS = 1e10

# types
@enum DissipationType begin
    dissipation = 1
    nodissipation = 2
end


@enum GateType begin
    zpiby2 = 1
    ypiby2 = 2
    xpiby2 = 3
end


@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end


DT_TO_STR = Dict(
    dissipation => "Dissipation",
    nodissipation => "No Dissipation",
)

GT_TO_STR = Dict(
    zpiby2 => "Z/2",
    ypiby2 => "Y/2",
    xpiby2 => "X/2",
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
            if occursin("_$save_file_name.$(extension)", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$save_file_name.$(extension)"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


function grab_controls(save_file_path; save_type=jl)
    data = h5open(save_file_path, "r") do save_file
        if save_type == jl
            cidx = read(save_file, "controls_idx")
            controls = read(save_file, "astates")[:, cidx]
            evolution_time = read(save_file, "evolution_time")
        elseif save_type == samplejl
            controls = read(save_file, "controls_sample")
            evolution_time = read(save_file, "evolution_time_sample")
        end
        return (controls, evolution_time)
    end

    return data
end


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


function plot_controls(save_file_path, plot_file_path;
                       save_type=jl, title=nothing)
    # Grab and prep data.
    (controls, evolution_time) = grab_controls(save_file_path; save_type=save_type)
    controls = controls ./ (2 * pi)
    (control_eval_count, control_count) = size(controls)
    control_eval_times = Array(1:1:control_eval_count) * DT_PREF
    file_name = split(basename(save_file_path), ".h5")[1]
    if isnothing(title)
        title = file_name
    end

    # Plot.
    fig = Plots.plot(dpi=DPI, title=title)
    for i = 1:control_count
        Plots.plot!(control_eval_times, controls[:, i],
                    label="controls $(i)")
    end
    Plots.xlabel!("Time (ns)")
    Plots.ylabel!("Amplitude (GHz)")
    Plots.savefig(fig, plot_file_path)
    println("Plotted to $(plot_file_path)")
    return
end


show_nice(x) = show(IOContext(stdout), "text/plain", x)


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
STATE_SIZE = 2
STATE_SIZE_ISO = 2 * STATE_SIZE
ZPIBY2_GATE_TIME = 17.86

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
NEG_I_2_PI_H_C1 = 2 * pi * NEG_I_H_C1
# dissipation ops
# L_{0} = |g> <e|
# L_{0}^{\dagger} = |e> <g|
# L_{0}^{\dagger} L_{0} = |e> <e|
# L_{1} = L_{0}^{\dagger} = |e> <g|
# L_{1}^{\dagger} = L_{0} = |g> <e|
# L_{1}^{\dagger} L_{1} = |g> <g|
G_E = SA_F64[0 1 0 0;
             0 0 0 0;
             0 0 0 1;
             0 0 0 0;]
E_G = SA_F64[0 0 0 0;
             1 0 0 0;
             0 0 0 0;
             0 0 1 0;]
NEG_G_G_BY2 = SA_F64[1 0 0 0;
                     0 0 0 0;
                     0 0 1 0;
                     0 0 0 0] * -0.5
NEG_E_E_BY2 = SA_F64[0 0 0 0;
                     0 1 0 0;
                     0 0 0 0;
                     0 0 0 1;] * -0.5
# gates
ZPIBY2 = SA_F64[1  0 1  0;
                0  1 0 -1;
                -1 0 1  0;
                0  1 0  1;] / sqrt(2)
YPIBY2 = SA_F64[1 -1 0  0;
                1  1 0  0;
                0  0 1 -1;
                0  0 1  1;] / sqrt(2)
XPIBY2 = SA_F64[1   0 0 1;
                0   1 1 0;
                0  -1 1 0;
                -1  0 0 1;] / sqrt(2)

GT_TO_GATE = Dict(
    xpiby2 => XPIBY2,
    ypiby2 => YPIBY2,
    zpiby2 => ZPIBY2,
)


# methods
function dynamics_lindblad_deqjl(density, (controls, control_knot_count, dt_inv), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    gamma = (get_t1_poly(controls[knot_point][1]))^(-1)
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls[knot_point][1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
        + gamma * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2)
        + gamma * (E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


function dynamics_lindblad_nodis_deqjl(density, (controls, control_knot_count, dt_inv), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls[knot_point][1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
    )
end


fidelity_mat(m1, m2) = abs(tr(m1' * m2)) / abs(tr(m2' * m2))


function gen_rand_state_iso(;seed=0)
    Random.seed!(seed)
    state = rand(STATE_SIZE) + 1im * rand(STATE_SIZE)
    return SVector{STATE_SIZE_ISO}(
        [real(state); imag(state)]
    )
end


function gen_rand_density_iso(;seed=0)
    Random.seed!(seed)
    state = rand(STATE_SIZE) + 1im * rand(STATE_SIZE)
    density = (state * state') / abs(state' * state)
    density_r = real(density)
    density_i = imag(density)
    density_iso = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}([
        density_r -density_i;
        density_i density_r;
    ])
    return density_iso
end


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


function run_sim_deqjl_single(gate_type, save_file_path; dt=DT_PREF,
                              dt_inv=DT_PREF_INV, dissipation_type=nodissipation,
                              save_type=jl, seed=0)
    # Grab data.
    (controls, evolution_time) = grab_controls(save_file_path)
    control_knot_count = Int(floor(evolution_time * dt_inv))
    save_times = [0; evolution_time]

    # Integrate.
    if dissipation_type == dissipation
        f = dynamics_lindblad_deqjl
    else
        f = dynamics_lindblad_nodis_deqjl
    end
    initial_density = gen_rand_density_iso(;seed=seed)
    tspan = (0., evolution_time)
    p = (controls, control_knot_count, dt_inv)
    prob = ODEProblem(f, initial_density, tspan, p)
    result = solve(prob, DifferentialEquations.Tsit5(), dt=dt, saveat=save_times,
                   maxiters=DEQJL_MAXITERS, adaptive=false)

    # Process.
    final_density = result.u[end]
    gate = GT_TO_GATE[gate_type]
    target_density = gate * initial_density * gate'
    fidelity = fidelity_mat(final_density, target_density)

    # Display.
    println("fidelity: $(fidelity)")
    println("initial_density")
    show_nice(initial_density)
    println("\ntarget_density")
    show_nice(target_density)
    println("\nfinal_density")
    show_nice(final_density)
    println("")
end


"""
sample_controls - Sample controls and d2controls_dt2
on the preferred time axis using a spline.
"""
function sample_controls(save_file_path; dt=DT_PREF, dt_inv=DT_PREF_INV,
                         plot=false, plot_file_path=nothing)
    # Grab data to sample from.
    save = read_save(save_file_path)
    controls = save["astates"][1:end - 1, (save["controls_idx"])]
    d2controls_dt2 = save["acontrols"][1:end, save["d2controls_dt2_idx"]]
    (control_knot_count, control_count) = size(controls)
    dts = save["acontrols"][1:end, save["dt_idx"]]
    time_axis = [0; cumsum(dts, dims=1)[1:end-1]]

    # Construct time axis to sample over.
    final_time_sample = sum(dts)
    knot_count_sample = Int(floor(final_time_sample * dt_inv))
    # The last control should be DT_PREF before final_time_sample.
    time_axis_sample = Array(0:1:knot_count_sample - 1) * dt

    # Sample time_axis_sample via spline.
    controls_sample = zeros(knot_count_sample, control_count)
    d2controls_dt2_sample = zeros(knot_count_sample, control_count)
    for i = 1:control_count
        controls_spline = Spline1D(time_axis, controls[:, i])
        controls_sample[:, i] = map(controls_spline, time_axis_sample)
        d2controls_dt2_spline = Spline1D(time_axis, d2controls_dt2[:, i])
        d2controls_dt2_sample[:, i] = map(d2controls_dt2_spline, time_axis_sample)
    end

    # Plot.
    if plot
        fig = Plots.plot(dpi=DPI)
        Plots.scatter!(time_axis, controls[:, 1], label="controls data", markersize=MS, alpha=ALPHA)
        Plots.scatter!(time_axis_sample, controls_sample[:, 1], label="controls fit", markersize=MS, alpha=ALPHA)
        Plots.scatter!(time_axis, d2controls_dt2[:, 1], label="d2_controls_dt2 data")
        Plots.scatter!(time_axis_sample, d2controls_dt2_sample[:, 1], label="d2_controls_dt2 fit")
        Plots.xlabel!("Time (ns)")
        Plots.ylabel!("Amplitude (GHz)")
        Plots.savefig(fig, plot_file_path)
        println("Plotted to $(plot_file_path)")
    end
    return (controls_sample, d2controls_dt2_sample, final_time_sample)
end


"""
t1_average - Compute the average t1 time for a control pulse.
"""
function t1_average(save_file_path; save_type=jl)
    # Grab and prep data.
    (controls, evolution_time) = grab_controls(save_file_path; save_type=save_type)
    (control_knot_count, control_count) = size(controls)
    t1_avgs = zeros(control_count)
    for i = 1:control_count
        t1s = map(get_t1_poly, controls[:, i] / (2 * pi))
        t1_avgs[i] = mean(t1s)
    end
    
    return t1_avgs
end
