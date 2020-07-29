"""
rbqoc.jl - common definitions for the rbqoc repo
"""

# imports
using Dates
using Dierckx
using DifferentialEquations
using HDF5
using Interpolations
using LinearAlgebra
using Plots
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
DEQJL_ADAPTIVE = false

# types
@enum DynamicsType begin
    schroed = 1
    lindbladnodis = 2
    lindbladdis = 3
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
    schroed => "Schroedinger",
    lindbladnodis => "Lindblad No Dissipation",
    lindbladdis => "Lindblad Dissipation",
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
MAX_CONTROL_NORM_0 = 2 * pi * 5e-1
FBFQ_A = 0.202407
FBFQ_B = 0.5
# coefficients are listed in descending order
# raw coefficients are in units of seconds
FBFQ_T1_COEFFS = [
    3276.06057; -7905.24414; 8285.24137; -4939.22432;
    1821.23488; -415.520981; 53.9684414; -3.04500484
] * 1e9
# raw T1 times are in units of microseconds
T1_ARRAY = [
    1597.923, 1627.93, 301.86, 269.03, 476.33, 1783.19, 2131.76, 2634.50, 
    4364.68, 2587.82, 1661.915, 1794.468, 2173.88, 1188.83, 
    1576.493, 965.183, 560.251, 310.88
] * 1e3
FBFQ_ARRAY = [
    0.26, 0.28, 0.32, 0.34, 0.36, 0.38, 0.4,
    0.42, 0.44, 0.46, 0.465, 0.47, 0.475,
    0.48, 0.484, 0.488, 0.492, 0.5
]
FBFQ_T1_SPLINE_DIERCKX = Spline1D(FBFQ_ARRAY, T1_ARRAY)
FBFQ_T1_SPLINE_ITP = extrapolate(interpolate((FBFQ_ARRAY,), T1_ARRAY, Gridded(Linear())), Flat())
STATE_SIZE_NOISO = 2
STATE_SIZE_ISO = 2 * STATE_SIZE_NOISO
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
function dynamics_schroed_deqjl(state, (controls, control_knot_count, dt_inv), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls[knot_point][1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * state
    )
end


function dynamics_lindblad_deqjl(density, (controls, control_knot_count, dt_inv), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    gamma_1 = (get_t1_spline(controls[knot_point][1]))^(-1)
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls[knot_point][1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
        + gamma_1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2)
        + gamma_1 * (E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
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
    state = rand(STATE_SIZE_NOISO) + 1im * rand(STATE_SIZE_NOISO)
    return SVector{STATE_SIZE_ISO}(
        [real(state); imag(state)]
    )
end


function gen_rand_density_iso(;seed=0)
    if seed == -1
        state = [1, 1] / sqrt(2)
    else
        Random.seed!(seed)
        state = rand(STATE_SIZE_NOISO) + 1im * rand(STATE_SIZE_NOISO)        
    end
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
get_fbfq - Compute flux by flux quantum. Reflects
over the flux frustration point.

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
get_t1_spline - Compute the t1 time for the given amplitude
in units of nanoseconds.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
get_t1_spline(amplitude) = FBFQ_T1_SPLINE_ITP(get_fbfq(amplitude))


"""
run_sim_deqjl_single - Run a single simulation
using differentialequations.jl for a gate, and measure its fidelity.
This is used mostly as a sanity check to ensure the gate has high fidelity
on a single application.
"""
function run_sim_deqjl_single(gate_type, save_file_path;
                              controls_dt_inv=DT_PREF_INV,
                              dt=DT_PREF, dynamics_type=lindbladnodis,
                              save_type=jl, seed=0)
    # Grab data.
    (controls, gate_time) = grab_controls(save_file_path; save_type=save_type)
    controls = controls ./ (2 * pi)
    control_knot_count = Int(floor(gate_time * controls_dt_inv))
    save_times = [0; gate_time]

    # Integrate.
    if dynamics_type == lindbladnodis
        f = dynamics_lindblad_nodis_deqjl
    else
        f = dynamics_lindblad_deqjl
    end
    initial_density = gen_rand_density_iso(;seed=seed)
    tspan = (0., gate_time)
    p = (controls, control_knot_count, controls_dt_inv)
    prob = ODEProblem(f, initial_density, tspan, p)
    result = solve(prob, DifferentialEquations.Vern9(), dt=dt, saveat=save_times,
                   maxiters=DEQJL_MAXITERS, adaptive=DEQJL_ADAPTIVE)

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
run_sim_deqjl - Apply a gate multiple times and measure the fidelity
after each application. Save the output.

Arguments:
save_file_path :: String - The file path to grab the controls from
"""
function run_sim_deqjl(
    gate_count, gate_type, save_file_path;
    controls_dt_inv=DT_PREF_INV,
    deqjl_adaptive=false, dynamics_type=lindbladnodis,
    dt=DT_PREF, print_final=false, save=true, save_type=jl, seed=-1,
    solver=DifferentialEquations.Tsit5)
    start_time = Dates.now()
    # grab
    (controls, gate_time) = grab_controls(save_file_path; save_type=save_type)
    controls = controls ./ (2 * pi)
    gate_knot_count = Int(floor(gate_time * controls_dt_inv))
    gate_times = Array(0:1:gate_count) * gate_time
    
    # integrate
    if dynamics_type == lindbladnodis
        f = dynamics_lindblad_nodis_deqjl
    else
        f = dynamics_lindblad_deqjl
    end
    initial_density = gen_rand_density_iso(;seed=seed)
    tspan = (0., gate_time * gate_count)
    p = (controls, gate_knot_count, controls_dt_inv)
    prob = ODEProblem(f, initial_density, tspan, p)
    result = solve(prob, solver(), dt=dt, saveat=gate_times,
                   maxiters=DEQJL_MAXITERS, adaptive=deqjl_adaptive)

    # Compute the fidelities.
    # All of the gates we consider are 4-cyclic.
    densities = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
    fidelities = zeros(gate_count + 1)
    g1 = GT_TO_GATE[gate_type]
    g2 = g1^2
    g3 = g1^3
    id0 = initial_density
    id1 = g1 * id0 * g1'
    id2 = g2 * id0 * g2'
    id3 = g3 * id0 * g3'
    target_dag = id0_dag = id0'
    id1_dag = id1'
    id2_dag = id2'
    id3_dag = id3'
    target_fnorm = id0_fnorm = abs(tr(id0_dag * id0))
    id1_fnorm = abs(tr(id1_dag * id1))
    id2_fnorm = abs(tr(id2_dag * id2))
    id3_fnorm = abs(tr(id3_dag * id3))
    # Compute the fidelity after each gate.
    for i = 1:gate_count + 1
        densities[i, :, :] = density = result.u[i]
        # 1-indexing means we are 1 ahead for modulo arithmetic.
        i_eff = i - 1
        if i_eff % 4 == 0
            target_dag = id0_dag
            target_fnorm = id0_fnorm
        elseif i_eff % 4 == 1
            target_dag = id1_dag
            target_fnorm = id1_fnorm
        elseif i_eff % 4 == 2
            target_dag = id2_dag
            target_fnorm = id2_fnorm
        elseif i_eff % 4 == 3
            target_dag = id3_dag
            target_fnorm = id3_fnorm
        end
        fidelities[i] = abs(tr(target_dag * density)) / target_fnorm
        # println("fidelity\n$(fidelities[i])")
        # println("density")
        # show_nice(density)
        # println("")
        # println("target")
        # show_nice(target_dag')
        # println("")
    end
    end_time = Dates.now()
    run_time = end_time - start_time
    if print_final
        println("fidelities[$(gate_count)]: $(fidelities[end])")
    end

    # Save the data.
    experiment_name = split(save_file_path, "/")[end - 1]
    save_path = dirname(save_file_path)
    data_file_path = nothing
    if save
        data_file_path = generate_save_file_path("h5", experiment_name, save_path)
        h5open(data_file_path, "w") do data_file
            write(data_file, "dynamics_type", Integer(dynamics_type))
            write(data_file, "gate_count", gate_count)
            write(data_file, "gate_time", gate_time)
            write(data_file, "gate_type", Integer(gate_type))
            write(data_file, "save_file_path", save_file_path)
            write(data_file, "seed", seed)
            write(data_file, "densities", densities)
            write(data_file, "fidelities", fidelities)
            write(data_file, "run_time", string(run_time))
        end
        println("Saved simulation to $(data_file_path)")
    end
    return data_file_path
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
        t1s = map(get_t1_spline, controls[:, i] / (2 * pi))
        t1_avgs[i] = mean(t1s)
    end
    
    return t1_avgs
end
