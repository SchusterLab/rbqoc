"""
spin16.jl - getting numerical accuracy for the long simulations
"""

using Dates
using DifferentialEquations
using HDF5
using LinearAlgebra
using Printf
import Plots
using Random
using StaticArrays
using Statistics

EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin16"
WDIR = ENV["ROBUST_QOC_PATH"]
META_PATH = joinpath(WDIR, "out", EXPERIMENT_META)
SAVE_PATH = joinpath(META_PATH, EXPERIMENT_NAME)
VE_SAVE_FILE_NAME = "ve_spin16"

# plotting configuration
ENV["GKSwstype"] = "nul"
Plots.gr()

# experimental constants
# qubit frequency at flux frustration point
OMEGA = 2 * pi * 1.4e-2 #GHz
FBFQ_A = 0.202407
FBFQ_B = 0.5
# coefficients are listed in descending order
# raw coefficients are in units of seconds
FBFQ_T1_COEFFS = [
    3276.06057; -7905.24414; 8285.24137; -4939.22432;
    1821.23488; -415.520981; 53.9684414; -3.04500484
] * 1e9

# Define the system.
NEG_I = SA_F64[0  0 1 0;
               0  0 0 1;
               -1 0 0 0;
               0 -1 0 0;]
SIGMA_X = SA_F64[0   1   0   0;
                 1   0   0   0;
                 0   0   0   1;
                 0   0   1   0;]
SIGMA_Z = SA_F64[1   0   0   0;
                 0  -1   0   0;
                 0   0   1   0;
                 0   0   0  -1;]
H_S = SIGMA_Z / 2
OMEGA_NEG_I_H_S = OMEGA * NEG_I * H_S
H_C1 = SIGMA_X / 2
NEG_I_2_PI_H_C1 = 2 * pi * NEG_I * H_C1
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
# system constants
STATE_SIZE = 2
STATE_SIZE_ISO = 2 * STATE_SIZE
ZPIBY2_GATE_TIME = 17.86

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

@enum PulseType begin
    analytic = 1
    vanilla = 2
    t1_m1 = 3
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

GT_TO_GATE = Dict(
    xpiby2 => XPIBY2,
    ypiby2 => YPIBY2,
    zpiby2 => ZPIBY2,
)

PT_TO_STR = Dict(
    analytic => "Analytic",
    vanilla => "Vanilla QOC",
    t1_m1 => "T1 QOC"
)


# data constants
NON_TRAJ_OPT_PULSE = -1
CIDX_KEY = 4

PULSE_DATA = Dict(
    analytic => Dict(
        zpiby2 => joinpath(META_PATH, "spin14", "00000_spin14.h5"),
        ypiby2 => joinpath(META_PATH, "spin14", "00001_spin14.h5"),
        xpiby2 => joinpath(META_PATH, "spin14", "00002_spin14.h5"),
        CIDX_KEY => NON_TRAJ_OPT_PULSE,
    ),
    vanilla => Dict(
        zpiby2 => joinpath(META_PATH, "spin15", "00013_spin15.h5"),
        ypiby2 => joinpath(META_PATH, "spin15", "00016_spin15.h5"),
        xpiby2 => joinpath(META_PATH, "spin15", "00012_spin15.h5"),
        CIDX_KEY => 10,
    ),
    t1_m1 => Dict(
        zpiby2 => joinpath(META_PATH, "spin15", "00023_spin15.h5"),
        ypiby2 => joinpath(META_PATH, "spin15", "00015_spin15.h5"),
        xpiby2 => joinpath(META_PATH, "spin15", "00024_spin15.h5"),
        CIDX_KEY => 10,
    ),
)

# other constants
MAXITERS = 1e8
DPI = 500


show_nice(x) = show(IOContext(stdout), "text/plain", x)


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


function grab_controls(gate_type, pulse_type)
    controls_file_path = PULSE_DATA[pulse_type][gate_type]
    controls_idx = PULSE_DATA[pulse_type][CIDX_KEY]
    (controls, gate_time) = h5open(controls_file_path, "r") do save_file
        if controls_idx == NON_TRAJ_OPT_PULSE
            controls = permutedims(read(save_file, "controls"), [2, 1])
            gate_time = read(save_file, "evolution_time")
        else
            controls = read(save_file, "states")[:, controls_idx]
            gate_time = read(save_file, "evolution_time")
        end
        controls = controls / (2 * pi)
        return (controls, gate_time)
    end

    data = (controls, controls_file_path, gate_time)
    return data
end


function gen_rand_state_iso(seed_)
    Random.seed!(seed_)
    state = rand(STATE_SIZE) + 1im * rand(STATE_SIZE)
    return SVector{STATE_SIZE_ISO}(
        [real(state); imag(state)]
    )
end


function gen_rand_density_iso(seed_)
    Random.seed!(seed_)
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


fidelity_mat(m1, m2) = abs(tr(m1' * m2)) / abs(tr(m2' * m2))

# Define integration.
DT = 1e-2
DT_INV = 1e2

function rk4_step(dynamics, x, u)
    k1 = dynamics(x, u) * DT
	k2 = dynamics(x + k1 / 2, u) * DT
	k3 = dynamics(x + k2 / 2, u) * DT
	k4 = dynamics(x + k3, u) * DT
	return x + (k1 + 2 * k2 + 2 * k3 + k4)/6
end


function horner(coeffs, val)
    run = coeffs[1]
    for i = 2:lastindex(coeffs)
        run = coeffs[i] + val * run
    end
    return run
end


function get_fbfq(amplitude)
    return -abs(amplitude) * FBFQ_A + FBFQ_B
end


function get_t1_poly(amplitude)
    fbfq = get_fbfq(amplitude)
    t1 = horner(FBFQ_T1_COEFFS, fbfq)
    return t1
end


function dynamics_lindblad_hb(density, controls)
    gamma = (get_t1_poly(controls[1]))^(-1)
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls[1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
        + gamma * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2)
        + gamma * (E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


function dynamics_lindblad_nodis_hb(density, controls)
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls[1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
    )
end


function run_sim_hb(dissipation_type, gate_count, gate_type, pulse_type;
                    save=true, seed=0, plot=false)
    # grab
    data = grab_controls(gate_type, pulse_type)
    (controls, controls_file_path, gate_time) = data
    gate_knot_count = Int(gate_time * DT_INV)
    total_knot_count = gate_knot_count * gate_count
    densities = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
    if dissipation_type == dissipation
        dynamics = dynamics_lindblad_hb
    else
        dynamics = dynamics_lindblad_nodis_hb
    end

    # integrate
    densities[1, :, :] = density = initial_density = gen_rand_density_iso(seed)
    for i = 1:gate_count
        for j = 1:gate_knot_count
            density = rk4_step(dynamics, density, controls[j])
        end
        densities[i + 1, :, :] = density
    end
    densities[end, :, :] = density
    fidelities = zeros(gate_count + 1)
    g = GT_TO_GATE[gate_type]
    g2 = g^2
    g3 = g^3
    id0 = initial_density
    id1 = g * id0 * g'
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
    for i = 1:gate_count + 1
        i_eff = i - 1
        if i_eff % 4 == 0
            target = id0
            target_dag = id0_dag
            target_fnorm = id0_fnorm
        elseif i_eff % 4 == 1
            target = id1
            target_dag = id1_dag
            target_fnorm = id1_fnorm
        elseif i_eff % 4 == 2
            target = id2
            target_dag = id2_dag
            target_fnorm = id2_fnorm
        elseif i_eff % 4 == 3
            target = id3
            target_dag = id3_dag
            target_fnorm = id3_fnorm
        end
        fidelities[i] = abs(tr(target_dag * densities[i, :, :])) / target_fnorm
        # println("fidelity\n$(fidelities[i])")
        # println("density")
        # show_nice(density)
        # println("")
        # println("target")
        # show_nice(target)
        # println("")
    end

    # save data
    if save
        data_file_path = generate_save_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        h5open(data_file_path, "cw") do data_file
            write(data_file, "controls_file_path", controls_file_path)
            write(data_file, "dissipation_type", Integer(dissipation_type))
            write(data_file, "gate_count", gate_count)
            write(data_file, "gate_time", gate_time)
            write(data_file, "gate_type", Integer(gate_type))
            write(data_file, "pulse_type", Integer(pulse_type))
            write(data_file, "seed", seed)
            write(data_file, "densities", densities)
            write(data_file, "fidelities", fidelities)
        end
        println("Saved to $(data_file_path)")
    end

    # plot
    if plot
        plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
        gate_axis = Array(0:1:gate_count)
        fig = Plots.plot(
            dpi=DPI, title="$(GT_TO_STR[gate_type]) $(PT_TO_STR[pulse_type])",
            ylims=(0, 1), yticks=(0:0.1:1),
        )
        Plots.plot!(gate_axis, fidelities, label=nothing)
        Plots.xlabel!("Gate Count")
        Plots.ylabel!("Fidelity")
        Plots.savefig(fig, plot_file_path)
        println("Plotted to $(plot_file_path)")
    end
end


function ve_hb(gate_count; seed=0)
    gate_knot_count = Int(ZPIBY2_GATE_TIME * DT_INV)
    total_knot_count = gate_knot_count * gate_count
    densities = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
    densities[1, :, :] = density = initial_density = gen_rand_density_iso(seed)
    for i = 1:gate_count
        for j = 1:gate_knot_count
            density = rk4_step(dynamics_lindblad_hb, density, 0)
        end
        densities[i + 1, :, :] = density
    end
    densities[end, :, :] = density
    fidelities = zeros(gate_count + 1)
    z = ZPIBY2
    z2 = z^2
    z3 = z^3
    id0 = initial_density
    id1 = z * id0 * z'
    id2 = z2 * id0 * z2'
    id3 = z3 * id0 * z3'
    target_dag = id0_dag = id0'
    id1_dag = id1'
    id2_dag = id2'
    id3_dag = id3'
    target_fnorm = id0_fnorm = abs(tr(id0_dag * id0))
    id1_fnorm = abs(tr(id1_dag * id1))
    id2_fnorm = abs(tr(id2_dag * id2))
    id3_fnorm = abs(tr(id3_dag * id3))
    for i = 1:gate_count + 1
        density = densities[i, :, :]
        i_eff = i - 1
        if i_eff % 4 == 0
            target = id0
            target_dag = id0_dag
            target_fnorm = id0_fnorm
        elseif i_eff % 4 == 1
            target = id1
            target_dag = id1_dag
            target_fnorm = id1_fnorm
        elseif i_eff % 4 == 2
            target = id2
            target_dag = id2_dag
            target_fnorm = id2_fnorm
        elseif i_eff % 4 == 3
            target = id3
            target_dag = id3_dag
            target_fnorm = id3_fnorm
        end
        fidelities[i] = abs(tr(target_dag * density)) / target_fnorm
        # println("fidelity\n$(fidelities[i])")
        # println("density")
        # show_nice(density)
        # println("")
        # println("target")
        # show_nice(target)
        # println("")
    end

    # save data
    data_file_path = generate_save_file_path("h5", VE_SAVE_FILE_NAME, SAVE_PATH)
    h5open(data_file_path, "cw") do data_file
        write(data_file, "densities", densities)
        write(data_file, "fidelities", fidelities)
    end
    println("Saved to $(data_file_path)")

    # plot
    plot_file_path = generate_save_file_path("png", VE_SAVE_FILE_NAME, SAVE_PATH)
    gate_axis = Array(0:1:gate_count)
    fig = Plots.plot(
        dpi=DPI, title="Verify Empty",
        ylims=(0, 1), yticks=(0:0.1:1),
    )
    Plots.plot!(gate_axis, fidelities, label=nothing)
    Plots.xlabel!("Gate Count")
    Plots.ylabel!("Fidelity")
    Plots.savefig(fig, plot_file_path)
    println("Plotted to $(plot_file_path)")
end


function plot_fidelity_by_gate_count_single(fig, path)
    (dissipation_type, fidelities, gate_type,
     pulse_type) = h5open(path) do data_file
         dissipation_type = DissipationType(read(data_file, "dissipation_type"))
         fidelities = read(data_file, "fidelities")
         gate_type = GateType(read(data_file, "gate_type"))
         pulse_type = PulseType(read(data_file, "pulse_type"))
        return (dissipation_type, fidelities, gate_type, pulse_type)
    end
    gate_count = size(fidelities)[1] - 1
    gate_count_axis = Array(0:1:gate_count)
    label = "$(GT_TO_STR[gate_type]) $(PT_TO_STR[pulse_type]) $(DT_TO_STR[dissipation_type])"
    Plots.plot!(fig, gate_count_axis, fidelities, label=label)
end


function plot_fidelity_by_gate_count(paths;title=nothing)
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    fig = Plots.plot(dpi=DPI, ylims=(0, 1), yticks=(0:0.1:1))
    for path in paths
        plot_fidelity_by_gate_count_single(fig, path)
    end
    Plots.ylabel!("Fidelity")
    Plots.xlabel!("Gate Count")
    Plots.savefig(fig, plot_file_path)
    println("Saved plot to $(plot_file_path)")
end


function dynamics_lindblad_deqjl(density, (controls, gate_knot_count), t)
    knot_point = (Int(floor(t * DT_INV)) % gate_knot_count) + 1
    controls_ = controls[knot_point]
    gamma = (get_t1_poly(controls_[1]))^(-1)
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls_[1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
        + gamma * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2)
        + gamma * (E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


function dynamics_lindblad_nodis_deqjl(density, (controls, gate_knot_count), t)
    knot_point = (Int(floor(t * DT_INV)) % gate_knot_count) + 1
    controls_ = controls[knot_point]
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls_[1] * NEG_I_2_PI_H_C1
    return (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
    )
end


function run_sim_deqjl(dissipation_type, gate_count, gate_type, pulse_type;
                       save=true, seed=0, plot=false)
    # grab
    data = grab_controls(gate_type, pulse_type)
    (controls, controls_file_path, gate_time) = data
    gate_knot_count = Int(gate_time * DT_INV)
    gate_times = Array(0:1:gate_count) * gate_time
    
    # integrate
    if dissipation_type == dissipation
        f = dynamics_lindblad_deqjl
    else
        f = dynamics_lindblad_nodis_deqjl
    end
    initial_density = gen_rand_density_iso(seed)
    tspan = (0., gate_time * gate_count)
    p = (controls, gate_knot_count)
    prob = ODEProblem(f, initial_density, tspan, p)
    result = solve(prob, DifferentialEquations.RK4(), dt=DT, saveat=gate_times, maxiters=MAXITERS)

    # compute fidelity
    densities = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
    fidelities = zeros(gate_count + 1)
    gate_knot_count = Int(gate_time * DT_INV)
    g = GT_TO_GATE[gate_type]
    g2 = g^2
    g3 = g^3
    id0 = initial_density
    id1 = g * id0 * g'
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
    for i = 1:gate_count + 1
        densities[i, :, :] = density = result.u[i]
        i_eff = i - 1
        if i_eff % 4 == 0
            target = id0
            target_dag = id0_dag
            target_fnorm = id0_fnorm
        elseif i_eff % 4 == 1
            target = id1
            target_dag = id1_dag
            target_fnorm = id1_fnorm
        elseif i_eff % 4 == 2
            target = id2
            target_dag = id2_dag
            target_fnorm = id2_fnorm
        elseif i_eff % 4 == 3
            target = id3
            target_dag = id3_dag
            target_fnorm = id3_fnorm
        end
        fidelities[i] = abs(tr(target_dag * density)) / target_fnorm
        # println("fidelity\n$(fidelities[i])")
        # println("density")
        # show_nice(density)
        # println("")
        # println("target")
        # show_nice(target)
        # println("")
    end

    # save data
    if save
        data_file_path = generate_save_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        h5open(data_file_path, "cw") do data_file
            write(data_file, "controls_file_path", controls_file_path)
            write(data_file, "dissipation_type", Integer(dissipation_type))
            write(data_file, "gate_count", gate_count)
            write(data_file, "gate_time", gate_time)
            write(data_file, "gate_type", Integer(gate_type))
            write(data_file, "pulse_type", Integer(pulse_type))
            write(data_file, "seed", seed)
            write(data_file, "densities", densities)
            write(data_file, "fidelities", fidelities)
        end
        println("Saved to $(data_file_path)")
    end

    # plot
    if plot
        plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
        gate_axis = Array(0:1:gate_count)
        fig = Plots.plot(
            dpi=DPI, title="$(GT_TO_STR[gate_type]) $(PT_TO_STR[pulse_type])",
            ylims=(0, 1), yticks=(0:0.1:1),
        )
        Plots.plot!(gate_axis, fidelities, label=nothing)
        Plots.xlabel!("Gate Count")
        Plots.ylabel!("Fidelity")
        Plots.savefig(fig, plot_file_path)
        println("Plotted to $(plot_file_path)")
    end
end


function ve_deqjl(gate_count; seed=0, data_file_path=nothing)
    gate_times = Array(0:1:gate_count) * ZPIBY2_GATE_TIME
    if isnothing(data_file_path)
        initial_density = u0 = gen_rand_density_iso(seed)
        tspan = (0., ZPIBY2_GATE_TIME * gate_count)
        prob = ODEProblem(dynamics_lindblad_deqjl, u0, tspan)
        result = solve(prob, DifferentialEquations.RK4(), dt=1e-3, saveat=gate_times)
    end
    densities = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
    gate_knot_count = Int(ZPIBY2_GATE_TIME * DT_INV)
    gate_knot_count_4 = 4 * gate_knot_count
    fidelities = zeros(gate_count + 1)
    z = ZPIBY2
    z2 = z^2
    z3 = z^3
    id0 = initial_density
    id1 = z * id0 * z'
    id2 = z2 * id0 * z2'
    id3 = z3 * id0 * z3'
    target_dag = id0_dag = id0'
    id1_dag = id1'
    id2_dag = id2'
    id3_dag = id3'
    target_fnorm = id0_fnorm = abs(tr(id0_dag * id0))
    id1_fnorm = abs(tr(id1_dag * id1))
    id2_fnorm = abs(tr(id2_dag * id2))
    id3_fnorm = abs(tr(id3_dag * id3))
    for i = 1:gate_count + 1
        density = result.u[i]
        densities[i, :, :] = Array(density)
        i_eff = i - 1
        if i_eff % 4 == 0
            target = id0
            target_dag = id0_dag
            target_fnorm = id0_fnorm
        elseif i_eff % 4 == 1
            target = id1
            target_dag = id1_dag
            target_fnorm = id1_fnorm
        elseif i_eff % 4 == 2
            target = id2
            target_dag = id2_dag
            target_fnorm = id2_fnorm
        elseif i_eff % 4 == 3
            target = id3
            target_dag = id3_dag
            target_fnorm = id3_fnorm
        end
        fidelities[i] = abs(tr(target_dag * density)) / target_fnorm
        # println("fidelity\n$(fidelities[i])")
        # println("density")
        # show_nice(density)
        # println("")
        # println("target")
        # show_nice(target)
        # println("")
    end

    # save data
    data_file_path = generate_save_file_path("h5", VE_SAVE_FILE_NAME, SAVE_PATH)
    h5open(data_file_path, "cw") do data_file
        write(data_file, "densities", densities)
        write(data_file, "fidelities", fidelities)
    end
    println("Saved to $(data_file_path)")

    # plot
    plot_file_path = generate_save_file_path("png", VE_SAVE_FILE_NAME, SAVE_PATH)
    gate_axis = Array(0:1:gate_count)
    fig = Plots.plot(
        dpi=DPI, title="Verify Empty",
        ylims=(0, 1), yticks=(0:0.1:1),
    )
    Plots.plot!(gate_axis, fidelities, label=nothing)
    Plots.xlabel!("Gate Count")
    Plots.ylabel!("Fidelity")
    Plots.savefig(fig, plot_file_path)
    println("Plotted to $(plot_file_path)")
end
