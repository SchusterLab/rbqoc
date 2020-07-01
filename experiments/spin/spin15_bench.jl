"""
spin15_bench.jl - lindblad evolution for spin15.jl in Julia
"""

using Dates
using HDF5
import LaTeXStrings
using LinearAlgebra
using MPI
import Plots
using Printf
using Random
using StaticArrays
using Statistics
using TrajectoryOptimization


EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin15_bench"
WDIR = ENV["ROBUST_QOC_PATH"]
META_PATH = joinpath(WDIR, "out", EXPERIMENT_META)
SAVE_PATH = joinpath(META_PATH, EXPERIMENT_NAME)
ZPIBY2_COMPARISON_FILE_PATH = joinpath(SAVE_PATH, "zpiby2_comparsion_spin15_bench.png")

# plotting configuration
ENV["GKSwstype"] = "nul"
Plots.gr()

# Define experimental constants.
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

STATE_SIZE = 2
DENSITY_SIZE_ISO = 2 * STATE_SIZE

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

# Define other constants.

@enum PulseType begin
    analytic = 1
    vanilla = 2
    t1_m1 = 3
end

@enum GateType begin
    zpiby2 = 1
    ypiby2 = 2
    xpiby2 = 3
end

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

OUT_DATA = Dict(
    analytic => Dict(
        zpiby2 => joinpath(SAVE_PATH, "00008_spin15_bench.h5"),
        ypiby2 => joinpath(SAVE_PATH, "00009_spin15_bench.h5"),
        xpiby2 => joinpath(SAVE_PATH, "00010_spin15_bench.h5"),
    ),
    vanilla => Dict(
        zpiby2 => joinpath(SAVE_PATH, "00011_spin15_bench.h5"),
        ypiby2 => joinpath(SAVE_PATH, "00012_spin15_bench.h5"),
        xpiby2 => joinpath(SAVE_PATH, "00013_spin15_bench.h5"),
    ),
    t1_m1 => Dict(
        zpiby2 => joinpath(SAVE_PATH, "00014_spin15_bench.h5"),
        ypiby2 => joinpath(SAVE_PATH, "00015_spin15_bench.h5"),
        xpiby2 => joinpath(SAVE_PATH, "00016_spin15_bench.h5"),
    ),
)

OUT_DATA_1 = Dict(
    analytic => Dict(
        zpiby2 => joinpath(SAVE_PATH, "00026_spin15_bench.h5"),
        ypiby2 => joinpath(SAVE_PATH, "00028_spin15_bench.h5"),
        xpiby2 => joinpath(SAVE_PATH, "00031_spin15_bench.h5"),
    ),
    vanilla => Dict(
        zpiby2 => joinpath(SAVE_PATH, "00024_spin15_bench.h5"),
        ypiby2 => joinpath(SAVE_PATH, "00027_spin15_bench.h5"),
        xpiby2 => joinpath(SAVE_PATH, "00030_spin15_bench.h5"),
    ),
    t1_m1 => Dict(
        zpiby2 => joinpath(SAVE_PATH, "00025_spin15_bench.h5"),
        ypiby2 => joinpath(SAVE_PATH, "00029_spin15_bench.h5"),
        xpiby2 => joinpath(SAVE_PATH, "00032_spin15_bench.h5"),
    ),
)

NO_DISSIPATION_KEY = 0
DISSIPATION_KEY = 1
COMPARISON_DATA = Dict(
    analytic => Dict(
        NO_DISSIPATION_KEY => joinpath(SAVE_PATH, "00022_spin15_bench.h5"),
        DISSIPATION_KEY => joinpath(SAVE_PATH, "00026_spin15_bench.h5"),
    ),
    vanilla => Dict(
        NO_DISSIPATION_KEY => joinpath(SAVE_PATH, "00023_spin15_bench.h5"),
        DISSIPATION_KEY => joinpath(SAVE_PATH, "00024_spin15_bench.h5"),
    ),
    t1_m1 => Dict(
        NO_DISSIPATION_KEY => joinpath(SAVE_PATH, "00021_spin15_bench.h5"),
        DISSIPATION_KEY => joinpath(SAVE_PATH, "00025_spin15_bench.h5")
    )
)

# plotting
DPI = 500
MARKER_ALPHA = 0.2
MARKER_SIZE = 10
GT_TO_PLOT_FILE_PATH = Dict(
    zpiby2 => joinpath(SAVE_PATH, "zpiby2_spin15_bench.png"),
    ypiby2 => joinpath(SAVE_PATH, "ypiby2_spin15_bench.png"),
    xpiby2 => joinpath(SAVE_PATH, "xpiby2_spin15_bench.png"),
)

GT_TO_PLOT_FILE_PATH_1 = Dict(
    zpiby2 => joinpath(SAVE_PATH, "00001_zpiby2_spin15_bench.png"),
    ypiby2 => joinpath(SAVE_PATH, "00001_ypiby2_spin15_bench.png"),
    xpiby2 => joinpath(SAVE_PATH, "00001_xpiby2_spin15_bench.png"),
)

PT_TO_STR = Dict(
    analytic => "Analytic",
    vanilla => "Vanilla QOC",
    t1_m1 => "T1 QOC"
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

# multiprocessing
MP = false
COMM = nothing
MAX_RANK = 0
ROOT_RANK = 0
if MP
    MPI.Init()
    COMM = MPI.COMM_WORLD
    MAX_RANK = MPI.Comm_size(COMM) - 1
end


show_nice(x) = show(IOContext(stdout), "text/plain", x)


function generate_save_file_path(save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$save_file_name.h5", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$save_file_name.h5"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


"""
gen_rand_density_iso - generate a random, normalized, density matrix
in the complex to real isomorphism
"""
function gen_rand_density_iso(seed_)
    Random.seed!(seed_)
    state = rand(STATE_SIZE)
    density = (state * state') / real(state' * state)
    density_r = real(density)
    density_i = imag(density)
    density_iso = SMatrix{DENSITY_SIZE_ISO, DENSITY_SIZE_ISO}([
        density_r -density_i;
        density_i density_r;
    ])
    return density_iso
end


fidelity_mat(m1, m2) = abs(tr(m1' * m2)) / abs(tr(m2' * m2))


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


function get_fbfq(amplitude)
    return -abs(amplitude) * FBFQ_A + FBFQ_B
end


function get_t1_poly(amplitude)
    fbfq = get_fbfq(amplitude)
    t1 = horner(FBFQ_T1_COEFFS, fbfq)
    return t1
end


function dynamics(density, controls)
    gamma = (get_t1_poly(controls[1]))^(-1)
    neg_i_hamiltonian = OMEGA_NEG_I_H_S + controls[1] * NEG_I_2_PI_H_C1
    delta_density = (
        neg_i_hamiltonian * density - density * neg_i_hamiltonian
        + gamma * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2)
        + gamma * (E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


# Define gate set.
ID = SA_F64[1 0 0 0;
            0 1 0 0;
            0 0 1 0;
            0 0 0 1;]
# 4 or 8 applications of Z/2, Y/2, and X/2 yield
GATE_4 = -ID
GATE_8 = ID
GATE_CYCLE_CONSTANT = 4

# Define integration.
DT = 1e-2
DT_INV = 1e2

function rk4_step(x, u)
    k1 = dynamics(x, u) * DT
	k2 = dynamics(x + k1 / 2, u) * DT
	k3 = dynamics(x + k2 / 2, u) * DT
	k4 = dynamics(x + k3, u) * DT
	return x + (k1 + 2 * k2 + 2 * k3 + k4)/6
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


function run_verify(gate_count, gate_type, pulse_type, seed)
    # grab controls
    (controls, gate_time) = grab_controls(gate_type, pulse_type)
    control_knot_count = size(controls)[1]
    density = initial_density = gen_rand_density_iso(seed)
    gate = GT_TO_GATE[gate_type]
    target_density = gate^gate_count * initial_density * gate'^gate_count

    for j = 1:gate_count
        for k = 1:control_knot_count
            density = rk4_step(density, controls[k])
        end
    end
    
    fidelity = fidelity_mat(target_density, density)
    println("fidelity\n$(fidelity)")
    println("initial_density")
    show_nice(initial_density)
    println("\ndensity")
    show_nice(density)
    println("\ntarget_density")
    show_nice(target_density)
    return
end


function run_sim(controls, evolution_time, gate_time, trial_count)
    # set variables
    if MP
        my_rank = MPI.Comm_rank(COMM)
    else
        my_rank = ROOT_RANK
    end
    my_trial_count, over = divrem(trial_count, MAX_RANK + 1)
    if my_rank + 1 <= over
        my_trial_count = my_trial_count + 1
    end
    knot_count = Int(evolution_time * DT_INV)
    gate_knot_count = Int(gate_time * DT_INV)
    gate_count = Int(floor(evolution_time / (4 * gate_time)))
    densities = zeros(my_trial_count, gate_count + 1, DENSITY_SIZE_ISO, DENSITY_SIZE_ISO)
    fidelities = zeros(my_trial_count, gate_count + 1)

    for trial_index = 1:my_trial_count
        # generate initial density
        density = initial_density = gen_rand_density_iso((my_rank + 1) * trial_index)
        densities[trial_index, 1, :, :] = Array(initial_density)
        # integrate and save density
        for gate_index = 2:gate_count + 1
            # Advance the integration by 4 gates.
            for j = 1:4
                for k = 1:gate_knot_count
                    density = rk4_step(density, controls[k])
                end
            end
            # Save the density every 4 gates.
            densities[trial_index, gate_index, :, :] = Array(density)
        end
        # compute fidelity on all densities
        initial_density_h = initial_density'
        initial_density_fnorm = abs(tr(initial_density_h * initial_density))
        fidelity_(density_) = abs(tr(initial_density_h * density_)) / initial_density_fnorm
        fidelities[trial_index, :] = mapslices(fidelity_, densities[trial_index, :, :, :], dims=[2, 3])
    end
    gate_error_avgf = mapslices(x -> 1 - mean(x), fidelities, dims=[1])


    return densities, gate_error_avgf
end


function run_all(evolution_time, gate_type, pulse_type, trial_count)
    if MP
        my_rank = MPI.Comm_rank(COMM)
    else
        my_rank = ROOT_RANK
    end
    
    if my_rank == ROOT_RANK
        start_time = Dates.now()
            
        # grab controls
        (controls, controls_file_path,
         gate_time) = grab_controls(gate_type, pulse_type)

        # send data to peers
        data = (controls, evolution_time, gate_time, trial_count)
        for peer_rank = 1:MAX_RANK
            MPI.send(data, peer_rank, peer_rank, COMM)
        end
        
        # run
        densities, gate_error_avgf = run_sim(controls, evolution_time, gate_time, trial_count)
        # println("densities")
        # show_nice(densities)
        # println("\ngate_error_avgf")
        # show_nice(gate_error_avgf)
        # println("")
        # quit()

        # receive results from peers
        for peer_rank = 1:MAX_RANK
            (result, status) = MPI.recv(peer_rank, peer_rank, COMM)
            (densities_, gate_error_avgf_) = result
            densities = vcat(densities, densities_)
            gate_error_avgf = vcat(gate_error_avgf, gate_error_avgf_)
        end

        # save all trials
        end_time = Dates.now()
        run_time = end_time - start_time
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        h5open(save_file_path, "cw") do save_file
            write(save_file, "densities", densities)
            write(save_file, "gate_error_avgf", gate_error_avgf)
            write(save_file, "evolution_time", evolution_time)
            write(save_file, "controls_file_path", controls_file_path)
            write(save_file, "run_time", string(run_time))
        end
        println("Saved to $(save_file_path)")
    else
        # receive data from root
        (data, status) = MPI.recv(ROOT_RANK, my_rank, COMM)
        (controls, evolution_time, gate_time, trial_count) = data
        
        # run
        result = run_sim(controls, evolution_time, gate_time, trial_count)

        # send result to root
        MPI.send(result, ROOT_RANK, my_rank, COMM)
    end
end


function plot_single(fig, gate_type, pulse_type)
    # grab fidelities
    (gate_error_avgf,) = h5open(OUT_DATA_1[pulse_type][gate_type], "r") do save_file
        gate_error_avgf = read(save_file, "gate_error_avgf")[1, :]
        return (gate_error_avgf,)
    end
    fidelities = map(x -> 1 - x, gate_error_avgf)
    # construct xaxis
    gate_count = size(gate_error_avgf)[1] - 1
    gate_count_axis = Array(0:4:4 * gate_count)
    label = PT_TO_STR[pulse_type]
    title = GT_TO_STR[gate_type]
    Plots.plot!(
        fig, gate_count_axis, fidelities,
        label=label, title=title, dpi=DPI,
        ylims=(0, 1), yticks=(0:0.1:1),
        # markeralpha=MARKER_ALPHA, ms=MARKER_SIZE
    )
end


function plot_all()
    # fig = Plots.plot()
    # plot_single(fig, zpiby2, analytic)
    # plot_single(fig, zpiby2, vanilla)
    # plot_single(fig, zpiby2, t1_m1)
    # Plots.ylabel!(fig,"log10 Gate Error")
    # Plots.xlabel!(fig, "Gate Count")
    # Plots.savefig(fig, GT_TO_PLOT_FILE_PATH[zpiby2])
    
    for gate_type in instances(GateType)
        fig = Plots.plot()
        for pulse_type in instances(PulseType)
            plot_single(fig, gate_type, pulse_type)
        end
        Plots.ylabel!(fig, "Fidelity")
        Plots.xlabel!(fig, "Gate Count")
        Plots.savefig(fig, GT_TO_PLOT_FILE_PATH_1[gate_type])
    end
end


"""
Prepend initial density and recalculate gate error for
the first batch 00008-00016
"""
function initial_density_and_gate_error_avgf_update()
    for pulse_type in instances(PulseType)
        for gate_type in instances(GateType)
            file_path = OUT_DATA[pulse_type][gate_type]
            fd = h5open(file_path, "cw")
            densities_old = read(fd, "densities")
            trial_count = size(densities_old)[1]
            gate_count = size(densities_old)[2]
            fidelities = zeros(trial_count, gate_count + 1)
            densities = zeros(trial_count, gate_count + 1, DENSITY_SIZE_ISO, DENSITY_SIZE_ISO)
            densities[:, 2:gate_count + 1, :, :] = densities_old
            for trial_index in 1:trial_count
                densities[trial_index, 1, :, :] = initial_density = gen_rand_density_iso(trial_index)
                initial_density_h = initial_density'
                initial_density_fnorm = abs(tr(initial_density_h * initial_density))
                fidelity_(density_) = abs(tr(initial_density_h * density_)) / initial_density_fnorm
                fidelities[trial_index, :] = (
                    mapslices(fidelity_, densities[trial_index, :, :, :], dims=[2, 3])
                )
            end
            close(fd)
        end
    end
end


function plot_zpiby2_comparison()
    fig = Plots.plot(dpi=DPI)
    for pulse_type in instances(PulseType)
        if pulse_type == vanilla
            continue
        end
        # grab
        no_dis_file_path = COMPARISON_DATA[pulse_type][NO_DISSIPATION_KEY]
        dis_file_path = COMPARISON_DATA[pulse_type][DISSIPATION_KEY]
        controls_file_path = PULSE_DATA[pulse_type][zpiby2]
        (gate_time,) = h5open(controls_file_path, "r") do fd
            gate_time = read(fd, "evolution_time")
            return (gate_time,)
        end
        (gea_no_dis,) = h5open(no_dis_file_path, "r") do fd
            gea = read(fd, "gate_error_avgf")[1, :]
            return (gea,)
        end
        (gea_dis,) = h5open(dis_file_path, "r") do fd
            gea = read(fd, "gate_error_avgf")[1, :]
            return (gea,)
        end

        # plot
        pulse_label = PT_TO_STR[pulse_type]
        gate_count = size(gea_dis)[1]
        gate_axis = Array(0:4:4 * (gate_count - 1))
        fid_dis = map(x -> 1 - x, gea_dis)
        fid_no_dis = map(x -> 1 - x, gea_no_dis)
        Plots.plot!(fig, gate_axis, fid_dis, label="$(pulse_label) Dissipation")
        Plots.plot!(fig, gate_axis, fid_no_dis, label="$(pulse_label)")
    end
    Plots.ylabel!(fig, "Fidelity")
    Plots.xlabel!(fig, "Gate Count")
    Plots.savefig(fig, ZPIBY2_COMPARISON_FILE_PATH)
end
