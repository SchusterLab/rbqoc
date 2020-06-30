"""
spin15_bench.jl - lindblad evolution for spin15.jl in Julia
"""

using Dates
using HDF5
using LaTeXStrings
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

# plotting configuration
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300


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



# Define other constants.

@enum PulseType begin
    analytic = 1
    vanilla = 2
    t1_sense = 3
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
        CIDX_KEY => 9,
    ),
    t1_sense => Dict(
        zpiby2 => joinpath(META_PATH, "spin15", "00023_spin15.h5"),
        ypiby2 => joinpath(META_PATH, "spin15", "00015_spin15.h5"),
        xpiby2 => joinpath(META_PATH, "spin15", "00024_spin15.h5"),
        CIDX_KEY => 9,
    ),
)

# multiprocessing
MP = true
COMM = nothing
MAX_RANK = 0
ROOT_RANK = 0
if MP
    MPI.Init()
    COMM = MPI.COMM_WORLD
    MAX_RANK = MPI.Comm_size(COMM) - 1
end


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


function run_sim(controls, evolution_time, gate_time, trial_count)
    # set variables
    my_rank = MPI.Comm_rank(COMM)
    # my_rank = ROOT_RANK
    my_trial_count, over = divrem(trial_count, MAX_RANK + 1)
    if my_rank + 1 <= over
        my_trial_count = my_trial_count + 1
    end
    knot_count = Int(evolution_time * DT_INV)
    gate_knot_count = Int(gate_time * DT_INV)
    gate_count = Int(floor(evolution_time / (4 * gate_time)))
    densities = zeros(my_trial_count, gate_count, DENSITY_SIZE_ISO, DENSITY_SIZE_ISO)
    fidelities = zeros(my_trial_count, gate_count)
    
    for trial_index = 1:my_trial_count
        # generate initial density
        density = initial_density = gen_rand_density_iso((my_rank + 1) * trial_index)
        # integrate and save density
        for gate_index = 1:gate_count
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
        id_adjoint = initial_density'
        id_fnorm = tr(id_adjoint * initial_density)
        fidelity_(density_) = tr(id_adjoint * density_) / id_fnorm
        fidelities[trial_index, :] = mapslices(fidelity_, densities[trial_index, :, :, :], dims=[2, 3])
    end

    return densities, fidelities
end


function run_all(evolution_time, gate_type, pulse_type, trial_count)
    my_rank = MPI.Comm_rank(COMM)
    # my_rank = ROOT_RANK
    
    if my_rank == ROOT_RANK
        start_time = Dates.now()
            
        # grab controls
        controls_file_path = PULSE_DATA[pulse_type][gate_type]
        controls_idx = PULSE_DATA[pulse_type][CIDX_KEY]
        (
            controls,
            gate_time,
        ) = h5open(controls_file_path, "r") do save_file
            if controls_idx == NON_TRAJ_OPT_PULSE
                controls = permutedims(read(save_file, "controls"), [2, 1])
                gate_time = read(save_file, "evolution_time")
            else
                controls = read(save_file, "states")[:, controls_idx]
                gate_time = read(save_file, "evolution_time")
            end

            return (controls, gate_time)
        end
        controls = controls / (2 * pi)

        # send data to peers
        data = (controls, evolution_time, gate_time, trial_count)
        for peer_rank = 1:MAX_RANK
            MPI.send(data, peer_rank, peer_rank, COMM)
        end
        
        # run
        densities, fidelities = run_sim(controls, evolution_time, gate_time, trial_count)

        # receive fidelities from peers
        for peer_rank = 1:MAX_RANK
            (densities_, fidelities_), status = MPI.recv(peer_rank, peer_rank, COMM)
            densities = vcat(densities, densities_)
            fidelities = vcat(fidelities, fidelities_)
        end

        # save all trials
        end_time = Dates.now()
        run_time = end_time - start_time
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        h5open(save_file_path, "cw") do save_file
            write(save_file, "densities", densities)
            write(save_file, "fidelities", fidelities)
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
        fidelities = run_sim(controls, evolution_time, gate_time, trial_count)

        # send fidelities to root
        MPI.send(fidelities, ROOT_RANK, my_rank, COMM)
    end
    MPI.Barrier(COMM)
end


function main()
    evolution_time = 160
    gate_type = zpiby2
    pulse_type = analytic
    trial_count = 2
    run_all(evolution_time, gate_type, pulse_type, trial_count)
end



