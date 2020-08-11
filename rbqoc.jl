"""
rbqoc.jl - common definitions for the rbqoc repo
"""

# imports
using Dates
using Dierckx
using DifferentialEquations
using ForwardDiff
using HDF5
using Interpolations
using LinearAlgebra
using Plots
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
DPI_FINAL = Integer(2e3)
MS_SMALL = 2
MS_MED = 6
ALPHA = 0.2

# other constants
DEQJL_MAXITERS = 1e10
DEQJL_ADAPTIVE = false

# types
@enum DynamicsType begin
    schroed = 1
    lindbladnodis = 2
    lindbladdis = 3
    ypiby2nodis = 4
    ypiby2dis = 5
    xpiby2nodis = 6
    xpiby2dis = 7
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


@enum SolverType begin
    ilqr = 1
    alilqr = 2
    altro = 3
end


DT_STR = Dict(
    schroed => "Schroedinger",
    lindbladnodis => "Lindblad No Dissipation",
    lindbladdis => "Lindblad Dissipation",
)

GT_STR = Dict(
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


function plot_controls(save_file_paths, plot_file_path;
                       save_types=[jl,], labels=nothing,
                       title="", colors=nothing, print_out=true,
                       legend=nothing)
    fig = Plots.plot(dpi=DPI, title=title, legend=legend)
    for (i, save_file_path) in enumerate(save_file_paths)
        # Grab and prep data.
        (controls, evolution_time) = grab_controls(save_file_path; save_type=save_types[i])
        controls = controls ./ (2 * pi)
        (control_eval_count, control_count) = size(controls)
        control_eval_times = Array(1:1:control_eval_count) * DT_PREF
        
        # Plot.
        for j = 1:control_count
            if labels == nothing
                label = nothing
            else
                label = labels[i][j]
            end
            if colors == nothing
                color = :auto
            else
                color = colors[i][j]
            end
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


show_nice(x) = show(IOContext(stdout), "text/plain", x)


### SPIN ###

# Define experimental constants.
# qubit frequency at flux frustration point
FQ = 1.4e-2 #GHz
SIGMAFQ = FQ * 5e-2
S1FQ = FQ + SIGMAFQ
S2FQ = FQ - SIGMAFQ
MAX_CONTROL_NORM_0 = 5e-1 #GHz
FBFQ_A = 0.202407
FBFQ_B = 0.5
AYPIBY2 = 1.25e-1 #GHz
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
# ISO indicates the object is defined in the complex to real isomorphism.
# NEGI is the negative complex unit.
NEGI = SA_F64[0   0  1  0 ;
              0   0  0  1 ;
              -1  0  0  0 ;
              0  -1  0  0 ;]
# SIGMAX, SIGMAZ are the X and Z pauli matrices
SIGMAX_ISO = SA_F64[0   1   0   0;
                    1   0   0   0;
                    0   0   0   1;
                    0   0   1   0]
SIGMAZ_ISO = SA_F64[1   0   0   0;
                    0  -1   0   0;
                    0   0   1   0;
                    0   0   0  -1]
NEGI_H0_ISO = pi * NEGI * SIGMAZ_ISO
NEGI_H1_ISO = pi * NEGI * SIGMAX_ISO
FQ_NEGI_H0_ISO = FQ * NEGI_H0_ISO
S1FQ_NEGI_H0_ISO = S1FQ * NEGI_H0_ISO
S2FQ_NEGI_H0_ISO = S2FQ * NEGI_H0_ISO
AYPIBY2_NEGI_H1_ISO = AYPIBY2 * NEGI_H1_ISO
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
ZPIBY2 = [1-1im 0;
          0 1+1im] / sqrt(2)
ZPIBY2_ISO = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}(get_mat_iso(ZPIBY2))
ZPIBY2_ISO_1 = SVector{STATE_SIZE_ISO}(ZPIBY2[:,1])
ZPIBY2_ISO_2 = SVector{STATE_SIZE_ISO}(ZPIBY2[:,2])
YPIBY2 = [1 -1;
          1  1] / sqrt(2)
YPIBY2_ISO = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}(get_mat_iso(YPIBY2))
YPIBY2_ISO_1 = SVector{STATE_SIZE_ISO}(YPIBY2[:,1])
YPIBY2_ISO_2 = SVector{STATE_SIZE_ISO}(YPIBY2[:,2])
XPIBY2 = [1 -1im;
          -1im 1] / sqrt(2)
XPIBY2_ISO = SMatrix{STATE_SIZE_ISO, STATE_SIZE_ISO}(get_mat_iso(XPIBY2))
XPIBY2_ISO_1 = SVector{STATE_SIZE_ISO}(XPIBY2[:,1])
XPIBY2_ISO_2 = SVector{STATE_SIZE_ISO}(XPIBY2[:,2])

GT_GATE = Dict(
    xpiby2 => XPIBY2_ISO,
    ypiby2 => YPIBY2_ISO,
    zpiby2 => ZPIBY2_ISO,
)


# methods

"""
amp_fbfq - Compute flux by flux quantum. Reflects
over the flux frustration point.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
amp_fbfq(amplitude) = -abs(amplitude) * FBFQ_A + FBFQ_B


"""
fbfq_amp - Compute the amplitude from the flux by
flux quantum. Reflects over the flux frustration point.
"""
fbfq_amp(fbfq) = (fbfq - FBFQ_B) / FBFQ_A


"""
amp_t1_poly - Compute the t1 time for the given amplitude in units
of nanoseconds.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
amp_t1_poly(amplitude) = horner(FBFQ_T1_COEFFS, amp_fbfq(amplitude))


"""
amp_t1_spline - Compute the t1 time in nanoseconds
for the given amplitude.

Arguments
amplitude :: Array(N) - amplitude in units of GHz (no 2 pi)
"""
# amp_t1_spline(amplitude::Float64) = Dierckx.evaluate(FBFQ_T1_SPLINE_DIERCKX, amp_fbfq(amplitude))
# damp_t1_spline(amplitude::Float64) = Dierckx.derivative(FBFQ_T1_SPLINE_DIERCKX, amp_fbfq(amplitude))
amp_t1_spline(amplitude) = FBFQ_T1_SPLINE_ITP(amp_fbfq(amplitude))


"""
Schroedinger dynamics.
"""
function dynamics_schroed_deqjl(state, (controls, control_knot_count, dt_inv, negi_h0), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    negi_h = (
        negi_h0
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        negi_h * state
    )
end


function dynamics_lindbladdis_deqjl(density, (controls, control_knot_count, dt_inv, negi_h0), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    gamma_1 = (amp_t1_spline(controls[knot_point][1]))^(-1)
    negi_h = (
        FQ_NEGI_H0_ISO
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        negi_h * density - density * negi_h
        + gamma_1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2)
        + gamma_1 * (E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


function dynamics_lindbladnodis_deqjl(density, (controls, control_knot_count, dt_inv, negi_h0), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    negi_h = (
        FQ_NEGI_H0_ISO
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        negi_h * density - density * negi_h
    )
end


H1_YPIBY2 = FQ_NEGI_H0_ISO + AYPIBY2_NEGI_H1_ISO
H2_YPIBY2 = FQ_NEGI_H0_ISO
H3_YPIBY2 = FQ_NEGI_H0_ISO - AYPIBY2_NEGI_H1_ISO
TX_YPIBY2 = 2.1656249366575766
TZ_YPIBY2 = 15.1423305995572655
TTOT_YPIBY2 = 19.4735804728724204
T1_YPIBY2 = TX_YPIBY2
T2_YPIBY2 = T1_YPIBY2 + TZ_YPIBY2
GAMMA11_YPIBY2 = amp_t1_spline(AYPIBY2)^(-1)
GAMMA12_YPIBY2 = amp_t1_spline(0)^(-1)
function dynamics_ypiby2nodis_deqjl(density, (controls, control_knot_count, dt_inv, negi_h0), t)
    t = t - Int(floor(t / TTOT_YPIBY2)) * TTOT_YPIBY2
    if t <= T1_YPIBY2
        negi_h = H1_YPIBY2
    elseif t <= T2_YPIBY2
        negi_h = H2_YPIBY2
    else
        negi_h = H3_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
    )
end


function dynamics_ypiby2dis_deqjl(density, (controls, control_knot_count, dt_inv, negi_h0), t)
    t = t - Int(floor(t / TTOT_YPIBY2)) * TTOT_YPIBY2
    if t <= T1_YPIBY2
        negi_h = H1_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T2_YPIBY2
        negi_h = H2_YPIBY2
        gamma1 = GAMMA12_YPIBY2
    else
        negi_h = H3_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
        + gamma1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2
                    + E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


TTOT_ZPIBY2 = 17.857142857142858
TTOT_XPIBY2 = 4 * TX_YPIBY2 + 2 * TZ_YPIBY2 + TTOT_ZPIBY2
T1_XPIBY2 = TX_YPIBY2
T2_XPIBY2 = T1_YPIBY2 + TZ_YPIBY2
T3_XPIBY2 = T2_XPIBY2 + TX_YPIBY2
T4_XPIBY2 = T3_XPIBY2 + TTOT_ZPIBY2
T5_XPIBY2 = T4_XPIBY2 + TX_YPIBY2
T6_XPIBY2 = T5_XPIBY2 + TZ_YPIBY2
function dynamics_xpiby2nodis_deqjl(density, (controls, control_knot_count, dt_inv), t)
    t = t - Int(floor(t / TTOT_XPIBY2)) * TTOT_XPIBY2
    if t <= T1_XPIBY2
        negi_h = H1_YPIBY2
    elseif t <= T2_XPIBY2
        negi_h = H2_YPIBY2
    elseif t <= T3_XPIBY2
        negi_h = H3_YPIBY2
    elseif t <= T4_XPIBY2
        negi_h = H2_YPIBY2
    elseif t <= T5_XPIBY2
        negi_h = H3_YPIBY2
    elseif t <= T6_XPIBY2
        negi_h = H2_YPIBY2
    else
        negi_h = H1_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
    )
end


function dynamics_xpiby2dis_deqjl(density, (controls, control_knot_count, dt_inv), t)
    t = t - Int(floor(t / TTOT_XPIBY2)) * TTOT_XPIBY2
    if t <= T1_XPIBY2
        negi_h = H1_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T2_XPIBY2
        negi_h = H2_YPIBY2
        gamma1 = GAMMA12_YPIBY2
    elseif t <= T3_XPIBY2
        negi_h = H3_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T4_XPIBY2
        negi_h = H2_YPIBY2
        gamma1 = GAMMA12_YPIBY2
    elseif t <= T5_XPIBY2
        negi_h = H3_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    elseif t <= T6_XPIBY2
        negi_h = H2_YPIBY2
        gamma1 = GAMMA12_YPIBY2
    else
        negi_h = H1_YPIBY2
        gamma1 = GAMMA11_YPIBY2
    end
    return(
        negi_h * density - density * negi_h
        + gamma1 * (G_E * density * E_G + NEG_E_E_BY2 * density + density * NEG_E_E_BY2
                    + E_G * density * G_E + NEG_G_G_BY2 * density + density * NEG_G_G_BY2)
    )
end


@inline fidelity_vec_iso2(s1, s2) = (
    (s1's2)^2 + (s1[1] * s2[3] + s1[2] * s2[4] - s1[3] * s2[1] - s1[4] * s2[2])^2
)


@inline fidelity_mat_iso(m1, m2) = abs(tr(m1' * m2)) / abs(tr(m2' * m2))


@inline fidelity_mat_iso2(m1, m2) = 0


function gen_rand_state_iso(;seed=0)
    if seed == -1
        state = [1, 1] / sqrt(2)
    else
        Random.seed!(seed)
        state = rand(STATE_SIZE_NOISO) + 1im * rand(STATE_SIZE_NOISO)
    end
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


function get_vec_iso(vec)
    return vcat(real(vec),
                imag(vec))
end


function get_mat_iso(mat)
    len = size(mat)[1]
    mat_r = real(mat)
    mat_i = imag(mat)
    return vcat(hcat(mat_r, -mat_i),
                hcat(mat_i,  mat_r))
end


"""
run_sim_deqjl - Apply a gate multiple times and measure the fidelity
after each application. Save the output.

Arguments:
save_file_path :: String - The file path to grab the controls from
"""
function run_sim_deqjl(
    gate_count, gate_type;
    save_file_path=nothing,
    controls_dt_inv=DT_PREF_INV,
    deqjl_adaptive=false, dynamics_type=lindbladnodis,
    dt=DT_PREF, save=true, save_type=jl, seed=-1,
    solver=DifferentialEquations.Vern9, print_seq=false, print_final=false,
    negi_h0=FQ_NEGI_H0_ISO)
    start_time = Dates.now()
    # grab
    if isnothing(save_file_path)
        controls = control_knot_count = nothing
        if dynamics_type == ypiby2nodis || dynamics_type == ypiby2dis
            gate_time = TTOT_YPIBY2
        elseif dynamics_type == xpiby2nodis || dynamics_type == xpiby2dis
            gate_time = TTOT_XPIBY2
        end
    else
        (controls, gate_time) = grab_controls(save_file_path; save_type=save_type)
        # controls = controls ./ (2 * pi)
        control_knot_count = Int(floor(gate_time * controls_dt_inv))
    end
    save_times = Array(0:1:gate_count) * gate_time
    
    # integrate
    if dynamics_type == lindbladnodis
        f = dynamics_lindbladnodis_deqjl
    elseif dynamics_type == lindbladdis
        f = dynamics_lindbladdis_deqjl
    elseif dynamics_type == schroed
        f = dynamics_schroed_deqjl
    elseif dynamics_type == ypiby2nodis
        f = dynamics_ypiby2nodis_deqjl
    elseif dynamics_type == ypiby2dis
        f = dynamics_ypiby2dis_deqjl
    elseif dynamics_type == xpiby2nodis
        f = dynamics_xpiby2nodis_deqjl
    elseif dynamics_type == xpiby2dis
        f = dynamics_xpiby2dis_deqjl
    end
    is_state = is_density = false
    if dynamics_type == schroed
        is_state = true
    else
        is_density = true
    end
    if is_state
        initial_state = gen_rand_state_iso(;seed=seed)
    elseif is_density
        initial_state =  gen_rand_density_iso(;seed=seed)
    end
    tspan = (0., gate_time * gate_count)
    p = (controls, control_knot_count, controls_dt_inv, negi_h0)
    prob = ODEProblem(f, initial_state, tspan, p)
    result = solve(prob, solver(), dt=dt, saveat=save_times,
                   maxiters=DEQJL_MAXITERS, adaptive=DEQJL_ADAPTIVE)

    # Compute the fidelities.
    # All of the gates we consider are 4-cyclic up to phase.
    fidelities = zeros(gate_count + 1)
    g1 = GT_GATE[gate_type]
    g2 = g1^2
    g3 = g1^3
    id0 = initial_state
    if is_state
        states = zeros(gate_count + 1, STATE_SIZE_ISO)
        id1 = g1 * id0
        id2 = g2 * id0
        id3 = g3 * id0
    elseif is_density
        states = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
        id1 = g1 * id0 * g1'
        id2 = g2 * id0 * g2'
        id3 = g3 * id0 * g3'
    end
    id0_dag = id0'
    id1_dag = id1'
    id2_dag = id2'
    id3_dag = id3'
    id1_fnorm = abs(tr(id1_dag * id1))
    id2_fnorm = abs(tr(id2_dag * id2))
    id3_fnorm = abs(tr(id3_dag * id3))
    id0_fnorm = abs(tr(id0_dag * id0))
    # Compute the fidelity after each gate.
    for i = 1:gate_count + 1
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
        if is_state
            states[i, :] = state = result.u[i]
            fidelities[i] = abs(target_dag * state)^2
        elseif is_density
            states[i, :, :] = state = result.u[i]
            fidelities[i] = abs(tr(target_dag * state)) / target_fnorm
        end

        if print_seq
            println("fidelities[$(i)]\n$(fidelities[i])")
            println("state")
            show_nice(state)
            println("")
            println("target")
            show_nice(target_dag')
            println("")
        end
    end
    end_time = Dates.now()
    run_time = end_time - start_time
    if print_final
        println("fidelities[$(gate_count)]: $(fidelities[end])")
    end

    # Save the data.
    experiment_name = save_path = nothing
    if isnothing(save_file_path)
        if (dynamics_type == ypiby2nodis || dynamics_type == ypiby2dis
            || dynamics_type == xpiby2nodis || dynamics_type == xpiby2dis)
            experiment_name = "spin14"
            save_path = joinpath(ENV["RBQOC_PATH"], "out", "spin", "spin14")
        end
    else
        experiment_name = split(save_file_path, "/")[end - 1]
        save_path = dirname(save_file_path)
    end
    data_file_path = nothing
    if save
        data_file_path = generate_save_file_path("h5", experiment_name, save_path)
        h5open(data_file_path, "w") do data_file
            write(data_file, "dynamics_type", Integer(dynamics_type))
            write(data_file, "gate_count", gate_count)
            write(data_file, "gate_time", gate_time)
            write(data_file, "gate_type", Integer(gate_type))
            write(data_file, "save_file_path", isnothing(save_file_path) ? "" : save_file_path)
            write(data_file, "seed", seed)
            write(data_file, "states", states)
            write(data_file, "fidelities", fidelities)
            write(data_file, "run_time", string(run_time))
            write(data_file, "dt", dt)
            write(data_file, "negi_h0", negi_h0)
        end
        println("Saved simulation to $(data_file_path)")
    end
    return data_file_path
end


"""
1 gate, many h0s
"""
function run_sim_h0sweep_deqjl(
    gate_type, negi_h0s;
    save_file_path=nothing,
    controls_dt_inv=DT_PREF_INV,
    deqjl_adaptive=false, dynamics_type=schroed,
    dt=DT_PREF, save=true, save_type=jl, seed=-1,
    solver=DifferentialEquations.Vern9, print_seq=false)
    
    start_time = Dates.now()
    # grab
    if isnothing(save_file_path)
        controls = control_knot_count = nothing
        if dynamics_type == ypiby2nodis || dynamics_type == ypiby2dis
            gate_time = TTOT_YPIBY2
        elseif dynamics_type == xpiby2nodis || dynamics_type == xpiby2dis
            gate_time = TTOT_XPIBY2
        end
    else
        (controls, gate_time) = grab_controls(save_file_path; save_type=save_type)
        # controls = controls ./ (2 * pi)
        control_knot_count = Int(floor(gate_time * controls_dt_inv))
    end
    save_times = [0., gate_time]
    
    # set up integration
    if dynamics_type == lindbladnodis
        dynamics = dynamics_lindbladnodis_deqjl
    elseif dynamics_type == lindbladdis
        dynamics = dynamics_lindbladdis_deqjl
    elseif dynamics_type == schroed
        dynamics = dynamics_schroed_deqjl
    elseif dynamics_type == ypiby2nodis
        dynamics = dynamics_ypiby2nodis_deqjl
    elseif dynamics_type == ypiby2dis
        dynamics = dynamics_ypiby2dis_deqjl
    elseif dynamics_type == xpiby2nodis
        dynamics = dynamics_xpiby2nodis_deqjl
    elseif dynamics_type == xpiby2dis
        dynamics = dynamics_xpiby2dis_deqjl
    end
    is_state = is_density = false
    if dynamics_type == schroed
        is_state = true
    else
        is_density = true
    end
    if is_state
        initial_state = gen_rand_state_iso(;seed=seed)
    elseif is_density
        initial_state =  gen_rand_density_iso(;seed=seed)
    end
    tspan = (0., gate_time)

    # integrate and compute fidelity
    sample_count = size(negi_h0s)[1]
    fidelities = zeros(sample_count)
    gate = GT_GATE[gate_type]
    if is_state
        target_state = gate * initial_state
    elseif is_density
        target_state = gate * initial_state * gate'
    end
    for i = 1:sample_count
        dargs = (controls, control_knot_count, controls_dt_inv, negi_h0s[i])
        prob = ODEProblem(dynamics, initial_state, tspan, dargs)
        result = solve(prob, solver(), dt=dt, saveat=save_times,
                       maxiters=DEQJL_MAXITERS, adaptive=DEQJL_ADAPTIVE)
        final_state = result.u[end]
        if is_state
            fidelities[i] = fidelity_vec_iso2(final_state, target_state)
        elseif is_density
            fidelities[i] = fidelity_mat_iso2(final_state, target_state)
        end
        if print_seq
            println("fidelities[$(i)] = $(fidelities[i])")
        end
    end
    end_time = Dates.now()
    run_time = end_time - start_time
    
    # save
    experiment_name = save_path = nothing
    if isnothing(save_file_path)
        if (dynamics_type == ypiby2nodis || dynamics_type == ypiby2dis
            || dynamics_type == xpiby2nodis || dynamics_type == xpiby2dis)
            experiment_name = "spin14"
            save_path = joinpath(ENV["RBQOC_PATH"], "out", "spin", "spin14")
        end
    else
        experiment_name = split(save_file_path, "/")[end - 1]
        save_path = dirname(save_file_path)
    end
    data_file_path = nothing
    if save
        data_file_path = generate_save_file_path("h5", experiment_name, save_path)
        h5open(data_file_path, "cw") do data_file
            write(data_file, "dynamics_type", Integer(dynamics_type))
            write(data_file, "gate_time", gate_time)
            write(data_file, "gate_type", Integer(gate_type))
            write(data_file, "save_file_path", isnothing(save_file_path) ? "" : save_file_path)
            write(data_file, "seed", seed)
            write(data_file, "fidelities", fidelities)
            write(data_file, "run_time", string(run_time))
            write(data_file, "dt", dt)
        end
        println("Saved run_sim_h0sweep_deqjl to $(data_file_path)")
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
    if "dt_idx" in keys(save)
        dts = save["acontrols"][1:end, save["dt_idx"]]
    elseif "dt" in keys(save)
        dts = save["dt"] * ones(control_knot_count)
    end
    time_axis = [0; cumsum(dts, dims=1)[1:end - 1]]

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
        Plots.scatter!(time_axis, controls[:, 1], label="controls data", markersize=MS_SMALL, alpha=ALPHA)
        Plots.scatter!(time_axis_sample, controls_sample[:, 1], label="controls fit",
                       markersize=MS_SMALL, alpha=ALPHA)
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
        t1s = map(amp_t1_spline, controls[:, i] / (2 * pi))
        t1_avgs[i] = mean(t1s)
    end
    
    return t1_avgs
end
