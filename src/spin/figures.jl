"""
figures.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using LaTeXStrings
using Printf
import Plots
using Statistics

# paths
const EXPERIMENT_NAME = "figures"
const SAVE_PATH = joinpath(SPIN_OUT_PATH, EXPERIMENT_NAME)
const F3C_DATA_FILE_PATH = joinpath(SAVE_PATH, "f3c.h5")
const F3D_DATA_FILE_PATH = joinpath(SAVE_PATH, "f3d.h5")

# types
@enum PulseType begin
    analytic = 1
    qoc = 2
    s2 = 3
    s4 = 4
    d2 = 5
    d3 = 6
    s2b = 7
    s4b = 8
    d2b = 9
    d3b = 10
    corpse = 11
    d1 = 12
    sut8 = 13
    d1b = 14
    sut8b = 15
    d1bb = 16
    d1bbb = 17
end

const GT_LIST = [zpiby2, ypiby2, xpiby2]

# common dict keys
const SAVEFP_KEY = 1
const SAVET_KEY = 2
const DATAFP_KEY = 3
const COLOR_KEY = 4
const ACORDS_KEY = 5
const MARKER_KEY = 6
const LCORDS_KEY = 7
const DATA2FP_KEY = 8
const INVAL = 99999

### FIGURE 1 ###

const F1_DATA = Dict(
    zpiby2 => Dict(
        qoc => joinpath(SPIN_OUT_PATH, "spin15/00209_spin15.h5"),
        analytic => joinpath(SPIN_OUT_PATH, "spin14/00000_spin14.h5"),
    ),
    ypiby2 => Dict(
        qoc => joinpath(SPIN_OUT_PATH, "spin15/00205_spin15.h5"),
        analytic => joinpath(SPIN_OUT_PATH, "spin14/00003_spin14.h5"),
    ),
    xpiby2 => Dict(
        qoc => joinpath(SPIN_OUT_PATH, "spin15/00239_spin15.h5"),
        analytic => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
    ),
)
const F1_PT_LIST = [analytic, qoc]


function gen_1a()
    gate_types = [Integer(gate_type) for gate_type in GT_LIST]
    pulse_types = [Integer(pulse_type) for pulse_type in F1_PT_LIST]
    save_file_paths = Array{String, 1}([])
    for (i, gate_type) in enumerate(GT_LIST)
        for (j, pulse_type) in enumerate(F1_PT_LIST)
            save_file_path = F1_DATA[gate_type][pulse_type]
            push!(save_file_paths, save_file_path)
        end
    end

    data_file_path = generate_file_path("h5", "f1a", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "gate_types", gate_types)
        write(data_file, "pulse_types", pulse_types)
    end
    println("Saved f1a data to $(data_file_path)")
end


const F1B_SAMPLE_COUNT = Integer(5e2)
function gen_1b()
    gate_types = GT_LIST
    gate_types_integer = [Integer(gt) for gt in gate_types]
    pulse_types = F1_PT_LIST
    pulse_types_integer = [Integer(pt) for pt in pulse_types]
    amps_fit = Array(range(0, stop=MAX_CONTROL_NORM_0, length=F1B_SAMPLE_COUNT))
    t1s_fit =  map(amp_t1_spline_cubic, amps_fit)
    amps_data = -1 .* map(fbfq_amp, FBFQ_ARRAY)
    t1s_data = T1_ARRAY
    t1s_data_err = T1_ARRAY_ERR
    avg_amps = Array{Float64, 1}([])
    avg_amps_t1 = Array{Float64, 1}([])

    for (i, gate_type) in enumerate(gate_types)
        for (j, pulse_type) in enumerate(pulse_types)
            save_file_path = F1_DATA[gate_type][pulse_type]
            (controls, controls_dt_inv, evolution_time) = grab_controls(save_file_path)
            avg_amp = mean(map(abs, controls))
            avg_amp_t1 = amp_t1_spline_cubic(avg_amp)
            push!(avg_amps, avg_amp)
            push!(avg_amps_t1, avg_amp_t1)
        end
    end
    
    data_file_path = generate_file_path("h5", "f1b", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "amps_fit", amps_fit)
        write(data_file, "t1s_fit", t1s_fit)
        write(data_file, "amps_data", amps_data)
        write(data_file, "t1s_data", t1s_data)
        write(data_file, "t1s_data_err", t1s_data_err)
        write(data_file, "avg_amps", avg_amps)
        write(data_file, "avg_amps_t1", avg_amps_t1)
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "gate_types", gate_types_integer)
    end
    println("Saved f1b data to $(data_file_path)")
end


const F1C_GATE_COUNT = Integer(1.6e3)
const F1C_AVG_COUNT = 10
const F1C_DT = 1e-3
function gen_1c(;use_previous=true)
    gate_types_integer = [Integer(gt) for gt in GT_LIST]
    gate_type_count = size(GT_LIST)[1]
    pulse_types_integer = [Integer(pt) for pt in F1_PT_LIST]
    pulse_type_count = size(F1_PT_LIST)[1]
    gate_errors = zeros(gate_type_count, pulse_type_count, F1C_AVG_COUNT, F1C_GATE_COUNT + 1)
    save_file_paths = Array{String, 2}(undef, gate_type_count, pulse_type_count)

    # check for previous computation
    if use_previous
        data_file_path_old = latest_file_path("h5", "f1c", SAVE_PATH)
    else
        data_file_path_old = nothing
    end
    if isnothing(data_file_path_old)
        (save_file_paths_old = gate_errors_old = gate_type_count_old = pulse_type_count_old
         = avg_count_old = nothing)
    else
        (save_file_paths_old, gate_errors_old,
         ) = h5open(data_file_path_old, "r") do data_file_old
             save_file_paths_old = read(data_file_old, "save_file_paths")
             gate_errors_old = read(data_file_old, "gate_errors")
             gate_type_count_old = size(gate_errors_old)[1]
             pulse_type_count_old = size(gate_errors_old)[2]
             avg_count_old = size(gate_errors_old)[3]
             gate_count_old = size(gate_errors_old)[4] - 1
             if gate_count_old != F1C_GATE_COUNT
                 save_file_paths_old = gate_errors_old = nothing
             end
             return (save_file_paths_old, gate_errors_old, gate_type_count_old,
                     pulse_type_count_old, avg_count_old)
         end
    end
    
    for (i, gate_type) in enumerate(GT_LIST)
        println("gt[$(i)]: $(gate_type)")
        for (j, pulse_type) in enumerate(F1_PT_LIST)
            print("pt[$(j)]: $(pulse_type) ")
            save_file_path = F1_DATA[gate_type][pulse_type]
            save_file_paths[i, j] = save_file_path
            save_file_path_sim = pulse_type == analytic ? nothing : save_file_path
            save_file_path_old = ((isnothing(save_file_paths_old) || i > gate_type_count_old
                                   || j > pulse_type_count_old)
                                  ? nothing : save_file_paths_old[i, j])
            if pulse_type == analytic
                if gate_type == zpiby2
                    dynamics_type = zpiby2t1
                elseif gate_type == ypiby2
                    dynamics_type = ypiby2t1
                elseif gate_type == xpiby2
                    dynamics_type = xpiby2t1
                end
            else
                dynamics_type = lindbladt1
            end
            for k = 1:F1C_AVG_COUNT
                # skip redundant computation
                if (!isnothing(save_file_path_old) && save_file_path == save_file_path_old
                    && k <= avg_count_old)
                    gate_errors[i, j, k, :] = gate_errors_old[i, j, k, :]
                    print("s")
                else
                    res = run_sim_deqjl(
                        F1C_GATE_COUNT, gate_type; dynamics_type=dynamics_type,
                        save_file_path=save_file_path_sim, save=false, dt=F1C_DT,
                        seed=k
                    )
                    gate_errors[i, j, k, :] = 1 .- res["fidelities"]
                    print(".")
                end
            end
            println("")
        end
    end
    
    data_file_path = generate_file_path("h5", "f1c", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "gate_types", gate_types_integer)
        write(data_file, "gate_errors", gate_errors)
        write(data_file, "save_file_paths", save_file_paths)
    end
    println("Saved f1c data to $(data_file_path)")
end


### FIGURE 2 ###
const F2A_DATA_ZPIBY2 = Dict(
    analytic => joinpath(SPIN_OUT_PATH, "spin14/00000_spin14.h5"),
    s2 => joinpath(SPIN_OUT_PATH, "spin12/00731_spin12.h5"), #"spin12/00692_spin12.h5"),
    sut8 => "", #joinpath(SPIN_OUT_PATH, "spin23/00094_spin23.h5"),
    d1 => joinpath(SPIN_OUT_PATH, "spin11/00539_spin11.h5"), #"spin11/00468_spin11.h5"),
    d2 => joinpath(SPIN_OUT_PATH, "spin11/00547_spin11.h5"), #"spin11/00487_spin11.h5"),
)
const F2A_PT_LIST = [analytic, s2, sut8, d1, d2]
function gen_2a(;gate_type=zpiby2)
    if gate_type == zpiby2
        data = F2A_DATA_ZPIBY2
    end
    pulse_types_integer = [Integer(pulse_type) for pulse_type in F2A_PT_LIST]
    pulse_type_count = size(F2A_PT_LIST)[1]
    save_file_paths = Array{String, 1}(undef, pulse_type_count)
    for (i, pulse_type) in enumerate(F2A_PT_LIST)
        save_file_paths[i] = data[pulse_type]
    end

    data_file_path = generate_file_path("h5", "f2a", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "gate_type", Integer(gate_type))
    end
    println("Saved f2a data to $(data_file_path)")
end



const F2B_DATA_ZPIBY2 = Dict(
    analytic => joinpath(SPIN_OUT_PATH, "spin14/00000_spin14.h5"),
    d1 => joinpath(SPIN_OUT_PATH, "spin11/00468_spin11.h5"),
    d1b => joinpath(SPIN_OUT_PATH, "spin11/00539_spin11.h5"),
    d1bb => joinpath(SPIN_OUT_PATH, "spin11/00556_spin11.h5"),
    d1bbb => joinpath(SPIN_OUT_PATH, "spin11/00585_spin11.h5"),
)
const F2B_TRIAL_COUNT = Integer(1e2)
const F2B_FQ_DEV = 3e-2
const F2B_PT_LIST = [analytic, d1, d1b, d1bb, d1bbb]
const F2B_AVG_COUNT = 10
function gen_2b(;use_previous=true, gate_type=zpiby2)
    @assert iseven(F2B_TRIAL_COUNT)
    if gate_type == zpiby2
        data = F2B_DATA_ZPIBY2
        dynamics_type_analytic = zpiby2nodis
    end
    pulse_types_integer = [Integer(pt) for pt in F2B_PT_LIST]
    pulse_type_count = size(F2B_PT_LIST)[1]
    fq_devs = Array(range(-F2B_FQ_DEV, stop=F2B_FQ_DEV, length=2 * F2B_TRIAL_COUNT))
    push!(fq_devs, 0)
    negi_h0s = [(FQ + FQ * fq_dev) * NEGI_H0_ISO for fq_dev in fq_devs]
    gate_errors = ones(pulse_type_count, 2 * F2B_TRIAL_COUNT + 1, F2B_AVG_COUNT)
    save_file_paths = Array{String, 1}(undef, pulse_type_count)

    # check for previous computation
    if use_previous
        data_file_path_old = latest_file_path("h5", "f2b", SAVE_PATH)
    else
        data_file_path_old = nothing
    end
    if isnothing(data_file_path_old)
        save_file_paths_old = nothing
    else
        (save_file_paths_old, gate_errors_old, avg_count_old, pulse_type_count_old
         ) = h5open(data_file_path_old, "r") do data_file_old
             save_file_paths_old = read(data_file_old, "save_file_paths")
             gate_errors_old = read(data_file_old, "gate_errors")
             pulse_type_count_old = size(gate_errors_old)[1]
             avg_count_old = read(data_file_old, "avg_count")
             fq_devs_old = read(data_file_old, "fq_devs")
             if fq_devs_old != fq_devs
                 save_file_paths_old = nothing
             end
            return (save_file_paths_old, gate_errors_old, avg_count_old, pulse_type_count_old)
        end
    end

    for (i, pulse_type) in enumerate(F2B_PT_LIST)
        println("pt[$(i)]: $(pulse_type)")
        save_file_path = data[pulse_type]
        save_file_paths[i] = save_file_path
        save_file_path_old = ((isnothing(save_file_paths_old) || i > pulse_type_count_old)
                              ? nothing : save_file_paths_old[i])
        dynamics_type = pulse_type == analytic ? dynamics_type_analytic : schroed
        for j = 1:2 * F2B_TRIAL_COUNT + 1
            negi_h0 = negi_h0s[j]
            for k = 1:F2B_AVG_COUNT
                # skip redundant computation
                if (!isnothing(save_file_path_old) && save_file_path == save_file_path_old
                    && k <= avg_count_old)
                    gate_errors[i, j, k] = gate_errors_old[i, j, k]
                else
                    res = run_sim_prop(
                        1, gate_type; save_file_path=save_file_path,
                        dynamics_type=dynamics_type, save=false, negi_h0=negi_h0, seed=k
                    )
                    gate_errors[i, j, k] = 1 - res["fidelities"][end]
                end
            end
        end
    end
    
    data_file_path = generate_file_path("h5", "f2b", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "gate_errors", gate_errors)
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "avg_count", F2B_AVG_COUNT)
        write(data_file, "fq_devs", fq_devs)
    end
    println("Saved f2b data to $(data_file_path)")
end


const F2C_GATE_TIMES_XPIBY2 = [50., 56.8, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] # 13
const F2C_DATA_XPIBY2 = Dict(
    analytic => [joinpath(SPIN_OUT_PATH, "spin14/$(lpad(index, 5, '0'))_spin14.h5") for index in [
        INVAL, 4, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL
    ]],
    s2 => [joinpath(SPIN_OUT_PATH, "spin12/$(lpad(index, 5, '0'))_spin12.h5") for index in [
        629, 627, 638, 655, 654, 666, 658, 665, 661, 667, 668, 681, 685
    ]],
    sut8 => [joinpath(SPIN_OUT_PATH, "spin23/$(lpad(index, 5, '0'))_spin23.h5") for index in [
        38, 94, 103, 113, 132, 135, 125, 140, 143, 145, 158, 154, 162
    ]],
    d1 => [joinpath(SPIN_OUT_PATH, "spin11/$(lpad(index, 5, '0'))_spin11.h5") for index in [
        505, 504, 506, 508, 509, 510, 512, 514, 516, 517, 519, 521, 524
    ]],
    d2 => [joinpath(SPIN_OUT_PATH, "spin11/$(lpad(index, 5, '0'))_spin11.h5") for index in [
        520, 528, 522, 523, 525, 526, 527, 530, 531, 533, 535, 536, 537
    ]],
)
const F2C_GATE_TIMES_ZPIBY2 = Array(18:2:72) # 28
const F2C_DATA_ZPIBY2 = Dict(
    analytic => [joinpath(SPIN_OUT_PATH, "spin14/$(lpad(index, 5, '0'))_spin14.h5") for index in [
        0, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL,
        INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL,
        INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL,
    ]],
    s2 => [joinpath(SPIN_OUT_PATH, "spin12/$(lpad(index, 5, '0'))_spin12.h5") for index in [
        779, 781, 782, 783, 784, 785, 786, 787, 788, 789,
        790, 793, 791, 792, 794, 795, 796, 797, 798, 799,
        800, 801, 802, 803, 805, 804, 806, 807,
    ]],
    sut8 => [joinpath(SPIN_OUT_PATH, "spin23/$(lpad(index, 5, '0'))_spin23.h5") for index in [
        INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL,
        INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL,
        INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL, INVAL
    ]],
    d1 => [joinpath(SPIN_OUT_PATH, "spin11/$(lpad(index, 5, '0'))_spin11.h5") for index in [
        468, 471, 472, 474, 475, 476, 477, 478, 480, 539,
        541, 542, 543, 544, 545, 551, 553, 554, 556, 569,
        570, 571, 572, 573, 574, 576, 567, 585,
    ]],
    d2 => [joinpath(SPIN_OUT_PATH, "spin11/$(lpad(index, 5, '0'))_spin11.h5") for index in [
        487, 488, 489, 491, 492, 493, 494, 498, 499, 547,
        549, 552, 558, 560, 561, 562, 563, 564, 565, 575,
        577, 588, 597, 590, 591, 592, 600, 598
    ]],
)
const F2C_PT_LIST = [analytic, s2, sut8, d1, d2]
const F2C_AVG_COUNT = 1000
const F2C_SIGMA = 1e-2
const F2C_S1_NEGI_H0 = (FQ + FQ * F2C_SIGMA) * NEGI_H0_ISO
const F2C_S2_NEGI_H0 = (FQ - FQ * F2C_SIGMA) * NEGI_H0_ISO
function gen_2c(;use_previous=true, gate_type=zpiby2)
    if gate_type == zpiby2
        gate_times = F2C_GATE_TIMES_ZPIBY2
        data = F2C_DATA_ZPIBY2
        dynamics_type_analytic = zpiby2nodis
    elseif gate_type == xpiby2
        gate_times = F2C_GATE_TIMES_XPIBY2
        data = F2C_DATA_XPIBY2
        dynamics_type_analytic = xpiby2nodis
    end
    pulse_type_count = size(F2C_PT_LIST)[1]
    pulse_types_integer = [Integer(pulse_type) for pulse_type in F2C_PT_LIST]
    gate_time_count = size(gate_times)[1]
    gate_errors = ones(pulse_type_count, gate_time_count, 2 * F2C_AVG_COUNT)
    save_file_paths = Array{String, 2}(undef, pulse_type_count, gate_time_count)

    # check for previous computation
    if use_previous
        data_file_path_old = latest_file_path("h5", "f2c", SAVE_PATH)
    else
        data_file_path_old = nothing
    end
    if isnothing(data_file_path_old)
        save_file_paths_old = nothing
    else
        (save_file_paths_old, gate_errors_old, avg_count_old
         ) = h5open(data_file_path_old, "r") do data_file_old
            save_file_paths_old = read(data_file_old, "save_file_paths")
            gate_errors_old = read(data_file_old, "gate_errors")
            avg_count_old = read(data_file_old, "avg_count")
            return (save_file_paths_old, gate_errors_old, avg_count_old)
         end
        pulse_type_count_old = size(gate_errors_old)[1]
        gate_time_count_old = size(gate_errors_old)[2]
    end

    for (i, pulse_type) in enumerate(F2C_PT_LIST)
        println("pt[$(i)]: $(pulse_type)")
        for (j, gate_time) in enumerate(gate_times)
            print("gt[$(j)]: $(gate_time) ")
            save_file_path = data[pulse_type][j]
            save_file_paths[i, j] = save_file_path
            save_file_path_old = ((isnothing(save_file_paths_old) ||
                                   i > pulse_type_count_old || j > gate_time_count_old)
                                  ? nothing : save_file_paths_old[i, j])
            dynamics_type = pulse_type == analytic ? dynamics_type_analytic : schroed
            if !isnothing(findfirst("$(INVAL)", save_file_path))
                println("INVAL")
                continue
            end
            for k = 1:F2C_AVG_COUNT
                # skip redundant computation
                if (!isnothing(save_file_path_old) && save_file_path == save_file_path_old
                    && k <= avg_count_old)
                    gate_errors[i, j, (k - 1) * 2 + 1] = gate_errors_old[i, j, (k - 1) * 2 + 1]
                    gate_errors[i, j, (k - 1) * 2 + 2] = gate_errors_old[i, j, (k - 1) * 2 + 2]
                else
                    res1 = run_sim_prop(
                        1, gate_type; save_file_path=save_file_path, dynamics_type=dynamics_type,
                        negi_h0=F2C_S1_NEGI_H0, save=false, seed=k
                    )
                    res2 = run_sim_prop(
                        1, gate_type; save_file_path=save_file_path, dynamics_type=dynamics_type,
                        negi_h0=F2C_S2_NEGI_H0, save=false, seed=k
                    )
                    ge1 = 1 - res1["fidelities"][end]
                    ge2 = 1 - res2["fidelities"][end]
                    gate_errors[i, j, (k - 1) * 2 + 1] = ge1
                    gate_errors[i, j, (k - 1) * 2 + 2] = ge2
                end
            end
            println(" $(mean(gate_errors[i, j, :]))")
        end
    end

    data_file_path = generate_file_path("h5", "f2c", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "gate_errors", gate_errors)
        write(data_file, "gate_times", gate_times)
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "avg_count", F2C_AVG_COUNT)
    end
    print("Saved f2c data to $(data_file_path)")
end


### FIGURE 3 ###
const F3_DATA = Dict(
    analytic => joinpath(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
    s2 => joinpath(SPIN_OUT_PATH, "spin18/00057_spin18.h5"),
    sut8 => "", #joinpath(SPIN_OUT_PATH, "$(INVAL)"),
    d1 => joinpath(SPIN_OUT_PATH, "spin17/00063_spin17.h5"),
    d2 => joinpath(SPIN_OUT_PATH, "spin17/00069_spin17.h5"),
)

const F3A_PT_LIST = [analytic, s2, sut8, d1, d2]
function gen_3a()
    pulse_types_integer = [Integer(pulse_type) for pulse_type in F3A_PT_LIST]
    pulse_type_count = size(F3A_PT_LIST)[1]
    save_file_paths = Array{String, 1}(undef, pulse_type_count)
    for (i, pulse_type) in enumerate(F3A_PT_LIST)
        save_file_paths[i] = F3_DATA[pulse_type]
    end

    data_file_path = generate_file_path("h5", "f3a", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "pulse_types", pulse_types_integer)
    end
    println("Saved f3a data to $(data_file_path)")
end


const F3B_PT_LIST = [analytic, s2, sut8, d1, d2]
const F3B_AVG_COUNT = 10
const F3B_GATE_COUNT = 500
function gen_3b(;use_previous=true)
    gate_type = xpiby2
    pulse_types_integer = [Integer(pulse_type) for pulse_type in F3B_PT_LIST]
    pulse_type_count = size(F3B_PT_LIST)[1]
    save_file_paths = Array{String, 1}(undef, pulse_type_count)
    gate_errors = ones(pulse_type_count, F3B_GATE_COUNT + 1, F3B_AVG_COUNT)

    # check for previous computation
    if use_previous
        data_file_path_old = latest_file_path("h5", "f3b", SAVE_PATH)
    else
        data_file_path_old = nothing
    end
    if !isnothing(data_file_path_old)
        (save_file_paths_old, gate_errors_old, avg_count_old
         ) = h5open(data_file_path_old, "r") do data_file_old
             gate_count_old = read(data_file_old, "gate_count")
             pulse_types_integer_old = read(data_file_old, "pulse_types")
             if gate_count_old == F3B_GATE_COUNT && pulse_types_integer_old == pulse_types_integer
                 save_file_paths_old = read(data_file_old, "save_file_paths")
                 gate_errors_old = read(data_file_old, "gate_errors")
                 avg_count_old = read(data_file_old, "avg_count")
             else
                 save_file_paths_old = gate_errors_old = avg_count_old = nothing
             end
             return (save_file_paths_old, gate_errors_old, avg_count_old)
         end
    else
        save_file_paths_old = gate_errors_old = avg_count_old = nothing
    end
    

    for (i, pulse_type) in enumerate(F3B_PT_LIST)
        print("pt[$(i)]: $(pulse_type) ")
        save_file_path = F3_DATA[pulse_type]
        save_file_paths[i] = save_file_path
        if isempty(save_file_path)
            println("")
            continue
        end
        save_file_path_old = isnothing(save_file_paths_old) ? nothing : save_file_paths_old[i]
        save_file_path_sim = pulse_type == analytic ? nothing : save_file_path
        dynamics_type = pulse_type == analytic ? xpiby2da : schroedda
        for j = 1:F3B_AVG_COUNT
            # avoid redundant computation
            if (!isnothing(save_file_path_old) && save_file_path == save_file_path_old
                && j <= avg_count_old)
                gate_errors[i, :, j] = gate_errors_old[i, :, j]
                print("s")
            else
                res = run_sim_prop(F3B_GATE_COUNT, gate_type; dynamics_type=dynamics_type,
                                   save_file_path=save_file_path_sim, seed=j, save=false)
                gate_errors[i, :, j] = 1 .- res["fidelities"]
                print(".")
            end
        end
        println("")
    end

    data_file_path = generate_file_path("h5", "f3b", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "gate_errors", gate_errors)
        write(data_file, "avg_count", F3B_AVG_COUNT)
        write(data_file, "gate_count", F3B_GATE_COUNT)
    end
    println("Saved f3b data to $(data_file_path)")
end


"""
gen_3c - generate an average of the flux noise used in fig3b,
see rbqoc/src/spin/spin.jl/run_sim_deqjl for noise construction
"""
function gen_3c()
    noise_dt_inv = DT_NOISE_INV
    noise_dist = STD_NORMAL
    noise_amp = NAMP_PREFACTOR
    evolution_time = 56.8 * 5e3
    noise_count = Int(ceil(evolution_time * noise_dt_inv)) + 1

    # generate noise
    noise = pink_noise_from_white(noise_count, noise_dt_inv, noise_dist; seed=0)
    for seed = 1:9
        noise .+= pink_noise_from_white(noise_count, noise_dt_inv, noise_dist; seed=seed)
    end
    # average
    noise ./= 10

    # take fft
    noise_fft = Array{Complex{Float64}, 1}(noise)
    fft!(noise_fft)
    # normalize and take absolute value
    for i = 1:size(noise_fft)[1]
        noise_fft[i] = abs(noise_fft[i]) / noise_count
    end
    noise_fft = Array{Float64, 1}(noise_fft)

    # heh, linearity
    noise .*= NAMP_PREFACTOR
    noise_fft .*= NAMP_PREFACTOR

    times = Array(0:(noise_count - 1)) / noise_dt_inv
    freqs = Array(fftfreq(noise_count, noise_dt_inv))

    h5open(F3C_DATA_FILE_PATH, "w") do data_file
        write(data_file, "noise", noise)
        write(data_file, "times", times)
        write(data_file, "noise_fft", noise_fft)
        write(data_file, "freqs", freqs)
    end
end


const F3D_SAVE_STEP = 1e-1
const F3D_PTS = [analytic, s2, s4, d2, d3]
"""
gen_3d - used for generating expectation values of an evolution
to plot on bloch sphere
"""
function gen_3d()
    pulse_types_int = [Integer(pulse_type) for pulse_type in F3D_PTS]
    point_count = Integer(ceil(56.80 * F3D_SAVE_STEP^(-1)))
    pointss = zeros(size(F3D_PTS)[1], point_count, 3)
    for (i, pulse_type) in enumerate(F3D_PTS)
        data = F3_DATA[pulse_type]
        if !(DATAFP_KEY in keys(data))
            if pulse_type == analytic
                data_file_path = run_sim_fine_deqjl(
                    dynamics_type=xpiby2nodis, dt=1e-3,
                    save_step=F3D_SAVE_STEP)
            else
                data_file_path = run_sim_fine_deqjl(
                    save_file_path=data[SAVEFP_KEY],
                    save_type=data[SAVET_KEY],
                    dt=1e-3, save_step=F3D_SAVE_STEP,
                )
            end
        else
            data_file_path = data[DATAFP_KEY]
        end

        (states_) = h5open(data_file_path, "r") do data_file
            states_ = read(data_file, "states")
            return (states_)
        end
        for j = 1:point_count
            state = get_vec_uniso(states_[j, :])
            state_dag = state'
            pointss[i, j, 1] = Real(state_dag * SIGMAX * state)
            pointss[i, j, 2] = Real(state_dag * SIGMAY * state)
            pointss[i, j, 3] = Real(state_dag * SIGMAZ * state)
        end
    end
    h5open(F3D_DATA_FILE_PATH, "w") do data_file
        write(data_file, "pointss", pointss)
        write(data_file, "pulse_types", pulse_types_int)
    end
end


const F4_DATA = Dict(
    corpse => joinpath(SPIN_OUT_PATH, "spin14/00124_spin14.h5"),
    s2 => nothing,
    s4 => nothing,
    d1 => joinpath(SPIN_OUT_PATH, "spin19/00025_spin19.h5"),
    d2 => joinpath(SPIN_OUT_PATH, "spin19/00026_spin19.h5"),
)

const F4A_PTS = [corpse, s2, s4, d1, d2]
function gen_4a()
    pulse_types_integer = [Integer(pulse_type) for pulse_type in F4A_PTS]
    pulse_type_count = size(F4A_PTS)[1]
    save_file_paths = Array{String, 1}(undef, pulse_type_count)
    for (i, pulse_type) in enumerate(F4A_PTS)
        save_file_path = F4_DATA[pulse_type]
        save_file_paths[i] = isnothing(save_file_path) ? "" : save_file_path
    end

    data_file_path = generate_file_path("h5", "f4a", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "pulse_types", pulse_types_integer)
    end
    println("Saved f4a data to $(data_file_path)")
end


const F4B_PTS = [corpse, s2, s4, d1, d2]
const F4B_TRIAL_COUNT = Integer(1e2)
const F4B_FQ_DEV = 2e-2
const F4B_AVG_COUNT = 10
const F4B_DT_INV = Int64(1e4) # go down to 1e5 if want ge < 10^-10
function gen_4b(;use_previous=true)
    gate_type = xpi
    pulse_types_integer = [Integer(pt) for pt in F4B_PTS]
    pulse_type_count = size(F4B_PTS)[1]
    fq_devs = Array(range(-F4B_FQ_DEV, stop=F4B_FQ_DEV, length=2 * F4B_TRIAL_COUNT))
    push!(fq_devs, 0)
    negi_h0s = [fq_dev * NEGI_H0_ISO for fq_dev in fq_devs]
    gate_errors = ones(pulse_type_count, 2 * F4B_TRIAL_COUNT + 1, F4B_AVG_COUNT)
    save_file_paths = Array{String, 1}(undef, pulse_type_count)

    # check for previous computation
    if use_previous
        data_file_path_old = latest_file_path("h5", "f4b", SAVE_PATH)
    else
        data_file_path_old = nothing
    end
    if isnothing(data_file_path_old)
        save_file_paths_old = nothing
    else
        (save_file_paths_old, gate_errors_old, avg_count_old, pulse_type_count_old
         ) = h5open(data_file_path_old, "r") do data_file_old
             save_file_paths_old = read(data_file_old, "save_file_paths")
             gate_errors_old = read(data_file_old, "gate_errors")
             pulse_type_count_old = size(gate_errors_old)[1]
             avg_count_old = read(data_file_old, "avg_count")
             fq_devs_old = read(data_file_old, "fq_devs")
             if fq_devs_old != fq_devs
                 save_file_paths_old = nothing
             end
            return (save_file_paths_old, gate_errors_old, avg_count_old, pulse_type_count_old)
        end
    end

    for (i, pulse_type) in enumerate(F4B_PTS)
        println("pt[$(i)]: $(pulse_type)")
        save_file_path = F4_DATA[pulse_type]
        save_file_paths[i] = isnothing(save_file_path) ? "" : save_file_path
        if isnothing(save_file_path)
            continue
        end
        save_file_path_old = ((isnothing(save_file_paths_old) || i > pulse_type_count_old)
                              ? nothing : save_file_paths_old[i])
        save_file_path_sim = pulse_type == corpse ? nothing : save_file_path
        if pulse_type == corpse
            dynamics_type = xpicorpse
        else
            dynamics_type = schroed
        end

        for j = 1:2 * F4B_TRIAL_COUNT + 1
            negi_h0 = negi_h0s[j]
            for k = 1:F4B_AVG_COUNT
                # skip redundant computation
                if (!isnothing(save_file_path_old) && save_file_path == save_file_path_old
                    && k <= avg_count_old)
                    gate_errors[i, j, k] = gate_errors_old[i, j, k]
                else
                    res = run_sim_deqjl(
                        1, gate_type; dynamics_type=dynamics_type,
                        save_file_path=save_file_path_sim,
                        dt_inv=F4B_DT_INV, save=false, negi_h0=negi_h0, seed=k
                    )
                    gate_errors[i, j, k] = 1 - res["fidelities"][end]
                end
            end
        end
    end
    
    data_file_path = generate_file_path("h5", "f4b", SAVE_PATH)
    h5open(data_file_path, "w") do data_file
        write(data_file, "pulse_types", pulse_types_integer)
        write(data_file, "gate_errors", gate_errors)
        write(data_file, "save_file_paths", save_file_paths)
        write(data_file, "avg_count", F4B_AVG_COUNT)
        write(data_file, "fq_devs", fq_devs)
    end
    println("Saved f4b data to $(data_file_path)")
end


"""
generate all required figure data
"""
function gen_all(;use_previous=true)
    gen_1a()
    gen_1b()
    gen_1c(;use_previous=use_previous)
    gen_2a()
    gen_2b(;use_previous=use_previous)
    gen_2c(;use_previous=use_previous)
    gen_3a()
    gen_3b(;use_previous=use_previous)
end


# If you run `julia figures.jl` all of the figure data will be
# generated from scratch.
if String(@__FILE__) == String(joinpath(pwd(), PROGRAM_FILE))
    gen_all(;use_previous=false)
end
