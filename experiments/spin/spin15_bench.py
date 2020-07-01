"""
spin15_bench.py - benchmark the spin15.jl t1 pulses
"""

from argparse import ArgumentParser
from copy import copy
from enum import Enum
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from qoc.standard import (
    conjugate_transpose, generate_save_file_path,
    matmuls
)
from qutip import (
    mesolve, Qobj, basis
)


# paths
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin15_bench"
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
META_PATH = os.path.join(WDIR, "out", EXPERIMENT_META)
SAVE_PATH = os.path.join(META_PATH, EXPERIMENT_NAME)

# comp. onstants
OMEGA = 2 * np.pi * 1.4e-2
INV_ROOT_2 = 2 ** (-1/2)
DT = 1e-2
DT_INV = 1e2
CONTROL_COUNT = 1
CONTROLS_ZERO = np.zeros((1, CONTROL_COUNT))
FBFQ_A = 0.202407
FBFQ_B = 0.5
# Sorted from highest order to lowest order.
# These coeffecients are in units of s.
FBFQ_T1_COEFFS = np.array([
    3276.06057, -7905.24414, 8285.24137, -4939.22432,
    1821.23488, -415.520981, 53.9684414, -3.04500484
]) * 1e9

INITIAL_STATE = np.array([[1.], [0.]])
INITIAL_DENSITY = np.matmul(INITIAL_STATE, conjugate_transpose(INITIAL_STATE))
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2
XPIBY2 = np.array([[1.,  -1j],
                   [-1.j, 1.]]) * INV_ROOT_2
YPIBY2 = np.array([[1., -1.],
                   [1.,  1.]]) * INV_ROOT_2
ZPIBY2 = np.array([[1. - 1.j, 0],
                   [0,        1. + 1.j]]) * INV_ROOT_2
C_G_TO_E = np.array([[0., 1.],
                     [0., 0.]])
C_E_TO_G = np.array([[0., 0.],
                     [1., 0.]])

# data constants
PULSE_DATA = {
    # "vanillat1_alt": {
    #   "zpiby2": {
    #       "experiment_name": "spin15",
    #       "controls_file_name": "00008_spin15.h5"
    #   }  
    # },
    "t1_m1": {
        "xpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00024_spin15.h5"
        },
        "ypiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00015_spin15.h5"
        },
        "zpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00023_spin15.h5"
        },
    },
    "vanilla": {
        "xpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00012_spin15.h5"
        },
        "ypiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00016_spin15.h5"
        },
        "zpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00013_spin15.h5"
        },
    },
    "analytic": {
        "xpiby2": {
            "experiment_name": "spin14",
            "controls_file_name": "00002_spin14.h5"
        },
        "ypiby2": {
            "experiment_name": "spin14",
            "controls_file_name": "00001_spin14.h5"
        },
        "zpiby2": {
            "experiment_name": "spin14",
            "controls_file_name": "00000_spin14.h5"
        },
    }
}

GT_TO_GATE = {
    "zpiby2": ZPIBY2,
    "ypiby2": YPIBY2,
    "xpiby2": XPIBY2,
}

PT_TO_CIDX = {
    "t1_m1": 9,
    "vanilla": 9,
    "analytic": -1,
}

ZPIBY2_DATA_FILE_PATH = os.path.join(SAVE_PATH, "00004_spin15_bench.h5")
ZPIBY2_PLOT_FILE_PATH = os.path.join(SAVE_PATH, "00004_spin15_bench.png")
YPIBY2_DATA_FILE_PATH = os.path.join(SAVE_PATH, "00005_spin15_bench.h5")
YPIBY2_PLOT_FILE_PATH = os.path.join(SAVE_PATH, "00005_spin15_bench.png")
XPIBY2_DATA_FILE_PATH = os.path.join(SAVE_PATH, "00006_spin15_bench.h5")
XPIBY2_PLOT_FILE_PATH = os.path.join(SAVE_PATH, "00006_spin15_bench.png")


# misc constants
SEED = 0
GATE_COUNT = 1
CONTROLS_IDX = 9
CIDX_PY = -1
DPI = 700
ZPIBY2_TIME = 17.86
ZPIBY2_TIME_ALT = 20.0
YPIBY2_TIME = 19.47
XPIBY2_TIME = 56.80
ZPIBY2_GATE_COUNT = 300
YPIBY2_GATE_COUNT = 300
XPIBY2_GATE_COUNT = 100


COLORS = [
    "yellow", "pink", "orange",
    "blue", "green", "purple",
    "black", "grey", "brown",
    "red"
]


class GateType(Enum):
    ZPIBY2 = 0
    YPIBY2 = 1
    XPIBY2 = 2
#ENDDEF


def horner(coeffs, val):
    run = coeffs[0]
    for i in range(1, coeffs.shape[0]):
        run = coeffs[i] + val * run
    #ENDFOR
    return run
#ENDDEF


def get_fbfq(amplitude):
    return -abs(amplitude) * FBFQ_A + FBFQ_B
#ENDDEF


def get_t1_poly(amplitude):
    fbfq = get_fbfq(amplitude)
    t1 = horner(FBFQ_T1_COEFFS, fbfq)
    return t1
#ENDDEF


def rms_norm(x1, x2):
    diff = x1 - x2
    return np.sqrt(np.sum(np.real(diff * np.conjugate(diff))) / np.prod(x1.shape))
#ENDDEF


def fidelity_mat(m1, m2):
    ip = np.trace(np.matmul(conjugate_transpose(m1), m2))
    fidelity_ = np.real(ip) ** 2 + np.imag(ip) ** 2
    ip_self = np.trace(np.matmul(conjugate_transpose(m2), m2))
    fidelity_self = np.real(ip_self) ** 2 + np.imag(ip_self) ** 2
    # print("fidelity_: {}, fidelity_self: {}, f/f_self: {}"
    #       "".format(fidelity_, fidelity_self, fidelity_/fidelity_self))

    return fidelity_ / fidelity_self
#ENDDEF


def t1_average(experiment_name, controls_file_name):
    save_file_path = os.path.join(META_PATH, experiment_name, controls_file_name)
    with h5py.File(save_file_path) as save_file:
        controls = save_file["controls"][()]
    #ENDWITH
    t1s = get_t1_poly(controls / (2 * np.pi))
    t1_average = np.mean(t1s)
    return t1_average
#ENDDEF


def run_sim(class_key, pulse_data, gate_sequence):
    class_data = pulse_data[class_key]
    
    # build controls and matrices from gate sequence
    gate_count = gate_sequence.shape[0]
    controls_list = list()
    running_state = INITIAL_STATE
    running_time = 0.
    running_ccount = 0
    state_array = np.zeros((gate_count, 2, 1), dtype=np.complex128)
    time_array = np.zeros(gate_count, dtype=np.float64)
    for i, gate_id in enumerate(gate_sequence):
        if gate_id == GateType.XPIBY2.value:
            controls_list.append(class_data["xpiby2"]["controls"])
            running_time = running_time + class_data["xpiby2"]["evolution_time"]
            running_state = np.matmul(XPIBY2, running_state)
        elif gate_id == GateType.YPIBY2.value:
            controls_list.append(class_data["ypiby2"]["controls"])
            running_time = running_time + class_data["ypiby2"]["evolution_time"]
            running_state = np.matmul(YPIBY2, running_state)
        else:
            controls_list.append(class_data["zpiby2"]["controls"])
            running_time = running_time + class_data["zpiby2"]["evolution_time"]
            running_state = np.matmul(ZPIBY2, running_state)
        state_array[i] = running_state
        time_array[i] = running_time
        # running_ccount = running_ccount + controls_list[-1].shape[0]
        # print("rt: {}, rc: {}"
        #       "".format(running_time, running_ccount))
    #ENDFOR
    controls = np.concatenate(controls_list)
    # extrude to match tlist shape
    # if class_key == "vanillat1" or class_key == "vanilla":
    #     controls = np.vstack((controls, CONTROLS_ZERO))
    
    # build simulation
    control_eval_count = controls.shape[0]
    evolution_time = running_time
    
    h_sys = Qobj(OMEGA * H_S)
    h_c1 = Qobj(H_C1)
    hlist = [
        [h_sys, np.ones(control_eval_count)],
        [h_c1, controls[:, 0]]
    ]
    rho0 = Qobj(INITIAL_DENSITY)
    tlist = np.arange(0, control_eval_count, 1) * DT
    t1_array = get_t1_poly(controls[:, 0] / (2 * np.pi))
    sqrt_gamma_array = t1_array ** -0.5
    print(sqrt_gamma_array)
    # print("t1_array:\n{}\ngamma_t1_array:\n{}"
    #       "".format(t1_array, gamma_t1_array))
    # print("time_array:\n{}\ntlist:\n{}"
    #       "".format(time_array, tlist))
    # print("controls:\n{}\nt1:\n{}"
    #       "".format(controls[:50], t1_array[:50]))
    c_ops = [
        [Qobj(C_G_TO_E), sqrt_gamma_array],
        [Qobj(C_E_TO_G), sqrt_gamma_array],
    ]
    e_ops = []

    # run simulation
    result = mesolve(hlist, rho0, tlist, c_ops, e_ops)

    # analysis
    densities = result.states
    fidelity_array = np.zeros(time_array.shape[0])
    j = 0
    for i, t in enumerate(tlist):
        if np.round(t, 2) == np.round(time_array[j], 2):
            density = densities[i].full()
            target_state = state_array[j]
            target_density = np.matmul(target_state, conjugate_transpose(target_state))
            fidelity_array[j] = fidelity_mat(density, target_density)
            # fidelity_array[j] = 1 - rms_norm(density, target_density)
            # print("tlist[{}]: {}, time_array[{}]: {}, fidelity: {}"
            #       "".format(i, t, j, time_array[j], fidelity_array[j]))
            # print("density:\n{}\ntarget_density:\n{}"
            #       "".format(density, target_density))
            # print("np.round(tlist[{}], 2): {}, np.round(time_array[{}], 2): {}"
            #       "".format(i, t, j, time_array[j]))
            j = j + 1
        #ENDIF
    #ENDFOR

    # do last check, which is missed
    density = densities[-1].full()
    target_state = state_array[-1]
    target_density = np.matmul(target_state, conjugate_transpose(target_state))
    fidelity_array[-1] = fidelity_mat(density, target_density)

    densities = np.array([density.full() for density in densities])
    print("d[3]:\n{}\nd[7]:\n{}"
          "".format(densities[3], densities[7]))
    
    return fidelity_array
#ENDDEF


def run_all():
    pulse_data = copy(PULSE_DATA)
    # generate gate sequence
    # np.random.seed(SEED)
    # gate_sequence = np.random.randint(GateType.XPIBY2.value,
    #                                   GateType.ZPIBY2.value + 1, GATE_COUNT)

    gate_sequence = np.repeat(0, 8)
    
    # # perform initial save
    # save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    # with h5py.File(save_file_path, "a") as save_file:
    #     save_file["gate_sequence"] = gate_sequence
    #     for class_key in pulse_data.keys():
    #         for gate_key in pulse_data[class_key].keys():
    #             gate_dict = pulse_data[class_key][gate_key]
    #             save_file["{}_{}".format(class_key, gate_key)] = (
    #                 np.string_(os.path.join(
    #                     gate_dict["experiment_name"],
    #                     gate_dict["controls_file_name"]))
    #             )
    #         #ENDFOR
    #     #ENDFOR
    # #ENDWITH

    # save_file_path = os.path.join(SAVE_PATH, "00004_spin15_bench.h5")    
    # grab controls
    for class_key in pulse_data.keys():
        # if class_key != "vanillat1_alt":
        #     continue
        for gate_key in pulse_data[class_key].keys():
            gate_dict = pulse_data[class_key][gate_key]
            controls_save_file_path = os.path.join(
                META_PATH, gate_dict["experiment_name"], gate_dict["controls_file_name"]
            )
            with h5py.File(controls_save_file_path, "r") as save_file:
                gate_dict["evolution_time"] = save_file["evolution_time"][()]
                if class_key == "analytic":
                    gate_dict["controls"] = save_file["controls"][()]
                if (class_key == "vanillat1" or class_key == "vanilla"
                    or class_key == "vanillat1_alt"):
                    controls = save_file["states"][CONTROLS_IDX, 0:-1]
                    controls = np.reshape(controls, (controls.shape[0], 1))
                    gate_dict["controls"] = controls
            #ENDWITH
        #ENDFOR
    #ENDFOR

    fidelity_array = run_sim("analytic", pulse_data, gate_sequence)
    print(fidelity_array)

    # # run sim
    # for class_key in pulse_data.keys():
    #     # if class_key != "vanillat1_alt":
    #     #     continue
    #     fidelity_array = run_sim(class_key, pulse_data, gate_sequence)
    #     with h5py.File(save_file_path, "a") as save_file:
    #         save_file["{}_fidelities".format(class_key)] = fidelity_array
    #     #ENDWITH
    # #ENDFOR

    # print("Saved bench to {}"
    #       "".format(save_file_path))
#ENDDEF


def safe_log(arr):
    return np.where(arr, np.log(arr), arr)
#ENDDEF


def plot():
    class_keys = PULSE_DATA.keys()
    zpiby2 = False
    ypiby2 = False
    xpiby2 = True

    # extract fidelity arrays
    with h5py.File(ZPIBY2_DATA_FILE_PATH) as save_file:
        zpiby2_analytic_fidelities = safe_log(save_file["analytic_fidelities"][:ZPIBY2_GATE_COUNT])
        zpiby2_vanilla_fidelities = safe_log(save_file["vanilla_fidelities"][:ZPIBY2_GATE_COUNT])
        zpiby2_vanillat1_fidelities = safe_log(save_file["vanillat1_fidelities"][:ZPIBY2_GATE_COUNT])
        zpiby2_vanillat1_alt_fidelities = safe_log(save_file["vanillat1_alt_fidelities"][:ZPIBY2_GATE_COUNT])
    #ENDWITH
    with h5py.File(YPIBY2_DATA_FILE_PATH) as save_file:
        ypiby2_analytic_fidelities = safe_log(save_file["analytic_fidelities"][:YPIBY2_GATE_COUNT])
        ypiby2_vanilla_fidelities = safe_log(save_file["vanilla_fidelities"][:YPIBY2_GATE_COUNT])
        ypiby2_vanillat1_fidelities = safe_log(save_file["vanillat1_fidelities"][:YPIBY2_GATE_COUNT])
    #ENDWITH
    with h5py.File(XPIBY2_DATA_FILE_PATH) as save_file:
        xpiby2_analytic_fidelities = safe_log(save_file["analytic_fidelities"][:XPIBY2_GATE_COUNT])
        xpiby2_vanilla_fidelities = safe_log(save_file["vanilla_fidelities"][:XPIBY2_GATE_COUNT])
        xpiby2_vanillat1_fidelities = safe_log(save_file["vanillat1_fidelities"][:XPIBY2_GATE_COUNT])
    #ENDWITH

    # extract time arrays
    xpiby2_times = np.arange(0, XPIBY2_GATE_COUNT, 1) * XPIBY2_TIME
    ypiby2_times = np.arange(0, YPIBY2_GATE_COUNT, 1) * YPIBY2_TIME
    zpiby2_times = np.arange(0, ZPIBY2_GATE_COUNT, 1) * ZPIBY2_TIME
    zpiby2_times_alt = np.arange(0, ZPIBY2_GATE_COUNT, 1) * ZPIBY2_TIME_ALT
    

    # plot
    fig = plt.figure()

    if zpiby2:
        plt.plot(zpiby2_times, zpiby2_analytic_fidelities,
                 label="$Z/2$ Analytic", color=COLORS[0])
        plt.plot(zpiby2_times, zpiby2_vanilla_fidelities,
                 label="$Z/2$ Vanilla", color=COLORS[1])
        plt.plot(zpiby2_times, zpiby2_vanillat1_fidelities,
                 label="$Z/2$ T1 Sense", color=COLORS[2])
        plt.plot(zpiby2_times_alt, zpiby2_vanillat1_alt_fidelities,
                 label="$Z/2$ T1 Sense Alt", color=COLORS[9])
        plot_save_file_path = ZPIBY2_PLOT_FILE_PATH
        title = "Z/2 T1 Relaxation"
    if ypiby2:
        plt.plot(ypiby2_times, ypiby2_analytic_fidelities,
                 label="$Y/2$ Analytic", color=COLORS[3])
        plt.plot(ypiby2_times, ypiby2_vanilla_fidelities,
                 label="$Y/2$ Vanilla", color=COLORS[4])
        plt.plot(ypiby2_times, ypiby2_vanillat1_fidelities,
                 label="$Y/2$ T1 Sense", color=COLORS[5])
        plot_save_file_path = YPIBY2_PLOT_FILE_PATH
        title = "Y/2 T1 Relaxation"

    if xpiby2:
        plt.plot(xpiby2_times, xpiby2_analytic_fidelities,
                 label="$X/2$ Analytic", color=COLORS[6])
        plt.plot(xpiby2_times, xpiby2_vanilla_fidelities,
                 label="$X/2$ Vanilla", color=COLORS[7])
        plt.plot(xpiby2_times, xpiby2_vanillat1_fidelities,
                 label="$X/2$ T1 Sense", color=COLORS[8])
        plot_save_file_path = XPIBY2_PLOT_FILE_PATH
        title = "X/2 T1 Relaxation"
    
    plt.legend()
    plt.ylabel("log Fidelity")
    plt.xlabel("Time (ns)")
    # plt.ylim(0, 1)
    plt.xlim(0)
    plt.title(title)
    plt.savefig(plot_save_file_path, dpi=DPI)
#ENDDEF


def gen_rand_density_iso(seed):
    np.random.seed(seed)
    rands = np.random.rand(4)
    state = np.array([[rands[0] + 1j * rands[1]],
                      [rands[2] + 1j * rands[3]]])
    density = np.matmul(state, conjugate_transpose(state))
    return density
#ENDDEF


def grab_controls(gate_type, pulse_type):
    pulse_data = PULSE_DATA[pulse_type][gate_type]
    controls_file_path = os.path.join(
        META_PATH, pulse_data["experiment_name"], pulse_data["controls_file_name"]
    )
    controls_idx = PT_TO_CIDX[pulse_type]
    with h5py.File(controls_file_path) as save_file:
        if controls_idx == CIDX_PY:
            controls = save_file["controls"][:, 0]
        else:
            controls = save_file["states"][controls_idx, :-1]
        #ENDIF
        gate_time = save_file["evolution_time"][()]
    #ENDWITH
    return (controls, gate_time)
#ENDDEF


def run_verify(gate_count, gate_type, pulse_type, seed):
    (controls, gate_time) = grab_controls(gate_type, pulse_type)
    control_knot_count = controls.shape[0]
    density = initial_density = gen_rand_density_iso(seed)
    evolution_time = gate_time * gate_count
    knot_count = int(evolution_time * DT_INV)
    gate = GT_TO_GATE[gate_type]
    target_density = (
        matmuls(
            *([gate] * gate_count),
            initial_density,
            *([conjugate_transpose(gate)] * gate_count),
        )
    )
    
    controls = np.concatenate([controls] * gate_count)
    h_sys = Qobj(OMEGA * H_S)
    h_c1 = Qobj(H_C1)
    hlist = [
        [h_sys, np.ones(control_knot_count * gate_count)],
        [h_c1, controls],
    ]
    rho0 = Qobj(initial_density)
    tlist = np.arange(0, knot_count, 1) * DT
    t1_array = get_t1_poly(controls / (2 * np.pi))
    sqrt_gamma_array = t1_array ** -0.5
    # print("t1_array:\n{}\ngamma_t1_array:\n{}"
    #       "".format(t1_array, gamma_t1_array))
    # print("time_array:\n{}\ntlist:\n{}"
    #       "".format(time_array, tlist))
    # print("controls:\n{}\nt1:\n{}"
    #       "".format(controls[:50], t1_array[:50]))
    c_ops = [
        # [Qobj(C_G_TO_E), sqrt_gamma_array],
        # [Qobj(C_E_TO_G), sqrt_gamma_array],
    ]
    e_ops = []

    result = mesolve(hlist, rho0, tlist, c_ops, e_ops)
    densities = result.states
    density = densities[-1].full()
    fidelity = fidelity_mat(density, target_density)

    print("fidelity:\n{}\ninitial_density:\n{}\ndensity:\n{}\ntarget_density:\n{}"
          "".format(fidelity, initial_density, density, target_density))
    
    return
#ENDDEF

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--gate", action="store", type=str, default="zpiby2")
    parser.add_argument("--pulse", action="store", type=str, default="analytic")
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--gc", action="store", type=int, default=1)
    args = vars(parser.parse_args())
    do_run = args["run"]
    do_plot = args["plot"]
    do_verify = args["verify"]
    pulse_type = args["pulse"]
    gate_type = args["gate"]
    seed = args["seed"]
    gate_count = args["gc"]

    if do_run:
        run_all()
    if do_plot:
        plot()
    if do_verify:
        run_verify(gate_count, gate_type, pulse_type, seed)
    #ENDIF


if __name__ == "__main__":
    main()
