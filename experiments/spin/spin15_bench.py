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
VE_FILE_NAME = "ve_spin15_bench"

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


def run_all():
    pass
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


def run_verify_empty(gate_count, seed):
    gate_time = ZPIBY2_TIME
    initial_density = gen_rand_density_iso(seed)
    gate_knot_count = int(gate_time * DT_INV)
    gate_knot_count_4 = 4 * gate_knot_count
    knot_count = gate_knot_count * gate_count
    h_sys = Qobj(OMEGA * H_S)
    hlist = [
        h_sys
    ]
    rho0 = Qobj(initial_density)
    tlist = np.arange(0, knot_count, 1) * DT
    c_ops = [
    ]
    e_ops = []

    result = mesolve(hlist, rho0, tlist, c_ops, e_ops)
    densities = np.stack([result.states[i].full() for i in range(len(result.states))])
    fidelities = np.zeros(gate_count)
    z1 = ZPIBY2
    z2 = matmuls(ZPIBY2, z1)
    z3 = matmuls(ZPIBY2, z2)
    z1_dag = conjugate_transpose(z1)
    z2_dag = conjugate_transpose(z2)
    z3_dag = conjugate_transpose(z3)
    id0 = initial_density
    id1 = matmuls(z1, id0, z1_dag)
    id2 = matmuls(z2, id0, z2_dag)
    id3 = matmuls(z3, id0, z3_dag)
    for i in range(gate_count):
        knot_count = gate_count * i
        if i %  4 == 0:
            target_density = id0
        elif i % 4 == 1:
            target_density = id1
        elif i % 4 == 2:
            target_density = id2
        elif i % 4 == 3:
            target_density = id3
        #ENDIF
        fidelities[i] = fidelity_mat(densities[i * gate_knot_count], target_density)
    #ENDFOR

    save_file_path = generate_save_file_path(VE_FILE_NAME, SAVE_PATH)
    with h5py.File(save_file_path) as save_file:
        save_file["densities"] = densities
        save_file["fidelities"] = fidelities
    #ENDWITH

    print("Saved to {}"
          "".format(save_file_path))

    return
#ENDDEF


def plot_verify_empty(save_file_path):
    plot_save_file_path = "{}.png".format(save_file_path.split(".h5")[0])
    with h5py.File(save_file_path) as save_file:
        fidelities = save_file["fidelities"][()]
    #ENDWITH
    gate_count = fidelities.shape[0]
    gate_axis = np.arange(0, gate_count)
    fig = plt.figure()
    plt.plot(gate_axis, fidelities)
    plt.ylabel("Fidelity")
    plt.xlabel("Gate Count")
    plt.title("Verify Empty")
    plt.savefig(plot_save_file_path, dpi=DPI)

    print("Plotted to {}"
          "".format(plot_save_file_path))
    return
#ENDDEF

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--ve", action="store_true")
    parser.add_argument("--pve", action="store", type=str, default=None)
    parser.add_argument("--gate", action="store", type=str, default="zpiby2")
    parser.add_argument("--pulse", action="store", type=str, default="analytic")
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--gc", action="store", type=int, default=1)
    args = vars(parser.parse_args())
    do_run = args["run"]
    do_verify = args["verify"]
    do_verify_empty = args["ve"]
    do_plot_verify_empty = args["pve"] is not None
    pulse_type = args["pulse"]
    gate_type = args["gate"]
    seed = args["seed"]
    gate_count = args["gc"]

    if do_run:
        run_all()
    if do_verify:
        run_verify(gate_count, gate_type, pulse_type, seed)
    if do_verify_empty:
        run_verify_empty(gate_count, seed)
    if do_plot_verify_empty:
        plot_verify_empty(args["pve"])
    #ENDIF


if __name__ == "__main__":
    main()
