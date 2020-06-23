"""
spin15_bench.py - benchmark the spin15.jl t1 pulses
"""

from argparse import ArgumentParser
from copy import copy
from enum import Enum
import os

import h5py
import numpy as np
from qoc.standard import (
    conjugate_transpose, generate_save_file_path,
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

# data constants
PULSE_DATA = {
    "vanillat1": {
        "xpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00010_spin15.h5"
        },
        "ypiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00006_spin15.h5"
        },
        "zpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00008_spin15.h5"
        },
    },
    "vanilla": {
        "xpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00011_spin15.h5"
        },
        "ypiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00007_spin15.h5"
        },
        "zpiby2": {
            "experiment_name": "spin15",
            "controls_file_name": "00009_spin15.h5"
        },
    },
    "analytic": {
        "xpiby2": {
            "experiment_name": "spin14",
            "controls_file_name": "00004_spin14.h5"
        },
        "ypiby2": {
            "experiment_name": "spin14",
            "controls_file_name": "00002_spin14.h5"
        },
        "zpiby2": {
            "experiment_name": "spin14",
            "controls_file_name": "00003_spin14.h5"
        },
    }
}


# misc constants
SEED = 0
GATE_COUNT = 1000


class GateType(Enum):
    XPIBY2 = 0
    YPIBY2 = 1
    ZPIBY2 = 2
#ENDDEF


def fidelity_vec(v1, v2):
    ip = np.matmul(conjugate_transpose(v1), v2)[0, 0]
    return np.real(ip) ** 2 + np.imag(ip) ** 2


def fidelity_mat(m1, m2):
    ip = np.matmul(conjugate_transpose(m1), m2)[0, 0]
    return np.real(ip) ** 2 + np.imag(ip) ** 2


def run_sim(class_data, gate_sequence):
    # build controls and matrices from gate sequence
    gate_count = gate_sequence.shape[0]
    controls_list = list()
    running_state = INITIAL_STATE
    running_time = 0.
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
        #ENDIF
        state_array[i] = running_state
        time_array[i] = running_time
    #ENDFOR
    controls = np.concatenate(controls_list)
    
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
    tlist = np.linspace(0, evolution_time, control_eval_count)
    c_ops = []
    e_ops = []

    # run simulation
    result = mesolve(hlist, rho0, tlist, c_ops, e_ops)

    # analysis
    densities = result.states
    fidelity_array = np.zeros(time_array.shape[0])
    j = 0
    for i, t in enumerate(tlist):
        if t == time_array[j]:
            density = densities[i].full()
            target_state = state_array[j]
            target_density = np.matmul(target_state, conjugate_transpose(target_state))
            fidelity_ = fidelity_mat(density, target_density)
            print("tlist[{}]: {}, time_array[{}]: {}\n"
                  "density:\n{}\ntarget_density:\n{}\nfidelity: {}"
                  "".format(i, t, j, time_array[j],
                            density, target_density, fidelity_))
            fidelity_array[j] = fidelity_
            j = j + 1
        #ENDIF
    #ENDFOR
    
    return fidelity_array
#ENDDEF


def run_all():
    pulse_data = copy(PULSE_DATA)
    # generate gate sequence
    # np.random.seed(SEED)
    # gate_sequence = np.random.randint(GateType.XPIBY2.value,
    #                                   GateType.ZPIBY2.value + 1, GATE_COUNT)
    gate_sequence = np.array([1])
    
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
    
    # grab controls
    for class_key in pulse_data.keys():
        for gate_key in pulse_data[class_key].keys():
            gate_dict = pulse_data[class_key][gate_key]
            controls_save_file_path = os.path.join(
                META_PATH, gate_dict["experiment_name"], gate_dict["controls_file_name"]
            )
            with h5py.File(controls_save_file_path, "r") as save_file:
                gate_dict["evolution_time"] = save_file["evolution_time"][()]
                gate_dict["controls"] = save_file["controls"][()]
            #ENDWITH
        #ENDFOR
    #ENDFOR

    fidelity_array = run_sim(pulse_data["analytic"], gate_sequence)

    # # run sim
    # for class_key in pulse_data.keys():
    #     fidelity_array = run_sim(pulse_data[class_key], gate_sequence)
    #     with h5py.File(save_file_path, "a") as save_file:
    #         save_file["{}_fidelities".format(class_key)] = fidelity_array
    #     #ENDWITH
    # #ENDFOR
#ENDDEF

    
def main():
    run_all()


if __name__ == "__main__":
    main()
