"""
spin12_hpsweep.py - hp sweep for spin11 and spin12
"""

from argparse import ArgumentParser
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from qoc.standard import (
    conjugate_transpose,
    generate_save_file_path,
)
from qutip import (
    mesolve, Qobj,
)

# Directory.
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
OUT_PATH = os.path.join(WDIR, "out")
SAVE_PATH = os.path.join(OUT_PATH, "spin", "spin12")
PULSE_DATA = [
    ("spin12", "00005_spin12.h5", 21, "sample 2"),
    ("spin12", "00004_spin12.h5", 13, "sample"),
]
EXPERIMENT_NAME = "spin12_hpsweep"

# Computational constants.
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2
OMEGA_RAW = 2 * np.pi * 1.4e-2
DOMEGA = OMEGA_RAW * 5e-2
OMEGA_PLUS = OMEGA_RAW + DOMEGA
OMEGA_MINUS = OMEGA_RAW - DOMEGA


# Sweep building.
SWEEP_COUNT = int(2e2)
SWEEP_MULTIPLIERS = np.linspace(-0.1, 0.1, SWEEP_COUNT)

# other constants
DPI = int(5e2)
ALPHA = 0.5
MS = 2
LINE_WIDTH = 1
LINE_ALPHA = 0.8
LINE_COLOR = "black"
COLORS = [
    "blue", "red", "green", "orange", "purple"
]

def fidelity(v1, v2):
    ip = np.matmul(conjugate_transpose(v1), v2)[0, 0]
    return np.real(ip) ** 2 + np.imag(ip) ** 2


def grab_controls(experiment_name, controls_file_name,
                  controls_idx,
                  experiment_meta="spin"):
    save_path = os.path.join(OUT_PATH, experiment_meta, experiment_name)
    controls_file_path = os.path.join(save_path, controls_file_name)
    with h5py.File(controls_file_path, "r") as save_file:
        states = save_file["states"][()]
        controls = states[controls_idx, 0:-1]
    #ENDWITH
    return controls
#ENDDEF


def run_spin(controls, domega_multiplier):
    control_eval_count = controls.shape[0]
    
    # define constants
    evolution_time = 120.
    domega = OMEGA_RAW * domega_multiplier
    omega = (
        OMEGA_RAW
        + domega
    )
    initial_state = np.array([[1], [0]])
    target_state = np.array([[0], [1]])

    # build simulation
    h_sys = Qobj(omega * H_S)
    h_c1 = Qobj(H_C1)
    hlist = [
        [h_sys, np.ones(control_eval_count)],
        [h_c1, controls]
    ]
    rho0 = Qobj(initial_state)
    tlist = np.linspace(0, evolution_time, control_eval_count)

    # run simulation
    result = mesolve(hlist, rho0, tlist)

    # analysis
    final_state = result.states[-1].full()
    fidelity_ = fidelity(final_state, target_state)

    # log
    # print("fidelity:\n{}\nfinal_state:\n{}"
    #       "".format(fidelity_, final_state))
    return fidelity_
#ENDDEF


def run_sweep(save_file_path=None):
    pulse_count = len(PULSE_DATA)
    if save_file_path is None:
        # initialize
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        pulse_names = np.array(list(map(np.string_, np.array(PULSE_DATA)[:, 1])))
        with h5py.File(save_file_path, "a") as save_file:
            save_file["sweep_multipliers"] = SWEEP_MULTIPLIERS
            save_file["pulse_data"] = pulse_names
            save_file["omega"] = OMEGA_RAW
        #ENDWITH

        # sweep
        fidelities = np.zeros((pulse_count, SWEEP_COUNT))
        for i, (experiment_name, file_name, controls_idx, _) in enumerate(PULSE_DATA):
            controls = grab_controls(experiment_name, file_name, controls_idx)
            for j, sweep_multiplier in enumerate(SWEEP_MULTIPLIERS):
                fidelities[i][j] = run_spin(controls, sweep_multiplier)
            #ENDFOR
        #ENDFOR
        with h5py.File(save_file_path, "a") as save_file:
            save_file["fidelities"] = fidelities
        #ENDWITH
    else:
        with h5py.File(save_file_path, "a") as save_file:
            fidelities = save_file["fidelities"][()]
    #ENDIF

    # Plot
    save_file_path_prefix = os.path.basename(save_file_path).split(".h5")[0]
    save_image_path = os.path.join(SAVE_PATH, "{}.png".format(save_file_path_prefix))
    omegas = (OMEGA_RAW + OMEGA_RAW * SWEEP_MULTIPLIERS) / (2 * np.pi)
    plt.figure()
    for i in range(pulse_count):
        plt.scatter(omegas, fidelities[i], color=COLORS[i], label=PULSE_DATA[i][3],
                    s=MS, alpha=ALPHA)
    #ENDFOR
    # plt.axhline(1., lw=LINE_WIDTH, alpha=LINE_ALPHA, color=LINE_COLOR)
    plt.axvline(OMEGA_PLUS / (np.pi * 2), lw=LINE_WIDTH, alpha=LINE_ALPHA, color=LINE_COLOR)
    plt.axvline(OMEGA_MINUS / (np.pi * 2), lw=LINE_WIDTH, alpha=LINE_ALPHA, color=LINE_COLOR)
    plt.ylim(top=1., bottom=0.99)
    plt.ylabel("Fidelity")
    plt.xlabel("$\omega$ (GHz)")
    plt.legend()
    plt.savefig(save_image_path, dpi=DPI)
#ENDDEF

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--file", type=str, default=None, action="store")
    args = vars(parser.parse_args())
    do_sweep = args["sweep"]
    data_file = args["file"]
    if data_file is not None:
        data_file = os.path.join(SAVE_PATH, args["file"])
    #ENDIF
    
    if do_sweep:
        run_sweep(data_file)
    #ENDIF


if __name__ == "__main__":
    main()
