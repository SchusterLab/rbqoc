"""
spin12_hpsweep.py - hp sweep for spin11 and spin12
"""

from argparse import ArgumentParser
from enum import Enum
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

class PulseType(Enum):
    TRAJ = 0
    QOC = 1
#ENDDEF

# Directory.
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
OUT_PATH = os.path.join(WDIR, "out")
SAVE_PATH = os.path.join(OUT_PATH, "spin", "spin12")
EXPERIMENT_NAME = "spin12_hpsweep"
EXPERIMENT_META = "spin"

# Data constants.
PULSE_DATA = [
    ("spin12", "00018_spin12.h5", 13, PulseType.TRAJ, "sample"),
    ("spin12", "00027_spin12.h5", 21, PulseType.TRAJ, "sample 2"),
]

# Sweep building.
SWEEP_COUNT = int(2e2)
SWEEP_MULTIPLIERS = np.linspace(-0.1, 0.1, SWEEP_COUNT)


# other constants
DPI = int(5e2)
ALPHA = 0.5
MS = 15
LINE_WIDTH = 1
LINE_ALPHA = 0.8
LINE_COLOR = "black"
COLORS = [
    "blue", "red", "green", "orange", "purple"
]


# Computational constants.
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2
OMEGA_RAW = 2 * np.pi * 1.4e-2
OMEGA_STD = 5e-2
DOMEGA = OMEGA_RAW * OMEGA_STD
OMEGA_PLUS = OMEGA_RAW + DOMEGA
OMEGA_MINUS = OMEGA_RAW - DOMEGA


def fidelity(v1, v2):
    ip = np.matmul(conjugate_transpose(v1), v2)[0, 0]
    return np.real(ip) ** 2 + np.imag(ip) ** 2


def grab_controls(experiment_name, controls_file_name,
                  controls_idx,
                  pulse_type,
                  experiment_meta="spin"):
    save_path = os.path.join(OUT_PATH, experiment_meta, experiment_name)
    controls_file_path = os.path.join(save_path, controls_file_name)
    with h5py.File(controls_file_path, "r") as save_file:
        if pulse_type == PulseType.TRAJ:
            states = save_file["states"][()]
            controls = states[controls_idx, 0:-1]
            evolution_time = save_file["evolution_time"][()]
        elif pulse_type == PulseType.QOC:
            controls = save_file["controls"][controls_idx][()]
            evolution_time = save_file["evolution_time"][()]
    #ENDWITH
    return controls, evolution_time
#ENDDEF


def run_spin(controls, domega_multiplier, evolution_time):
    control_eval_count = controls.shape[0]
    
    # define constants
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


def run_tsweep(save_file_path=None):
    omegas = np.array([OMEGA_PLUS, OMEGA_MINUS])
    
    if save_file_path is None:
        # initialize
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        with h5py.File(save_file_path, "a") as save_file:
            save_file["omegas"] = omegas
        #ENDWITH

        # sweep
        for i, pulse_data in enumerate(PULSE_DATA):
            experiment_name = pulse_data[0]
            pulse_type = pulse_data[1]
            controls_idx = pulse_data[2]
            label = pulse_data[3]
            controls_file_names = pulse_data[4]
            fidelities = list()
            evolution_times = list()
            for j, controls_file_name in enumerate(controls_file_names):
                controls, evolution_time = grab_controls(experiment_name, controls_file_name,
                                                         controls_idx, pulse_type)
                fidelity_plus = run_spin(controls, OMEGA_STD, evolution_time)
                fidelity_minus = run_spin(controls, -OMEGA_STD, evolution_time)
                fidelity_avg = (fidelity_plus + fidelity_minus) / 2
                fidelities.append(fidelity_avg)
                evolution_times.append(evolution_time)
            #ENDFOR
            controls_file_names_ = np.array(list(map(np.string_, np.array(controls_file_names))))
            with h5py.File(save_file_path, "a") as save_file:
                save_file["{}/file_names".format(label)] = controls_file_names_
                save_file["{}/evolution_times".format(label)] = np.array(evolution_times)
                save_file["{}/fidelities".format(label)] = np.array(fidelities)
            #ENDWITH
        #ENDFOR
    #ENDIF

    # plot
    save_file_path_prefix = os.path.basename(save_file_path).split(".h5")[0]
    save_image_path = os.path.join(SAVE_PATH, "{}.png".format(save_file_path_prefix))
    pulse_count = len(PULSE_DATA)
    plt.figure()
    for i, pulse_data in enumerate(PULSE_DATA):
        label = pulse_data[3]
        # if label == "vanilla":
        #     continue
        with h5py.File(save_file_path) as save_file:
            fidelities = save_file[label]["fidelities"][()]
            evolution_times = save_file[label]["evolution_times"][()]
        #ENDWITH
        plt.scatter(evolution_times, fidelities, color=COLORS[i],
                    s=MS, alpha=ALPHA, label=label)
    #ENDFOR
    plt.axhline(SPIN14_AF, label="analytic (80 ns)", color=COLORS[pulse_count],
                lw=LINE_WIDTH, alpha=LINE_ALPHA)
    plt.xlabel("Evolution Time (ns)")
    plt.ylabel("Avg. Fidelity")
    plt.ylim(top=1.,)
    plt.legend()
    plt.savefig(save_image_path, dpi=DPI)
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
        for i, pulse_data in enumerate(PULSE_DATA):
            experiment_name = pulse_data[0]
            file_name = pulse_data[1]
            controls_idx = pulse_data[2]
            pulse_type = pulse_data[3]
            controls, evolution_time = grab_controls(experiment_name, file_name, controls_idx, pulse_type)
            for j, sweep_multiplier in enumerate(SWEEP_MULTIPLIERS):
                fidelities[i][j] = run_spin(controls, sweep_multiplier, evolution_time)
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
        plt.scatter(omegas, fidelities[i], color=COLORS[i], label=PULSE_DATA[i][4],
                    s=MS, alpha=ALPHA)
    #ENDFOR
    # plt.axhline(1., lw=LINE_WIDTH, alpha=LINE_ALPHA, color=LINE_COLOR)
    plt.axvline(OMEGA_PLUS / (np.pi * 2), lw=LINE_WIDTH, alpha=LINE_ALPHA, color=LINE_COLOR)
    plt.axvline(OMEGA_MINUS / (np.pi * 2), lw=LINE_WIDTH, alpha=LINE_ALPHA, color=LINE_COLOR)
    plt.ylim(top=1., bottom=0.998)
    plt.ylabel("Fidelity")
    plt.xlabel("$\omega$ (GHz)")
    plt.legend()
    plt.savefig(save_image_path, dpi=DPI)
#ENDDEF

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--tsweep", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--file", type=str, default=None, action="store")
    args = vars(parser.parse_args())
    data_file = args["file"]
    do_sweep = args["sweep"]
    do_tsweep = args["tsweep"]
    if data_file is not None:
        data_file = os.path.join(SAVE_PATH, args["file"])
    #ENDIF
    
    if do_sweep:
        run_sweep(data_file)
    #ENDIF
    if do_tsweep:
        run_tsweep(data_file)
    #ENDIF


if __name__ == "__main__":
    main()

PULSE_DATA_6 = (
    ("spin11", PulseType.TRAJ, 13, "derivative",
     (
         "00019_spin11.h5", "00020_spin11.h5",
         "00021_spin11.h5", "00022_spin11.h5",
         "00023_spin11.h5", "00024_spin11.h5",
         "00025_spin11.h5", "00026_spin11.h5",
         "00027_spin11.h5", "00028_spin11.h5",
         "00029_spin11.h5", "00030_spin11.h5",
         "00031_spin11.h5",
     ),
    ),
    ("spin12", PulseType.TRAJ, 13, "sample",
     (
         "00006_spin12.h5", "00007_spin12.h5",
         "00008_spin12.h5", "00009_spin12.h5",
         "00010_spin12.h5", "00011_spin12.h5",
         "00012_spin12.h5", "00013_spin12.h5",
         "00014_spin12.h5", "00015_spin12.h5",
         "00016_spin12.h5", "00017_spin12.h5",
         "00018_spin12.h5",
     ),
    ),
    ("spin12", PulseType.TRAJ, 21, "sample2",
     (
         "00019_spin12.h5", "00020_spin12.h5",
         "00021_spin12.h5", "00022_spin12.h5",
         "00023_spin12.h5", "00024_spin12.h5",
         "00025_spin12.h5", "00026_spin12.h5",
         "00027_spin12.h5", "00028_spin12.h5",
         "00029_spin12.h5", "00030_spin12.h5",
     ),
    ),
    ("spin13", PulseType.TRAJ, 5, "vanilla",
     (
         "00002_spin13.h5", "00003_spin13.h5",
         "00004_spin13.h5", "00005_spin13.h5",
         "00006_spin13.h5", "00007_spin13.h5",
         "00008_spin13.h5", "00009_spin13.h5",
         "00010_spin13.h5", "00011_spin13.h5",
         "00012_spin13.h5", "00013_spin13.h5",
         "00014_spin13.h5",
     ),
    ),
)
SPIN14_AF = 0.989004164533118

PULSE_DATA_5 = [
    ("spin11", "00023_spin11.h5", 13, PulseType.TRAJ, "derivative"),
    ("spin12", "00010_spin12.h5", 13, PulseType.TRAJ, "sample"),
    ("spin12", "00022_spin12.h5", 21, PulseType.TRAJ, "sample 2"),
    ("spin13", "00006_spin13.h5", 5, PulseType.TRAJ, "vanilla"),
    ("spin14", "00000_spin14.h5", 0, PulseType.QOC, "analytic"),
]
