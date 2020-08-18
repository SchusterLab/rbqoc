"""
spincnstr_hpsweep.py - Do a sweep of the hamiltonian parameters
for spincnstr.
"""

import os

import h5py
from filelock import FileLock, Timeout
import matplotlib.pyplot as plt
import numpy as np
from rbqoc.util import (hamiltonian_args_sweep,
                        plot_hamiltonian_args_sweep,)
from spincnstr import (EVOL_CONFIG, EXPERIMENT_META, EXPERIMENT_NAME, HAMILTONIAN_ARGS)

# Plot
DPI = 1000

# Sweep trials
SWEEP_TRIAL_COUNT = 100
SWEEP_MULTIPLIERS = np.linspace(0.9, 1.1, SWEEP_TRIAL_COUNT)
HAMILTONIAN_ARGS_SWEEP = np.stack([HAMILTONIAN_ARGS] * SWEEP_TRIAL_COUNT)
for i, multiplier in enumerate(SWEEP_MULTIPLIERS):
    HAMILTONIAN_ARGS_SWEEP[i] *= SWEEP_MULTIPLIERS[i]

# I/O
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
SAVE_FILE_PATH = os.path.join(SAVE_PATH, "{}_hpsweep_plot.png".format(EXPERIMENT_NAME))
PULSE_FILES = [
    # "00033_spincnstr.h5",
    "00034_spincnstr.h5",
    "00035_spincnstr.h5",
    "00036_spincnstr.h5",
    "00037_spincnstr.h5",
    # "00038_spincnstr.h5",
    # "00039_spincnstr.h5",
    "00040_spincnstr.h5",
    "00041_spincnstr.h5",
]
TIMES = [
    # 150,
    100,
    80,
    60,
    120,
    # 70,
    # 90,
    110,
    130,
]
PULSE_PATHS = list()
for pulse_file in PULSE_FILES:
    PULSE_PATHS.append(os.path.join(SAVE_PATH, pulse_file))
SWEEP_SUFFIX = "freq_sweep"
PLOT_SAVE_FILE = "{}_{}.png".format(EXPERIMENT_NAME, SWEEP_SUFFIX)
PLOT_SAVE_PATH = os.path.join(SAVE_PATH, PLOT_SAVE_FILE)

def plot(sweep_path_list):
    freq = HAMILTONIAN_ARGS[0]
    freq1_lo = freq * 0.99
    freq1_hi = freq * 1.01
    freq1_los = list()
    freq1_his = list()
    freq2_lo = freq * 0.999
    freq2_hi = freq * 1.001
    freq2_los = list()
    freq2_his = list()
    freq3_lo = freq * 1
    freq3_hi = freq * 1
    freq3_los = list()
    freq3_his = list()
    for sweep_path in sweep_path_list:
        lock_path = "{}.lock".format(sweep_path)
        try:
            with FileLock(lock_path):
                with h5py.File(sweep_path) as f:
                    freq1_lo_index = np.nonzero((f["hamiltonian_args"][()] - freq1_lo) > 0)[0][0]
                    freq1_hi_index = np.nonzero((f["hamiltonian_args"][()] - freq1_hi) > 0)[0][0]
                    freq1_los.append(f["error"][freq1_lo_index])
                    freq1_his.append(f["error"][freq1_hi_index])
                    freq2_lo_index = np.nonzero((f["hamiltonian_args"][()] - freq2_lo) > 0)[0][0]
                    freq2_hi_index = np.nonzero((f["hamiltonian_args"][()] - freq2_hi) > 0)[0][0]
                    freq2_los.append(f["error"][freq2_lo_index])
                    freq2_his.append(f["error"][freq2_hi_index])
                    freq3_lo_index = np.nonzero((f["hamiltonian_args"][()] - freq3_lo) > 0)[0][0]
                    freq3_hi_index = np.nonzero((f["hamiltonian_args"][()] - freq3_hi) > 0)[0][0]
                    freq3_los.append(f["error"][freq3_lo_index])
                    freq3_his.append(f["error"][freq3_hi_index])
                #ENDWITH
            #ENDWITH
        except Timeout:
            print("timeout")
        #ENDTRY
    #ENDFOR

    fig = plt.figure
    err1 = list()
    err2 = list()
    err3 = list()
    for index, time in enumerate(TIMES):
        err1_ = np.maximum(freq1_los[index], freq1_his[index])
        err1.append(err1_)
        err2_ = np.maximum(freq2_los[index], freq2_his[index])
        err2.append(err2_)
        err3_ = np.maximum(freq3_los[index], freq3_his[index])
        err3.append(err3_)
    plt.scatter(TIMES, err1, label="1%", color="blue")
    plt.scatter(TIMES, err2, label="0.1%", color="red")
    # plt.scatter(TIMES, err3, label="0.01%", color="green")
    plt.legend()
    plt.xlabel("Time (ns)")
    plt.ylabel("Infidelity")
    plt.savefig(SAVE_FILE_PATH, dpi=DPI)


def main():
    sweep_path_list = list()
    # Sweep the frequency parameter for each pulse.
    for i, pulse_path in enumerate(PULSE_PATHS):
        save_path = hamiltonian_args_sweep(EVOL_CONFIG, HAMILTONIAN_ARGS_SWEEP,
                                           pulse_path, SWEEP_SUFFIX)
        sweep_path_list.append(save_path)

    # Plot results
    plot(sweep_path_list)


if __name__ == "__main__":
    main()
