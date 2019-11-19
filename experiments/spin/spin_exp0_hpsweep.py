"""
spin_exp0_hpsweep.py - Do a sweep of the hamiltonian parameters
for spin_exp0.
"""

import os

import numpy as np
from rbqoc.util import (hamiltonian_args_sweep,
                        plot_hamiltonian_args_sweep,)
from spin_exp0 import (EVOL_ARGS, EXPERIMENT_META, EXPERIMENT_NAME, HAMILTONIAN_ARGS)

# Sweep trials
SWEEP_TRIAL_COUNT = 100
SWEEP_MULTIPLIERS = np.linspace(0, 3, SWEEP_TRIAL_COUNT)
HAMILTONIAN_ARGS_SWEEP = np.stack([HAMILTONIAN_ARGS] * SWEEP_TRIAL_COUNT)
for i, multiplier in enumerate(SWEEP_MULTIPLIERS):
    HAMILTONIAN_ARGS_SWEEP[i] *= SWEEP_MULTIPLIERS[i]

# I/O
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
PULSE_FILE_0 = "00000_spin_exp0.h5"
PULSE_FILE_1 = "00001_spin_exp0.h5"
PULSE_FILE_2 = "00013_spin_exp0.h5"
PULSE_FILE_3 = "00010_spin_exp0.h5"
PULSE_FILES = [PULSE_FILE_0, PULSE_FILE_1, PULSE_FILE_2, PULSE_FILE_3]
PULSE_PATH_0 = os.path.join(SAVE_PATH, PULSE_FILE_0)
PULSE_PATH_1 = os.path.join(SAVE_PATH, PULSE_FILE_1)
PULSE_PATH_2 = os.path.join(SAVE_PATH, PULSE_FILE_2)
PULSE_PATH_3 = os.path.join(SAVE_PATH, PULSE_FILE_3)
PULSE_PATHS = [PULSE_PATH_0, PULSE_PATH_1, PULSE_PATH_2, PULSE_PATH_3]
SWEEP_SUFFIX = "freq_sweep"
PLOT_SAVE_FILE = "{}_{}.png".format(EXPERIMENT_NAME, SWEEP_SUFFIX)
PLOT_SAVE_PATH = os.path.join(SAVE_PATH, PLOT_SAVE_FILE)

def main():
    sweep_path_list = list()
    # Sweep the frequency parameter for each pulse.
    for i, pulse_path in enumerate(PULSE_PATHS):
        save_path = hamiltonian_args_sweep(EVOL_ARGS, HAMILTONIAN_ARGS_SWEEP,
                                           pulse_path, SWEEP_SUFFIX)
        sweep_path_list.append(save_path)

    # Plot the sweeps together.
    plot_hamiltonian_args_sweep(PLOT_SAVE_PATH, sweep_path_list,
                                title=EXPERIMENT_NAME,
                                x_label="GHz",
                                y_label="Infidelity")


if __name__ == "__main__":
    main()
