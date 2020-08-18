"""
spin_pi_corpse_hpsweep.py - Do a sweep of the hamiltonian parameters
for spin_pi_corpse.
"""

import os

import numpy as np
from rbqoc.util import (hamiltonian_args_sweep,
                        plot_hamiltonian_args_sweep,)
from spin_pi_corpse import (EVOL_CONFIG, EXPERIMENT_META, EXPERIMENT_NAME, HAMILTONIAN_ARGS)

EVOL_CONFIG.pop("controls", None)
EVOL_CONFIG.pop("hamiltonian_args", None)

# Sweep trials
SWEEP_TRIAL_COUNT = 100
HAMILTONIAN_ARGS_SWEEP = np.linspace(2e-2, 1e-1, SWEEP_TRIAL_COUNT)
HAMILTONIAN_ARGS_SWEEP = np.expand_dims(HAMILTONIAN_ARGS_SWEEP, axis=1)
# SWEEP_MULTIPLIERS = np.linspace(0, 3, SWEEP_TRIAL_COUNT)
# HAMILTONIAN_ARGS_SWEEP = np.stack([HAMILTONIAN_ARGS] * SWEEP_TRIAL_COUNT)
# for i, multiplier in enumerate(SWEEP_MULTIPLIERS):
#     HAMILTONIAN_ARGS_SWEEP[i] *= SWEEP_MULTIPLIERS[i]


# I/O
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH_0 = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
SAVE_PATH_1 = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, "spin_exp0")
PULSE_FILE_0 = "00000_spin_pi_corpse.h5"
PULSE_FILE_1 = "00015_spin_exp0.h5"
PULSE_FILE_2 = "00017_spin_exp0.h5"
LABELS = [
    "CORPSE",
    "QOC",
    "RBQOC",
]
PULSE_FILES = [PULSE_FILE_0,
               PULSE_FILE_1,
               PULSE_FILE_2,
]
PULSE_PATH_0 = os.path.join(SAVE_PATH_0, PULSE_FILE_0)
PULSE_PATH_1 = os.path.join(SAVE_PATH_1, PULSE_FILE_1)
PULSE_PATH_2 = os.path.join(SAVE_PATH_1, PULSE_FILE_2)
PULSE_PATHS = [
    PULSE_PATH_0,
    PULSE_PATH_1,
    PULSE_PATH_2,
]
SWEEP_SUFFIX = "freq2"
PLOT_SAVE_FILE = "{}_{}.png".format(EXPERIMENT_NAME, SWEEP_SUFFIX)
PLOT_SAVE_PATH = os.path.join(SAVE_PATH_0, PLOT_SAVE_FILE)

def main():
    sweep_path_list = list()
    # Sweep the frequency parameter for each pulse.
    for i, pulse_path in enumerate(PULSE_PATHS):
        save_path = hamiltonian_args_sweep(EVOL_CONFIG, HAMILTONIAN_ARGS_SWEEP,
                                           pulse_path, SWEEP_SUFFIX)
        sweep_path_list.append(save_path)

    # Plot the sweeps together.
    plot_hamiltonian_args_sweep(PLOT_SAVE_PATH, sweep_path_list,
                                labels=LABELS,
                                title=EXPERIMENT_NAME,
                                x_label="GHz",
                                x_min=4e-2, x_max=8e-2,
                                y_label="Infidelity",
                                y_max=2e-2, y_min=0)



if __name__ == "__main__":
    main()
