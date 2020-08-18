"""
spin_exp0_ampsweep.py - Do a sweep of the control amplitude multiplier
for spin_exp0.
"""

import os

import numpy as np
from rbqoc.util import (amp_sweep, plot_amp_sweep)
from spin_exp0 import (EVOL_CONFIG, EXPERIMENT_NAME, EXPERIMENT_META)

# Sweep trials
SWEEP_TRIAL_COUNT = 100
AMP_MULTIPLIERS = np.linspace(0.98, 1.02, SWEEP_TRIAL_COUNT)


# I/O
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
PULSE_FILE_0 = "00000_spin_exp0.h5"
PULSE_FILE_1 = "00001_spin_exp0.h5"
PULSE_FILES = (
    PULSE_FILE_0,
    PULSE_FILE_1,
)
PULSE_PATH_0 = os.path.join(SAVE_PATH, PULSE_FILE_0)
PULSE_PATH_1 = os.path.join(SAVE_PATH, PULSE_FILE_1)
PULSE_PATHS = (
    PULSE_PATH_0,
    PULSE_PATH_1,
)
SWEEP_SUFFIX = "amp_sweep"
PLOT_SAVE_FILE = "{}_{}.png".format(EXPERIMENT_NAME, SWEEP_SUFFIX)
PLOT_SAVE_PATH = os.path.join(SAVE_PATH, PLOT_SAVE_FILE)

def main():
    sweep_path_list = list()
    # Sweep the amplitude parameter for each pulse.
    for i, pulse_path in enumerate(PULSE_PATHS):
        save_path = amp_sweep(AMP_MULTIPLIERS, EVOL_CONFIG,
                              pulse_path, SWEEP_SUFFIX)
        sweep_path_list.append(save_path)

    # Plot the sweeps together.
    plot_amp_sweep(PLOT_SAVE_PATH, sweep_path_list,
                                title=EXPERIMENT_NAME,
                                x_label="Amplitude Multiplier",
                                y_label="Infidelity")


if __name__ == "__main__":
    main()
