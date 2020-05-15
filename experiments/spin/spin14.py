"""
spin14.py - recreate the pulses from the paper

Refs:
[0] https://arxiv.org/abs/2002.10653
"""

from argparse import ArgumentParser
from enum import Enum
import os

import h5py
import numpy as np
from qoc.standard import (
    conjugate_transpose,
    generate_save_file_path,
)
from qutip import (
    mesolve, Qobj,
)

class Shape(Enum):
    SQUARE = 0
    TRIANGLE = 1

# Directory.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin14"
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
SAVE_PATH = os.path.join(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Computational constants.
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])
MAX_CONTROL_0 = 2 * np.pi * 3e-1
OMEGA = 2 * np.pi * 1.4e-2
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2

# Pulse parameters.
AMP_0 = 2 * np.pi * 1.25e-1
T_XZ = 4.800613
T_Z = 11.928691
T_ZPI = 35.714285714285715
# T_ZPI = 35.9
T_YPIBY2 = 2 * T_XZ + T_Z
T_TOT = 2 * T_YPIBY2 + T_ZPI
T0 = 0
T1 = T_XZ / 2
T2 = T_XZ
T3 = T2 + T_Z
T4 = T3 + T_XZ / 2
T5 = T3 + T_XZ
T6 = T5 + T_ZPI
T7 = T6 + T_XZ / 2
T8 = T6 + T_XZ
T9 = T8 + T_Z
T10 = T9 + T_XZ / 2
T11 = T9 + T_XZ

# Evolution parameters.
DT = 1e-2
N_TOT = int(T_TOT / DT)

print("t_tot: {}, t11: {}, n_tot: {}"
      "".format(T_TOT, T11, N_TOT))

def gen_controls(t, shape=Shape.SQUARE):
    # Y/2
    if t <= T5:
        if shape == Shape.SQUARE:
            if t <= T2:
                c1 = AMP_0 / 2
            elif t <= T3:
                c1 = 0
            elif t <= T5:
                c1 = -AMP_0 / 2
        elif shape == Shape.TRIANGLE:
            if t <= T1:
                c1 = 2 * AMP_0 * t / T_XZ
            elif t <= T2:
                c1 = 2 * AMP_0 * (1 - t / T_XZ)
            elif t <= T3:
                c1 = 0
            elif t <= T4:
                c1 = 2 * AMP_0 / T_XZ * (T3 - t)
            elif t <= T5:
                c1 = 2 * AMP_0 / T_XZ * (t - T4) - AMP_0
        #ENDIF
    # Z
    elif t <= T6:
        c1 = 0
    # -Y/2
    else:
        if shape == Shape.SQUARE:
            if t <= T8:
                c1 = -AMP_0 / 2
            elif t <= T9:
                c1 = 0
            elif t <= T11:
                c1 = AMP_0 / 2
        elif shape == Shape.TRIANGLE:
            if t <= T7:
                c1 = 2 * AMP_0 / T_XZ * (T6 - t)
            elif t <= T8:
                c1 = 2 * AMP_0 / T_XZ * (t - T7) - AMP_0
            elif t <= T9:
                c1 = 0
            elif t <= T10:
                c1 = 2 * AMP_0 / T_XZ * (t - T9)
            elif t <= T11:
                c1 = 2 * AMP_0 / T_XZ * (T10 - t) + AMP_0
            else:
                c1 = 0
        #ENDIF
    #ENDIF
    
    return [c1]


def save_controls():
    shape = Shape.TRIANGLE
    control_eval_times = np.linspace(0, T_TOT, N_TOT)
    controls = np.array([gen_controls(t, shape=shape) for t in control_eval_times])
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    with h5py.File(save_file_path, "a") as save_file:
        save_file["complex_controls"] = False
        save_file["evolution_time"] = T_TOT
        save_file["controls"] = np.stack([controls,])
    # ENDWITH
    print("saved controls to {}"
          "".format(save_file_path))

    
def main():
    do_controls = True
    # do_controls = False
    
    if do_controls:
        save_controls()


if __name__ == "__main__":
    main()

