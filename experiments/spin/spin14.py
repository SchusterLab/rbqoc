"""
spin14.py - recreate the pulses from the paper

Refs:
[0] https://arxiv.org/abs/2002.10653
"""

from argparse import ArgumentParser
from enum import Enum
import os

import h5py
import matplotlib.pyplot as plt
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

# paths
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin14"
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
SAVE_PATH = os.path.join(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# misc constants
DPI = 300

# computational constants
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])

MAX_CONTROL_0 = 2 * np.pi * 3e-1
OMEGA = 2 * np.pi * 1.4e-2
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2

AMP_0 = 2 * np.pi * 1.25e-1
DT = 1e-2
DT_INV = 1e2
T_ZPI = 35.714285714285715
T_TOT_ZPIBY2 = 17.85714285714286

# XPI pulse parameters
T_XZ_XPI = 4.800613
T_Z_XPI = 11.928691
T_YPIBY2_XPI = 2 * T_XZ_XPI + T_Z_XPI
T_TOT_XPI = 2 * T_YPIBY2_XPI + T_ZPI
N_TOT_XPI = int(T_TOT_XPI / DT)
T0_XPI = 0
T1_XPI = T_XZ_XPI / 2
T2_XPI = T_XZ_XPI
T3_XPI = T2_XPI + T_Z_XPI
T4_XPI = T3_XPI + T_XZ_XPI / 2
T5_XPI = T3_XPI + T_XZ_XPI
T6_XPI = T5_XPI + T_ZPI
T7_XPI = T6_XPI + T_XZ_XPI / 2
T8_XPI = T6_XPI + T_XZ_XPI
T9_XPI = T8_XPI + T_Z_XPI
T10_XPI = T9_XPI + T_XZ_XPI / 2
T11_XPI = T9_XPI + T_XZ_XPI


def gen_controls_xpi(t, shape=Shape.SQUARE):
    # Y/2
    if t <= T5_XPI:
        if shape == Shape.SQUARE:
            if t <= T2_XPI:
                c1 = AMP_0 / 2
            elif t <= T3_XPI:
                c1 = 0
            elif t <= T5_XPI:
                c1 = -AMP_0 / 2
        elif shape == Shape.TRIANGLE:
            if t <= T1_XPI:
                c1 = 2 * AMP_0 * t / T_XZ_XPI
            elif t <= T2_XPI:
                c1 = 2 * AMP_0 * (1 - t / T_XZ_XPI)
            elif t <= T3_XPI:
                c1 = 0
            elif t <= T4_XPI:
                c1 = 2 * AMP_0 / T_XZ_XPI * (T3_XPI - t)
            elif t <= T5_XPI:
                c1 = 2 * AMP_0 / T_XZ_XPI * (t - T4_XPI) - AMP_0
        #ENDIF
    # Z
    elif t <= T6_XPI:
        c1 = 0
    # -Y/2
    else:
        if shape == Shape.SQUARE:
            if t <= T8_XPI:
                c1 = -AMP_0 / 2
            elif t <= T9_XPI:
                c1 = 0
            elif t <= T11_XPI:
                c1 = AMP_0 / 2
        elif shape == Shape.TRIANGLE:
            if t <= T7_XPI:
                c1 = 2 * AMP_0 / T_XZ_XPI * (T6_XPI - t)
            elif t <= T8_XPI:
                c1 = 2 * AMP_0 / T_XZ_XPI * (t - T7_XPI) - AMP_0
            elif t <= T9_XPI:
                c1 = 0
            elif t <= T10_XPI:
                c1 = 2 * AMP_0 / T_XZ_XPI * (t - T9_XPI)
            elif t <= T11_XPI:
                c1 = 2 * AMP_0 / T_XZ_XPI * (T10_XPI - t) + AMP_0
            else:
                c1 = 0
        #ENDIF
    #ENDIF
    
    return np.array([c1,])


def save_controls_xpi():
    shape = Shape.TRIANGLE
    control_eval_times = np.linspace(0, T_TOT_XPI, N_TOT_XPI)
    controls = np.vstack([gen_controls_xpi(t, shape=shape) for t in control_eval_times])
    evolution_time = controls.shape[0] * DT
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    with h5py.File(save_file_path, "a") as save_file:
        save_file["complex_controls"] = False
        save_file["evolution_time"] = evolution_time
        save_file["controls"] = controls
    # ENDWITH
    print("saved xpi controls to {}"
          "".format(save_file_path))
#ENDDEF

    
T_XZ_YPIBY2 = 2.1656249366575766
T_Z_YPIBY2 = 15.142330599557274
T_TOT_YPIBY2 = 2 * T_XZ_YPIBY2 + T_Z_YPIBY2
T0_YPIBY2 = 0.
T1_YPIBY2 = T_XZ_YPIBY2 / 2
T2_YPIBY2 = T_XZ_YPIBY2
T3_YPIBY2 = T2_YPIBY2 + T_Z_YPIBY2
T4_YPIBY2 = T3_YPIBY2 + T_XZ_YPIBY2 / 2
T5_YPIBY2 = T3_YPIBY2 + T_XZ_YPIBY2

def gen_controls_ypiby2(t, shape=Shape.TRIANGLE):
    if shape == Shape.TRIANGLE:
        amp = 2 * AMP_0
        if t <= T1_YPIBY2:
            c1 = 2 * amp * t / T_XZ_YPIBY2
        elif t <= T2_YPIBY2:
            c1 = 2 * amp * (1 - t / T_XZ_YPIBY2)
        elif t <= T3_YPIBY2:
            c1 = 0
        elif t <= T4_YPIBY2:
            c1 = 2 * amp / T_XZ_YPIBY2 * (T3_YPIBY2 - t)
        else:
            c1 = 2 * amp / T_XZ_YPIBY2 * (t - T4_YPIBY2) - amp
        #ENDIF
    elif shape == Shape.SQUARE:
        amp = AMP_0
        if t <= T2_YPIBY2:
            c1 = amp
        elif t <= T3_YPIBY2:
            c1 = 0
        else:
            c1 = -amp
    #ENDIF

    return np.array([c1,])
#ENDDEF


def save_controls_ypiby2():
    # generate
    shape = Shape.SQUARE
    evolution_time = T_TOT_YPIBY2
    control_eval_count = int(np.floor(evolution_time * DT_INV))
    control_eval_times = np.arange(0, control_eval_count, 1) * DT
    controls = np.vstack([gen_controls_ypiby2(t, shape=shape) for t in control_eval_times])

    # save
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    with h5py.File(save_file_path, "a") as save_file:
        save_file["complex_controls"] = False
        save_file["evolution_time"] = evolution_time
        save_file["controls"] = controls
    # ENDWITH
    print("saved controls to {}"
          "".format(save_file_path))

    # plot
    file_prefix = save_file_path.split(".")[0]
    plot_file_path = "{}_controls.png".format(file_prefix)
    fig = plt.figure()
    plt.scatter(control_eval_times, controls,
                label="controls", color="blue")
    plt.title(file_prefix)
    plt.legend()
    plt.savefig(plot_file_path, dpi=DPI)
#ENDDEF


T_TOT_XPIBY2 = 2 * T_TOT_YPIBY2 + T_TOT_ZPIBY2
T1_XPIBY2 = T_TOT_YPIBY2
T2_XPIBY2 = T1_XPIBY2 + T_TOT_ZPIBY2


def gen_controls_xpiby2(t, shape=Shape.TRIANGLE):
    if t <= T1_XPIBY2:
        ret = gen_controls_ypiby2(t, shape=shape)
    elif t <= T2_XPIBY2:
        ret = np.array([0])
    else:
        ret = -gen_controls_ypiby2(t - T2_XPIBY2, shape=shape)
    #ENDIF

    return ret
#ENDDEF


def save_controls_xpiby2():
    # generate
    shape = Shape.SQUARE
    evolution_time = T_TOT_XPIBY2
    control_eval_count = int(np.floor(evolution_time * DT_INV))
    control_eval_times = np.arange(0, control_eval_count, 1) * DT
    controls = np.vstack([gen_controls_xpiby2(t, shape=shape) for t in control_eval_times])

    # save
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    with h5py.File(save_file_path, "a") as save_file:
        save_file["complex_controls"] = False
        save_file["evolution_time"] = evolution_time
        save_file["controls"] = controls
    # ENDWITH
    print("saved controls to {}"
          "".format(save_file_path))

    # plot
    file_prefix = save_file_path.split(".")[0]
    plot_file_path = "{}_controls.png".format(file_prefix)
    fig = plt.figure()
    plt.scatter(control_eval_times, controls,
                label="controls", color="blue")
    plt.title(file_prefix)
    plt.legend()
    plt.savefig(plot_file_path, dpi=DPI)
#ENDDEF

    
def main():
    # save_controls_xpi()
    # save_controls_ypiby2()
    save_controls_xpiby2()


if __name__ == "__main__":
    main()

