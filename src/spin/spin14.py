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

class Shape(Enum):
    SQUARE = 0
    TRIANGLE = 1
#ENDDEF

class SaveType(Enum):
    jl = 1
    samplejl = 2
    py = 3
#ENDDEF

# paths
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin14"
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
SAVE_PATH = os.path.join(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# plotting
DPI = 300

# constants
AMP_0 = 1.25e-1
DT = 1e-2
DT_INV = 1e2

# methods
def generate_file_path(extension, save_file_name, save_path):
    # Ensure the path exists.
    os.makedirs(save_path, exist_ok=True)
    
    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory. 
    max_numeric_prefix = -1
    for file_name in os.listdir(save_path):
        if ("_{}.{}".format(save_file_name, extension)) in file_name:
            max_numeric_prefix = max(int(file_name.split("_")[0]),
                                     max_numeric_prefix)
    #ENDFOR
    save_file_name_augmented = ("{:05d}_{}.{}"
                                "".format(max_numeric_prefix + 1,
                                          save_file_name, extension))
    
    return os.path.join(save_path, save_file_name_augmented)
#ENDDEF

### Z/2 ###

T_TOT_ZPIBY2 = 17.85714285714286
def gen_controls_zpiby2(t, shape=Shape.SQUARE):
    c1 = 0
    return np.array([c1,])
#ENDDEF


def save_controls_zpiby2(shape=Shape.SQUARE):
    pass
#ENDDEF


### Y/2 ###

T_XZ_YPIBY2 = 2.1656249366575766
T_Z_YPIBY2 = 15.142330599557274
T_TOT_YPIBY2 = 2 * T_XZ_YPIBY2 + T_Z_YPIBY2
T0_YPIBY2 = 0.
T1_YPIBY2 = T_XZ_YPIBY2 / 2
T2_YPIBY2 = T_XZ_YPIBY2
T3_YPIBY2 = T2_YPIBY2 + T_Z_YPIBY2
T4_YPIBY2 = T3_YPIBY2 + T_XZ_YPIBY2 / 2
T5_YPIBY2 = T3_YPIBY2 + T_XZ_YPIBY2
def gen_controls_ypiby2(t, shape=Shape.SQUARE):
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


def save_controls_ypiby2(plot=False):
    # generate
    shape = Shape.SQUARE
    evolution_time = T_TOT_YPIBY2
    control_eval_count = int(np.floor(evolution_time * DT_INV))
    control_eval_times = np.arange(0, control_eval_count, 1) * DT
    controls = np.vstack([gen_controls_ypiby2(t, shape=shape) for t in control_eval_times])
    controls[0, 0] = 0
    controls[-1, 0] = 0

    # save
    save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    with h5py.File(save_file_path, "a") as save_file:
        save_file["complex_controls"] = False
        save_file["evolution_time"] = evolution_time
        save_file["controls"] = controls
        save_file["save_type"] = int(SaveType.py.value)
    # ENDWITH
    print("saved ypiby2 controls to {}"
          "".format(save_file_path))

    # plot
    if plot:
        file_prefix = save_file_path.split(".")[0]
        plot_file_path = "{}_controls.png".format(file_prefix)
        fig = plt.figure()
        plt.scatter(control_eval_times, controls,
                    label="controls", color="blue")
        plt.title(file_prefix)
        plt.legend()
        plt.savefig(plot_file_path, dpi=DPI)
        print("plotted controls to {}"
              "".format(plot_file_path))
    #ENDIF
#ENDDEF


### X/2 ###

T_TOT_XPIBY2 = 2 * T_TOT_YPIBY2 + T_TOT_ZPIBY2
T1_XPIBY2 = T_TOT_YPIBY2
T2_XPIBY2 = T1_XPIBY2 + T_TOT_ZPIBY2
def gen_controls_xpiby2(t, shape=Shape.SQUARE):
    if t <= T1_XPIBY2:
        ret = -gen_controls_ypiby2(t, shape=shape)
    elif t <= T2_XPIBY2:
        ret = np.array([0])
    else:
        ret = gen_controls_ypiby2(t - T2_XPIBY2, shape=shape)
    #ENDIF

    return ret
#ENDDEF


def save_controls_xpiby2(plot=False):
    # generate
    shape = Shape.SQUARE
    evolution_time = T_TOT_XPIBY2
    control_eval_count = int(np.floor(evolution_time * DT_INV))
    control_eval_times = np.arange(0, control_eval_count, 1) * DT
    controls = np.vstack([gen_controls_xpiby2(t, shape=shape) for t in control_eval_times])
    controls[0, 0] = 0
    controls[-1, 0] = 0

    # save
    save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    with h5py.File(save_file_path, "a") as save_file:
        save_file["complex_controls"] = False
        save_file["evolution_time"] = evolution_time
        save_file["controls"] = controls
        save_file["save_type"] = int(SaveType.py.value)
    # ENDWITH
    print("saved xpiby2 controls to {}"
          "".format(save_file_path))

    # plot
    if plot:
        file_prefix = save_file_path.split(".")[0]
        plot_file_path = "{}_controls.png".format(file_prefix)
        fig = plt.figure()
        plt.scatter(control_eval_times, controls,
                    label="controls", color="blue")
        plt.title(file_prefix)
        plt.legend()
        plt.savefig(plot_file_path, dpi=DPI)
        print("plotted controls to {}"
              "".format(plot_file_path))
    #ENDIF
#ENDDEF


A_XPIC = 0.2166666666666667
T1_XPIC = 5.384615384615385
T2_XPIC = T1_XPIC + 3.846153846153847
TTOT_XPIC = 10.
def gen_xpicorpse(time):
    if time <= T1_XPIC:
        c1 = A_XPIC
    elif time <= T2_XPIC:
        c1 = -A_XPIC
    else:
        c1 = A_XPIC
    #ENDIF
    return np.array([c1,])
#ENDDEF


def save_xpicorpse():
    evolution_time = TTOT_XPIC
    control_eval_count = int(np.floor(evolution_time * DT_INV))
    control_eval_times = np.arange(0, control_eval_count, 1) * DT
    controls = np.vstack([gen_xpicorpse(t) for t in control_eval_times])
    controls[0, 0] = controls[-1, 0] = 0

    # save
    save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    with h5py.File(save_file_path, "a") as save_file:
        save_file["controls"] = controls
        save_file["save_type"]= int(SaveType.py.value)
        save_file["dt"] = DT
        save_file["evolution_time"] = evolution_time
    #ENDWITH
    print("saved xpicorpse controls to {}"
          "".format(save_file_path))
#ENDDEF
    

    
def main():
    # save_controls_ypiby2()
    # save_controls_xpiby2()
    save_xpicorpse()
#ENDDEF


if __name__ == "__main__":
    main()
#ENDIF
