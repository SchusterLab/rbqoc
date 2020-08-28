"""
figures.py - figures but in python
"""

from enum import Enum
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


## GENERAL ##

# CONSTANTS #

# paths
WDIR = os.environ.get("ROBUST_QOC_PATH", "../../")
META_NAME = "spin"
EXPERIMENT_NAME = "figures"
SPIN_OUT_PATH = os.path.join(WDIR, "out", META_NAME)
SAVE_PATH = os.path.join(WDIR, "out", META_NAME, EXPERIMENT_NAME)

# simulation
DT_PREF = 1e-2
DT_PREF_INV = 1e2

# plotting
DPI = 300
DPI_FINAL = int(1e3)

# TYPES #

class SaveType(Enum):
    jl = 1
    samplejl = 2
    py = 3
#ENDDEF

# METHODS #

def grab_controls(save_file_path, save_type=SaveType.jl):
    with h5py.File(save_file_path, "r") as save_file:
        if save_type == SaveType.jl:
            cidx = save_file["controls_idx"][()]
            controls = save_file["astates"][cidx - 1, :][()]
            controls = np.reshape(controls, (controls.shape[1], 1))
            evolution_time = save_file["evolution_time"][()]
        elif save_type == SaveType.samplejl:
            controls = np.swapaxes(save_file["controls_sample"][()], -1, -2)
            evolution_time = save_file["evolution_time_sample"][()]
        elif save_type == SaveType.py:
            controls = save_file["controls"][()]
            evolution_time = save_file["evolution_time"][()]
        #ENDIF
    #ENDWITH
    return (controls, evolution_time)
#ENDDEF


def generate_save_file_path(extension, save_file_name, save_path):
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


## PLOTTING ##

class GateType(Enum):
    zpiby2 = 1
    ypiby2 = 2
    xpiby2 = 3
#ENDDEF

GT_STR = {
    GateType.zpiby2: "Z/2",
    GateType.ypiby2: "Y/2",
    GateType.xpiby2: "X/2",
}

GT_LIST = [GateType.zpiby2, GateType.ypiby2, GateType.xpiby2]

class PulseType(Enum):
    analytic = 1
    qoc = 2
#ENDDEF

PT_STR = {
    PulseType.analytic: "Anl.",
    PulseType.qoc: "QOC",
}

PT_COLOR = {
    PulseType.analytic: "blue",
    PulseType.qoc: "red",
}

DATAFP_KEY = 1
SAVEFP_KEY = 2
SAVET_KEY = 3
ACORDS_KEY = 4
AVGAMP_KEY = 5
AVGT1_KEY = 6

# FIGURE 1 #

F1_PT_LIST = [PulseType.analytic, PulseType.qoc]

F1_DATA = {
    GateType.zpiby2: {
        PulseType.qoc: {
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00201_spin15.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00194_spin15.h5"),
            SAVET_KEY: SaveType.jl,
            AVGAMP_KEY: 0.10734410077202858,
            AVGT1_KEY: 1.435236159072783e6,
        },
        PulseType.analytic: {
            ACORDS_KEY: (0.2, 0.17),
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00043_spin14.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00000_spin14.h5"),
            SAVET_KEY: SaveType.py,
            AVGAMP_KEY: 0.,
            AVGT1_KEY: 310880.0,
        },
    },
    GateType.ypiby2: {
        PulseType.qoc: {
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00200_spin15.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00185_spin15.h5"),
            SAVET_KEY: SaveType.jl,
            AVGAMP_KEY: 0.14559425517772442,
            AVGT1_KEY: 1.8676076471694398e6,
        },
        PulseType.analytic: {
            ACORDS_KEY: (1.2, 0.26),
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00041_spin14.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00003_spin14.h5"),
            SAVET_KEY: SaveType.py,
            AVGAMP_KEY: 0.027799178222907037,
            AVGT1_KEY: 513020.2302254785,
        },
    },
    GateType.xpiby2: {
        PulseType.qoc: {
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00202_spin15.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00174_spin15.h5"),
            SAVET_KEY: SaveType.jl,
            AVGAMP_KEY: 0.10855232522260652,
            AVGT1_KEY: 1.4927200778820992e6,
        },
        PulseType.analytic: {
            ACORDS_KEY: (0.6, 0.4),
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00042_spin14.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
            SAVET_KEY: SaveType.py,
            AVGAMP_KEY: 0.019058098591549295,
            AVGT1_KEY: 497914.87979617505,
        },  
    },
}

def make_figure1a():
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    save_file_paths = list()
    save_types = list()
    labels = list()
    colors = list()
    subfigs = list()
    (fig, axs) = plt.subplots(3)
    for (i, gate_type) in enumerate(GT_LIST):
        # annotate gate type
        text_ = GT_STR[gate_type]
        (textx, texty) = F1_DATA[gate_type][PulseType.analytic][ACORDS_KEY]
        axs[i].text(textx, texty, text_)
        xmax = 0
        # plot controls
        for (j, pulse_type) in enumerate(F1_PT_LIST):
            data = F1_DATA[gate_type][pulse_type]
            color = PT_COLOR[pulse_type]
            save_file_path = data[SAVEFP_KEY]
            save_type = data[SAVET_KEY]
            (controls, evolution_time) = grab_controls(save_file_path, save_type=save_type)
            (control_eval_count, control_count) = controls.shape
            control_eval_times = np.linspace(0, control_eval_count - 1, control_eval_count) * DT_PREF
            xmax = max(xmax, control_eval_times[-1])
            axs[i].plot(control_eval_times, controls[:,0], color=color)
        #ENDFOR
        axs[i].set_xlim(0, xmax)
    #ENDFOR
    axs[1].set_ylabel("$a$ (GHz)")
    axs[2].set_xlabel("$t$ (ns)")
    axs[0].text(-2.1, 0.17, "(a)")
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=1., wspace=None, hspace=0.25)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure1a to {}"
          "".format(plot_file_path))
#ENDDEF


F1B_SAMPLE_COUNT = int(1e4)
F1B_GT_MK = {
    GateType.zpiby2: "s",
    GateType.ypiby2: "^",
    GateType.xpiby2: "*",
}
F1B_MS_M = 8
F1B_MS_DATA = 20
F1B_ELW = 1.
F1B_ALPHA = 1.
F1B_ALPHA_M = 1.
F1B_DATA_PATH = os.path.join(SPIN_OUT_PATH, "figures/f1b.h5")
def make_figure1b():
    plot_file_path = generate_save_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    t1_normalize = 1e6 # us
    with h5py.File(F1B_DATA_PATH, "r") as data_file:
        amps_fit = data_file["amps_fit"][()]
        t1s_fit = data_file["t1s_fit"][()] / t1_normalize
        amps_data = data_file["amps_data"][()]
        t1s_data = data_file["t1s_data"][()] / t1_normalize
        t1s_data_err = data_file["t1s_data_err"][()] / t1_normalize
    #ENDWITH
    fig = plt.figure()
    plt.plot(amps_fit, t1s_fit, color="green", zorder=1)
    plt.scatter(amps_data, t1s_data, color="purple",
                s=F1B_MS_DATA, marker="o", zorder=3)
    plt.errorbar(amps_data, t1s_data, yerr=t1s_data_err, linestyle="none",
                 elinewidth=F1B_ELW, zorder=2)
    for (i, gate_type) in enumerate(GT_LIST):
        for (j, pulse_type) in enumerate(F1_PT_LIST):
            data = F1_DATA[gate_type][pulse_type]
            avg_amp = data[AVGAMP_KEY]
            avg_t1 = data[AVGT1_KEY] / t1_normalize
            label = "{} {}".format(GT_STR[gate_type], PT_STR[pulse_type])
            color = PT_COLOR[pulse_type]
            marker = F1B_GT_MK[gate_type]
            zorder = 5 if pulse_type == PulseType.qoc and gate_type == GateType.xpiby2 else 4
            plt.plot(avg_amp, avg_t1, label=label, marker=marker, ms=F1B_MS_M,
                     color=color, alpha=F1B_ALPHA_M, zorder=zorder)
        #ENDFOR
    #ENDFOR
    plt.text(-0.06, 4.5, "(b)")
    plt.ylabel("$T_{1}$ (ms)")
    plt.xlabel(r"${\langle |a| \rangle}_{t}$ (GHz)")
    plt.xlim(-0.02, 0.5)
    plt.legend()
    plt.subplots_adjust(left=0.07, right=0.97, bottom=0.1, top=1., wspace=None, hspace=None)
    plt.savefig(plot_file_path, dpi=DPI)
    print("Plotted Figure1b to {}"
          "".format(plot_file_path))
#ENDDEF

def main():
    make_figure1b()
#ENDDEF


if __name__ == "__main__":
    main()
#ENDIF
