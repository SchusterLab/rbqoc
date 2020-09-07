"""
figures.py - figures but in python
"""

from argparse import ArgumentParser
from enum import Enum
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import qutip


## GENERAL ##

# CONSTANTS #

# paths
WDIR = os.environ.get("ROBUST_QOC_PATH", "../../")
META_NAME = "spin"
EXPERIMENT_NAME = "figures"
SPIN_OUT_PATH = os.path.join(WDIR, "out", META_NAME)
SAVE_PATH = os.path.join(WDIR, "out", META_NAME, EXPERIMENT_NAME)
F2C_DATA_FILE_PATH = os.path.join(SAVE_PATH, "f2c.h5")
F3C_DATA_FILE_PATH = os.path.join(SAVE_PATH, "f3c.h5")
F3D_DATA_FILE_PATH = os.path.join(SAVE_PATH, "f3d.h5")

# simulation
DT_PREF = 1e-2
DT_PREF_INV = 1e2

# plotting
DPI = 300
DPI_FINAL = int(1e3)
TICK_FS = LABEL_FS = LEGEND_FS = TEXT_FS = 10
PAPER_LW = 3.40457

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
    s2 = 3
    s4 = 4
    d2 = 5
    d3 = 6
    s2b = 7
    s4b = 8
    d2b = 9
    d3b = 10
#ENDDEF

PT_STR = {
    PulseType.analytic: "Anl.",
    PulseType.qoc: "QOC",
    PulseType.s2: "S-2",
    PulseType.s4: "S-4",
    PulseType.d2: "D-2",
    PulseType.d3: "D-3",
    PulseType.s2b: "S-2",
    PulseType.s4b: "S-4",
    PulseType.d2b: "D-2",
    PulseType.d3b: "D-2",
}

PT_COLOR = {
    PulseType.analytic: "blue",
    PulseType.qoc: "red",
    PulseType.s2: "lime",
    PulseType.s4: "green",
    PulseType.d2: "red",
    PulseType.d3: "darkred",
    PulseType.s2b: "lime",
    PulseType.s4b: "green",
    PulseType.d2b: "red",
    PulseType.d3b: "darkred",
}

PT_LS = {
    PulseType.analytic: "solid",
    PulseType.qoc: "solid",
    PulseType.s2: "solid",
    PulseType.s4: "solid",
    PulseType.d2: "solid",
    PulseType.d3: "solid",
    PulseType.s2b: "dashed",
    PulseType.s4b: "dashed",
    PulseType.d2b: "dashed",
    PulseType.d3b: "dashed",
}

PT_MARKER = {
    PulseType.s2: "o",
    PulseType.s4: "s",
    PulseType.d2: "^",
    PulseType.d3: "d",
}

DATAFP_KEY = 1
SAVEFP_KEY = 2
SAVET_KEY = 3
ACORDS_KEY = 4
AVGAMP_KEY = 5
AVGT1_KEY = 6
DATA2FP_KEY = 7

# GENERAL #
def plot_fidelity_by_gate_count(
        fidelitiess, inds=None, title="", ylim=None,
        yticks=None, labels=None, colors=None, linestyles=None,
        xlim=None, figlabel=None,
        adjust=[None, None, None, None, None, None],
        legend_frameon=False, legend_bbox_to_anchor=None,
        figsize=[8, 6], legend_fs=LEGEND_FS,
        ylabel="Gate Error"):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    gate_count = fidelitiess[0].shape[0] - 1
    gate_count_axis = np.arange(0, gate_count + 1)
    if inds is None:
        inds = np.arange(0, gate_count)
    #ENDIF
    if xlim is None:
        xlim = (0., gate_count)
    #ENDIF
    for (i, fidelities) in enumerate(fidelitiess):
        color = None if colors is None else colors[i]
        label = None if labels is None else labels[i]
        linestyle = "solid" if linestyles is None else linestyles[i]
        plt.plot(gate_count_axis[inds], 1 - fidelities[inds], label=label,
                 color=color, linestyle=linestyle)
    #ENDFOR
    if figlabel is not None:
        fig.text(figlabel[0], figlabel[1], figlabel[2], fontsize=TEXT_FS)
    #ENDIF
    plt.ylabel(ylabel, fontsize=LABEL_FS)
    plt.xlabel("Gate Count", fontsize=LABEL_FS)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.yticks(yticks)
    legend_loc = "best" if legend_bbox_to_anchor is None else "lower left"
    if labels is not None:
        plt.legend(frameon=legend_frameon, fontsize=legend_fs,
                   loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    #ENDIF
    plt.subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2],
                        top=adjust[3], wspace=adjust[4], hspace=adjust[5])
    ax.tick_params(direction="in", labelsize=TICK_FS)
#ENDDEF


# FIGURE 1 #

F1_PT_LIST = [PulseType.analytic, PulseType.qoc]

F1_DATA = {
    GateType.zpiby2: {
        PulseType.qoc: {
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00216_spin15.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00209_spin15.h5"),
            SAVET_KEY: SaveType.jl,
            AVGAMP_KEY: 0.20628855236611363,
            AVGT1_KEY: 2.94161031851321e6,
        },
        PulseType.analytic: {
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00043_spin14.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00000_spin14.h5"),
            SAVET_KEY: SaveType.py,
            AVGAMP_KEY: 0.,
            AVGT1_KEY: 310880.0,
        },
    },
    GateType.ypiby2: {
        PulseType.qoc: {
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00215_spin15.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin15/00205_spin15.h5"),
            SAVET_KEY: SaveType.jl,
            AVGAMP_KEY: 0.20386686839468854,
            AVGT1_KEY: 2.846659754345148e6,
        },
        PulseType.analytic: {
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
            DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00042_spin14.h5"),
            SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
            SAVET_KEY: SaveType.py,
            AVGAMP_KEY: 0.019058098591549295,
            AVGT1_KEY: 497914.87979617505,
        },  
    },
}

def make_figure1a():
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    save_file_paths = list()
    save_types = list()
    labels = list()
    colors = list()
    subfigs = list()
    (fig, axs) = plt.subplots(3, figsize=(PAPER_LW * 0.55, PAPER_LW * 0.8))
    for (i, gate_type) in enumerate(GT_LIST):
        # annotate gate type
        text_ = GT_STR[gate_type]
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
        axs[i].axhline(0, color="grey", alpha=0.2, zorder=1, linewidth=0.8)
        axs[i].set_xlim(0, xmax)
    #ENDFOR
    axs[0].set_xticks([0, 10, 20])
    axs[0].set_xticklabels(["0", "10", "20"])
    axs[0].set_yticks([-0.5, 0, 0.5])
    axs[0].set_yticklabels(["$-$0.5", "0.0", "0.5"])
    axs[0].tick_params(direction="in", labelsize=TICK_FS)
    axs[0].text(0.2, 0.26, "Z/2", fontsize=TEXT_FS)
    axs[1].set_xticks([0, 10, 20])
    axs[1].set_xticklabels(["0", "10", "20"])
    axs[1].set_yticks([-0.5, 0, 0.5])
    axs[1].set_yticklabels(["$-$0.5", "0.0", "0.5"])
    axs[1].tick_params(direction="in", labelsize=TICK_FS)
    axs[1].text(1., 0.26, "Y/2", fontsize=TEXT_FS)
    axs[2].set_xticks([0, 25, 50])
    axs[2].set_xticklabels(["0", "25", "50"])
    axs[2].set_yticks([-0.5, 0, 0.5])
    axs[2].set_yticklabels(["$-$0.5", "0.0", "0.5"])
    axs[2].set_ylim(-0.52, 0.52)
    axs[2].tick_params(direction="in", labelsize=TICK_FS)
    axs[2].text(2.2, 0.26, "X/2", fontsize=TEXT_FS)
    axs[1].set_ylabel("$a$ (GHz)", fontsize=LABEL_FS)
    axs[2].set_xlabel("$t$ (ns)", fontsize=LABEL_FS)
    fig.text(0, 0.96, "(a)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.32, right=0.997, bottom=0.14, top=.95, wspace=None, hspace=0.55)
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
F1B_GT_MS = {
    GateType.zpiby2: 6,
    GateType.ypiby2: 7,
    GateType.xpiby2: 10,
}
F1B_MS_DATA = 20
F1B_MEW_M = 0.5
F1B_MEW_DATA = 0.5
F1B_ELW = 1.
F1B_ALPHA = 1.
F1B_ALPHA_M = 1.
F1B_DATA_PATH = os.path.join(SPIN_OUT_PATH, "figures/f1b.h5")
def make_figure1b():
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    t1_normalize = 1e6 # us
    with h5py.File(F1B_DATA_PATH, "r") as data_file:
        amps_fit = data_file["amps_fit"][()]
        t1s_fit = data_file["t1s_fit"][()] / t1_normalize
        amps_data = data_file["amps_data"][()]
        t1s_data = data_file["t1s_data"][()] / t1_normalize
        t1s_data_err = data_file["t1s_data_err"][()] / t1_normalize
    #ENDWITH
    fig = plt.figure(figsize=(PAPER_LW * 0.4, PAPER_LW * 0.8))
    ax = plt.gca()
    plt.plot(amps_fit, t1s_fit, color="black", zorder=1)
    plt.scatter(amps_data, t1s_data, color="black",
                s=F1B_MS_DATA, marker="o", zorder=3,
                linewidths=F1B_MEW_DATA, edgecolors="black")
    plt.errorbar(amps_data, t1s_data, yerr=t1s_data_err, linestyle="none",
                 elinewidth=F1B_ELW, zorder=2, ecolor="black")
    for (i, gate_type) in enumerate(GT_LIST):
        for (j, pulse_type) in enumerate(F1_PT_LIST):
            data = F1_DATA[gate_type][pulse_type]
            avg_amp = data[AVGAMP_KEY]
            avg_t1 = data[AVGT1_KEY] / t1_normalize
            color = PT_COLOR[pulse_type]
            marker = F1B_GT_MK[gate_type]
            ms = F1B_GT_MS[gate_type]
            zorder = 5 if marker == "*" else 4
            plt.plot(avg_amp, avg_t1, marker=marker, ms=ms,
                     markeredgewidth=F1B_MEW_M, markeredgecolor="black",
                     color=color, alpha=F1B_ALPHA_M, zorder=zorder)
        #ENDFOR
        plt.plot([], [], label="{}".format(GT_STR[gate_type]), color="black",
                 marker=marker, ms=ms, markeredgewidth=F1B_MEW_M,
                 markeredgecolor="black", linewidth=0)
    #ENDFOR
    plt.ylabel("$T_{1}$ (ms)", fontsize=LABEL_FS)
    plt.yticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"], fontsize=LABEL_FS)
    plt.xlabel("$|a|$ (GHz)", fontsize=LABEL_FS)
    plt.xlim(-0.02, 0.5)
    plt.xticks([0, 0.2, 0.4], ["0", "0.2", "0.4"], fontsize=LABEL_FS)
    ax.tick_params(direction="in", labelsize=TICK_FS)
    plt.legend(frameon=False, fontsize=LEGEND_FS, loc="lower left",
               bbox_to_anchor=[0.3, 0.01], handletextpad=0.1)
    fig.text(0, .96, "(b)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.26, right=0.997, bottom=0.145, top=0.95, wspace=None, hspace=None)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure1b to {}"
          "".format(plot_file_path))
#ENDDEF


F1C_GT_LS = {
    GateType.zpiby2: "solid",
    GateType.ypiby2: "dashed",
    GateType.xpiby2: "dashdot",
}
def make_figure1c():
    fidelitiess = list()
    colors = list()
    linestyles = list()
    for gate_type in GT_LIST:
        for pulse_type in F1_PT_LIST:
            data = F1_DATA[gate_type][pulse_type]
            with h5py.File(data[DATAFP_KEY], "r") as data_file:
                fidelities = data_file["fidelities"][()]
            #ENDWITH
            color = PT_COLOR[pulse_type]
            linestyle = F1C_GT_LS[gate_type]
            fidelitiess.append(fidelities)
            colors.append(color)
            linestyles.append(linestyle)
        #ENDFOR
    #ENDFOR
    plot_file_path = plot_fidelity_by_gate_count(
        fidelitiess, ylim=(0, 0.05), yticks=np.arange(0, 0.06, 0.01),
        colors=colors, linestyles=linestyles,
        xlim=(0, 1700), figlabel=(0, 0.96, "(c)"),
        adjust=[0.18, 0.997, 0.14, 0.95, None, None],
        figsize=(PAPER_LW, PAPER_LW * 0.8),
        
    )
    plt.plot([], [], label="Z/2", linestyle=linestyles[0], color="black")
    plt.plot([], [], label="Y/2", linestyle=linestyles[2], color="black")
    plt.plot([], [], label="X/2", linestyle=linestyles[4], color="black")
    plt.legend(frameon=False, fontsize=LEGEND_FS, loc="lower left",
               bbox_to_anchor=(0.74, 0.25), handletextpad=0.4)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure1c to {}"
          "".format(plot_file_path))
#ENDDEF


# FIGURE 2 #

F2_DATA = {
    PulseType.analytic: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
        SAVET_KEY: SaveType.py,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00074_spin14.h5"),
    },
    PulseType.d2: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00091_spin11.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00092_spin11.h5"),
    },
    PulseType.d3: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00110_spin11.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00111_spin11.h5"),
    },
    PulseType.s2: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00295_spin12.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00297_spin12.h5"),
    },
    PulseType.s4: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00298_spin12.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00299_spin12.h5"),
    },
    PulseType.d2b: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00426_spin11.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00427_spin11.h5"),
    },
    PulseType.d3b: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00294_spin11.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin11/00425_spin11.h5"),
    },
    PulseType.s2b: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00303_spin12.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00495_spin12.h5"),
    },
    PulseType.s4b: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00310_spin12.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: os.path.join(SPIN_OUT_PATH, "spin12/00494_spin12.h5"),
    },
}

F2A_PT_LIST = [
    [PulseType.analytic],
    [PulseType.s2], [PulseType.s4],
    [PulseType.d2], [PulseType.d3],
]
def make_figure2a():
    subfigtot = len(F2A_PT_LIST)
    (fig, axs) = plt.subplots(subfigtot, figsize=(PAPER_LW * 0.55, PAPER_LW * 0.8))
    for (i, pulse_types) in enumerate(F2A_PT_LIST):
        for (j, pulse_type) in enumerate(pulse_types):
            data = F2_DATA[pulse_type]
            color = PT_COLOR[pulse_type]
            label = "{}".format(PT_STR[pulse_type])
            linestyle = "solid" #PT_LS[pulse_type]
            save_file_path = data[SAVEFP_KEY]
            save_type = data[SAVET_KEY]
            (controls, evolution_time) = grab_controls(save_file_path, save_type=save_type)
            (control_eval_count, control_count) = controls.shape
            control_eval_times = np.arange(0, control_eval_count, 1) * DT_PREF
            axs[i].plot(control_eval_times, controls[:, 0], color=color, label=label,
                        linestyle=linestyle, zorder=2)
            axs[i].set_xlim((0, 56.80))
            if pulse_type == PulseType.analytic:
                axs[i].set_ylim(-0.15, 0.15)
                axs[i].set_yticks([-0.1, 0, 0.1])
            else:
                axs[i].set_ylim((-0.63, 0.63))
                axs[i].set_yticks([-0.4, 0, 0.4])
            #ENDIF
        #ENDFOR
        axs[i].axhline(0, color="grey", alpha=0.2, zorder=1, linewidth=0.8)
        axs[i].tick_params(direction="in", labelsize=TICK_FS)
        axs[i].set_xticks([])
    #ENDFOR
    axs[0].text(3, 0.05, "Anl.", fontsize=TEXT_FS)
    axs[1].text(3, 0.25, "S-2", fontsize=TEXT_FS)
    axs[2].text(3, 0.25, "S-4", fontsize=TEXT_FS)
    axs[3].text(3, 0.25, "D-2", fontsize=TEXT_FS)
    axs[4].text(3, 0.25, "D-3", fontsize=TEXT_FS)
    axs[2].set_ylabel("$a$ (GHz)", fontsize=LABEL_FS)
    axs[4].set_xlabel("$t$ (ns)", fontsize=LABEL_FS)
    axs[4].set_xticks([0, 25, 50])
    axs[4].set_xticklabels(["0", "25", "50"])
    fig.text(0, 0.96, "(a)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.34, right=0.997, top=0.95, bottom=0.14, hspace=0., wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Saved Figure2a to {}"
          "".format(plot_file_path))
#ENDDEF


F2B_TRIAL_COUNT = int(1e3)
F2B_FQ_DEV = 1e-1
F2B_PT_LIST = [PulseType.analytic, PulseType.s2, PulseType.s4, PulseType.d2,
               PulseType.d3, PulseType.s2b, PulseType.s4b, PulseType.d2b, PulseType.d3b]
F2B_LB = 1 + 1e-8
def make_figure2b():
    fq = 1.4e-2
    fq_devs_ = np.linspace(-F2B_FQ_DEV, F2B_FQ_DEV, F2B_TRIAL_COUNT)
    fqs = (fq_devs_ * fq) + fq
    log_transform = lambda x: np.log10(x) / np.log10(F2B_LB)
    
    fig = plt.figure(figsize=(PAPER_LW * 0.4, PAPER_LW * 0.8))
    ax = plt.gca()
    for (i, pulse_type) in enumerate(F2B_PT_LIST):
        data = F2_DATA[pulse_type]
        if not DATAFP_KEY in data.keys():
            continue
        #ENDIF
        label = "{}".format(PT_STR[pulse_type])
        linestyle = PT_LS[pulse_type]
        color = PT_COLOR[pulse_type]
        data_file_path = data[DATAFP_KEY]
        with h5py.File(data_file_path, "r") as data_file:
            fidelities = data_file["fidelities"][()]
            fq_devs = fq_devs_ if not "fq_devs" in data_file.keys() else data_file["fq_devs"][()]
        #ENDWITH
        gate_errors = 1 - fidelities
        mid_idx = int(len(gate_errors) / 2)
        mean_gate_errors = (np.flip(gate_errors[0:mid_idx]) + gate_errors[mid_idx:]) / 2
        log_mge = log_transform(mean_gate_errors)
        fq_devs_top = fq_devs[mid_idx:]
        plt.plot(fq_devs_top, log_mge, color=color,
                 linestyle=linestyle)
    #ENDFOR
    plt.xlim(0, 3e-2)
    plt.xticks(
        np.array([0, 1, 2, 3]) * 1e-2,
        ["0", "1", "2", "3"],
    )
    plt.ylim(log_transform(1e-7), log_transform(1e-3))
    plt.yticks(
        log_transform(np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3])),
        ["$10^{-7}$", "$10^{-6}$", "$10^{-5}$",
         "$10^{-4}$", "$10^{-3}$"],
    )
    ax.tick_params(direction="in", labelsize=TICK_FS)
    plt.xlabel("$|\delta f_{q} / f_{q}| \; (\%)$", fontsize=LABEL_FS)
    plt.ylabel("Gate Error", fontsize=LABEL_FS)
    plt.plot([], [], label="fixed", linestyle="solid", color="black")
    plt.plot([], [], label="best", linestyle="dashed", color="black")
    plt.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.14, 0.02),
               fontsize=8, handlelength=1.5, handletextpad=0.4)
    fig.text(0, 0.96, "(b)")
    plt.subplots_adjust(left=0.45, right=0.97, bottom=0.15, top=0.95, hspace=None, wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure2b to {}"
          "".format(plot_file_path))
#ENDDEF


F2C_MS = 30
F2C_MEW = 0.5
def make_figure2c():
    gate_type = GateType.xpiby2
    # get data and plot
    fig = plt.figure(figsize=(PAPER_LW, PAPER_LW * 0.8))
    ax = plt.gca()
    data_file_path = F2C_DATA_FILE_PATH
    with h5py.File(data_file_path, "r") as data_file:
        gate_errors = np.swapaxes(data_file["gate_errors"][()], -1, -2)
        gate_times = data_file["gate_times"][()]
        pulse_types = [PulseType(ipt) for ipt in data_file["pulse_types"][()]]
    #ENDWITH
    for (i, pulse_type) in enumerate(pulse_types):
        label = "{}".format(PT_STR[pulse_type])
        color = PT_COLOR[pulse_type]
        marker = PT_MARKER[pulse_type]
        for (j, gate_time) in enumerate(gate_times):
            gate_error = gate_errors[i, j]
            plt.scatter([gate_time], [gate_error], color=color,
                        marker=marker, s=F2C_MS, linewidths=F2C_MEW,
                        edgecolors="black")
        #ENDFOR
        plt.scatter([], [], label=label, color=color, linewidths=F2C_MEW,
                    s=F2C_MS, marker=marker, edgecolors="black")
    #ENDFOR
    plt.ylim(0, 1e-4)
    plt.yticks(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1e-5,
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    )
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
               ["", "60", "", "80", "", "100", "", "120", "", "140", "", "160"])
    ax.tick_params(direction="in", labelsize=TICK_FS)
    plt.xlabel("$t_{N}$ (ns)")
    plt.ylabel("Gate Error $\\times10^{5}$")
    plt.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1),
               handletextpad=0.2, fontsize=8)
    fig.text(0, 0.96, "(c)")
    plt.subplots_adjust(left=0.14, right=0.997, top=0.95, bottom=0.14, hspace=None, wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure2c to {}".format(plot_file_path))
#ENDDEF


# FIGURE 3 #

F3_DATA = {
    PulseType.analytic: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin14/00004_spin14.h5"),
        SAVET_KEY: SaveType.py,
        DATAFP_KEY: [os.path.join(SPIN_OUT_PATH, "spin14", "{:05d}_spin14.h5".format(index)) for index in [
            73, 86, 87, 88, 89, 90, 91, 92, 93, 94
        ]],
        ACORDS_KEY: (2.5, 0.08),
    },
    PulseType.s2: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin18/00003_spin18.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: [os.path.join(SPIN_OUT_PATH, "spin18", "{:05d}_spin18.h5".format(index)) for index in [
            7, 36, 38, 40, 42, 43, 45, 48, 49, 51
        ]],
        ACORDS_KEY: (0.5, 0.4),
    },
    PulseType.s4: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin18/00005_spin18.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: [os.path.join(SPIN_OUT_PATH, "spin18", "{:05d}_spin18.h5".format(index)) for index in [
            8, 35, 37, 39, 41, 44, 46, 47, 50, 52
        ]],
        ACORDS_KEY: (0.5, 0.4),
    },
    PulseType.d2: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin17/00003_spin17.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: [os.path.join(SPIN_OUT_PATH, "spin17", "{:05d}_spin17.h5".format(index)) for index in [
            6, 33, 35, 37, 39, 41, 43, 45, 47, 50
        ]],
        ACORDS_KEY: (0.5, 0.4),
    },
    PulseType.d3: {
        SAVEFP_KEY: os.path.join(SPIN_OUT_PATH, "spin17/00005_spin17.h5"),
        SAVET_KEY: SaveType.jl,
        DATAFP_KEY: [os.path.join(SPIN_OUT_PATH, "spin17", "{:05d}_spin17.h5".format(index)) for index in [
            7, 34, 36, 38, 40, 42, 44, 46, 48, 49
        ]],
        ACORDS_KEY: (0.5, 0.4),
    },
}

F3_PT_LIST = [PulseType.analytic, PulseType.s2, PulseType.s4, PulseType.d2, PulseType.d3]


F3A_PT_LIST = [[PulseType.analytic], [PulseType.s2], [PulseType.s4], [PulseType.d2], [PulseType.d3]]
def make_figure3a():
    subfigtot = len(F3A_PT_LIST)
    (fig, axs) = plt.subplots(subfigtot, figsize=(PAPER_LW, PAPER_LW))
    for (i, pulse_types) in enumerate(F3A_PT_LIST):
        for (j, pulse_type) in enumerate(pulse_types):
            data = F3_DATA[pulse_type]
            color = PT_COLOR[pulse_type]
            label = "{}".format(PT_STR[pulse_type])
            linestyle = "solid" #PT_LS[pulse_type]
            save_file_path = data[SAVEFP_KEY]
            save_type = data[SAVET_KEY]
            (controls, evolution_time) = grab_controls(save_file_path, save_type=save_type)
            (control_eval_count, control_count) = controls.shape
            control_eval_times = np.arange(0, control_eval_count, 1) * DT_PREF
            axs[i].plot(control_eval_times, controls[:, 0], color=color, label=label,
                        linestyle=linestyle, zorder=2)
            axs[i].set_xlim((0, 56.80))
            if pulse_type == PulseType.analytic:
                axs[i].set_ylim(-0.15, 0.15)
                axs[i].set_yticks([-0.1, 0, 0.1])
            else:
                axs[i].set_ylim((-0.63, 0.63))
                axs[i].set_yticks([-0.4, 0, 0.4])
            #ENDIF
        #ENDFOR
        axs[i].axhline(0, color="grey", alpha=0.2, zorder=1, linewidth=0.8)
        axs[i].tick_params(direction="in", labelsize=TICK_FS)
        axs[i].set_xticks([])
    #ENDFOR
    axs[0].text(3, 0.07, "Anl.", fontsize=TEXT_FS)
    axs[1].text(2, 0.3, "S-2", fontsize=TEXT_FS)
    axs[2].text(2, 0.3, "S-4", fontsize=TEXT_FS)
    axs[3].text(2, 0.3, "D-2", fontsize=TEXT_FS)
    axs[4].text(2, 0.3, "D-3", fontsize=TEXT_FS)
    axs[2].set_ylabel("$a$ (GHz)", fontsize=LABEL_FS)
    axs[4].set_xlabel("$t$ (ns)", fontsize=LABEL_FS)
    axs[4].set_xticks([0, 25, 50])
    axs[4].set_xticklabels(["0", "25", "50"])
    fig.text(0, 0.96, "(a)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.18, right=0.997, top=0.95, bottom=0.12, hspace=0., wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Saved Figure3a to {}"
          "".format(plot_file_path))
#ENDDEF


def make_figure3b():
    colors = list()
    fidelitiess = list()
    labels = list()
    linestyles = list()
    for (i, pulse_type) in enumerate(F3_PT_LIST):
        data = F3_DATA[pulse_type]
        color = PT_COLOR[pulse_type]
        mfidelities = None
        fcount = len(data[DATAFP_KEY])
        for (j, data_file_path) in enumerate(data[DATAFP_KEY]):
            with h5py.File(data_file_path, "r") as data_file:
                fidelities = data_file["fidelities"][()]
            #ENDWITH
            if mfidelities is None:
                mfidelities = fidelities
            else:
                mfidelities = mfidelities + fidelities
            #ENDIF
        #ENDFOR
        mfidelities = mfidelities / fcount
        label = "{}".format(PT_STR[pulse_type])
        colors.append(color)
        fidelitiess.append(mfidelities)
        labels.append(label)
        linestyles.append("solid") # PT_LS[pulse_type]
    #ENDFOR
    plot_fidelity_by_gate_count(
        fidelitiess, colors=colors, linestyles=linestyles,
        adjust=(0.2, 0.95, 0.11, 0.95, None, None),
        figlabel=(0, 0.96, "(b)"), figsize=(PAPER_LW, PAPER_LW),
        ylabel="Mean Gate Error", ylim=(0, 1.3e-2),
    )
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure3b to {}"
            "".format(plot_file_path))
#ENDDEF


def make_figure3c():
    with h5py.File(F3C_DATA_FILE_PATH, "r") as data_file:
        noise = data_file["noise"][()]
        times = data_file["times"][()]
        noise_fft = data_file["noise_fft"][()]
        freqs = data_file["freqs"][()]
    #ENDWITH
    noise_inds = np.arange(0, noise.shape[0], 30)
    times = times[noise_inds] / 1e3
    noise = noise[noise_inds] * 1e3
    freqs = freqs * 1e3

    (fig, axs) = plt.subplots(2)
    axs[0].scatter(times, noise, s=0.1, alpha=0.8)
    axs[0].set_xlim(times[0], times[-1])
    axs[0].set_ylim(0, np.max(noise))
    axs[0].set_xlabel("$t$ ($\mu$s)")
    axs[0].set_ylabel("${\\xi}(t)$ (MHz)")
    axs[1].scatter(freqs, noise_fft ** 2, s=0.1, alpha=0.8)
    axs[1].set_xlim(-2.5e1, 2.5e1)
    axs[1].set_ylim(0, 3e-7 ** 2)
    axs[1].set_xlabel("$f$ (MHz)")
    axs[1].set_ylabel("$|\hat{\\xi}(t)|^{2}$ (a.u.)")
    fig.text(0.02, 0.97, "(c)")
    plt.subplots_adjust(left=0.1, right=1., bottom=0.12, top=1., hspace=0.3, wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI)
    print("Saved Figure3c to {}"
          "".format(plot_file_path))
#ENDDEF


def make_figure3d():
    # initialize
    fig = plt.figure(figsize=(PAPER_LW, PAPER_LW))
    bloch = qutip.Bloch(fig=fig)

    # add points
    with h5py.File(F3D_DATA_FILE_PATH, "r") as data_file:
        pointss = np.swapaxes(data_file["pointss"][()], 0, 2)
        pulse_types = [PulseType(pt_int) for pt_int in data_file["pulse_types"][()]]
    #ENDWITH
    for (i, pulse_type) in enumerate(pulse_types):
        if i in [4]:
            bloch.add_points(np.swapaxes(pointss[i, :, :], -1, -2), meth="l")
        #ENDIF
    #ENDFOR


    # save
    bloch.render(bloch.fig, bloch.axes)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    bloch.fig.savefig(plot_file_path, dpi=DPI)
    print("Saved Figure3d to {}"
          "".format(plot_file_path))
#ENDDEF


def main():
    parser = ArgumentParser()
    parser.add_argument("--fig", type=str, default="")
    args = vars(parser.parse_args())

    fig_str = args["fig"]
    if fig_str == "1a":
        make_figure1a()
    elif fig_str == "1b":
        make_figure1b()
    elif fig_str == "1c":
        make_figure1c()
    elif fig_str == "2a":
        make_figure2a()
    elif fig_str == "2b":
        make_figure2b()
    elif fig_str == "2c":
        make_figure2c()
    elif fig_str == "3a":
        make_figure3a()
    elif fig_str == "3b":
        make_figure3b()
    elif fig_str == "3c":
        make_figure3c()
    elif fig_str == "3d":
        make_figure3d()
    #ENDIF
#ENDDEF


if __name__ == "__main__":
    main()
#ENDIF
