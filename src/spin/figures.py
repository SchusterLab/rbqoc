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
F3C_DATA_FILE_PATH = os.path.join(SAVE_PATH, "f3c.h5")
F3D_DATA_FILE_PATH = os.path.join(SAVE_PATH, "f3d.h5")

# simulation
DT_PREF = 1e-2
DT_PREF_INV = 1e2

# plotting
DPI = 300
DPI_FINAL = int(1e3)
TICK_FS = LABEL_FS = LEGEND_FS = TEXT_FS = 8
LW = 1.
DASH_LS = (0, (3.0, 2.0))
DDASH_LS = (0, (3.0, 2.0, 1.0, 2.0))
DDDASH_LS = (0, (3.0, 2.0, 1.0, 2.0, 1.0, 2.0))
PAPER_LW = 3.40457
PAPER_TW = 7.05826

# keys
DATAFP_KEY = 1
SAVEFP_KEY = 2
SAVET_KEY = 3
ACORDS_KEY = 4
AVGAMP_KEY = 5
AVGT1_KEY = 6
DATA2FP_KEY = 7


# TYPES #

class SaveType(Enum):
    jl = 1
    samplejl = 2
    py = 3
#ENDDEF

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
    corpse = 11
    d1 = 12
    sut8 = 13
    d1b = 14
    sut8b = 15
    d1bb = 16
    d1bbb = 17
#ENDDEF

PT_STR = {
    PulseType.analytic: "Anl.",
    PulseType.qoc: "QOC",
    PulseType.s2: "S-8",
    PulseType.s4: "S-4",
    PulseType.d1: "D-1",
    PulseType.d2: "D-2",
    PulseType.d3: "D-3",
    PulseType.s2b: "S-2",
    PulseType.s4b: "S-4",
    PulseType.d2b: "D-2",
    PulseType.d3b: "D-2",
    PulseType.corpse: "C-2",
    PulseType.sut8: "SU-10",
}

PT_COLOR = {
    PulseType.analytic: "skyblue",
    PulseType.qoc: "red",
    PulseType.s2: "lime",
    PulseType.s4: "green",
    PulseType.d2: "darkred",
    PulseType.d3: "darkred",
    PulseType.s2b: "lime",
    PulseType.s4b: "green",
    PulseType.d2b: "darkred",
    PulseType.d3b: "darkred",
    PulseType.corpse: "pink",
    PulseType.d1: "lightcoral",
    PulseType.sut8: "green",
    PulseType.d1: "red",
    PulseType.d1b: "red",
    PulseType.d1bb: "red",
    PulseType.d1bbb: "red",
    PulseType.sut8b: "green",
}

PT_LS = {
    PulseType.analytic: "solid",
    PulseType.qoc: "solid",
    PulseType.s2: "solid",
    PulseType.s4: "solid",
    PulseType.d2: "solid",
    PulseType.d3: "solid",
    PulseType.s2b: DASH_LS,
    PulseType.s4b: DASH_LS,
    PulseType.d2b: DASH_LS,
    PulseType.d3b: DASH_LS,
    PulseType.corpse: "solid",
    PulseType.d1: "solid",
    PulseType.sut8: "solid",
    PulseType.d1b: DASH_LS,
    PulseType.sut8b: DASH_LS,
    PulseType.d1bb: DDASH_LS,
    PulseType.d1bbb: DDDASH_LS,
}

PT_MARKER = {
    PulseType.analytic: "*",
    PulseType.s2: "o",
    PulseType.sut8: "s",
    PulseType.d1: "^",
    PulseType.d2: "d",
    PulseType.corpse: "x",
    PulseType.d3: "d",
    PulseType.s4: "s",
}

# METHODS #

def grab_controls(save_file_path):
    with h5py.File(save_file_path, "r") as save_file:
        save_type = SaveType(save_file["save_type"][()])
        if save_type == SaveType.jl:
            cidx = save_file["controls_idx"][()]
            controls = save_file["astates"][cidx - 1, :-1][()]
            controls = np.reshape(controls, (controls.shape[1], 1))
            evolution_time = save_file["evolution_time"][()]
            dt = DT_PREF if not "dt" in save_file else save_file["dt"][()]
        elif save_type == SaveType.py:
            controls = save_file["controls"][()]
            evolution_time = save_file["evolution_time"][()]
            dt = DT_PREF
        #ENDIF
    #ENDWITH
    return (controls, evolution_time, dt)
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


def latest_file_path(extension, save_file_name, save_path):
    max_numeric_prefix = -1
    for file_name in os.listdir(save_path):
        if ("_{}.{}".format(save_file_name, extension)) in file_name:
            max_numeric_prefix = max(int(file_name.split("_")[0]),
                                     max_numeric_prefix)
        #ENDIF
    #ENDFOR
    if max_numeric_prefix > -1:
        save_file_name = ("{:05d}_{}.{}"
                          "".format(max_numeric_prefix, save_file_name, extension))
        save_file_path = os.path.join(save_path, save_file_name)
    else:
        save_file_path = None
    #ENDIF
    
    return save_file_path
#ENDDEF


# GENERAL #

def plot_fidelity_by_gate_count(
        fidelitiess, inds=None,
        labels=None, colors=None, linestyles=None,
        ylabel="Gate Error", label_fontsize=LABEL_FS):
    gate_count = fidelitiess[0].shape[0] - 1
    gate_count_axis = np.arange(0, gate_count + 1)
    if inds is None:
        inds = np.arange(0, gate_count)
    #ENDIF
    for (i, fidelities) in enumerate(fidelitiess):
        color = None if colors is None else colors[i]
        label = None if labels is None else labels[i]
        linestyle = "solid" if linestyles is None else linestyles[i]
        plt.plot(gate_count_axis[inds], 1 - fidelities[inds], label=label,
                 color=color, linestyle=linestyle)
    #ENDFOR
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xlabel("Gate Count", fontsize=label_fontsize)
#ENDDEF


# FIGURE 1 #

F1A_XEPS = [1e-1, 1e-1, 3e-1]
F1A_YEPS = [0, 0, 2e-2]
def make_figure1a():
    data_file_path = latest_file_path("h5", "f1a", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        save_file_paths = data_file["save_file_paths"][()]
        gate_types = [GateType(gt) for gt in data_file["gate_types"][()]]
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
    #ENDWITH
    gate_type_count = len(gate_types)
    pulse_type_count = len(pulse_types)
    (fig, axs) = plt.subplots(3, figsize=(PAPER_TW * 0.315, PAPER_TW * 0.315))
    for (i, gate_type) in enumerate(gate_types):
        xeps = F1A_XEPS[i]
        xmax = 0
        for (j, pulse_type) in enumerate(pulse_types):
            idx = i * pulse_type_count + j
            color = PT_COLOR[pulse_type]
            save_file_path = save_file_paths[idx]
            (controls, evolution_time, dt) = grab_controls(save_file_path)
            (control_eval_count, control_count) = controls.shape
            control_eval_times = np.linspace(0, control_eval_count - 1, control_eval_count) * dt
            xmax = max(xmax, control_eval_times[-1])
            axs[i].plot(control_eval_times, controls[:, 0], color=color, linewidth=LW)
        #ENDFOR
        axs[i].axhline(0, color="grey", alpha=0.2, zorder=1, linewidth=0.8)
        axs[i].set_xlim(0 - xeps, xmax + xeps)
    #ENDFOR
    axs[0].set_xticks([0 - F1A_XEPS[0], 10, 20])
    axs[0].set_xticklabels(["0", "10", "20"])
    axs[0].set_yticks([-0.5, 0, 0.5])
    axs[0].set_yticklabels(["$-$0.5", "0.0", "0.5"])
    axs[0].tick_params(direction="in", labelsize=TICK_FS)
    fig.text(0.38, 0.95, "Z/2", fontsize=TEXT_FS)
    axs[1].set_xticks([0 - F1A_XEPS[1], 10, 20])
    axs[1].set_xticklabels(["0", "10", "20"])
    axs[1].set_yticks([-0.5, 0, 0.5])
    axs[1].set_yticklabels(["$-$0.5", "0.0", "0.5"])
    axs[1].tick_params(direction="in", labelsize=TICK_FS)
    fig.text(0.38, 0.65, "Y/2", fontsize=TEXT_FS)
    axs[2].set_xticks([0 - F1A_XEPS[2], 25, 50])
    axs[2].set_xticklabels(["0", "25", "50"])
    axs[2].set_yticks([-0.5 - F1A_YEPS[2], 0, 0.5 + F1A_YEPS[2]])
    axs[2].set_yticklabels(["$-$0.5", "0.0", "0.5"])
    axs[2].set_ylim(-0.5 - F1A_YEPS[2], 0.5 + F1A_YEPS[2])
    axs[2].tick_params(direction="in", labelsize=TICK_FS)
    fig.text(0.38, 0.35, "X/2", fontsize=TEXT_FS)
    axs[1].set_ylabel("$a$ (GHz)", fontsize=LABEL_FS)
    axs[2].set_xlabel("$t$ (ns)", fontsize=LABEL_FS)
    fig.text(0, 0.955, "(a)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.22, right=0.997, bottom=0.14, top=.94, wspace=None, hspace=0.55)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
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
def make_figure1b():
    t1_normalize = 1e6 # us
    data_file_path = latest_file_path("h5", "f1b", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        amps_fit = data_file["amps_fit"][()]
        t1s_fit = data_file["t1s_fit"][()]
        amps_data = data_file["amps_data"][()]
        t1s_data = data_file["t1s_data"][()]
        t1s_data_err = data_file["t1s_data_err"][()]
        avg_amps = data_file["avg_amps"][()]
        avg_amps_t1 = data_file["avg_amps_t1"][()]
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
        gate_types = [GateType(gt) for gt in data_file["gate_types"][()]]
    #ENDWITH
    gate_count = len(gate_types)
    pulse_count = len(pulse_types)
    
    fig = plt.figure(figsize=(PAPER_TW * 0.23, PAPER_TW * 0.315))
    ax = plt.gca()
    plt.plot(amps_fit, t1s_fit, color="black", zorder=1, linewidth=LW)
    plt.scatter(amps_data, t1s_data, color="black",
                s=F1B_MS_DATA, marker="o", zorder=3,
                linewidths=F1B_MEW_DATA, edgecolors="black")
    plt.errorbar(amps_data, t1s_data, yerr=t1s_data_err, linestyle="none",
                 elinewidth=F1B_ELW, zorder=2, ecolor="black")
    for (i, gate_type) in enumerate(gate_types):
        for (j, pulse_type) in enumerate(pulse_types):
            avg_amp = avg_amps[i * pulse_count + j]
            avg_amp_t1 = avg_amps_t1[i * pulse_count + j]
            color = PT_COLOR[pulse_type]
            marker = F1B_GT_MK[gate_type]
            ms = F1B_GT_MS[gate_type]
            zorder = 5 if marker == "*" else 4
            plt.plot(avg_amp, avg_amp_t1, marker=marker, ms=ms,
                     markeredgewidth=F1B_MEW_M, markeredgecolor="black",
                     color=color, alpha=F1B_ALPHA_M, zorder=zorder)
        #ENDFOR
        plt.plot([], [], label="{}".format(GT_STR[gate_type]), color="black",
                 marker=marker, ms=ms, markeredgewidth=F1B_MEW_M,
                 markeredgecolor="black", linewidth=0)
    #ENDFOR
    plt.ylabel("$T_{1}$ (ms)", fontsize=LABEL_FS)
    plt.yticks(np.array([0, 1, 2, 3, 4]) * t1_normalize,
               ["0", "1", "2", "3", "4"], fontsize=LABEL_FS)
    plt.xlabel("$|a|$ (GHz)", fontsize=LABEL_FS)
    plt.xlim(-0.02, 0.5)
    plt.xticks([0, 0.2, 0.4], ["0", "0.2", "0.4"], fontsize=LABEL_FS)
    ax.tick_params(direction="in", labelsize=TICK_FS)
    plt.legend(frameon=False, fontsize=LEGEND_FS, loc="lower left",
               bbox_to_anchor=[0.5, 0.05], handletextpad=0.1)
    fig.text(0, 0.955, "(b)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.19, right=0.997, bottom=0.15, top=0.96, wspace=None, hspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure1b to {}"
          "".format(plot_file_path))
#ENDDEF


F1C_GT_LS = {
    GateType.zpiby2: "solid",
    GateType.ypiby2: "dashed",
    GateType.xpiby2: "dashdot",
}
F1C_LB = 1.1
F1C_ALPHA = np.array([[1, 0.4], [1, 0.4], [1, 1]])
def make_figure1c():
    """
    Refs:
    [0] https://stackoverflow.com/questions/17458580/embedding-small
    -plots-inside-subplots-in-matplotlib
    """
    log_transform = lambda x: np.log10(x) / np.log10(F1C_LB)
    
    data_file_path = latest_file_path("h5", "f1c", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
        gate_types = [GateType(gt) for gt in data_file["gate_types"][()]]
        gate_errors = np.moveaxis(data_file["gate_errors"][()], (0, 1, 2, 3), (3, 2, 1, 0))
    #ENDWITH
    gate_errors = np.mean(gate_errors, axis=2)
    gate_count = gate_errors.shape[2] - 1
    pulse_type_count = len(pulse_types)
    gate_count_axis = np.arange(0, gate_count + 1)
    inds = np.arange(0, gate_count)

    fig = plt.figure(figsize=(PAPER_TW * 0.4, PAPER_TW * 0.315))
    ax = plt.gca()
    inax = fig.add_axes((0.71, 0.4, 0.25, 0.25))
    for (i, gate_type) in enumerate(gate_types):
        for (j, pulse_type) in enumerate(pulse_types):
            color = PT_COLOR[pulse_type]
            linestyle = F1C_GT_LS[gate_type]
            gate_errors_ = gate_errors[i, j, :]
            alpha = F1C_ALPHA[i, j]
            ax.plot(gate_count_axis[inds], gate_errors_[inds],
                     color=color, linestyle=linestyle, linewidth=LW, alpha=alpha)
            inax.plot(gate_count_axis[inds], log_transform(gate_errors_[inds]),
                      color=color, linestyle=linestyle, linewidth=LW, alpha=alpha)
        #ENDFOR
    #ENDFOR

    # configure main 
    ax.set_ylabel("Gate Error", fontsize=LABEL_FS)
    ax.set_xlabel("Gate Count", fontsize=LABEL_FS)
    ax.set_xlim(0, gate_count)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xticklabels(["0", "500", "1000", "1500"])
    # ax.set_ylim(0, 1e-3)
    ax.tick_params(direction="in", labelsize=TICK_FS)
    ax.plot([], [], label="Z/2", linestyle=F1C_GT_LS[GateType.zpiby2], color="black")
    ax.plot([], [], label="Y/2", linestyle=F1C_GT_LS[GateType.ypiby2], color="black")
    ax.plot([], [], label="X/2", linestyle=F1C_GT_LS[GateType.xpiby2], color="black")
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="lower left",
               bbox_to_anchor=(-0.02, 0.67), handletextpad=0.4, handlelength=1.7)

    # configure inset
    inax.set_xlim(0, 50)
    inax.set_xticks([0, 20, 40])
    inax.set_xticklabels(["0", "20", "40"])
    inax.set_ylim(log_transform(1e-5), log_transform(1e-2))
    inax.set_yticks(log_transform(np.array([1e-5, 1e-4, 1e-3, 1e-2])))
    inax.set_yticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", ""])
    inax.tick_params(direction="in", labelsize=6)

    # configure all
    fig.text(0, 0.955, "(c)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.17, right=0.96, bottom=0.15, top=0.94, wspace=None, hspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure1c to {}"
          "".format(plot_file_path))
#ENDDEF


# FIGURE 2 #

F2A_XEPS = 1e-1
F2A_TF = 36.
def make_figure2a():
    data_file_path = latest_file_path("h5", "f2a", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
        save_file_paths = data_file["save_file_paths"][()]
    #ENDWITH
    pulse_type_count = len(pulse_types)
    (fig, axs) = plt.subplots(pulse_type_count, figsize=(PAPER_TW * 0.315, PAPER_TW * 0.315))
    for (i, pulse_type) in enumerate(pulse_types):
        color = PT_COLOR[pulse_type]
        label = "{}".format(PT_STR[pulse_type])
        linestyle = "solid" #PT_LS[pulse_type]
        save_file_path = save_file_paths[i]
        if save_file_path:
            (controls, evolution_time, dt) = grab_controls(save_file_path)
            (control_eval_count, control_count) = controls.shape
            control_eval_times = np.arange(0, control_eval_count, 1) * dt
            axs[i].plot(control_eval_times, controls[:, 0], color=color, label=label,
                        linestyle=linestyle, linewidth=LW, zorder=2)
        #ENDWITH
        axs[i].set_xlim((0 - F2A_XEPS, F2A_TF + F2A_XEPS))
        axs[i].set_ylim((-0.63, 0.63))
        axs[i].set_yticks([-0.4, 0, 0.4])
        #ENDIF
        axs[i].axhline(0, color="grey", alpha=0.2, zorder=1, linewidth=0.8)
        axs[i].tick_params(direction="in", labelsize=TICK_FS)
        axs[i].set_xticks([])
    #ENDFOR
    fig.text(0.27, 0.91, "Anl.", fontsize=TEXT_FS)
    fig.text(0.27, 0.75, "S-8", fontsize=TEXT_FS)
    fig.text(0.27, 0.59, "SU-10", fontsize=TEXT_FS)
    fig.text(0.27, 0.425, "D-1", fontsize=TEXT_FS)
    fig.text(0.27, 0.265, "D-2", fontsize=TEXT_FS)
    axs[2].set_ylabel("$a$ (GHz)", fontsize=LABEL_FS)
    axs[4].set_xlabel("$t$ (ns)", fontsize=LABEL_FS)
    xticks = [0 - F2A_XEPS, 10, 20, 30]
    xtick_labels = ["{}".format(int(np.ceil(xtick))) for xtick in xticks]
    axs[4].set_xticks(xticks)
    axs[4].set_xticklabels(xtick_labels)
    fig.text(0, 0.955, "(a)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.23, right=0.997, top=0.96, bottom=0.15, hspace=0., wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Saved Figure2a to {}"
          "".format(plot_file_path))
#ENDDEF


F2B_LB = 1.1
F2B_PT_ZO = {
    PulseType.d1: 1,
    PulseType.d1b: 1,
    PulseType.d1bb: 1,
    PulseType.d1bbb: 1,
    PulseType.analytic: 2,
}
def make_figure2b():
    log_transform = lambda x: np.log10(x) / np.log10(F2B_LB)
    
    data_file_path = latest_file_path("h5", "f2b", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        fq_devs = data_file["fq_devs"][()]
        gate_errors = np.moveaxis(data_file["gate_errors"][()], (0, 1, 2), (2, 1, 0))
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
    #ENDWITH
    dev_count_by2 = int((fq_devs.shape[0] - 1) / 2)
    fq_devs_abs = np.concatenate((fq_devs[-1:], fq_devs[dev_count_by2:-1]))
    gate_errors = np.mean(gate_errors, axis=2)
    gate_errors_z = gate_errors[:, -1:]
    gate_errors_lo = gate_errors[:, 0:dev_count_by2]
    gate_errors_hi = gate_errors[:, dev_count_by2:-1]
    gate_errors_avg = (np.flip(gate_errors_lo, axis=1) + gate_errors_hi) / 2
    gate_errors = np.concatenate((gate_errors_z, gate_errors_avg), axis=1)
    
    fig = plt.figure(figsize=(PAPER_TW * 0.23, PAPER_TW * 0.315))
    ax = plt.gca()
    for (i, pulse_type) in enumerate(pulse_types):
        zorder = F2B_PT_ZO[pulse_type]
        linestyle = PT_LS[pulse_type]
        color = PT_COLOR[pulse_type]
        log_mge = log_transform(gate_errors[i, :])
        plt.plot(fq_devs_abs, log_mge, color=color,
                 linestyle=linestyle, linewidth=LW, zorder=zorder)
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
    fig.text(0.78, 0.69, "$t_{N}$", fontsize=TEXT_FS)
    plt.plot([], [], label="18ns", linestyle="solid", color="black")
    plt.plot([], [], label="36ns", linestyle=DASH_LS, color="black")
    plt.plot([], [], label="54ns", linestyle=DDASH_LS, color="black")
    plt.plot([], [], label="72ns", linestyle=DDDASH_LS, color="black")
    plt.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.05, 0.24),
               fontsize=8, handlelength=1.85, handletextpad=0.4)
    fig.text(0, 0.955, "(c)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.3, right=0.97, bottom=0.15, top=0.96, hspace=None, wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure2b to {}"
          "".format(plot_file_path))
#ENDDEF


F2C_MS = 30
F2C_MSS = 40
F2C_MEW = 0.5
F2C_PT_ZO = {
    PulseType.analytic: 6,
    PulseType.s2: 3,
    PulseType.sut8: 2,
    PulseType.d1: 5,
    PulseType.d2: 4,
}
F2C_YEPS = 1e-5
def make_figure2c():
    fig = plt.figure(figsize=(PAPER_TW * 0.4, PAPER_TW * 0.315))
    ax = plt.gca()
    data_file_path = latest_file_path("h5", "f2c", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        gate_errors = np.moveaxis(data_file["gate_errors"][()], (0, 1, 2), (2, 1, 0))
        gate_times = data_file["gate_times"][()]
        pulse_types = [PulseType(ipt) for ipt in data_file["pulse_types"][()]]
    #ENDWITH
    gate_errors = np.mean(gate_errors, axis=2)
    ge_max = 0
    
    for (i, pulse_type) in enumerate(pulse_types):
        label = "{}".format(PT_STR[pulse_type])
        color = PT_COLOR[pulse_type]
        marker = PT_MARKER[pulse_type]
        marker_size = F2C_MSS if marker == "*" else F2C_MS
        zorder = F2C_PT_ZO[pulse_type]
        for (j, gate_time) in enumerate(gate_times):
            gate_error = gate_errors[i, j]
            if not np.allclose(gate_error, 1):
                ge_max = np.maximum(ge_max, gate_error)
            #ENDIF
            plt.scatter([gate_time], [gate_error], color=color,
                        marker=marker, s=marker_size, linewidths=F2C_MEW,
                        edgecolors="black", zorder=zorder)
        #ENDFOR
        plt.scatter([], [], label=label, color=color, linewidths=F2C_MEW,
                    s=F2C_MS, marker=marker, edgecolors="black")
    #ENDFOR
    yticks_ = np.arange(0, 5 + 1, 1)
    yticks = 1e-5 * yticks_
    ytick_labels = ["{:d}".format(ytick) for ytick in yticks_]
    plt.yticks(yticks, ytick_labels)
    plt.ylim(yticks[0], yticks[-1])
    # xticks_ = np.arange(50, 160 + 1, 20)
    # xtick_labels = ["{:d}".format(xtick) for xtick in xticks_]
    # plt.xticks(xticks_, xtick_labels)
    ax.tick_params(direction="in", labelsize=TICK_FS)
    plt.xlabel("$t_{N}$ (ns)", fontsize=LABEL_FS)
    plt.ylabel("Gate Error ($10^{-5}$)", fontsize=LABEL_FS)
    plt.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.03, 0),
               handletextpad=0.2, fontsize=8, ncol=2, columnspacing=0.)
    fig.text(0, 0.955, "(b)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.11, right=0.997, top=0.96, bottom=0.15, hspace=None, wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure2c to {}".format(plot_file_path))
#ENDDEF


# FIGURE 3 #

F3A_XEPS = [2e-1] * 5
def make_figure3a():
    data_file_path = latest_file_path("h5", "f3a", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
        save_file_paths = data_file["save_file_paths"][()]
    #ENDWITH
    pulse_type_count = len(pulse_types)
    
    (fig, axs) = plt.subplots(pulse_type_count, figsize=(PAPER_LW, PAPER_LW * 0.8))
    for (i, pulse_type) in enumerate(pulse_types):
        color = PT_COLOR[pulse_type]
        label = "{}".format(PT_STR[pulse_type])
        linestyle = "solid" #PT_LS[pulse_type]
        save_file_path = save_file_paths[i]
        if save_file_path:
            (controls, evolution_time, dt) = grab_controls(save_file_path)
            (control_eval_count, control_count) = controls.shape
            control_eval_times = np.arange(0, control_eval_count, 1) * dt
            axs[i].plot(control_eval_times, controls[:, 0], color=color, label=label,
                        linestyle=linestyle, zorder=2, linewidth=LW)
        #ENDIF
        axs[i].set_xlim((0 - F3A_XEPS[i], 56.80 + F3A_XEPS[i]))
        if pulse_type == PulseType.analytic:
            axs[i].set_ylim(-0.15, 0.15)
            axs[i].set_yticks([-0.1, 0, 0.1])
        else:
            axs[i].set_ylim((-0.63, 0.63))
            axs[i].set_yticks([-0.4, 0, 0.4])
        #ENDIF
        axs[i].axhline(0, color="grey", alpha=0.2, zorder=1, linewidth=0.8)
        axs[i].tick_params(direction="in", labelsize=TICK_FS)
        axs[i].set_xticks([])
    #ENDFOR
    axs[0].text(3, 0.07, "Anl.", fontsize=TEXT_FS)
    axs[1].text(2, 0.3, "S-8", fontsize=TEXT_FS)
    axs[2].text(2, 0.3, "SU-10", fontsize=TEXT_FS)
    axs[3].text(2, 0.3, "D-1", fontsize=TEXT_FS)
    axs[4].text(2, 0.3, "D-2", fontsize=TEXT_FS)
    axs[2].set_ylabel("$a$ (GHz)", fontsize=LABEL_FS)
    axs[4].set_xlabel("$t$ (ns)", fontsize=LABEL_FS)
    axs[4].set_xticks([0 - F3A_XEPS[i], 10, 20, 30, 40, 50])
    axs[4].set_xticklabels(["0", "10", "20", "30", "40", "50"])
    fig.text(0, 0.955, "(a)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.15, right=0.997, top=0.96, bottom=0.116, hspace=0., wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Saved Figure3a to {}"
          "".format(plot_file_path))
#ENDDEF


F3B_LB = 1.1
def make_figure3b():
    log_transform = lambda x: np.log10(x) / np.log10(F3B_LB)
    data_file_path = latest_file_path("h5", "f3b", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
        save_file_paths = data_file["save_file_paths"][()]
        gate_errors = np.moveaxis(data_file["gate_errors"][()], (0, 1, 2), (2, 1, 0))
    #ENDWITH
    pulse_type_count = len(pulse_types)
    gate_count = gate_errors.shape[1] - 1
    gate_count_axis = np.arange(0, gate_count + 1)
    gate_errors = np.mean(gate_errors, axis=2)

    fig = plt.figure(figsize=(PAPER_LW, PAPER_LW * 0.8))
    ax = plt.gca()
    for (i, pulse_type) in enumerate(pulse_types):
        color = PT_COLOR[pulse_type]
        linestyle = "solid"
        ax.plot(gate_count_axis, log_transform(gate_errors[i, :]),
                color=color, linestyle=linestyle)
    #ENDFOR

    ax.set_ylabel("Gate Error", fontsize=LABEL_FS)
    ax.set_xlabel("Gate Count", fontsize=LABEL_FS)
    ax_xticks = np.arange(0, gate_count + 1, 100)
    ax.set_xticks(ax_xticks)
    ax.set_xticklabels(["{}".format(ax_xtick) for ax_xtick in ax_xticks])
    ax.set_xlim(0, gate_count)
    ax_yticks_ = np.arange(-6., -2. + 1, 1.)
    ax_yticks = log_transform(10 ** ax_yticks_)
    ax.set_yticks(ax_yticks)
    ax.set_yticklabels(["$10^{{{0}}}$".format(int(ax_ytick)) for ax_ytick in ax_yticks_])
    ax.set_ylim(ax_yticks[0], ax_yticks[-1])
    ax.tick_params(direction="in", labelsize=TICK_FS)

    fig.text(0, 0.96, "(b)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.11, hspace=0., wspace=None)
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


# FIGURE 4 #

F4A_XEPS = 5e-2
F4A_YEPS = [5e-2, 0, 0, 5e-2, 0]
def make_figure4a():
    data_file_path = latest_file_path("h5", "f4a", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
        save_file_paths = data_file["save_file_paths"][()]
    #ENDWITH
    pulse_type_count = len(pulse_types)
    
    (fig, axs) = plt.subplots(pulse_type_count, figsize=(PAPER_LW * 0.6, PAPER_LW * 0.8))
    for (i, pulse_type) in enumerate(pulse_types):
        color = PT_COLOR[pulse_type]
        label = "{}".format(PT_STR[pulse_type])
        linestyle = "solid" #PT_LS[pulse_type]
        save_file_path = save_file_paths[i]
        if save_file_path:
            (controls, evolution_time, dt) = grab_controls(save_file_path)
            (control_eval_count, control_count) = controls.shape
            control_eval_times = np.arange(0, control_eval_count, 1) * dt
            controls[0, 0] = controls[-1, 0] = 0.
            axs[i].plot(control_eval_times, controls[:, 0], color=color, label=label,
                        linestyle=linestyle, zorder=2, linewidth=LW)
        #ENDIF
        axs[i].set_xlim((0 - F4A_XEPS, 10. + F4A_XEPS))
        axs[i].axhline(0, color="grey", alpha=0.2, zorder=1, linewidth=0.8)
        axs[i].set_xticks([])
        if pulse_type == PulseType.corpse:
            axs[i].set_ylim([-0.35, 0.35])
            axs[i].set_yticks([-0.2, 0, 0.2])
        else:
            axs[i].set_ylim([-0.65, 0.65])
            axs[i].set_yticks([-0.4, 0, 0.4])
        #ENDIF
        axs[i].tick_params(direction="in", labelsize=TICK_FS)
    #ENDFOR
    # axs[0].text(3, 0.05, "Anl.", fontsize=TEXT_FS)
    # fig.text(0.3, 0.75, "S-2", fontsize=TEXT_FS)
    # fig.text(0.3, 0.59, "S-4", fontsize=TEXT_FS)
    # axs[3].text(3, 0.25, "D-2", fontsize=TEXT_FS)
    # axs[4].text(3, 0.25, "D-3", fontsize=TEXT_FS)
    axs[4].set_xticks([0 - F4A_XEPS, 5, 10 + F4A_XEPS])
    axs[4].set_xticklabels(["0", "5", "10"])
    axs[4].tick_params(direction="in", labelsize=TICK_FS)
    axs[2].set_ylabel("$a$ (GHz)", fontsize=LABEL_FS)
    axs[4].set_xlabel("$t$ (ns)", fontsize=LABEL_FS)
    fig.text(0, 0.955, "(a)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.25, right=0.96, top=0.96, bottom=0.12, hspace=0., wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Saved Figure4a to {}"
          "".format(plot_file_path))
#ENDDEF


F4B_LB = 1.1
def make_figure4b():
    log_transform = lambda x: np.log10(x) / np.log10(F4B_LB)
    
    data_file_path = latest_file_path("h5", "f4b", SAVE_PATH)
    with h5py.File(data_file_path, "r") as data_file:
        fq_devs = data_file["fq_devs"][()]
        gate_errors = np.moveaxis(data_file["gate_errors"][()], (0, 1, 2), (2, 1, 0))
        pulse_types = [PulseType(pt) for pt in data_file["pulse_types"][()]]
    #ENDWITH
    dev_count_by2 = int((fq_devs.shape[0] - 1) / 2)
    fq_devs_abs = np.concatenate((fq_devs[-1:], fq_devs[dev_count_by2:-1]))
    gate_errors = np.mean(gate_errors, axis=2)
    gate_errors_z = gate_errors[:, -1:]
    gate_errors_lo = gate_errors[:, 0:dev_count_by2]
    gate_errors_hi = gate_errors[:, dev_count_by2:-1]
    gate_errors_avg = (np.flip(gate_errors_lo, axis=1) + gate_errors_hi) / 2
    gate_errors = np.concatenate((gate_errors_z, gate_errors_avg), axis=1)
    
    fig = plt.figure(figsize=(PAPER_LW * 0.6, PAPER_LW * 0.8))
    ax = plt.gca()
    for (i, pulse_type) in enumerate(pulse_types):
        if pulse_type in [PulseType.s2, PulseType.s4, PulseType.d3]:
            continue
        #ENDIF
        linestyle = PT_LS[pulse_type]
        color = PT_COLOR[pulse_type]
        log_mge = log_transform(gate_errors[i, :])
        label = PT_STR[pulse_type]
        plt.plot(fq_devs_abs, log_mge, color=color,
                 linestyle=linestyle, linewidth=LW, label=label)
    #ENDFOR
    plt.xlim(min(fq_devs_abs), max(fq_devs_abs))
    plt.xticks(
        np.array([0, 1, 2]) * 1e-2,
    )
    yticks_ = np.arange(-12., -3 + 1, 1)
    yticklabels = ["$10^{{{0}}}$".format(int(ytick)) for ytick in yticks_]
    yticks = 10 ** yticks_
    plt.ylim(log_transform(yticks[0]), log_transform(yticks[-1]))
    plt.yticks(log_transform(yticks), yticklabels)
    ax.tick_params(direction="in", labelsize=TICK_FS)
    plt.xlabel("$|\delta f_{q}| \; (GHz)$", fontsize=LABEL_FS)
    plt.ylabel("Gate Error", fontsize=LABEL_FS)
    fig.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.9, 0.15),
               fontsize=8, handlelength=1.5, handletextpad=0.4)
    fig.text(0, 0.955, "(b)", fontsize=TEXT_FS)
    plt.subplots_adjust(left=0.27, right=0.92, bottom=0.12, top=0.96, hspace=None, wspace=None)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    plt.savefig(plot_file_path, dpi=DPI_FINAL)
    print("Plotted Figure4b to {}"
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
    elif fig_str == "4a":
        make_figure4a()
    elif fig_str == "4b":
        make_figure4b()
    elif fig_str == "all":
        make_figure1a()
        make_figure1b()
        make_figure1c()
        make_figure2a()
        make_figure2b()
        make_figure2c()
        make_figure3a()
        make_figure3b()
    #ENDIF
#ENDDEF


if __name__ == "__main__":
    main()
#ENDIF
