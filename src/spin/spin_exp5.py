"""
spin_exp5.py
"""

from argparse import ArgumentParser
import os

import autograd.numpy as anp
from filelock import FileLock, Timeout
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from qoc import grape_schroedinger_discrete, evolve_schroedinger_discrete
from qoc.standard import (
    TargetStateInfidelity, ControlVariation, ControlNorm,
    conjugate_transpose,
    get_annihilation_operator,
    get_creation_operator,
    SIGMA_Z, SIGMA_X,
    generate_save_file_path,
    matrix_to_column_vector_list,
    krons, plot_controls,
    plot_state_population,
    Adam, LBFGSB,
)

CORE_COUNT = 8
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define experimental constants.
CONTROL_SAMPLING_RATE = 1.2 # GS/s
MAX_AMP_CONTROL_0 = 2 * anp.pi * 1e-1 # GHz
MAX_AMP_BANDWIDTH_CONTROL_0 = 2 * anp.pi * 5e-1 # GHz
OMEGA_Q = 2 * anp.pi * 1e-2 # GHz

# Define the system.
SYSTEM_HAMILTONIAN_0 = SIGMA_Z / 2
CONTROL_HAMILTONIAN_0 = SIGMA_X / 2
HAMILTONIAN_ARGS = anp.array([OMEGA_Q])
hamiltonian = lambda controls, time: (
    OMEGA_Q * SYSTEM_HAMILTONIAN_0
    + controls[0] * CONTROL_HAMILTONIAN_0)
MAX_CONTROL_NORMS = anp.array((MAX_AMP_CONTROL_0,))
MAX_CONTROL_BANDWIDTHS = anp.array((MAX_AMP_BANDWIDTH_CONTROL_0,))

# Define the optimization.
WDIR = os.environ["ROBUST_QOC_PATH"]
# CONTROLS_PATH = os.path.join(WDIR, "out/spin/spin_exp5/00001_spin_exp5.h5")
# CONTROLS_PATH_LOCK = "{}.lock".format(CONTROLS_PATH)
# try:
#     with FileLock(CONTROLS_PATH_LOCK):
#         with h5py.File(CONTROLS_PATH) as controls_file:
#             index = anp.argmin(controls_file["error"][()])
#             initial_controls = controls_file["controls"][index]
#     #ENDWITH
# except Timeout:
#     initial_controls = None
#     print("Unable to load initial controls.")
# INITIAL_CONTROLS = initial_controls
INITIAL_CONTROLS = None
ITERATION_COUNT = int(1e4)
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 50 # nanoseconds
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = EVOLUTION_TIME + 1
LEARNING_RATE = 5e-2
OPTIMIZER = LBFGSB()
def impose_control_conditions(controls):
    controls[0, :]= 0
    controls[-1, :] = 0
    return controls

# Define the problem.
INITIAL_STATE_0 = anp.array(((1,),
                             (0,)))
INITIAL_STATE_1 = anp.array(((0,),
                             (1,)))
INITIAL_STATES = anp.stack((INITIAL_STATE_0, INITIAL_STATE_1,))
def RX(theta):
    return anp.array([[anp.cos(theta / 2), -1j * anp.sin(theta / 2)],
                     [-1j * anp.sin(theta / 2), anp.cos(theta / 2)]])
THETA = anp.pi
THETA_2 = 2 * anp.pi - 2 * anp.arcsin(anp.sin(THETA / 2) / 2)
TARGET_UNITARY = RX(-THETA_2)
TARGET_STATES = matrix_to_column_vector_list(TARGET_UNITARY)
def gauss(x):
    b = anp.mean(x)
    c = anp.std(x)
    return anp.exp(-((x - b) ** 2) / (c ** 2))
CONTROL_NORM_WEIGHTS = 1 - gauss(anp.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT))
CONTROL_NORM_WEIGHTS = anp.repeat(CONTROL_NORM_WEIGHTS[:, anp.newaxis], CONTROL_COUNT, axis=1)
CONTROL_NORM_MULTIPLIER = 0.5
FIDELITY_MULTIPLIER = 1.
COSTS = (
    TargetStateInfidelity(TARGET_STATES,
                          cost_multiplier=FIDELITY_MULTIPLIER,),
     # ControlNorm(CONTROL_COUNT, CONTROL_EVAL_COUNT,
     #             control_weights=CONTROL_NORM_WEIGHTS,
     #             cost_multiplier=CONTROL_NORM_MULTIPLIER,
     #             max_control_norms=MAX_CONTROL_NORMS,),
    )

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES = False
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin_exp5"
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
SAVE_FILE = EXPERIMENT_NAME

GRAPE_CONFIG = {
    "control_count": CONTROL_COUNT,
    "control_eval_count": CONTROL_EVAL_COUNT,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "complex_controls": COMPLEX_CONTROLS,
    # "hamiltonian_args": HAMILTONIAN_ARGS,
    "impose_control_conditions": impose_control_conditions,
    "initial_controls": INITIAL_CONTROLS,
    "iteration_count": ITERATION_COUNT,
    "log_iteration_step": LOG_ITERATION_STEP,
    "max_control_norms": MAX_CONTROL_NORMS,
    "optimizer": OPTIMIZER,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES,
    "save_iteration_step": SAVE_ITERATION_STEP,
}

EVOL_CONFIG = {
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "controls": INITIAL_CONTROLS,
    "costs": COSTS,
    # "hamiltonian_args": HAMILTONIAN_ARGS,
}

def main():
    parser = ArgumentParser()
    parser.add_argument("--grape", action="store_true")
    parser.add_argument("--evol", action="store_true")
    args = vars(parser.parse_args())
    do_grape = args["grape"]
    do_evol = args["evol"]
    if do_grape:
        grape_save_file_path = generate_save_file_path(SAVE_FILE, SAVE_PATH)
        GRAPE_CONFIG.update({
                "save_file_path": grape_save_file_path,
                })
        result = grape_schroedinger_discrete(**GRAPE_CONFIG)
    elif do_evol:
        result = evolve_schroedinger_discrete(**EVOL_CONFIG)
        print("error: {}\nfinal_states:\n{}\ntarget_states:\n{}"
              "".format(result.error, column_vector_list_to_matrix(result.final_states),
                        TARGET_UNITARY))


if __name__ == "__main__":
    main()
