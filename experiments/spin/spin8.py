"""
spin8.py - spin experiment 8
"""

from argparse import ArgumentParser
from copy import copy
import os

import autograd.numpy as anp
from filelock import FileLock
import h5py
import numpy as np
from qoc import (
    evolve_schroedinger_discrete,
    grape_schroedinger_discrete,
)
from qoc.standard import (
    Adam, LBFGSB,
    ControlArea,
    TargetStateInfidelity,
    TargetStateInfidelityTime,
    conjugate_transpose, matmuls, krons,
    get_annihilation_operator, get_creation_operator,
    generate_save_file_path,
    SIGMA_X, SIGMA_Z
)

# Define paths
META_NAME = "spin"
EXPERIMENT_NAME = "spin8"
SAVE_PATH = os.path.join(os.environ["ROBUST_QOC_PATH"], "out", META_NAME, EXPERIMENT_NAME)


# Define experimental constants.
FLUXONIUM_FREQ = 2 * np.pi * 1e-2 #GHz
MAX_AMP_NORM_FLUXONIUM = 2 * np.pi * 3e-1 #GHz
SAMPLE_RATE = 1.2 #samples/ns


# Define the system
FLUXONIUM_STATE_COUNT = 2
FLUXONIUM_VACUUM = np.zeros((FLUXONIUM_STATE_COUNT, 1))
FLUXONIUM_G = np.copy(FLUXONIUM_VACUUM)
FLUXONIUM_G[0][0] = 1.
FLUXONIUM_E = np.copy(FLUXONIUM_VACUUM)
FLUXONIUM_E[1][0] = 1.
HAMILTONIAN_ARGS = np.array([FLUXONIUM_FREQ])
H_SYSTEM = (
    SIGMA_Z / 2
)
H_CONTROL_0 = (
    SIGMA_X / 2
)

hamiltonian = lambda controls, hargs, time: (
    hargs[0] * H_SYSTEM
    + controls[0] * H_CONTROL_0
)
CONTROL_COUNT = 1
COMPLEX_CONTROLS = False
MAX_CONTROL_NORMS = np.array((
    MAX_AMP_NORM_FLUXONIUM,
))


# Define the problem
EVOLUTION_TIME = 1.2e2 #ns
CONTROL_EVAL_COUNT = int(np.ceil(SAMPLE_RATE * EVOLUTION_TIME))
SYSTEM_EVAL_COUNT = 4 * int(EVOLUTION_TIME) + 1
INITIAL_STATE_0 = FLUXONIUM_G
INITIAL_STATES = np.stack((INITIAL_STATE_0,))
TARGET_STATE_0 = np.array([[2 ** (-1/2)],
                           [2 ** (-1/2)]])
TARGET_STATES = np.stack((TARGET_STATE_0,))
COSTS = (
    TargetStateInfidelity(TARGET_STATES),
    ControlArea(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                max_control_norms=MAX_CONTROL_NORMS,),
)


# Define the optimization
LEARNING_RATE = 1e-3
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
# OPTIMIZER = LBFGSB()
ITERATION_COUNT = int(3e3)
GRAB_CONTROLS = True
CREATE_CONTROLS = False
if GRAB_CONTROLS:
    controls_path = os.path.join(SAVE_PATH, "00072_spin8.h5")
    controls_path_lock = "{}.lock".format(controls_path)
    with FileLock(controls_path_lock):
        with h5py.File(controls_path) as save_file:
            index = np.argmin(save_file["error"])
            controls = save_file["controls"][index][()]
        #ENDWITH
    #ENDWITH
    INITIAL_CONTROLS = controls
elif CREATE_CONTROLS:
    max_ = np.ones((CONTROL_EVAL_COUNT, CONTROL_COUNT)) * MAX_CONTROL_NORMS
    INITIAL_CONTROLS = max_ * 1e-4
else:
    INITIAL_CONTROLS = None


# Define constraints
def impose_control_conditions(controls):
    controls[0,:] = 0
    controls[-1,:] = 0
    return controls


# Define the output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES_EVOL = False

GRAPE_CONFIG = {
    "control_count": CONTROL_COUNT,
    "control_eval_count": CONTROL_EVAL_COUNT,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "hamiltonian_args": HAMILTONIAN_ARGS,
    "impose_control_conditions": impose_control_conditions,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "complex_controls": COMPLEX_CONTROLS,
    "initial_controls": INITIAL_CONTROLS,
    "iteration_count": ITERATION_COUNT,
    "log_iteration_step": LOG_ITERATION_STEP,
    "max_control_norms": MAX_CONTROL_NORMS,
    "optimizer": OPTIMIZER,
    "save_iteration_step": SAVE_ITERATION_STEP,
}

EVOL_CONFIG = {
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "hamiltonian_args": HAMILTONIAN_ARGS,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "controls": INITIAL_CONTROLS,
    "costs": COSTS,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES_EVOL,
}

def run_grape():
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    config = copy(GRAPE_CONFIG)
    config.update({
        "save_file_path": save_file_path
    })
    result = grape_schroedinger_discrete(**config)


def run_evolve():
    config = copy(EVOL_CONFIG)
    if SAVE_INTERMEDIATE_STATES_EVOL:
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        config.update({
            "save_file_path": save_file_path
        })
    result = evolve_schroedinger_discrete(**config)
    print(result.error)


def main():
    parser = ArgumentParser()
    parser.add_argument("--grape", action="store_true")
    parser.add_argument("--evolve", action="store_true")
    args = vars(parser.parse_args())
    do_grape = args["grape"]
    do_evolve = args["evolve"]

    if do_grape:
        run_grape()
    elif do_evolve:
        run_evolve()


if __name__ == "__main__":
    main()
