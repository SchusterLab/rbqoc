"""
spin_exp1.py - Spin experiment 1.
"""

import os


import autograd.numpy as anp
from filelock import FileLock, Timeout
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from qoc import grape_schroedinger_discrete, evolve_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z, SIGMA_X,
                          generate_save_file_path,
                          krons, plot_controls,
                          plot_state_population,
                          Adam, LBFGSB,)

CORE_COUNT = 8
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define experimental constants.
CONTROL_SAMPLING_RATE = 1.2 # GS/s
MAX_AMP_CONTROL_0 = 1e-1 # GHz
MAX_AMP_BANDWIDTH_CONTROL_0 = 5e-1 # GHz
OMEGA_Q = 1e-2 # GHz

# Define the system.
SYSTEM_HAMILTONIAN_0 = SIGMA_Z / 2
CONTROL_HAMILTONIAN_0 = SIGMA_X / 2
HAMILTONIAN_ARGS = anp.array([OMEGA_Q])
hamiltonian = lambda controls, time, args: (args[0] * SYSTEM_HAMILTONIAN_0
                                            + controls[0] * CONTROL_HAMILTONIAN_0)
MAX_CONTROL_NORMS = anp.array((MAX_AMP_CONTROL_0,))
MAX_CONTROL_BANDWIDTHS = anp.array((MAX_AMP_BANDWIDTH_CONTROL_0,))

# Define the optimization.
ITERATION_COUNT = int(1e4)
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 200 # nanoseconds
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = EVOLUTION_TIME + 1
OPTIMIZER = Adam(learning_rate=1e-4)

# Define the problem.
INITIAL_STATE_0 = anp.array(((1,),
                             (0,)))
INITIAL_STATE_1 = anp.array(((0,),
                             (1,)))
INITIAL_STATES = anp.stack((INITIAL_STATE_0, INITIAL_STATE_1,))
r2_by_2 = anp.sqrt(2) / 2
TARGET_STATE_0 = anp.array(((r2_by_2,),
                            (-1j * r2_by_2,)))
TARGET_STATE_1 = anp.array(((-1j * r2_by_2,),
                            (r2_by_2,)))
TARGET_STATES = anp.stack((TARGET_STATE_0, TARGET_STATE_1,))
COSTS = [TargetStateInfidelity(TARGET_STATES)]

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES = False
EVOL_SAVE_INTERMEDIATE_STATES = True
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin_exp1"
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
SAVE_FILE = EXPERIMENT_NAME

EVOL_ARGS = {
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "costs": COSTS,
}

def main():
    initial_controls = None
    grape_save_file_path = generate_save_file_path(SAVE_FILE, SAVE_PATH)
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES, SYSTEM_EVAL_COUNT,
                                         complex_controls=COMPLEX_CONTROLS,
                                         hamiltonian_args=HAMILTONIAN_ARGS,
                                         initial_controls=initial_controls,
                                         iteration_count=ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_control_norms=MAX_CONTROL_NORMS,
                                         optimizer=OPTIMIZER,
                                         save_file_path=grape_save_file_path,
                                         save_intermediate_states=SAVE_INTERMEDIATE_STATES,
                                         save_iteration_step=SAVE_ITERATION_STEP,)


if __name__ == "__main__":
    main()
