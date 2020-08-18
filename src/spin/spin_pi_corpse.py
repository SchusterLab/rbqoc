"""
spin_pi_corpse.py
"""

from argparse import ArgumentParser
import os

import autograd.numpy as anp
from filelock import FileLock, Timeout
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from qoc import grape_schroedinger_discrete, evolve_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z, SIGMA_X, RX,
                          generate_save_file_path,
                          matrix_to_column_vector_list,
                          column_vector_list_to_matrix,
                          matmuls,
                          krons, plot_controls,
                          plot_state_population,
                          Adam, LBFGSB,)
from qutip import mesolve, Qobj

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
hamiltonian = lambda controls, hargs, time: (
    hargs[0] * SYSTEM_HAMILTONIAN_0
    + controls[0] * CONTROL_HAMILTONIAN_0)
MAX_CONTROL_NORMS = anp.array((MAX_AMP_CONTROL_0,))
MAX_CONTROL_BANDWIDTHS = anp.array((MAX_AMP_BANDWIDTH_CONTROL_0,))

# Define the optimization.
WDIR = os.environ["ROBUST_QOC_PATH"]
CONTROLS_PATH_1 = os.path.join(WDIR, "out/spin/spin_exp4/00002_spin_exp4.h5")
CONTROLS_PATH_LOCK_1 = "{}.lock".format(CONTROLS_PATH_1)
CONTROLS_PATH_2 = os.path.join(WDIR, "out/spin/spin_exp5/00002_spin_exp5.h5")
CONTROLS_PATH_LOCK_2 = "{}.lock".format(CONTROLS_PATH_2)
CONTROLS_PATH_3 = os.path.join(WDIR, "out/spin/spin_exp6/00001_spin_exp6.h5")
CONTROLS_PATH_LOCK_3 = "{}.lock".format(CONTROLS_PATH_3)
def grab_controls(controls_path, controls_path_lock):
    try:
        with FileLock(controls_path_lock):
            with h5py.File(controls_path) as controls_file:
                index = anp.argmin(controls_file["error"][()])
                controls = controls_file["controls"][index][()]
                unitary = column_vector_list_to_matrix(controls_file["final_states"][index][()])
        #ENDWITH
    except Timeout:
        controls = None
        print("Timeout while locking {}"
              "".format(controls_path_lock))
    return controls, unitary
CONTROLS_1, UNITARY_1 = grab_controls(CONTROLS_PATH_1, CONTROLS_PATH_LOCK_1)
CONTROLS_2, UNITARY_2 = grab_controls(CONTROLS_PATH_2, CONTROLS_PATH_LOCK_2)
CONTROLS_3, UNITARY_3 = grab_controls(CONTROLS_PATH_3, CONTROLS_PATH_LOCK_3)
CONTROLS = np.vstack((
    CONTROLS_1,
    CONTROLS_2[1:,],
    CONTROLS_3[1:,],
    ))
# print("c1:\n{}\nc2:\n{}c:\n{}"
#       "".format(CONTROLS_3, CONTROLS_2, CONTROLS))
# print("predicted_unitary:\n{}"
#       "".format(matmuls(UNITARY_3, UNITARY_2, UNITARY_1)))
INITIAL_CONTROLS = CONTROLS
ITERATION_COUNT = int(1e4)
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 150 # nanoseconds
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = EVOLUTION_TIME + 1
LEARNING_RATE = 1e-2
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

# Define the problem.
INITIAL_STATE_0 = anp.array(((1,),
                             (0,)))
INITIAL_STATE_1 = anp.array(((0,),
                             (1,)))
INITIAL_STATES = anp.stack((INITIAL_STATE_0, INITIAL_STATE_1,))
THETA = anp.pi
THETA_1 = 2 * anp.pi + THETA / 2 - anp.arcsin(anp.sin(THETA / 2) / 2)
THETA_2 = 2 * anp.pi - 2 * anp.arcsin(anp.sin(THETA / 2) / 2)
THETA_3 = THETA / 2 - anp.arcsin(anp.sin(THETA / 2) / 2)
RX1 = RX(THETA_1)
RX2 = RX(-THETA_2)
RX3 = RX(THETA_3)
# print("t1: {}\nt2: {}\nt3: {}\nrx1:\n{}\nrx2:\n{}\nrx3:\n{}\nrx32:\n{}\nrx321:\n{}"
#       "".format(THETA_1, -THETA_2, THETA_3, RX1, RX2, RX3, matmuls(RX3, RX2),
#                 matmuls(RX3, RX2, RX1)))
TARGET_UNITARY = RX(THETA)
TARGET_STATES = matrix_to_column_vector_list(TARGET_UNITARY)
COSTS = (
    TargetStateInfidelity(TARGET_STATES),
    )

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES = False
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin_pi_corpse"
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
    "hamiltonian_args": HAMILTONIAN_ARGS,
}

# Qutip
H0_QUTIP = Qobj(OMEGA_Q * SYSTEM_HAMILTONIAN_0)
H0_AMP_QUTIP = np.ones(CONTROLS.shape[0])
H1_QUTIP = Qobj(CONTROL_HAMILTONIAN_0)
H1_AMP_QUTIP = CONTROLS
HLIST_QUTIP = [[H0_QUTIP, H0_AMP_QUTIP], [H1_QUTIP, H1_AMP_QUTIP]]
INITIAL_STATE_QUTIP = Qobj(INITIAL_STATE_0)
COPS_QUTIP = list()
EOPS_QUTIP = list()
TLIST_QUTIP = np.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT)

QUTIP_CONFIG = {
    "H": HLIST_QUTIP,
    "rho0": INITIAL_STATE_QUTIP,
    "tlist": TLIST_QUTIP,
    "c_ops": COPS_QUTIP,
    "e_ops": EOPS_QUTIP,
}

def main():
    parser = ArgumentParser()
    parser.add_argument("--grape", action="store_true")
    parser.add_argument("--evol", action="store_true")
    parser.add_argument("--qutip", action="store_true")
    args = vars(parser.parse_args())
    do_grape = args["grape"]
    do_evol = args["evol"]
    do_qutip = args["qutip"]
    
    if do_grape:
        grape_save_file_path = generate_save_file_path(SAVE_FILE, SAVE_PATH)
        GRAPE_CONFIG.update({
                "save_file_path": grape_save_file_path,
                })
        result = grape_schroedinger_discrete(**GRAPE_CONFIG)
    elif do_evol:
        evol_save_file_path = generate_save_file_path(SAVE_FILE, SAVE_PATH)
        EVOL_CONFIG.update({
            "save_file_path": evol_save_file_path,
        })
        result = evolve_schroedinger_discrete(**EVOL_CONFIG)
        print("error: {}\nfinal_states:\n{}\ntarget_unitary:\n{}"
              "".format(result.error, column_vector_list_to_matrix(result.final_states),
                        TARGET_UNITARY))
    elif do_qutip:
        result = mesolve(**QUTIP_CONFIG)
        final_states_qutip = result.states[-1].full()
        print("final_states_qutip:\n{}"
              "".format(final_states_qutip))


if __name__ == "__main__":
    main()
