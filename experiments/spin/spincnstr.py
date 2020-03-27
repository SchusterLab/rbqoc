"""
spincnstr.py - constrained robust optimization of the spin system
"""

from argparse import ArgumentParser
from copy import copy
import os

import autograd.numpy as anp
from filelock import FileLock, Timeout
import h5py
import numpy as np
from qoc import (grape_schroedinger_discrete,
                 evolve_schroedinger_discrete,)
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z, SIGMA_X,
                          generate_save_file_path,
                          SGD)

CORE_COUNT = 8
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)

MAX_AMP_0 = 2 * anp.pi * 3e-1
OMEGA = 2 * anp.pi * 1e-2

# Define the system.
HILBERT_SIZE = 2
H_SYSTEM_0 = SIGMA_Z / 2
H_CONTROL_0 = SIGMA_X / 2
hamiltonian = lambda controls, hargs, time: (
    hargs[0] * H_SYSTEM_0
    + controls[0] * H_CONTROL_0
)
HAMILTONIAN_ARGS = anp.array([OMEGA])
MAX_CONTROL_NORMS = anp.array([MAX_AMP_0])

# Define the problem.
INITIAL_STATE_0 = anp.array([[1], [0]])
TARGET_STATE_0 = anp.array([[0], [1]])
INITIAL_STATES = anp.stack((INITIAL_STATE_0,),)
TARGET_STATES = anp.stack((TARGET_STATE_0,),)
TARGET_STATE_CONSTRAINT = 1e-3
TARGET_STATE_MULT = 1e2
COSTS = [
    TargetStateInfidelity(TARGET_STATES,
                          cost_multiplier=TARGET_STATE_MULT,
                          constraint=TARGET_STATE_CONSTRAINT,)
]

# Define the optimization.
OPTIMIZER = SGD()
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 130
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = 2 * int(EVOLUTION_TIME) + 1
ITERATION_COUNT = 1000

def gs(X, row_vecs=True, norm=True):
    """
    References:
    [0] https://gist.github.com/iizukak/1287876/edad3c337844fac34f7e56ec09f9cb27d4907cc7
    """
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

    
# Generate orthonormal basis for zero net flux constraint manifold.
ZF_BASIS = np.zeros((CONTROL_EVAL_COUNT - 1, CONTROL_EVAL_COUNT))
for i in range(CONTROL_EVAL_COUNT - 1):
    ZF_BASIS[i][0] = 1
    ZF_BASIS[i][i + 1] = -1
ZF_BASIS = gs(ZF_BASIS)


def project(x, basis):
    """
    Assumes basis is orthonormal
    """
    y = np.zeros_like(x)
    for i in range(basis.shape[0]):
        y += np.dot(x, basis[i]) * basis[i]
    return y


def impose_control_conditions(controls):
    # Project onto zero net flux constraint manifold.
    controls_ = np.zeros_like(controls)
    for control_index in range(controls.shape[1]):
        controls_[:, control_index] = project(controls[:, control_index], ZF_BASIS)
    controls = controls_

    # Impose zero at boundaries.
    controls[0, :] = 0
    controls[-1, :] = 0

    return controls


# Define output.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spincnstr"
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
SAVE_FILE = EXPERIMENT_NAME
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES_GRAPE = False
SAVE_EVOL = False
SAVE_INTERMEDIATE_STATES_EVOL = False

GRAB_CONTROLS = False
GEN_CONTROLS = False
if GRAB_CONTROLS:
    controls_file_path = os.path.join(SAVE_PATH, "00005_{}.h5".format(EXPERIMENT_NAME))
    controls_lock_file_path = "{}.lock".format(controls_file_path)
    try:
        with FileLock(controls_lock_file_path):
            with h5py.File(controls_file_path) as f:
                index = np.argmin(f["error"][()])
                controls_ = f["controls"][index][()]
    except Timeout:
        print("Timeout on {}."
              "".format(controls_lock_file_path))
elif GEN_CONTROLS:
    controls_ = np.ones((CONTROL_EVAL_COUNT, CONTROL_COUNT))
    mid_index = int(np.floor(CONTROL_EVAL_COUNT / 2))
    controls_[:mid_index] = -controls_[:mid_index]
    controls_ = controls_ * (3 * MAX_CONTROL_NORMS / 4)
else:
    controls_ = None
INITIAL_CONTROLS = controls_


GRAPE_CONFIG = {
    "complex_controls": COMPLEX_CONTROLS,
    "control_count": CONTROL_COUNT,
    "control_eval_count": CONTROL_EVAL_COUNT,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "hamiltonian_args": HAMILTONIAN_ARGS,
    "impose_control_conditions": impose_control_conditions,
    "initial_states": INITIAL_STATES,
    "initial_controls": INITIAL_CONTROLS,
    "iteration_count": ITERATION_COUNT,
    "log_iteration_step": LOG_ITERATION_STEP,
    "max_control_norms": MAX_CONTROL_NORMS,
    "optimizer": OPTIMIZER,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES_GRAPE,
    "save_iteration_step": SAVE_ITERATION_STEP,
    "system_eval_count": SYSTEM_EVAL_COUNT,
}

EVOL_CONFIG = {
    "controls": INITIAL_CONTROLS,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "hamiltonian_args": HAMILTONIAN_ARGS,
    "initial_states": INITIAL_STATES,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES_EVOL,
    "system_eval_count": SYSTEM_EVAL_COUNT,
}


def do_grape():
    save_file_path = generate_save_file_path(SAVE_FILE, SAVE_PATH)
    config = copy(GRAPE_CONFIG)
    config.update({
        "save_file_path": save_file_path
    })
    result = grape_schroedinger_discrete(**config)


def do_evol():
    config = copy(EVOL_CONFIG)
    if SAVE_EVOL:
        save_file_path = generate_save_file_path(SAVE_FILE, SAVE_PATH)
        config.update({
                "save_file_path": save_file_path
        })
    result = evolve_schroedinger_discrete(**config)
    print("e: {}\ns:\n{}"
          "".format(result.error, result.final_states))


def main():
    parser = ArgumentParser()
    parser.add_argument("--grape", dest="grape", action="store_true", default=False)
    parser.add_argument("--evol", dest="evol", action="store_true", default=False)
    args = vars(parser.parse_args())
    run_grape = args["grape"]
    run_evol = args["evol"]
    
    if run_grape:
        do_grape()
    elif run_evol:
        do_evol()


if __name__ == "__main__":
    main()
