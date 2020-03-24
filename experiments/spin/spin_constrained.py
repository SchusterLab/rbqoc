"""
spin_constrained.py
"""

from argparse import ArgumentParser
from copy import copy

import autograd.numpy as anp
import numpy as np
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z, SIGMA_X,
                          generate_save_file_path,)

MAX_AMP_0 = 2 * anp.pi * 3e-1
OMEGA = 2 * anp.pi * 1e-2

# Define the system.
HILBERT_SIZE = 2
H_SYSTEM_0 = OMEGA * SIGMA_Z / 2
H_CONTROL_0 = SIGMA_X / 2
hamiltonian = lambda controls, hamiltonian_args, time: (
    H_SYSTEM_0
    + controls[0] * H_CONTROL_0
)
MAX_CONTROL_NORMS = anp.array([MAX_AMP_0])

# Define the problem.
INITIAL_STATE_0 = anp.array([[1], [0]])
TARGET_STATE_0 = anp.array([[0], [1]])
INITIAL_STATES = anp.stack((INITIAL_STATE_0,),)
TARGET_STATES = anp.stack((TARGET_STATE_0,),)
COSTS = [
    TargetStateInfidelity(TARGET_STATES)
]

# Define the optimization.
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 100 # nanoseconds
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = 2 * int(EVOLUTION_TIME) + 1
ITERATION_COUNT = 1000

def gs(X, row_vecs=True, norm=True):
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
    controls_ = np.zeros_like(controls)
    
    # Project onto zero net flux constraint manifold.
    for control_index in range(controls.shape[1]):
        controls_[:, control_index] = project(controls[:, control_index], ZF_BASIS)

    # TODO: Project onto maximum amplitude constraint manifold.
    
    return controls_

# Define output.
SAVE_PATH = "./out"
SAVE_FILE_NAME = "spin"
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES_GRAPE = True

GRAPE_CONFIG = {
    "complex_controls": COMPLEX_CONTROLS,
    "control_count": CONTROL_COUNT,
    "control_eval_count": CONTROL_EVAL_COUNT,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "impose_control_conditions": impose_control_conditions,
    "initial_states": INITIAL_STATES,
    "iteration_count": ITERATION_COUNT,
    "log_iteration_step": LOG_ITERATION_STEP,
    "max_control_norms": MAX_CONTROL_NORMS,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES_GRAPE,
    "save_iteration_step": SAVE_ITERATION_STEP,
    "system_eval_count": SYSTEM_EVAL_COUNT,
}

def do_grape():
    save_file_path = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)
    config = copy(GRAPE_CONFIG)
    config.update({
        "save_file_path": save_file_path
    })
    result = grape_schroedinger_discrete(**config)



def main():
    parser = ArgumentParser()
    parser.add_argument("-g", dest="grape", action="store_true", default=False)
    args = vars(parser.parse_args())
    run_grape = args["grape"]
    
    if run_grape:
        do_grape()


if __name__ == "__main__":
    main()
