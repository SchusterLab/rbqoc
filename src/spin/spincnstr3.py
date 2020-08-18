"""
spincnstr3.py - benchmark against julia
"""

from argparse import ArgumentParser
from copy import copy
import os

from autograd.extend import Box
import autograd.numpy as anp
import jax.numpy as jnp
from filelock import FileLock, Timeout
import h5py
import numpy as np
from qoc import (
    grape_schroedinger_discrete,
    evolve_schroedinger_discrete,
    lqr,
)
from qoc.core.common import(
    clip_control_norms,
)
from qoc.standard import (
    TargetStateInfidelity,
    conjugate_transpose,
    gram_schmidt,
    matmuls,
    project,
    get_annihilation_operator,
    get_creation_operator,
    SIGMA_Z, SIGMA_X,
    generate_save_file_path,
    SGD, Adam,
)
from qoc.models import Cost
from qoc.core.mathmethods import interpolate_linear_set, integrate_rkdp5
from qutip import essolve, mesolve, Qobj


# Define paths.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spincnstr3"
WDIR_PATH = os.environ["ROBUST_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR_PATH, "out", EXPERIMENT_META, EXPERIMENT_NAME)
SAVE_FILE = EXPERIMENT_NAME


# Define compute resources.
CORE_COUNT = 8
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)


# Define experimental constants.
MAX_AMP_0 = 2 * np.pi * 3e-1
OMEGA = 2 * np.pi * 1e-2


# Define the system.
HILBERT_SIZE = 2
H_S = SIGMA_Z / 2
H_C_0 = SIGMA_X / 2

# Define the optimization.
EVOLUTION_TIME = 120

# Define the problem.
INITIAL_STATE = np.array([[1], [0]], dtype=np.float64)
TARGET_STATE = np.array([[0], [1]], dtype=np.float64)
TARGET_STATE_DAGGER = conjugate_transpose(TARGET_STATE)

INITIAL_ASTATE = np.hstack([
    INITIAL_STATE.ravel(),
    # np.zeros_like(INITIAL_STATE).ravel(),
],)
state_offset = 0
state_shape = INITIAL_STATE.shape
state_size = np.prod(state_shape)
get_state = lambda astate: anp.reshape(astate[state_offset:state_offset + state_size], state_shape)

# Define the problem.
class Fidelity(Cost):
    """
    A cost function to promote fidelity.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def cost(self, controls, final_astate):
        final_state = get_state(final_astate)
        inner_product = anp.matmul(TARGET_STATE_DAGGER, final_state)[0, 0]        
        fidelity = anp.real(inner_product * anp.conjugate(inner_product))
        print("fidelity:\n{}".format(fidelity))
        cost_ = 1 - fidelity
        return cost_

    
class FidelityRobustness(Cost):
    """
    A cost function to promote robustness of fidelity.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def cost(self, controls, final_astate):
        cost_ = anp.abs(final_astate[3][0][1][0])
        augmented_cost = self.augment_cost(cost_)
        return augmented_cost

    
COSTS = [
    Fidelity(),
    # FidelityRobustness(),
]

def interpolate_controls(controls, time):
    return controls[0]

def rhs(astate, controls, time):
    delta_astate = anp.zeros_like(astate)
    controls_ = interpolate_controls(controls, time)
    hamiltonian_ = OMEGA * H_S + controls[0] * H_C_0
    state = get_state(astate)
    delta_state = -1j * anp.matmul(hamiltonian_, state)
    # # dpsi_dw
    # delta_dpsi_dw = -1j * anp.matmul(H_S, astate[0]) -1j * anp.matmul(hamiltonian_, astate[1])
    # # d2psi_dw2
    # delta_d2psi_d21 = -2j * anp.matmul(H_S, astate[1]) -1j * anp.matmul(hamiltonian_, astate[2])
    # # d2f_dw2
    # delta_d2f_dw2 = anp.sum(
    #     -1j * anp.matmul(TARGET_STATES_DAGGER,
    #                      (2 * anp.matmul(H_S, astate[1])
    #                       + anp.matmul(hamiltonian_, astate[2]))
    #     )[:, 0, 0])

    delta_astate = anp.hstack([
        delta_state.ravel(),
    ])
    return delta_astate


INITIAL_CONTROLS = anp.array([1.])
COMPLEX_CONTROLS = False
ITERATION_COUNT = 1

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1


LQR_CONFIG = {
    "complex_controls": COMPLEX_CONTROLS,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "initial_astate": INITIAL_ASTATE,
    "initial_controls": INITIAL_CONTROLS,
    "iteration_count": ITERATION_COUNT,
    "rhs": rhs,
}


def do_lqr():
    config = copy(LQR_CONFIG)
    result = lqr(**config)
    final_state = get_state(result.final_astate)
    grads = result.grads
    cost = result.cost
    while isinstance(final_state, Box):
        final_state = final_state._value
    print("cost:\n{}\nfinal_state:\n{}\ngrads:\n{}"
          "".format(cost, final_state, grads))

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--lqr", dest="lqr", action="store_true", default=False)
    args = vars(parser.parse_args())
    run_lqr = args["lqr"]

    if run_lqr:
        do_lqr()


if __name__ == "__main__":
    main()
