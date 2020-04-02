"""
spincnstr2.py - constrained robust optimization of the spin system
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
EXPERIMENT_NAME = "spincnstr2"
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
H_SYSTEM_0 = SIGMA_Z / 2
H_CONTROL_0 = SIGMA_X / 2
hamiltonian = lambda controls, hargs, time: (
    hargs[0] * H_SYSTEM_0
    + controls[0] * H_CONTROL_0
)
HAMILTONIAN_ARGS = np.array([OMEGA])
MAX_CONTROL_NORMS = np.array([MAX_AMP_0])

# Define the optimization.
OPTIMIZER = Adam()
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 120
CONTROL_EVAL_COUNT = 2 * int(EVOLUTION_TIME) + 1
CONTROL_EVAL_TIMES = np.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT)
ITERATION_COUNT = 100

# Define the problem.
INITIAL_STATE_0 = np.array([[1], [0]], dtype=np.float64)
TARGET_STATE_0 = np.array([[0], [1]], dtype=np.float64)
INITIAL_STATES = np.stack((INITIAL_STATE_0,),)
TARGET_STATES = np.stack((TARGET_STATE_0,),)
TARGET_STATES_DAGGER = conjugate_transpose(TARGET_STATES)
FIDELITY_CONSTRAINT = 1e-3
FIDELITY_MULTIPLIER = 5
FIDELITY_MULTIPLIER_SCALE = 5
INITIAL_FIDELITY_ROBUSTNESS = 0

INITIAL_ASTATE = np.hstack([
    INITIAL_STATES.ravel(),
],)
states_count = INITIAL_STATES.shape[0]
states_offset = 0
states_shape = INITIAL_STATES.shape
states_size = np.prod(states_shape)
get_states = lambda astate: anp.reshape(astate[states_offset:states_offset + states_size], states_shape)

# Define the problem.
class Fidelity(Cost):
    """
    A cost function to promote fidelity.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def cost(self, controls, final_astate):
        final_states = get_states(final_astate)
        inner_products = anp.matmul(TARGET_STATES_DAGGER, final_states)[:, 0, 0]        
        fidelities = anp.real(inner_products * anp.conjugate(inner_products))
        fidelity_normalized = anp.sum(fidelities) / states_count
        cost_ = 1 - fidelity_normalized

        augmented_cost = self.augment_cost(cost_)
        return augmented_cost

    
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


def rhs(astate, controls, time):
    delta_astate = anp.zeros_like(astate)
    # controls_ = interpolate_linear_set(time, CONTROL_EVAL_TIMES, controls)
    controls_ = controls[0]
    hamiltonian_ = hamiltonian(controls_, HAMILTONIAN_ARGS, time)
    states = get_states(astate)
    delta_states = -1j * anp.matmul(hamiltonian_, states)
    # # dpsi_dw
    # delta_dpsi_dw = -1j * anp.matmul(H_SYSTEM_0, astate[0]) -1j * anp.matmul(hamiltonian_, astate[1])
    # # d2psi_dw2
    # delta_d2psi_d21 = -2j * anp.matmul(H_SYSTEM_0, astate[1]) -1j * anp.matmul(hamiltonian_, astate[2])
    # # d2f_dw2
    # delta_d2f_dw2 = anp.sum(
    #     -1j * anp.matmul(TARGET_STATES_DAGGER,
    #                      (2 * anp.matmul(H_SYSTEM_0, astate[1])
    #                       + anp.matmul(hamiltonian_, astate[2]))
    #     )[:, 0, 0])

    delta_astate = anp.hstack([
        delta_states.ravel(),
    ])
    return delta_astate


# Define controls update procedure.
# Generate orthonormal basis for zero net flux constraint manifold.
ZF_BASIS = np.zeros((CONTROL_EVAL_COUNT - 1, CONTROL_EVAL_COUNT))
for i in range(CONTROL_EVAL_COUNT - 1):
    ZF_BASIS[i][0] = 1
    ZF_BASIS[i][i + 1] = -1
ZF_BASIS = gram_schmidt(ZF_BASIS)


def impose_control_conditions(controls):
    # Impose zero at boundaries.
    # controls[0, :] = 0
    # controls[-1, :] = 0
    
    # Project onto zero net flux constraint manifold.
    # controls_ = anp.zeros_like(controls)
    # for control_index in range(controls.shape[1]):
    #     controls_[:, control_index] = project(controls[:, control_index], ZF_BASIS)
    # controls = controls_

    # Impose control norm constraints.
    # clip_control_norms(controls, MAX_CONTROL_NORMS)

    return controls


# Choose initial controls.
GRAB_CONTROLS = False
GEN_CONTROLS = True
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


# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1


LQR_CONFIG = {
    "complex_controls": COMPLEX_CONTROLS,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "impose_control_conditions": impose_control_conditions,
    "initial_astate": INITIAL_ASTATE,
    "initial_controls": INITIAL_CONTROLS,
    "iteration_count": ITERATION_COUNT,
    "rhs": rhs,
}


def do_lqr():
    config = copy(LQR_CONFIG)
    result = lqr(**config)
    final_states = get_states(result.final_astate)
    while isinstance(final_states, Box):
        final_states = final_states._value
    c = np.matmul(TARGET_STATES_DAGGER, final_states)[0, 0, 0]
    print("final_states:\n{}\nc:\n{}"
          "".format(final_states, c))

    
def do_qutip():
    h0_qutip = Qobj(OMEGA * H_SYSTEM_0)
    c0_qutip = np.ones_like(INITIAL_CONTROLS[:, 0])
    h1_qutip = Qobj(H_CONTROL_0)
    c1_qutip = INITIAL_CONTROLS[:, 0]
    h_list = [
        [h0_qutip, c0_qutip],
        [h1_qutip, c1_qutip],
    ]
    initial_state_qutip = Qobj(INITIAL_STATE_0)
    tlist = np.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT)
    c_op_list = []
    e_ops = []
    result_qutip = mesolve(h_list, initial_state_qutip, tlist, c_op_list, e_ops)
    final_state_qutip = result_qutip.states[-1].full()
    ip = np.matmul(conjugate_transpose(final_state_qutip), final_state_qutip)
    print("final_state_qutip:\n{}\nip:\n{}"
          "".format(final_state_qutip, ip))


def do_evolve():
    controls = INITIAL_CONTROLS
    pc = np.polyfit(CONTROL_EVAL_TIMES, controls, 10)
    initial_state = INITIAL_STATE_0
    hargs = HAMILTONIAN_ARGS
    initial_time = 0
    final_times = np.array([EVOLUTION_TIME])
    
def rhs_(time, state):
        
        # controls_ = interpolate_linear_set(time, CONTROL_EVAL_TIMES, controls)
        # hamiltonian_ = hamiltonian(controls_, hargs, time)
        # return -1j * np.matmul(hamiltonian_, state)
        return -1j * np.matmul(OMEGA * H_SYSTEM_0, state)
    result = integrate_rkdp5(rhs_, final_times, initial_time, initial_state)
    final_state = result
    ip = np.matmul(conjugate_transpose(final_state), final_state)[0, 0]
    print("final_state:\n{}\nip:\n{}"
          "".format(final_state, ip))
    


def main():
    parser = ArgumentParser()
    parser.add_argument("--evolve", dest="evolve", action="store_true", default=False)
    parser.add_argument("--lqr", dest="lqr", action="store_true", default=False)
    parser.add_argument("--qutip", dest="qutip", action="store_true", default=False)
    args = vars(parser.parse_args())
    run_evolve = args["evolve"]
    run_lqr = args["lqr"]
    run_qutip = args["qutip"]

    if run_evolve:
        do_evolve()
    if run_lqr:
        do_lqr()
    if run_qutip:
        do_qutip()


if __name__ == "__main__":
    main()
