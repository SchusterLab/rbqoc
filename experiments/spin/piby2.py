"""
piby2.py - This module searches for pi/2 pulses
for the spin system.
"""

import os

import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (conjugate_transpose,
                          generate_save_file_path,
                          SIGMA_X, SIGMA_Z,
                          TargetStateInfidelity,
                          ControlArea,
                          ControlBandwidth,
                          LBFGSB,
                          Adam,)

CORE_COUNT = 8
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define experimental constants.
CONTROL_SAMPLING_RATE = 1.2 # GS/s
MAX_AMP_CONTROL_0 = 1e-1 # GHz
MAX_AMP_BANDWIDTH_CONTROL_0 = 5e-1 # GHz
OMEGA = 1e-2 # GHz


# Define the system.
SYSTEM_HAMILTONIAN_0 = OMEGA * SIGMA_Z / 2
CONTROL_HAMILTONIAN_0 = SIGMA_X / 2
hamiltonian = lambda controls, time: (SYSTEM_HAMILTONIAN_0
                                      + controls[0] * CONTROL_HAMILTONIAN_0)
MAX_CONTROL_NORMS = anp.array((MAX_AMP_CONTROL_0,))
MAX_CONTROL_BANDWIDTHS = anp.array((MAX_AMP_BANDWIDTH_CONTROL_0,))

# Define the optimization.
ITERATION_COUNT = int(1e4)
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 150 # nanoseconds
CONTROL_STEP_COUNT = int(EVOLUTION_TIME * CONTROL_SAMPLING_RATE)
OPTIMIZER = Adam()

# Define the problem.
INITIAL_STATE_0 = anp.array(((1,), (0,)))
INITIAL_STATE_1 = anp.array(((0,), (1,)))
INITIAL_STATES = anp.stack((INITIAL_STATE_0, INITIAL_STATE_1,))
# Rx(pi/2)
r2_by_2 = anp.divide(anp.sqrt(2), 2)
TARGET_STATE_0 = anp.array(((r2_by_2,), (-1j * r2_by_2,)))
TARGET_STATE_1 = anp.array(((-1j * r2_by_2,), (r2_by_2,)))
TARGET_STATES = anp.stack((TARGET_STATE_0, TARGET_STATE_1,))
COSTS = (ControlArea(CONTROL_COUNT,
                     CONTROL_STEP_COUNT,
                     MAX_CONTROL_NORMS,),
         ControlBandwidth(CONTROL_COUNT, MAX_CONTROL_BANDWIDTHS,
                          MAX_CONTROL_NORMS,),
         TargetStateInfidelity(TARGET_STATES),)

# Define the output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
EXPERIMENT_NAME = "piby2"
SAVE_PATH = os.path.join("out", EXPERIMENT_NAME)
SAVE_FILE_PATH = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)



def main():
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_STEP_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES,
                                         complex_controls=COMPLEX_CONTROLS,
                                         iteration_count=ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_control_norms=MAX_CONTROL_NORMS,
                                         optimizer=OPTIMIZER,
                                         save_iteration_step=SAVE_ITERATION_STEP,
                                         save_file_path=SAVE_FILE_PATH,)


if __name__ == "__main__":
    main()
