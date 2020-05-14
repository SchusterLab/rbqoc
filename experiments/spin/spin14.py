"""
spin14.py - recreate the pulses from the paper

Refs:
[0] https://arxiv.org/abs/2002.10653
"""

from argparse import ArgumentParser
import os

import h5py
import numpy as np
from qoc.standard import conjugate_transpose
from qutip import (
    mesolve, Qobj,
)

# Directory.
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin14"
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
SAVE_PATH = os.path.join(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Computational constants.
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])
MAX_CONTROL_0 = 2 * np.pi * 3e-1
OMEGA = 2 * np.pi * 1.4e-2
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2


def fidelity(v1, v2):
    ip = np.matmul(conjugate_transpose(v1), v2)[0, 0]
    return np.real(ip) ** 2 + np.imag(ip) ** 2


def main(evol_time):
    pass


if __name__ == "__main__":
    main()

