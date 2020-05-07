"""
verify.py - Quitp verification.
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
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
OUT_PATH = os.path.join(WDIR, "out")

# Computational constants.
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2


def fidelity(v1, v2):
    ip = np.matmul(conjugate_transpose(v1), v2)[0, 0]
    return np.real(ip) ** 2 + np.imag(ip) ** 2


def run_spin():
    # get controls
    experiment_meta = "spin"
    experiment_name = "spin12"
    save_path = os.path.join(OUT_PATH, experiment_meta, experiment_name)
    controls_file_path = os.path.join(save_path, "00000_spin12.h5")
    with h5py.File(controls_file_path, "r") as save_file:
        # controls = save_file["controls"][0, :]
        states = save_file["states"][()]
        controls = states[13, 0:-1]
        # print(save_file["Q"][()])
    #ENDWITH
    control_eval_count = controls.shape[0]
    
    # define constants
    evolution_time = 120.
    omega_raw = 2 * np.pi * 1e-2
    domega = omega_raw * 5e-2
    omega = (
        omega_raw
        + domega
    )
    initial_state = np.array([[1], [0]])
    target_state = np.array([[0], [1]])

    # build simulation
    h_sys = Qobj(omega * H_S)
    h_c1 = Qobj(H_C1)
    hlist = [
        [h_sys, np.ones(control_eval_count)],
        [h_c1, controls]
    ]
    rho0 = Qobj(initial_state)
    tlist = np.linspace(0, evolution_time, control_eval_count)

    # run simulation
    result = mesolve(hlist, rho0, tlist)

    # analysis
    final_state = result.states[-1].full()
    fidelity_ = fidelity(final_state, target_state)

    # log
    print("fidelity:\n{}\nfinal_state:\n{}"
          "".format(fidelity_, final_state))
#ENDDEF

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--spin", action="store_true")
    args = vars(parser.parse_args())
    do_spin = args["spin"]
    
    if do_spin:
        run_spin()


if __name__ == "__main__":
    main()
