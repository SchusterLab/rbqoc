"""
verify.py - Quitp verification.
"""

from argparse import ArgumentParser
from functools import reduce
import os

import h5py
import numpy as np
from qoc.standard import conjugate_transpose, matmuls
from qutip import (
    mesolve, Qobj,
)

# paths
WDIR = os.environ.get("ROBUST_QOC_PATH", ".")
OUT_PATH = os.path.join(WDIR, "out")


# types
class SaveType(Enum):
    QOCJL = 1
    QOCPY = 2
    SAMPLEJL = 3
#ENDCLASS


# defs
def fidelity_vec(v1, v2):
    ip = np.matmul(conjugate_transpose(v1), v2)[0, 0]
    return np.real(ip) ** 2 + np.imag(ip) ** 2
#ENDDEF


def fidelity_mat(m1, m2):
    return (
        np.abs(np.trace(np.matmul(conjugate_transpose(m1), m2)))
        / np.abs(np.trace(np.matmul(conjugate_transpose(m2), m2)))
    )
#ENDDEF


def gen_initial_state(seed):
    np.random.rand(0)
    rands = np.random.rand(4)
    state = np.array([[rands[0] + 1j * rands[1],],
                      [rands[2] + 1j * rands[3],]])
    state = state / np.sqrt(np.abs(np.matmul(conjugate_transpose(state), state)[0, 0]))
    return state
#ENDDEF


def grab_controls(controls_file_path, save_type=SaveType.QOCJL):
    with h5py.File(controls_file_path, "r") as save_file:
        if save_type == SaveType.QOCJL:
            x = 0
        elif save_type == SaveType.SAMPLEJL:
            x = 0
        elif save_type == SaveType.QOCPY:
            x = 0
        #ENDIF
        controls = save_file["controls"][()]
    #ENDWITH

    return (controls, evolution_time)
#ENDDEF


# computational constants
OMEGA = 2 * np.pi * 1.4e-2
INV_ROOT_2 = 2 ** (-1/2)
DT = 1e-2
DT_INV = 1e2

SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2
XPIBY2 = np.array([[1.,  -1j],
                   [-1.j, 1.]]) * INV_ROOT_2
YPIBY2 = np.array([[1., -1.],
                   [1.,  1.]]) * INV_ROOT_2
NEG_YPIBY2 = np.array([[1., 1.],
                       [-1., 1.]]) * INV_ROOT_2
ZPIBY2 = np.array([[1. - 1.j, 0],
                   [0,        1. + 1.j]]) * INV_ROOT_2
INITIAL_STATE = gen_initial_state(2)
TARGET_STATE = np.matmul(YPIBY2, INITIAL_STATE)
INITIAL_DENSITY = np.matmul(INITIAL_STATE, conjugate_transpose(INITIAL_STATE))
TARGET_DENSITY = np.matmul(TARGET_STATE, conjugate_transpose(TARGET_STATE))

# other constants
CIDX_PY = -1


def run_spin(experiment_name, controls_file_name, controls_idx):
    # get controls
    experiment_meta = "spin"
    save_path = os.path.join(OUT_PATH, experiment_meta, experiment_name)
    controls_file_path = os.path.join(save_path, controls_file_name)
    with h5py.File(controls_file_path, "r") as save_file:
        if controls_idx == CIDX_PY:
            controls = save_file["controls"][()]
        else:
            states = save_file["states"][()]
            controls = states[controls_idx, 0:-1]
        evolution_time = save_file["evolution_time"][()]
    #ENDWITH
    control_eval_count = controls.shape[0]
    
    # define constants
    omega = OMEGA

    # build simulation
    h_sys = Qobj(omega * H_S)
    h_c1 = Qobj(H_C1)
    hlist = [
        [h_sys, np.ones(control_eval_count)],
        [h_c1, controls]
    ]
    rho0 = Qobj(INITIAL_DENSITY)
    tlist = np.arange(0, control_eval_count, 1) * DT

    # run simulation
    result = mesolve(hlist, rho0, tlist)

    # analysis
    final_state = result.states[-1].full()
    # fidelity_ = fidelity_vec(final_state, TARGET_STATE)
    fidelity_ = fidelity_mat (final_state, TARGET_DENSITY)

    # log
    # print("fidelity:\n{}\nfinal_state:\n{}\ntarget_state:\n{}"
    #       "".format(fidelity_, final_state, TARGET_STATE))
    print("fidelity:\n{}\nfinal_density:\n{}\ntarget_density:\n{}\ninitial_density:\n{}"
          "".format(fidelity_, final_state, TARGET_DENSITY, INITIAL_DENSITY))
#ENDDEF

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--spin", action="store_true")
    parser.add_argument("--ename", action="store", type=str)
    parser.add_argument("--cname", action="store", type=str)
    parser.add_argument("--cidx", action="store", type=int)
    parser.add_argument("--cidx", action="store", type=str)
    args = vars(parser.parse_args())
    do_spin = args["spin"]
    experiment_name = args["ename"]
    controls_file_name = args["cname"]
    controls_idx = args["cidx"]
    
    if do_spin:
        run_spin(experiment_name, controls_file_name, controls_idx)


if __name__ == "__main__":
    main()
