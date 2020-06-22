"""
t1.py - Some t1 things in python.

FBFQ = Flux by Flux Quantum
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scqubits as qubit
from scqubits import HilbertSpace, InteractionTerm
from scqubits.utils.spectrum_utils import matrix_element

EXPERIMENT_META = "spin"
WDIR = os.environ["ROBUST_QOC_PATH"]
OUT_PATH = os.path.join(WDIR, "out", EXPERIMENT_META)
FBFQ_VS_T1_POLY_SAVE_FILE_PATH = os.path.join(OUT_PATH, "fluxvst1poly.png")
AMP_VS_FBFQ_POLY_SAVE_FILE_PATH = os.path.join(OUT_PATH, "ampvsfluxpoly.png")

# Define experimental constants.
T1_ARRAY = 10 ** (-6) * np.array([
    1597.923, 1627.93, 301.86, 269.03, 476.33, 1783.19, 2131.76, 2634.50, 
    4364.68, 2587.82, 1661.915, 1794.468, 2173.88, 1188.83, 
    1576.493, 965.183, 560.251, 310.88
])
FBFQ_ARRAY = np.array([
    0.26, 0.28, 0.32, 0.34, 0.36, 0.38, 0.4,
    0.42, 0.44, 0.46, 0.465, 0.47, 0.475,
    0.48, 0.484, 0.488, 0.492, 0.5
])
FBFQ_VS_T1_POLY_TEST_COUNT = 100
FBFQ_VS_T1_POLY_LINSPACE = np.linspace(FBFQ_ARRAY[0], FBFQ_ARRAY[-1],
                            FBFQ_VS_T1_POLY_TEST_COUNT)
FBFQ_VS_T1_POLY_DEGREE = 7

AMP_VS_FBFQ_POLY_SAMPLE_COUNT = int(1e3)
AMP_VS_FBFQ_POLY_LINSPACE = np.linspace(0.25, 0.5, AMP_VS_FBFQ_POLY_SAMPLE_COUNT)
AMP_VS_FBFQ_POLY_DEG = 10

def get_fbfq_vs_t1_poly():
    coeffs = np.polyfit(
        FBFQ_ARRAY, T1_ARRAY,
        FBFQ_VS_T1_POLY_DEGREE
    )
    poly = np.poly1d(coeffs)
    fig = plt.figure()
    plt.scatter(FBFQ_ARRAY, T1_ARRAY, color="blue")
    plt.plot(FBFQ_VS_T1_POLY_LINSPACE, poly(FBFQ_VS_T1_POLY_LINSPACE), color="red")
    plt.savefig(FBFQ_VS_T1_POLY_SAVE_FILE_PATH)
    return coeffs
#ENDDEF


def get_amp_coeff():
    resonator = qubit.Oscillator(
        E_osc = 5.7286,
        truncated_dim=20
    )
    
    fluxonium = qubit.Fluxonium(
        EJ = 3.395,
        EL = 0.132,
        EC = 0.479,
        flux = 0.5,
        cutoff = 110,
        truncated_dim = 60
    )
    
    hilbertspc = HilbertSpace([
        fluxonium,
        resonator
    ])
    
    adag = resonator.creation_operator()
    a = resonator.annihilation_operator()
    int_term = InteractionTerm(
        g_strength = 0.076, 
        subsys1 = fluxonium, 
        op1 = fluxonium.n_operator(), 
        subsys2 = resonator, 
        op2 = a + adag
    )
    interaction_list = [int_term]
    hilbertspc.interaction_list = interaction_list

    dressed_hamiltonian = hilbertspc.get_hamiltonian()    
    evals, evecs = dressed_hamiltonian.eigenstates(eigvals=10)

    g_phi_e = matrix_element(evecs[0],
                             hilbertspc.identity_wrap(fluxonium.phi_operator(), fluxonium),
                             evecs[1]
    )

    return g_phi_e


def get_amp_vs_fbfq_poly():
    fbfqs = AMP_VS_FBFQ_POLY_LINSPACE
    amps = get_amp(fbfqs)
    coeffs = np.polyfit(amps, fbfqs, AMP_VS_FBFQ_POLY_DEG)
    p = np.poly1d(coeffs)
    fig = plt.figure()
    plt.scatter(amps, fbfqs, color="red", label="data")
    plt.plot(p(fbfqs), fbfqs, color="blue", label="poly")
    plt.legend()
    plt.savefig(AMP_VS_FBFQ_POLY_SAVE_FILE_PATH)
    return coeffs
    
#ENDDEF


def main():
    # coeff0 = get_amp_coeff()
    # print(np.abs(coeff0))

    coeffs1 = get_fbfq_vs_t1_poly()
    coeffs1_reversed = coeffs1[::-1]
    print(coeffs1_reversed)


if __name__ == "__main__":
    main()
