"""
twospin.jl - common definitions for the twospin directory derived from spin.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "rbqoc.jl"))

using Dierckx
using DifferentialEquations
using Distributions
using FFTW
using Random
using StaticArrays
using Statistics
using Zygote

# types
@enum GateType begin
    cnot = 1
    iswap = 2
    sqrtiswap = 3
end

const HDIM_TWOSPIN = 4
const HDIM_TWOSPIN_ISO = HDIM_TWOSPIN * 2
const HDIM_TWOSPIN_VISO = HDIM_TWOSPIN^2
const HDIM_TWOSPIN_VISO_ISO = HDIM_TWOSPIN_VISO * 2

# initial states
TWOSPIN_ISO_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

const SIGMAX = [0 1;
                1 0]
const SIGMAY = [0   -1im;
                1im 0]
const SIGMAZ = [1 0;
                0 -1]
const PAULI_IDENTITY = [1 0;
                        0 1]

# qubit frequencies in MHz
const FQ_1 = 0.047
const FQ_2 = 0.060

const XX = kron(SIGMAX, SIGMAX)
const IZ = kron(PAULI_IDENTITY, SIGMAZ)
const ZI = kron(SIGMAZ, PAULI_IDENTITY)

const ZI_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(ZI))
const IZ_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(IZ))
const XX_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(XX))

const CNOT = [1 0 0 0;
              0 1 0 0;
              0 0 0 1;
              0 0 1 0]
const sqrt2 = sqrt(2.0)
const sqrtiSWAP = [1    0          0       0;
                   0    1/sqrt2 -1im/sqrt2 0;
                   0 -1im/sqrt2    1/sqrt2 0;
                   0    0          0       1]
const iSWAP = [1    0    0 0;
               0    0 -1im 0;
               0 -1im    0 0;
               0    0    0 1]

const CNOT_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(CNOT))
const iSWAP_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(iSWAP))
const sqrtiSWAP_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Float64}(get_mat_iso(sqrtiSWAP))

const NEGI_H0_TWOSPIN_ISO = pi * (-1im) * (FQ_1 * ZI_ISO + FQ_2 * IZ_ISO)
const NEGI_H1_TWOSPIN_ISO_3 = 2.0 * pi * (-1im) * XX_ISO

const CNOT_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,1]))
const CNOT_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,2]))
const CNOT_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,3]))
const CNOT_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,4]))

const iSWAP_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,1]))
const iSWAP_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,2]))
const iSWAP_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,3]))
const iSWAP_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,4]))

const sqrtiSWAP_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(sqrtiSWAP[:,1]))
const sqrtiSWAP_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(sqrtiSWAP[:,2]))
const sqrtiSWAP_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(sqrtiSWAP[:,3]))
const sqrtiSWAP_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(sqrtiSWAP[:,4]))

function target_states(gate_type)
    if gate_type == cnot
        target_state1 = Array(CNOT_ISO_1)
        target_state2 = Array(CNOT_ISO_2)
        target_state3 = Array(CNOT_ISO_3)
        target_state4 = Array(CNOT_ISO_4)
    elseif gate_type == iswap
        target_state1 = Array(iSWAP_ISO_1)
        target_state2 = Array(iSWAP_ISO_2)
        target_state3 = Array(iSWAP_ISO_3)
        target_state4 = Array(iSWAP_ISO_4)
    elseif gate_type == sqrtiswap
        target_state1 = Array(sqrtiSWAP_ISO_1)
        target_state2 = Array(sqrtiSWAP_ISO_2)
        target_state3 = Array(sqrtiSWAP_ISO_3)
        target_state4 = Array(sqrtiSWAP_ISO_4)
    end
    return (target_state1, target_state2, target_state3, target_state4)
end
