"""
twospin.jl - common definitions for the twospin directory derived from spin.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

# types
@enum GateType begin
    zzpiby2 = 1
    yypiby2 = 2
    xxpiby2 = 3
end

# const GT_STR = Dict(
#     zzpiby2 => "Z/2 ⊗ Z/2",
#     yypiby2 => "Y/2 ⊗ Y/2",
#     xxpiby2 => "X/2 ⊗ X/2",
# )

const HDIM_TWOSPIN = 4
const HDIM_TWOSPIN_ISO = HDIM_TWOSPIN * 2
const HDIM_TWOSPIN_VISO = HDIM_TWOSPIN^2
const HDIM_TWOSPIN_VISO_ISO = HDIM_TWOSPIN_VISO * 2

TWOSPIN_ISO_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TWOSPIN_ISO_4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

const PAULI_IDENTITY = [1 0;
                        0 1]

# Operators for two coupled spins

const FQ_1 = 0.071
const FQ_2 = 0.080

const XX = kron(SIGMAX, SIGMAX)
const ZZ = kron(SIGMAZ, SIGMAZ)
const IZ = kron(PAULI_IDENTITY, SIGMAZ)
const ZI = kron(SIGMAZ, PAULI_IDENTITY)
const IX = kron(PAULI_IDENTITY, SIGMAX)
const XI = kron(SIGMAX, PAULI_IDENTITY)

const XX_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(XX))
const ZZ_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(ZZ))
const IZ_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(IZ))
const ZI_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(ZI))
const IX_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(IX))
const XI_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(XI))

const NEGI_TWOSPIN = SA_F64[0   0   0   0  1  0  0  0;
                            0   0   0   0  0  1  0  0;
                            0   0   0   0  0  0  1  0;
                            0   0   0   0  0  0  0  1;
                            -1  0   0   0  0  0  0  0;
                            0  -1   0   0  0  0  0  0;
                            0   0  -1   0  0  0  0  0;
                            0   0   0  -1  0  0  0  0]

const FQ_1_ZI_ISO = FQ_1 * ZI_ISO
const FQ_2_IZ_ISO = FQ_2 * IZ_ISO
const NEGI_H0_TWOSPIN_ISO = -pi * NEGI_TWOSPIN * (FQ_1_ZI_ISO + FQ_2_IZ_ISO)
const NEGI_H1_TWOSPIN_ISO_1 = pi * NEGI_TWOSPIN * XI_ISO
const NEGI_H1_TWOSPIN_ISO_2 = pi * NEGI_TWOSPIN * IX_ISO
const NEGI_H1_TWOSPIN_ISO_3 = pi * NEGI_TWOSPIN * XX_ISO

# two qubit gates

ZZPIBY2_ = kron(ZPIBY2_, ZPIBY2_)
XXPIBY2_ = kron(XPIBY2_, XPIBY2_)
YYPIBY2_ = kron(YPIBY2_, YPIBY2_)
const ZZPIBY2_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO}(get_mat_iso(ZZPIBY2_))
const XXPIBY2_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO}(get_mat_iso(XXPIBY2_))
const YYPIBY2_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO}(get_mat_iso(YYPIBY2_))

const ZZPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZZPIBY2_[:,1]))
const ZZPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZZPIBY2_[:,2]))
const ZZPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZZPIBY2_[:,3]))
const ZZPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZZPIBY2_[:,4]))

const XXPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XXPIBY2_[:,1]))
const XXPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XXPIBY2_[:,2]))
const XXPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XXPIBY2_[:,3]))
const XXPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XXPIBY2_[:,4]))

const YYPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YYPIBY2_[:,1]))
const YYPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YYPIBY2_[:,2]))
const YYPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YYPIBY2_[:,3]))
const YYPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YYPIBY2_[:,4]))
