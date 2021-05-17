"""
twospin.jl - common definitions for the twospin directory derived from spin.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))
using CSV
using DataFrames


chi_minus_df = CSV.read(joinpath(WDIR, "src/twospin/chi_minus_values.csv"), DataFrame, header=false)
gs_minus_expect_df = CSV.read(joinpath(WDIR, "src/twospin/gs_minus_expect_values.csv"), DataFrame, header=false)
J_eff_vals_df = CSV.read(joinpath(WDIR, "src/twospin/J_eff_values.csv"), DataFrame, header=false)
x_vals_df = CSV.read(joinpath(WDIR, "src/twospin/flux_vals.csv"), DataFrame, header=false)
chi_minus_vals = convert(Matrix{Float64}, chi_minus_df)[:, 1]
gs_minus_expect_vals = convert(Matrix{Float64}, gs_minus_expect_df)[:, 1]
J_eff_vals = convert(Matrix{Float64}, J_eff_vals_df)[:, 1]
x_vals = convert(Matrix{Float64}, x_vals_df)[:, 1]

J_eff_spline_itp = extrapolate(interpolate((x_vals,), J_eff_vals,
                                Gridded(Linear())), Periodic())
J_eff_spline_dkx = Spline1D(x_vals[:, 1], J_eff_vals[:, 1], periodic=true)
chi_minus_spline_itp = extrapolate(interpolate((x_vals,), chi_minus_vals,
                                   Gridded(Linear())), Periodic())
gs_minus_spline_itp = extrapolate(interpolate((x_vals,), gs_minus_expect_vals,
                                  Gridded(Linear())), Periodic())


function J_eff_func(x)
    return (-1.03822555e+02*x^10 + 24.2426102 * x^8 + 8.81815156e-05 * x^7 + -2.89820033 * x^6
            + -2.91231770e-05 * x^5 + -1.27445653e+00 * x^4 + 2.26535175e-06 * x^3
            + -2.52489290e-01 * x^2 + -3.15538863e-08 * x + 3.28790244e-02)
end
# types
@enum GateType begin
    zzpiby2 = 1
    yypiby2 = 2
    xxpiby2 = 3
    xipiby2 = 4
    ixpiby2 = 5
    yipiby2 = 6
    iypiby2 = 7
    zipiby2 = 8
    izpiby2 = 9
    cnot = 10
    iswap = 11
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

const FQ_1 = 0.0072 #0.070
const FQ_2 = 0.0085 #0.078
const EL_a = 0.21
const EL_b = 0.21
const EL_1 = 4.0
const EL_2 = 4.0
const EL_twiddle = EL_a + EL_b + EL_1 + EL_2

const XX = kron(SIGMAX, SIGMAX)
const ZZ = kron(SIGMAZ, SIGMAZ)
const IZ = kron(PAULI_IDENTITY, SIGMAZ)
const ZI = kron(SIGMAZ, PAULI_IDENTITY)
const IX = kron(PAULI_IDENTITY, SIGMAX)
const XI = kron(SIGMAX, PAULI_IDENTITY)
const CNOT = [1 0 0 0;
              0 1 0 0;
              0 0 0 1;
              0 0 1 0]
const sqrt2 = sqrt(2.0)
# const sqrt_iSWAP = [1.     0          0      0;
#                     0    1./sqrt2 -1im/sqrt2 0;
#                     0  -1im/sqrt2   1./sqrt2 0;
#                     0      0          0      1]
const iSWAP = [1    0    0 0;
               0    0 -1im 0;
               0 -1im    0 0;
               0    0    0 1]

const XX_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(XX))
const ZZ_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(ZZ))
const IZ_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(IZ))
const ZI_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(ZI))
const IX_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(IX))
const XI_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(XI))
const CNOT_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(CNOT))
const iSWAP_ISO = SMatrix{HDIM_TWOSPIN_ISO, HDIM_TWOSPIN_ISO, Int64}(get_mat_iso(iSWAP))

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
const NEGI_H0_TWOSPIN_ISO = pi * NEGI_TWOSPIN * (FQ_1_ZI_ISO + FQ_2_IZ_ISO)
const NEGI_H1_TWOSPIN_ISO_1 = NEGI_TWOSPIN * XI_ISO
const NEGI_H1_TWOSPIN_ISO_2 = NEGI_TWOSPIN * IX_ISO
const NEGI_H1_TWOSPIN_ISO_3 = 2.0 * pi * NEGI_TWOSPIN * XX_ISO

const XX_coeff = 2.0 * pi * 2.94698960 * 2.97691877

function EL_bar(flux_c)
    return EL_a - 0.5 * EL_a^2 * chi_minus_spline_itp(flux_c) - EL_a^2 / EL_twiddle
end

# function J_eff(flux_c)
#     return EL_a^2 * (0.5 * chi_minus_spline_itp(flux_c) - (1. / EL_twiddle))
# end

function J_eff(flux_c)
    return J_eff_spline_itp(flux_c)
end

function XI_coeff(flux_a, flux_b, flux_c)
    return (2.94698960 * 2.0 * pi * (0.5 * EL_a * gs_minus_spline_itp(flux_c)
            - J_eff(flux_c) * 2.0 * pi * flux_b - EL_bar(flux_c) * 2.0 * pi * flux_a))
end

function IX_coeff(flux_a, flux_b, flux_c)
    return - (2.97691877 * 2.0 * pi * (0.5 * EL_b * gs_minus_spline_itp(flux_c)
              - J_eff(flux_c) * 2.0 * pi * flux_a - EL_bar(flux_c) * 2.0 * pi * flux_b))
end
# two qubit gates

ZZPIBY2_ = kron(ZPIBY2_, ZPIBY2_)
XXPIBY2_ = kron(XPIBY2_, XPIBY2_)
YYPIBY2_ = kron(YPIBY2_, YPIBY2_)
XIPIBY2_ = kron(XPIBY2_, PAULI_IDENTITY)
IXPIBY2_ = kron(PAULI_IDENTITY, XPIBY2_)
YIPIBY2_ = kron(YPIBY2_, PAULI_IDENTITY)
IYPIBY2_ = kron(PAULI_IDENTITY, YPIBY2_)
ZIPIBY2_ = kron(ZPIBY2_, PAULI_IDENTITY)
IZPIBY2_ = kron(PAULI_IDENTITY, ZPIBY2_)

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

const CNOT_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,1]))
const CNOT_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,2]))
const CNOT_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,3]))
const CNOT_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(CNOT[:,4]))

const iSWAP_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,1]))
const iSWAP_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,2]))
const iSWAP_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,3]))
const iSWAP_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(iSWAP[:,4]))

const XIPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XIPIBY2_[:,1]))
const XIPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XIPIBY2_[:,2]))
const XIPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XIPIBY2_[:,3]))
const XIPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(XIPIBY2_[:,4]))

const IXPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IXPIBY2_[:,1]))
const IXPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IXPIBY2_[:,2]))
const IXPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IXPIBY2_[:,3]))
const IXPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IXPIBY2_[:,4]))

const YIPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YIPIBY2_[:,1]))
const YIPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YIPIBY2_[:,2]))
const YIPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YIPIBY2_[:,3]))
const YIPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(YIPIBY2_[:,4]))

const IYPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IYPIBY2_[:,1]))
const IYPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IYPIBY2_[:,2]))
const IYPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IYPIBY2_[:,3]))
const IYPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IYPIBY2_[:,4]))

const ZIPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZIPIBY2_[:,1]))
const ZIPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZIPIBY2_[:,2]))
const ZIPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZIPIBY2_[:,3]))
const ZIPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(ZIPIBY2_[:,4]))

const IZPIBY2_ISO_1 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IZPIBY2_[:,1]))
const IZPIBY2_ISO_2 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IZPIBY2_[:,2]))
const IZPIBY2_ISO_3 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IZPIBY2_[:,3]))
const IZPIBY2_ISO_4 = SVector{HDIM_TWOSPIN_ISO}(get_vec_iso(IZPIBY2_[:,4]))

function target_states(gate_type)
    if gate_type == xxpiby2
        target_state1 = Array(XXPIBY2_ISO_1)
        target_state2 = Array(XXPIBY2_ISO_2)
        target_state3 = Array(XXPIBY2_ISO_3)
        target_state4 = Array(XXPIBY2_ISO_4)
    elseif gate_type == yypiby2
        target_state1 = Array(YYPIBY2_ISO_1)
        target_state2 = Array(YYPIBY2_ISO_2)
        target_state3 = Array(YYPIBY2_ISO_3)
        target_state4 = Array(YYPIBY2_ISO_4)
    elseif gate_type == zzpiby2
        target_state1 = Array(ZZPIBY2_ISO_1)
        target_state2 = Array(ZZPIBY2_ISO_2)
        target_state3 = Array(ZZPIBY2_ISO_3)
        target_state4 = Array(ZZPIBY2_ISO_4)
    elseif gate_type == xipiby2
        target_state1 = Array(XIPIBY2_ISO_1)
        target_state2 = Array(XIPIBY2_ISO_2)
        target_state3 = Array(XIPIBY2_ISO_3)
        target_state4 = Array(XIPIBY2_ISO_4)
    elseif gate_type == ixpiby2
        target_state1 = Array(IXPIBY2_ISO_1)
        target_state2 = Array(IXPIBY2_ISO_2)
        target_state3 = Array(IXPIBY2_ISO_3)
        target_state4 = Array(IXPIBY2_ISO_4)
    elseif gate_type == yipiby2
        target_state1 = Array(YIPIBY2_ISO_1)
        target_state2 = Array(YIPIBY2_ISO_2)
        target_state3 = Array(YIPIBY2_ISO_3)
        target_state4 = Array(YIPIBY2_ISO_4)
    elseif gate_type == iypiby2
        target_state1 = Array(IYPIBY2_ISO_1)
        target_state2 = Array(IYPIBY2_ISO_2)
        target_state3 = Array(IYPIBY2_ISO_3)
        target_state4 = Array(IYPIBY2_ISO_4)
    elseif gate_type == zipiby2
        target_state1 = Array(ZIPIBY2_ISO_1)
        target_state2 = Array(ZIPIBY2_ISO_2)
        target_state3 = Array(ZIPIBY2_ISO_3)
        target_state4 = Array(ZIPIBY2_ISO_4)
    elseif gate_type == izpiby2
        target_state1 = Array(IZPIBY2_ISO_1)
        target_state2 = Array(IZPIBY2_ISO_2)
        target_state3 = Array(IZPIBY2_ISO_3)
        target_state4 = Array(IZPIBY2_ISO_4)
    elseif gate_type == cnot
        target_state1 = Array(CNOT_ISO_1)
        target_state2 = Array(CNOT_ISO_2)
        target_state3 = Array(CNOT_ISO_3)
        target_state4 = Array(CNOT_ISO_4)
    elseif gate_type == iswap
        target_state1 = Array(iSWAP_ISO_1)
        target_state2 = Array(iSWAP_ISO_2)
        target_state3 = Array(iSWAP_ISO_3)
        target_state4 = Array(iSWAP_ISO_4)
    end
    return (target_state1, target_state2, target_state3, target_state4)
end
