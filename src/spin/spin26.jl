"""
spin26.jl - This is a julia implementation of backwards diff
GRAPE with projected gradient descent via Adam.
"""

WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using ForwardDiff
using HDF5
using LinearAlgebra
using ReverseDiff
using StaticArrays
using Statistics
using Zygote

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin26"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

### PADE DUE TO HIGHAM 2005 ###

# Pade approximants from algorithm 2.3.
const B = (
    64764752532480000,
    32382376266240000,
    7771770303897600,
    1187353796428800,
    129060195264000,
    10559470521600,
    670442572800,
    33522128640,
    1323241920,
    40840800,
    960960,
    16380,
    182,
    1,
)

function one_norm(a)
    """
    Return the one-norm of the matrix.
    References:
    [0] https://www.mathworks.com/help/dsp/ref/matrix1norm.html
    Arguments:
    a :: ndarray(N x N) - The matrix to compute the one norm of.
    
    Returns:
    one_norm_a :: float - The one norm of a.
    """
    return maximum(sum(map(abs, a), dims=1))
end


function pade3(a, i)
    a2 = a * a
    u = a * (B[3] * a2) + B[2] * a
    v = B[3] * a2 + B[1] * i
    return u, v
end

function pade5(a, i)
    a2 = a * a
    a4 = a2 * a2
    u = a * (B[6] * a4 + B[4] * a2) + B[2] * a
    v = B[5] * a4 + B[3] * a2 + B[1] * i
    return u, v
end

function pade7(a, i)
    a2 = a * a
    a4 = a2 * a2
    a6 = a2 * a4
    u = a * (B[8] * a6 + B[6] * a4 + B[4] * a2) + B[2] * a
    v = B[7] * a6 + B[5] * a4 + B[3] * a2 + B[1] * i
    return u, v
end


function pade9(a, i)
    a2 = a * a
    a4 = a2 * a2
    a6 = a2 * a4
    a8 = a2 * a6
    u = a * (B[10] * a8 + B[8] * a6 + B[6] * a4 + B[4] * a2) + B[2] * a
    v = B[9] * a8 + B[7] * a6 + B[5] * a4 + B[3] * a2 + B[1] * i
    return u, v
end

function pade13(a, i)
    a2 = a * a
    a4 = a2 * a2
    a6 = a2 * a4
    u = a * (a6 * (B[14] * a6 + B[12] * a4 + B[10] * a2) + B[8] * a6 + B[6] * a4 + B[4] * a2) + B[2] * a
    v = a6* (B[13] * a6 + B[11] * a4 + B[9] * a2) + B[7] * a6 + B[5] * a4 + B[3] * a2 + B[1] * i
    return u, v
end

# Valid pade orders for algorithm 2.3.
const PADE_ORDERS = (
    3,
    5,
    7,
    9,
    13,
)


# Pade approximation functions.
const PADE = [
    nothing,
    nothing,
    nothing,
    pade3,
    nothing,
    pade5,
    nothing,
    pade7,
    nothing,
    pade9,
    nothing,
    nothing,
    nothing,
    pade13,
]


# Constants taken from table 2.3.
const THETA = (
    0,
    0,
    0,
    1.495585217958292e-2,
    0,
    2.539398330063230e-1,
    0,
    9.504178996162932e-1,
    0,
    2.097847961257068,
    0,
    0,
    0,
    5.371920351148152,
)


function expm_pade(a)
    """
    Compute the matrix exponential via pade approximation.
    References:
    [0] http://eprints.ma.man.ac.uk/634/1/high05e.pdf
    [1] https://github.com/scipy/scipy/blob/v0.14.0/scipy/linalg/_expm_frechet.py#L10
    [2] https://github.com/SchusterLab/qoc/blob/dev-robust/qoc/standard/functions/expm.py#L212
    Arguments:
    a :: ndarray(N x N) - The matrix to exponentiate.
    
    Returns:
    expm_a :: ndarray(N x N) - The exponential of a.
    """
    # If the one norm is sufficiently small,
    # pade orders up to 13 are well behaved.
    scale = 0
    size_ = size(a, 1)
    pade_order = nothing
    one_norm_ = one_norm(a)
    for pade_order_ in PADE_ORDERS
        if one_norm_ < THETA[pade_order_]
            pade_order = pade_order_
        end
    end

    # If the one norm is large, scaling and squaring
    # is required.
    if isnothing(pade_order)
        pade_order = 14
        scale = max(0, Int(ceil(log2(one_norm_ / THETA[14]))))
        a = a * 2^(-scale)
    end

    # Execute pade approximant.
    i = I(size_)
    u, v = PADE[pade_order](a, i)
    r = (u + v)/(-u + v)

    # Do squaring if necessary.
    for i = 1:scale
        r = r * r
    end

    return r
end
        

"""
gram_schmidt

Arguments:
    basis::Matrix - cols are basis vectors
Returns:
    basis_ ::Vector{Vector} - elements are 
    orthogonal basis vectors
References:
[0] http://vmls-book.stanford.edu/vmls-julia-companion.pdf
"""
function gram_schmidt(basis)
    basis_ = Vector{Vector}(undef, size(basis, 2))
    for i = 1:size(basis, 2)
        q = basis[:, i]
        for j = 1:i - 1
            q -= basis_[j]'basis[:, i] * basis_[j]
        end
        basis_[i] = q / norm(q)
    end
    return basis_
end


function adam_update(params::Vector, grads::Vector, iter::Int, learning_rate::Float64,
                     gradient_moment::Vector, gradient_square_moment::Vector;
                     eps=1e-8, β1=0.9, β2=0.999)
    gradient_moment .= β1 * gradient_moment + (1 - β1) * grads
    gradient_square_moment .= β2 * gradient_square_moment + (1 - β2) * grads.^2
    gradient_moment_hat = gradient_moment / (1 - β1^iter)
    gradient_square_moment_hat = gradient_square_moment / (1 - β2^iter)
    params .= params .- learning_rate .* gradient_moment ./ (map(sqrt, gradient_square_moment) .+ eps)
end    


"""
znf_basis - construct an orthonormal basis for the tangent space where all elements
of a vector sum to zero
"""
function znf_basis(knot_count)
    basis = zeros(knot_count, knot_count - 1)
    for i = 2:knot_count - 1
        basis[i, i - 1] = -1
    end
    basis[end, end] = -1
    basis[1, :] .= 1
    basis = gram_schmidt(basis)
    return basis
end


"""
project - project a vector onto an orthonormal basis
"""
function project(vec::Vector, basis::Vector{<:Vector})
    vec_ = zeros(size(vec, 1))
    for i = 1:size(basis, 1)
        vec_ = vec_ + vec'basis[i] * basis[i]
    end
    return vec_
end


"""
rollout - integrate the Schrodinger equation like it's hot
"""
function rollout(controls::AbstractVector, state1::SVector{HDIM_ISO},
                 state2::SVector{HDIM_ISO}, dt::Real, knot_count::Int64)
    for i = 1:knot_count
        h_prop = expm_pade(dt .* (FQ_NEGI_H0_ISO + (controls[i] .* NEGI_H1_ISO)))
        state1 = h_prop * state1
        state2 = h_prop * state2
    end

    return (state1, state2)
end


function objective(controls::AbstractVector, state1::SVector{HDIM_ISO},
                   state2::SVector{HDIM_ISO}, target1::SVector{HDIM_ISO}, target2::SVector{HDIM_ISO},
                   qs::SVector{3}, dt::Real, knot_count::Int64)
    # fidelity
    (fstate1, fstate2) = rollout(controls, state1, state2, dt, knot_count)
    d1 = fstate1 - target1
    d2 = fstate2 - target2
    state_cost = qs[1] * (d1'd1 + d2'd2)

    # control variation
    dcontrols_diff = diff(controls)
    dcontrols_cost = qs[2] * dcontrols_diff'dcontrols_diff
    d2controls_diff = diff(dcontrols_diff)
    d2controls_cost = qs[3] * d2controls_diff'd2controls_diff
    
    return state_cost + dcontrols_cost + d2controls_cost
end


@inline max_violation(state1::SVector{HDIM_ISO}, state2::SVector{HDIM_ISO},
                      target1::SVector{HDIM_ISO}, target2::SVector{HDIM_ISO}) = (
    maximum((1 - fidelity_vec_iso2(state1, target1), 1 - fidelity_vec_iso2(state2, target2)))
)


"""
run_pgrape

Arguments:
qs :: Array - (ψ, ∂a, ∂2a)
"""
function run_pgrape(;gate_type=zpiby2, evolution_time=18.,
                    ctol=1e-8, dt=1e-1, qs=SVector{3}([1e0, 1e-1, 1e-1]),
                    learning_rate=1e-2, max_iterations=Int64(1e3),
                    min_control=-0.5, max_control=0.5)
    # setup
    knot_count = Int(div(evolution_time, dt))
    controls = fill(1e-2, knot_count)
    znf_basis_ = znf_basis(knot_count)
    
    # initial state
    state1 = SVector{HDIM_ISO}(IS1_ISO_)
    state2 = SVector{HDIM_ISO}(IS2_ISO_)

    # final state
    if gate_type == xpiby2
        target1 = XPIBY2_ISO_1
        target2 = XPIBY2_ISO_2
    elseif gate_type == zpiby2
        target1 = ZPIBY2_ISO_1
        target2 = ZPIBY2_ISO_2
    end
    
    # initialize optimization
    max_viol = 1
    objective_(controls_) = objective(controls_, state1, state2,
                                      target1, target2, qs, dt, knot_count)
    grad_controls = zeros(knot_count)
    gradient_moment = zeros(knot_count)
    gradient_square_moment = zeros(knot_count)
    # perform optimization
    for i = 1:max_iterations
        # check if converged
        (fs1, fs2) = rollout(controls, state1, state2, dt, knot_count)
        max_viol = max_violation(fs1, fs2, target1, target2)
        println("max_viol: $(max_viol)")
        if max_viol < ctol
            return (fs1, fs2)
            break
        end
        # compute gradient
        (grad_controls,) = Zygote.gradient(objective_, controls)
        # update controls
        adam_update(controls, grad_controls, i, learning_rate,
                    gradient_moment, gradient_square_moment)
        # project controls onto zero net flux constraint manifold
        controls = project(controls, znf_basis_)
        # ensure controls satisfy bound constraints
        controls[1] = controls[end] = 0.
        for j = 2:knot_count - 1
            controls[j] = clamp(controls[j], min_control, max_control)
        end
    end
end
