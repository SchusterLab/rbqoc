#=
# foo.jl - A module to try out julia.
=#
module Foo

using DiffEqSensitivity
using DifferentialEquations
using Zygote

# Define experimental constants.
OMEGA = 2 * pi * 1e-2

# Define the system.
SIGMA_X = [0 1; 1  0]
SIGMA_Z = [1 0; 0 -1]
H_S = SIGMA_Z / 2
H_C1 = SIGMA_X / 2

"""
Interpolate the controls at a given time.
This function is a dummy. We would normally use polynomial fitting.
"""
function interpolate(controls, time)
    return controls[1]
end


"""
The right-hand-side of the Schroedinger equation for the dynamics
we consider.
"""
# function rhs(state, controls, time)
#     controls_ = interpolate(controls, time)
#     hamiltonian_ = OMEGA * H_S + controls_[1] * H_C1
#     return -1im * hamiltonian_ * state
# end


# Define the optimization.
EVOLUTION_TIME = 120.
TSPAN = (0., EVOLUTION_TIME)

# Define the problem.
INITIAL_STATE = Complex{Float64}[1; 0]
TARGET_STATE = Complex{Float64}[0; 1]
TARGET_STATE_DAGGER = adjoint(TARGET_STATE)
INITIAL_ASTATE = [
    INITIAL_STATE;
]
INITIAL_CONTROLS = [1.;]

"""
Infidelity is a functional of the final state that captures the difference between
the final state and the target final state.
"""
function infidelity(final_state)
    inner_product = TARGET_STATE_DAGGER * final_state
    fidelity = real(inner_product * conj(inner_product))
    infidelity = 1 - fidelity
    return infidelity
end


"""
Evolve a state and compute associated costs.
"""
function evolve(controls, state)
    problem = ODEProblem(rhs, state, TSPAN, p=controls)
    sol = concrete_solve(problem, Tsit5(), state, controls, sensealg=QuadratureAdjoint())
    final_state = sol.u[length(sol.u)]
    return infidelity(final_state)
end


function rhs(du, u, p, t)
  du[1] = p[1] * u[1]
  du[2] = p[2] * u[2]
end

function cost(u0, p, prob, end_time)
    sol = Array(concrete_solve(prob, Tsit5(), u0, p, saveat=end_time, sensealg=InterpolatingAdjoint()))
    final_u = sol[3:4]
    cost_ = final_u[1]
    return cost_
end

function main()
    p = [1.0, 1.0, 3.0, 1.0]
    u0 = [1.0 ; 1.0]
    end_time = 1.0
    tspan = (0.0, end_time)
    prob = ODEProblem(rhs, u0, tspan, p=p)
    cost_(p_) = cost(u0, p_, prob, end_time)
    (dp,) = Zygote.gradient(cost_, p)
    println(dp)
end

    
end
