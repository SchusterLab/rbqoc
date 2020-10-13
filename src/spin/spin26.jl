"""
spin26.jl - p-grape
"""

WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using HDF5
using LinearAlgebra
using StaticArrays
using Zygote

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin13"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 1
const STATE_COUNT = 2
const ASTATE_SIZE = STATE_COUNT * HDIM_ISO + 3 * CONTROL_COUNT
const ACONTROL_SIZE = CONTROL_COUNT
const INITIAL_STATE1 = [1., 0, 0, 0]
const INITIAL_STATE2 = [0., 1, 0, 0]
# state indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const INTCONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INTCONTROLS_IDX[end] + 1:INTCONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT


function rollout(controls::SizedVector{knot_count}, state1::SVector{HDIM_ISO},
                 state2::SVector{HDIM_ISO}, dt::Real)
    for i = 1:knot_count
        h_prop = exp(dt * (FQ_NEGI_H0_ISO + controls[i] * NEGI_H1_ISO))
        state1 = h_prop * state1
        state2 = h_prop * state2
    end

    return (state1, state2)
end


function objective(controls::SizedVector{knot_count}, state1::SVector{HDIM_ISO},
                   state2::SVector{HDIM_ISO}, target1::SVector{HDIM_ISO}, target2::SVector{HDIM_ISO},
                   qs::SVector{3}, dt::Real)
    (fstate1, fstate2) = rollout(controls, state1, state2, dt)
    d1 = fstate1 - target1
    d2 = fstate2 - target2
    state_cost = qs[1] * (d1'd1 + d2'd2)
    
    dcontrols_diff = diff(controls)
    dcontrols_cost = qs[2] * dcontrols_diff'dcontrols_diff
    d2controls_diff = diff(dcontrols_diff)
    d2controls_cost = qs[3] * d2controls_diff'd2controls_diff
    
    return state_cost + dcontrols_cost + d2controls_cost
end


function max_violation(state1::SVector{HDIM_ISO}, state2::SVector{HDIM_ISO},
                       target1::SVector{HDIM_ISO}, target2::SVector{HDIM_ISO})
    d1 = state1 - target1
    s1_viol = d1'd1
    d2 = state2 - target2
    s2_viol = d2'd2

    return maximum((s1_viol, s2_viol))
end


function run_pgrape(;gate_type=xpiby2, evolution_time=56.8,
                    ctol=1e-8, dt=1e-1, qs=SVector{3}([1e0, 1e-1, 1e-1]))
    # setup
    knot_count = Int(div(evolution_time, dt))
    controls = SizedVector{knot_count}(fill(1e-2, knot_count))
    
    # initial state
    state1 = SVector{HDIM_ISO}(INITIAL_STATE1)
    state2 = SVector{HDIM_ISO}(INITIAL_STATE2)

    # final state
    if gate_type == xpiby2
        target1 = XPIBY2_ISO_1
        target2 = XPIBY2_ISO_1
    end
    
    # perform optimization
    max_viol = 1
    while max_viol > ctol
        (grad_controls,) = Zygote.gradient(objective_, controls)
        (fs1, fs2) = rollout(controls, state1, state2)
        max_viol = max_violation(fs1, fs2, target1, target2)
    end
end
