"""
spin11.jl - use trajectory optimization for the spin system
"""

using HDF5
using LinearAlgebra
using TrajectoryOptimization
using Printf
using StaticArrays

EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin11"
WDIR = ENV["ROBUST_QOC_PATH"]
SAVE_PATH = joinpath(WDIR, "out/$EXPERIMENT_META/$EXPERIMENT_NAME")

function generate_save_file_path(save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$save_file_name.h5", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$save_file_name.h5"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end

# Define experimental constants.
OMEGA = 2 * pi * 1e-2
MAX_CONTROL_NORM_0 = 2 * pi * 3e-1

# Define the system.
NEG_I = SA_F64[0   0  1  0 ;
               0   0  0  1 ;
               -1  0  0  0 ;
               0  -1  0  0 ;]
SIGMA_X = SA_F64[0   1   0   0;
                 1   0   0   0;
                 0   0   0   1;
                 0   0   1   0]
SIGMA_Z = SA_F64[1   0   0   0;
                 0  -1   0   0;
                 0   0   1   0;
                 0   0   0  -1]
H_S = SIGMA_Z / 2
neg_i_H_S = NEG_I * H_S
H_C1 = SIGMA_X / 2

# Define the optimization.
EVOLUTION_TIME = 120.
COMPLEX_CONTROLS = false
CONTROL_COUNT = 1
DT = 1e-2
N = Int(EVOLUTION_TIME / DT) + 1

# Define the problem.
INITIAL_STATE = SA[1., 0, 0, 0]
STATE_SIZE, = size(INITIAL_STATE)
TARGET_STATE = SA[0, 1., 0, 0]

struct Model <: AbstractModel
    n :: Int
    m :: Int
    dynamics :: Any
end

function dynamics(model::Model, x, u, t)
    hamiltonian = OMEGA * H_S + u[1] * H_C1
    return NEG_I * hamiltonian * x
end


function main()
    dt = DT
    n = STATE_SIZE
    m = CONTROL_COUNT
    t0 = 0.
    tf = EVOLUTION_TIME
    x0 = INITIAL_STATE
    xf = TARGET_STATE
    
    model = Model(n, m, dynamics)

    # initial trajectory
    U0 = [@SVector randn(m) for k = 1:N-1]
    X0 = [@SVector fill(NaN, n) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    Q = 1e-2 * Diagonal(I, STATE_SIZE)
    Qf = 1. * Diagonal(I, STATE_SIZE)
    R = 1e-1 * Diagonal(I, CONTROL_COUNT)
    obj = LQRObjective(Q, R, Qf, TARGET_STATE, N)

    # bnd = BoundConstraint(STATE_SIZE, CONTROL_COUNT, u_max=MAX_CONTROL_NORM_0, u_min=-MAX_CONTROL_NORM_0)
    goal_constraint = GoalConstraint(TARGET_STATE)
    goal_constraint = ConstraintVals(goal_constraint, N:N)
    constraints = ConstraintSet(STATE_SIZE, CONTROL_COUNT, [goal_constraint], N)

    # constraints = Constraints(N)
    # for k = 1:N-1
    #     constraints[k] += bnd
    # end
    # constraints[N] += goal

    prob = Problem{RK3}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    solver = iLQRSolver(prob)
    solve!(solver)
end
