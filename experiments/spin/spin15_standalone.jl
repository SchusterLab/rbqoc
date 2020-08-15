"""
spin15_standalone.jl - t1 optimized pulses
"""

using Altro
using Dierckx
using HDF5
using Interpolations
using LinearAlgebra
using Plots
using Printf
using RobotDynamics
using StaticArrays
using TrajectoryOptimization
const TO = TrajectoryOptimization

# paths
WDIR = get(ENV, "ROBUST_QOC_PATH", "../../")
EXPERIMENT_META = "spin"
EXPERIMENT_NAME = "spin15"
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# defs
# this type is used for specifying a target transformation
@enum GateType begin
    zpiby2 = 1
    ypiby2 = 2
    xpiby2 = 3
end

# this type specifies common save formats i use
@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end

# this type is used for choosing an Altro solver
@enum SolverType begin
    ilqr = 1
    alilqr = 2
    altro = 3
end


"""
get an unused file name like XXXXX_<save_file_path>/<save_file_name>.<extension>
where XXXXX is a unique numeric prefix
"""
function generate_save_file_path(extension, save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$(save_file_name).$(extension)", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$(save_file_name).$(extension)"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


"""
read_save - Read all data from an h5 file into memory.
"""
function read_save(save_file_path)
    dict = h5open(save_file_path, "r") do save_file
        dict = Dict()
        for key in names(save_file)
            dict[key] = read(save_file, key)
        end
        return dict
    end

    return dict
end


"""
sample_controls - Sample controls and d2controls_dt2
on the preferred time axis using a spline.
"""
function sample_controls(save_file_path; dt=DT_PREF, dt_inv=DT_PREF_INV,
                         plot=false, plot_file_path=nothing)
    # Grab data to sample from.
    save = read_save(save_file_path)
    controls = save["astates"][1:end - 1, (save["controls_idx"])]
    d2controls_dt2 = save["acontrols"][1:end, save["d2controls_dt2_idx"]]
    (control_knot_count, control_count) = size(controls)
    if "dt_idx" in keys(save)
        dts = save["acontrols"][1:end, save["dt_idx"]]
    elseif "dt" in keys(save)
        dts = save["dt"] * ones(control_knot_count)
    end
    time_axis = [0; cumsum(dts, dims=1)[1:end - 1]]

    # Construct time axis to sample over.
    final_time_sample = sum(dts)
    knot_count_sample = Int(floor(final_time_sample * dt_inv))
    # The last control should be DT_PREF before final_time_sample.
    time_axis_sample = Array(0:1:knot_count_sample - 1) * dt

    # Sample time_axis_sample via spline.
    controls_sample = zeros(knot_count_sample, control_count)
    d2controls_dt2_sample = zeros(knot_count_sample, control_count)
    for i = 1:control_count
        controls_spline = Spline1D(time_axis, controls[:, i])
        controls_sample[:, i] = map(controls_spline, time_axis_sample)
        d2controls_dt2_spline = Spline1D(time_axis, d2controls_dt2[:, i])
        d2controls_dt2_sample[:, i] = map(d2controls_dt2_spline, time_axis_sample)
    end

    # Plot.
    if plot
        DPI = 300
        MS_SMALL = 2
        ALPHA = 0.2
        fig = Plots.plot(dpi=DPI)
        Plots.scatter!(time_axis, controls[:, 1], label="controls data", markersize=MS_SMALL, alpha=ALPHA)
        Plots.scatter!(time_axis_sample, controls_sample[:, 1], label="controls fit",
                       markersize=MS_SMALL, alpha=ALPHA)
        Plots.scatter!(time_axis, d2controls_dt2[:, 1], label="d2_controls_dt2 data")
        Plots.scatter!(time_axis_sample, d2controls_dt2_sample[:, 1], label="d2_controls_dt2 fit")
        Plots.xlabel!("Time (ns)")
        Plots.ylabel!("Amplitude (GHz)")
        Plots.savefig(fig, plot_file_path)
        println("Plotted to $(plot_file_path)")
    end
    return (controls_sample, d2controls_dt2_sample, final_time_sample)
end


# system defs
FQ = 1.4e-2 #GHz
MAX_CONTROL_NORM_0 = 5e-1 #GHz
FBFQ_A = 0.202407
FBFQ_B = 0.5
NEGI = SA_F64[0   0  1  0 ;
              0   0  0  1 ;
              -1  0  0  0 ;
              0  -1  0  0 ;]
SIGMAX_ISO = SA_F64[0   1   0   0;
                    1   0   0   0;
                    0   0   0   1;
                    0   0   1   0]
SIGMAZ_ISO = SA_F64[1   0   0   0;
                    0  -1   0   0;
                    0   0   1   0;
                    0   0   0  -1]
NEGI_H0_ISO = pi * NEGI * SIGMAZ_ISO
NEGI_H1_ISO = pi * NEGI * SIGMAX_ISO
const FQ_NEGI_H0_ISO = FQ * NEGI_H0_ISO

# raw T1 times are in units of microseconds
T1_ARRAY = [
    1597.923, 1627.93, 301.86, 269.03, 476.33, 1783.19, 2131.76, 2634.50, 
    4364.68, 2587.82, 1661.915, 1794.468, 2173.88, 1188.83, 
    1576.493, 965.183, 560.251, 310.88
] * 1e3
FBFQ_ARRAY = [
    0.26, 0.28, 0.32, 0.34, 0.36, 0.38, 0.4,
    0.42, 0.44, 0.46, 0.465, 0.47, 0.475,
    0.48, 0.484, 0.488, 0.492, 0.5
]
const FBFQ_T1_SPLINE_ITP = extrapolate(interpolate((FBFQ_ARRAY,), T1_ARRAY, Gridded(Linear())), Flat())
@inline amp_fbfq_lo(amplitude) = -abs(amplitude) * FBFQ_A + FBFQ_B
@inline amp_t1_spline(amplitude) = FBFQ_T1_SPLINE_ITP(amp_fbfq_lo(amplitude))


# Define the optimization.
CONTROL_COUNT = 1
DT_PREF = 1e-2
DT_PREF_INV = 1e2
# static is the time step used for non time-optimal problems
DT_STATIC = DT_PREF
DT_STATIC_INV = DT_PREF_INV
# dt_init is the initial time step at all knot points for time-optimal problems
DT_INIT = DT_PREF
DT_INIT_INV = DT_PREF_INV
DT_MIN = DT_INIT / 2
DT_MAX = DT_INIT * 2
CONSTRAINT_TOLERANCE = 1e-8
AL_KICKOUT_TOLERANCE = 1e-7
PN_STEPS = 5

# Define the problem.
INITIAL_STATE1 = SA[1., 0, 0, 0]
INITIAL_STATE2 = SA[0., 1, 0, 0]
STATE_SIZE, = size(INITIAL_STATE1)
INITIAL_ASTATE = [
    INITIAL_STATE1; # state1, this is a quantum evolving according to the schroedinger equation
    INITIAL_STATE2; # state2, also a quantum state
    @SVector zeros(CONTROL_COUNT); # int_control, this is the integral of the control
    @SVector zeros(CONTROL_COUNT); # control, this is the control
    @SVector zeros(CONTROL_COUNT); # dcontrol_dt, this is the first time derivative of the control
    @SVector zeros(1); # int_gamma, this is a thing i want to minimize subject to the constraints of the state evolution
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
# definitions of target quantum states
ZPIBY2_1 = SA[1., 0, -1, 0] / sqrt(2)
ZPIBY2_2 = SA[0., 1, 0, 1] / sqrt(2)
YPIBY2_1 = SA[1., 1, 0, 0] / sqrt(2)
YPIBY2_2 = SA[-1., 1, 0, 0] / sqrt(2)
XPIBY2_1 = SA[1., 0, 0, -1] / sqrt(2)
XPIBY2_2 = SA[0., 1, -1, 0] / sqrt(2)

# state indices
const STATE1_IDX = 1:STATE_SIZE
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + STATE_SIZE
const INT_CONTROLS_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + CONTROL_COUNT
const CONTROLS_IDX = INT_CONTROLS_IDX[end] + 1:INT_CONTROLS_IDX[end] + CONTROL_COUNT
const DCONTROLS_DT_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const INT_GAMMA_IDX = DCONTROLS_DT_IDX[end] + 1:DCONTROLS_DT_IDX[end] + 1
# control indices
const D2CONTROLS_DT2_IDX = 1:CONTROL_COUNT
const DT_IDX = D2CONTROLS_DT2_IDX[end] + 1:D2CONTROLS_DT2_IDX[end] + 1

# Specify logging.
VERBOSE = true
SAVE = true

struct Model <: AbstractModel
    n :: Int
    m :: Int
end


Base.size(model::Model) = (model.n, model.m)


# Running this function with the default arguments given here reproduces
# the problem that I want to solve.
function run_traj(;evolution_time=60., gate_type=xpiby2,
                  initial_save_file_path=nothing,
                  initial_save_type=jl, time_optimal=true,
                  solver_type=altro)
    # Choose dynamics
    # It is likely the first time you run the script than an error will be thrown
    # saying that the dynamics can not be found. You should be able to reinclude
    # the file and try running it again. There is some weird behavior with world
    # contexts that I don't understand yet due to the eval statements. There
    # might be another way to define the dynamics function conditionally
    # like this but I do not know one yet.
    if time_optimal
        expr = :(
            function RobotDynamics.dynamics(model::Model, astate, acontrols, time)
            negi_h = (
                FQ_NEGI_H0_ISO
                + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
            )
            delta_state1 = negi_h * astate[STATE1_IDX]
            delta_state2 = negi_h * astate[STATE2_IDX]
            delta_int_control = astate[CONTROLS_IDX]
            delta_control = astate[DCONTROLS_DT_IDX]
            delta_dcontrol_dt = acontrols[D2CONTROLS_DT2_IDX]
            # int_gamma is a bad thing that I want to be small. the
            # value of amp_t1_spline is proportional to the value of astate[CONTROLS_IDX][1]
            # so making int_gamma small is often at odds with keeping the controls small
            delta_int_gamma = amp_t1_spline(astate[CONTROLS_IDX][1])^(-1)
            return [
                delta_state1;
                delta_state2;
                delta_int_control;
                delta_control;
                delta_dcontrol_dt;
                delta_int_gamma;
                # I have been doing time optimal problems by multiplying the dynamics
                # by the dt in the acontrols vector, and telling TO that my
                # dt is 1.
            ] .* acontrols[DT_IDX][1]^2
        end
        )
        eval(expr)
    else
        expr = :(
            function RobotDynamics.dynamics(model::Model, astate, acontrols, time)
            negi_h = (
                FQ_NEGI_H0_ISO
                + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
            )
            delta_state1 = negi_h * astate[STATE1_IDX]
            delta_state2 = negi_h * astate[STATE2_IDX]
            delta_int_control = astate[CONTROLS_IDX]
            delta_control = astate[DCONTROLS_DT_IDX]
            delta_dcontrol_dt = acontrols[D2CONTROLS_DT2_IDX]
            delta_int_gamma = amp_t1_spline(astate[CONTROLS_IDX][1])^(-1)
            return [
                delta_state1;
                delta_state2;
                delta_int_control;
                delta_control;
                delta_dcontrol_dt;
                delta_int_gamma;
            ]
        end
        )
        eval(expr)
    end
    # Convert to trajectory optimization language.
    n = ASTATE_SIZE
    t0 = 0.
    x0 = INITIAL_ASTATE
    if time_optimal
        # add one to make dt a decision variable
        m = CONTROL_COUNT + 1
    else
        m = CONTROL_COUNT
    end
    if gate_type == xpiby2
        target_state1 = XPIBY2_1
        target_state2 = XPIBY2_2
    elseif gate_type == ypiby2
        target_state1 = YPIBY2_1
        target_state2 = YPIBY2_2
    elseif gate_type == zpiby2
        target_state1 = ZPIBY2_1
        target_state2 = ZPIBY2_2
    end
    xf = [
        target_state1;
        target_state2;
        @SVector zeros(CONTROL_COUNT); # int_control
        @SVector zeros(CONTROL_COUNT); # control
        @SVector zeros(CONTROL_COUNT); # dcontrol_dt
        @SVector zeros(1); # int_gamma
    ]
    
    # Bound the control amplitude.
    x_max = SVector{n}([
        fill(Inf, STATE_SIZE);
        fill(Inf, STATE_SIZE);
        fill(Inf, CONTROL_COUNT);
        fill(MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, 1)
    ])
    x_min = SVector{n}([
        fill(-Inf, STATE_SIZE);
        fill(-Inf, STATE_SIZE);
        fill(-Inf, CONTROL_COUNT);
        fill(-MAX_CONTROL_NORM_0, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, 1)
    ])
    # Controls start and end at 0.
    x_max_boundary = SVector{n}([
        fill(Inf, STATE_SIZE);
        fill(Inf, STATE_SIZE);
        fill(Inf, CONTROL_COUNT);
        fill(0, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
        fill(Inf, 1)
    ])
    x_min_boundary = SVector{n}([
        fill(-Inf, STATE_SIZE);
        fill(-Inf, STATE_SIZE);
        fill(-Inf, CONTROL_COUNT);
        fill(0, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
        fill(-Inf, 1)
    ])
    # Bound dt.
    if time_optimal
        u_min = SVector{m}([
            fill(-Inf, CONTROL_COUNT);
            fill(sqrt(DT_MIN), 1); # dt
        ])
        u_max = SVector{m}([
            fill(Inf, CONTROL_COUNT);
            fill(sqrt(DT_MAX), 1); # dt
        ])
    else
        u_min = SVector{m}([
            fill(-Inf, CONTROL_COUNT);
        ])
        u_max = SVector{m}([
            fill(Inf, CONTROL_COUNT);
        ])
    end

    # Generate initial trajectory.
    model = Model(n, m)
    U0 = nothing
    if time_optimal
        # Default initial guess w/ optimization over dt.
        dt = 1
        N = Int(floor(evolution_time * DT_INIT_INV)) + 1
        U0 = [SVector{m}([
            fill(1e-4, CONTROL_COUNT);
            fill(DT_INIT, 1);
        ]) for k = 1:N - 1]
    else
        if initial_save_file_path == nothing
            # Default initial guess.
            dt = DT_STATIC
            N = Int(floor(evolution_time * DT_STATIC_INV)) + 1
            U0 = [SVector{m}(
                fill(1e-4, CONTROL_COUNT)
            ) for k = 1:N - 1]
        else
            # Initial guess pulled from initial_save_file_path.
            (d2controls_dt2, evolution_time) = h5open(initial_save_file_path, "r") do save_file
                 if initial_save_type == jl
                     d2controls_dt2_idx = read(save_file, "d2controls_dt2_idx")
                     d2controls_dt2 = read(save_file, "acontrols")[:, d2controls_dt2_idx]
                     evolution_time = read(save_file, "evolution_time")
                 elseif initial_save_type == samplejl
                     d2controls_dt2 = read(save_file, "d2controls_dt2_sample")
                     evolution_time = read(save_file, "evolution_time_sample")
                 end
                 return (d2controls_dt2, evolution_time)
            end
            # Without variable dts, evolution time will be a multiple of DT_STATIC.
            evolution_time = Int(floor(evolution_time * DT_STATIC_INV)) * DT_STATIC
            dt = DT_STATIC
            N = Int(floor(evolution_time * DT_STATIC_INV)) + 1
            U0 = [SVector{m}(d2controls_dt2[k, 1]) for k = 1:N-1]
        end
    end
    X0 = [SVector{n}(
        fill(NaN, n)
    ) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))


    # Define penalties.
    Q = Diagonal(SVector{n}([
        # The state1 and state2 evolution is the key thing. I want to simulate their evolution
        # for very long periods of time using the controls obtained from this optimization.
        # Therefore, I need to make sure the optimized controls get very low tolerances
        # on reaching the target state, otherwise when I put the controls into the long
        # time simulation I get numerical error that builds with each iterate.
        fill(1e1, STATE_SIZE); # state 1
        fill(1e1, STATE_SIZE); # state 2
        fill(1e2, CONTROL_COUNT); # int
        fill(1e2, CONTROL_COUNT); # control
        fill(1e-1, CONTROL_COUNT); # dcontrol_dt
        # I would like this value to be as large as possible so int_gamma is as small as possible.
        # int_gamma is typically on the order of 1e-5 so this 1e7 is not that large.
        # I can typically afford to use 1e9 for similar problems.
        fill(1e7, 1); # int_gamma
    ]))
    Qf = Q * N
    if time_optimal
        R = Diagonal(SVector{m}([
            fill(1e-1, CONTROL_COUNT); # d2control_dt2
            fill(5e2, 1); # dt, I have found this typically needs to be high
        ]))
    else
        R = Diagonal(SVector{m}([
            fill(1e-1, CONTROL_COUNT); # d2control_dt2
        ]))
    end
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must satisfy controls start and stop at zero.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must satisfy dt bound.
    dt_bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    # States must reach target. Controls must have zero net flux.
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; INT_CONTROLS_IDX])
    # States must obey unit norm. This constraint can be removed if necessary.
    # This constraint is rarely the reason for tolerance violation if dt is
    # adequately small / integration is accurate.
    normalization_constraint_1 = NormConstraint(n, m, 1, TO.Equality(), STATE1_IDX)
    normalization_constraint_2 = NormConstraint(n, m, 1, TO.Equality(), STATE2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    add_constraint!(constraints, normalization_constraint_1, 2:N-1)
    add_constraint!(constraints, normalization_constraint_2, 2:N-1)
    if time_optimal
        add_constraint!(constraints, dt_bnd, 1:N-1)
    end

    # Instantiate problem and solve.
    prob = Problem{RobotDynamics.RK4}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
    solver = nothing
    opts = SolverOptions(verbose=VERBOSE)
    if solver_type == alilqr
        solver = AugmentedLagrangianSolver(prob, opts)
        solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.opts.constraint_tolerance_intermediate = CONSTRAINT_TOLERANCE
    elseif solver_type == altro
        solver = ALTROSolver(prob, opts)
        solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.solver_al.opts.constraint_tolerance = AL_KICKOUT_TOLERANCE
        solver.solver_al.opts.constraint_tolerance_intermediate = AL_KICKOUT_TOLERANCE
        solver.solver_pn.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
        solver.solver_pn.opts.n_steps = PN_STEPS
        solver.solver_al.opts.iterations = 1
        solver.solver_al.solver_uncon.opts.iterations = 1
    end
    Altro.solve!(solver)

    # Post-process.
    acontrols_raw = controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TO.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    cidx_arr = Array(CONTROLS_IDX)
    d2cdt2idx_arr = Array(D2CONTROLS_DT2_IDX)
    dtidx_arr = Array(DT_IDX)
    # Square the dts.
    if time_optimal
        acontrols_arr[:, DT_IDX] = acontrols_arr[:, DT_IDX] .^2
    end
    cmax = TO.max_violation(solver)
    cmax_info = TO.findmax_violation(get_constraints(solver))
    
    # Save.
    if SAVE
        save_file_path = generate_save_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            write(save_file, "acontrols", acontrols_arr)
            write(save_file, "controls_idx", cidx_arr)
            write(save_file, "d2controls_dt2_idx", d2cdt2idx_arr)
            write(save_file, "dt_idx", dtidx_arr)
            write(save_file, "evolution_time", evolution_time)
            write(save_file, "astates", astates_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
            write(save_file, "solver_type", Integer(solver_type))
            write(save_file, "cmax", cmax)
            write(save_file, "cmax_info", cmax_info)
        end
        if time_optimal
            # Sample the important metrics.
            (controls_sample, d2controls_dt2_sample, evolution_time_sample) = sample_controls(save_file_path)
            h5open(save_file_path, "r+") do save_file
                write(save_file, "controls_sample", controls_sample)
                write(save_file, "d2controls_dt2_sample", d2controls_dt2_sample)
                write(save_file, "evolution_time_sample", evolution_time_sample)
            end
        end
    end
end
