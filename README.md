# rbqoc
Repository for [Robust Quantum Optimal Control with Trajectory Optimization]()

## Contents
- [About](#about)
- [Quick Start](#quick-start)
  - [Who's Julia?](#whos-julia)
  - [I have Julia, how do I work with this repo?](#i-have-julia-how-do-i-work-with-this-repo)
  - [I have Julia and the repo, what should I run?](#i-have-julia-and-the-repo-what-should-i-run)
- [Beyond Quick Start](#beyond-quick-start)
- [Related Work](#related-work)
- [Cite this Work](#cite-this-work)

## About
This repository is associated with the paper
[Robust Quantum Optimal Control with Trajectory Optimization](), a collaboration
between [Schuster Lab](http://schusterlab.uchicago.edu) and
the [Robotic Exploration Lab](http://roboticexplorationlab.org).
This repo contains the files and information
necessary to reproduce the work.

This repo is NOT a package. The optimization problems
are defined using
[TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl)
and [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl).
The optimization problems are solved with
[Altro.jl](https://github.com/RoboticExplorationLab/Altro.jl).
For those familiar with the quantum optimal control (QOC) literature,
ALTRO is a solver in the same sense that GOAT, GRAPE, and Krotov are solvers.
This repo is merely a set of files to demonstrate how to use ALTRO for
QOC, and in particular, how to engineer robustness to parameter uncertainties and
mitigate decoherence using the techniques we introduced in the paper.

This repo will NOT be updated to reflect new versions
of its dependencies. However, this repo will be updated for clarity.
If you feel that an aspect of
the documentation for this work is lacking, e.g. a part of this README is ambiguous
or a file could be better commented / explained, or you find a bug,
please file a GitHub issue. Other inquiries about this work can be directed to
[Thomas Propson](mailto:tcpropson@protonmail.com)
or [David Schuster](mailto:David.Schuster@uchicago.edu).


## Quick Start


### Who's Julia?
To execute the code in this repo, you will need to
[install Julia](https://julialang.org/downloads/).

[Julia](https://julialang.org) is a dynamically-typed and just-in-time (JIT) compiled
programming language designed for high performance computing.
Julia is similar to Python in terms of the
easy-to-read syntax you have come to love, but dissimilar in terms of the
slow for-loops you have come to not so love.
Julia provides substantial
performance benefits for this work through compiler optimizations,
most importantly those in
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
We encourage the interested reader to check out the links in the
[Related Work](#related-work) section to find out more about Julia's performance
benefits.

### I have Julia, how do I work with this repo?
Julia uses a different package-management scheme than Python.
With Python, you use third-party installers like pip or conda
to manage your global environment.
With Julia, you use the [Pkg](https://docs.julialang.org/en/v1/stdlib/Pkg/)
module from the standard library 
to manage an environment for each _project_. This is similar to the
concept of a pipfile. The packages used by this project
are defined in `Manifest.toml` and `Project.toml`
at the top level of the repo.

First, clone the repo.
```
$ git clone https://github.com/SchusterLab/rbqoc.git
```
Navigate to the top level.
```
$ cd rbqoc
```
Enter the Julia read-eval-print-loop (REPL).
```
$ julia
```
Import the Pkg module.
```
julia> using Pkg
```
Activate the project.
```
julia> Pkg.activate(".")
```
Instantiate the project.
```
julia> Pkg.instantiate()
```

You have now downloaded all of the necessary packages.

### I have Julia and the repo, what should I run?
The base optimization outlined in section III of the paper
is a good starting point. It can be found in `src/spin/spin13.jl`.

Navigate to the file.
```
$ cd src/spin
```
Enter the Julia REPL.
```
$ julia
```
Include the file. If this is your first time including
the file, all of the dependent packages will be precompiled.
Precompiling will take some time, but it will only happen once.
```
julia> include("spin13.jl")
```
Run the optimization. All of the optimizations in this
repo are called with a function named `run_traj`. The hyperparameters
and output of the optimization can be modified by passing
arguments to this function, see the corresponding
[file](https://github.com/SchusterLab/rbqoc/blob/master/src/spin/spin13.jl#L62)
for details.
```
juila> run_traj()
```

## Beyond Quick Start

`src/rbqoc.jl` and `src/spin/spin.jl` contain common definitions.

The analytic pulses were generated with `src/spin/spin14.py`.

In Figure 1, the depolarization aware pulses were generated with `src/spin/spin15.jl`

In Figure 2, the sampling method corresponds to `src/spin/spin12.jl`, the unscented
sampling method corresponds to `src/spin/spin30.jl`, and the derivative methods
correspond to `src/spin/spin11.jl`.

In Figure 3, the sampling method corresponds to `src/spin/spin18.jl`,
the unscented sampling method corresponds to `src/spin/spin25.jl`,
and the derivative methods correspond to `src/spin/spin17.jl`.

The data for the figures in the paper can be produced with `src/spin/figures.jl`
and the figures can be produced with `src/spin/figures.py`. In `src/spin/figures.jl`
you will find references to HDF5 files with the name structure `XXXXX_spinYY.h5`.
These files are output by each of the optimization programs named `src/spin/spinYY.jl`
and contain the optimized pulse. The HDF5 files used for the paper are
available upon request from the authors--we did not put them in the repo because
they are large binary files--but they can
be generated on your machine by running the corresponding optimization program
`src/spin/spinYY.jl` with the hyperparameters listed in `nb/trails.xlsx`, or
its Google Sheet counterpart
[trials.xlsx](https://docs.google.com/spreadsheets/d/1DrW6S13RZ-FpTsDDSbfPPRQWIXcY5S-4/edit#gid=1396699849).



## Related Work
- Trajectory Optimization
  - [Trajectory Optimization (Wikipedia)](https://en.wikipedia.org/wiki/Trajectory_optimization)
  - [LQR (Wikipedia)](https://en.wikipedia.org/wiki/Linear–quadratic_regulator)
  - [ALTRO paper](https://roboticexplorationlab.org/papers/altro-iros.pdf)
  - [ALTRO tutorial paper](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf)
  - [Direct Collocation tutorial paper](https://epubs.siam.org/doi/pdf/10.1137/16M1062569)
- Julia
  - [Julia micro-benchmarks](https://julialang.org/benchmarks/)
  - [What scientists must know about hardware to write fast code](https://biojulia.net/post/hardware/)
  - [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl) Julia's QuTiP
- QOC
  - [c3](https://github.com/q-optimize/c3) control, calibration and characterization
  of quantum machines
  - [CRAB](https://github.com/Bilal092/CRAB) CRAB in MATLAB
  - [Dynamo](https://github.com/shaimach/Dynamo) DYNAMO in MATLAB
  - [GOAT-QuantumControl](https://github.com/Bilal092/GOAT-QuantumControl) GOAT in MATLAB
  - [Juqbox.jl](https://github.com/LLNL/Juqbox.jl) QOC from LLNL in Julia,
  symplectic integrators, pseudo-spectral
  - [krotov](https://github.com/qucontrol/krotov) Krotov in Python
  - [python-open-controls](https://github.com/qctrl/python-open-controls) QOC from Q-CTRL in Python
  - [qoc](https://github.com/SchusterLab/qoc) GRAPE via autograd in Python
  - [quantum-optimal-control](https://github.com/SchusterLab/quantum-optimal-control)
  GRAPE via TensorFlow in Python
  - [QuOptimalControl.jl](https://github.com/alastair-marshall/QuOptimalControl.jl)
	a library of QOC algorithms, written in Julia
  - [QuTiP](http://qutip.org/docs/latest/guide/guide-control.html)
  QOC with QuTip

  
## Cite this Work


