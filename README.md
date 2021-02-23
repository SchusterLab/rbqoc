# rbqoc
Repository for [Robust Quantum Optimal Control with Trajectory Optimization]().

## Contents
- About
- Quick Start
  - What's Julia?
  - I have Julia, how do I work with this repo?
  - I have Julia and the repo, what should I run?
- Beyond Quick Start
- Related Work
- Cite this Work

## About
This repository is associated with the paper
[Robust Quantum Optimal Control with Trajectory Optimization](), a collaboration
between [Schuster Lab](http://schusterlab.uchicago.edu)
at the University of Chicago and
the [Robotic Exploration Lab](http://roboticexplorationlab.org)
at Carnegie Mellon University.
This repository contains all of the files and information
necessary to reproduce the work.

This repository is NOT a package. All of the optimization problems in this repository
are set up with the packages
[TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl)
and [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl)
and solved with the package
[Altro.jl](https://github.com/RoboticExplorationLab/Altro.jl).
For those in the quantum optimal control (QOC) community,
ALTRO is a solver in the same sense that GOAT, GRAPE, and Krotov are solvers.
This repository is merely a set of files to demonstrate how to use ALTRO for
QOC, and in particular, how to engineer robustness to parameter uncertainties and
mitigate decoherence using the techniques we introduced in the paper.

Additionally, this repo contains the versions of Altro.jl, RobotDynamics.jl, and
TrajectoryOptimization.jl that the repo is meant to be used with.
These versions don't correspond to exact releases of the corresponding packages and
they contain minor bug fixes. You are welcome to use the packages contained in
this repo for setting up your own optimizations; however, depending on when you are reading this,
there may be new functionality added to these packages that the versions in this repo
do not reflect. Should you want to use this new functionality, you can obtain
the latest versions of these packages via their
respective GitHub repositories or through Julia's built-in package management.

Lastly, this repo will not be actively updated as its dependencies
evolve. If you feel that an aspect of
the documentation for this work is lacking, e.g. a part of this README is ambiguous
or a file could be better commented / explained, or you find a bug,
please file a GitHub issue. Other inquiries about this work can be directed to
[Thomas Propson](mailto:tcpropson@protonmail.com)
or [David Schuster](mailto:David.Schuster@uchicago.edu).


## Quick Start


### What's Juila?


### I have Julia, how do I work with this repo?


### I have Julia and the repo, what should I run?


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
QOC is a very active area of research and there are many great tools available for
performing optimization on quantum systems. Here are some projects and references
we would like to share.

## Cite this Work


