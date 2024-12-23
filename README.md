# Randomization Tests for Conditional Group Symmetry

This repository contains the code used to generate the experiment results discussed in the article [Randomization Tests for Conditional Group Symmetry](https://arxiv.org/abs/2412.14391), which supersedes the article [Non-parametric Hypothesis Tests for Distributional Group Symmetry](https://arxiv.org/abs/2307.15834). The repository for the previous article can be found [here](https://github.com/chiukenny/Tests-for-Distributional-Symmetry).

* To preprocess data for the LHC and Lorentz demos/experiments, see the file `preprocess_data.jl` for instructions.
* To run one of the demo files, execute the command `julia <demo_file>.jl` in the command line.
* To run one of the experiments described in the article, see the file `run_experiments.jl` for instructions.
    * See `/src/global_variables.jl` for default parameter values.
* To create the plots and figures described in the article, execute the command `julia make_plots.jl` in the command line after running all experiments.

---
#### Organization

This GitHub repo is organized as follows:

* `/data/`: raw data to be processed for the LHC and Lorentz demos/experiments
* `/experiments/`: scripts for experiments described in the article
* `/outputs/`: experiment outputs and plots
* `/src/`: implementations of tests and helper functions