## USAGE
## -----
##
## From the command line, execute the following command to run ./experiments/<script>.jl:
##     julia --threads 1 run_experiments.jl <script>.jl [args...]
##
## See the experiments below for argument inputs
##
## Edit the 'Experiment settings' and 'Paths' variables below as necessary


# Paths
dir_src = "./src/"               # Source code directory
dir_exp = "./experiments/"       # Experiment directory
dir_out = "./outputs/"           # Output directory
dir_out_dat = "./outputs/data/"  # Cleaned data directory


# -------------------


# For logging
using Dates
include(dir_src * "global_variables.jl")  # Global variables
println("[$(Dates.format(now(),GV_DT))] Loading packages and modules")
using Base.Threads
using StaticArrays
using Statistics
using Distributions
using LinearAlgebra
using Random
using InvertedIndices
using TriangularIndices
using Distances
using StatsBase
using GLM
using DataFrames
using JLD2
using CSV
using HDF5
import H5Zblosc

include(dir_src * "util.jl")                            # Shared functions
include(dir_src * "groups.jl")                          # Groups and related functions
include(dir_src * "test.jl")                            # Data structures and functions for tests
include(dir_src * "kernel.jl")                          # Kernel functions
include(dir_src * "resampler.jl")                       # Resampling functions
include(dir_src * "maximum_mean_discrepancy.jl")        # MMD test
include(dir_src * "baseline_test.jl")                   # Baseline test
include(dir_src * "conditional_randomization_test.jl")  # Conditional randomization test
include(dir_src * "aggregate_test.jl")                  # Aggregate tests
include(dir_src * "experiment_helpers.jl")              # Experiment helper functions


# Read arguments
ARG_script = ARGS[1]

# Synthetic experiment: changing covariance + isolated non-equivariance
#      Args: [sample size n] [covariance p] [dimension d]
if ARG_script=="gaussian_equivariance_covariance.jl" || ARG_script=="gaussian_nonequivariance_covariance.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_p = parse(Float64, ARGS[3])
    ARG_d = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n), p=$(ARG_p), and d=$(ARG_d)")
    
# Synthetic experiment: approximate versus exact conditional sampling
#      Args: [sample size n] [covariance p] [dimension d]
elseif ARG_script == "gaussian_equivariance_truth.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_p = parse(Float64, ARGS[3])
    ARG_d = parse(Int, ARGS[4])
    ARG_exp = ARGS[5]
    if ARG_exp == "truth"
        println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) (truth) with n=$(ARG_n), p=$(ARG_p), and d=$(ARG_d)")
    else
        println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n), p=$(ARG_p), and d=$(ARG_d)")
    end
    
# Synthetic experiment: permutation
#      Args: [sample size n] [shift s] [dimension d]
elseif ARG_script == "gaussian_equivariance_permutation.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_s = parse(Float64, ARGS[3])
    ARG_d = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n), s=$(ARG_s), and d=$(ARG_d)")
    
# Synthetic experiment: number of randomizations
#      Args: [sample size n] [covariance p] [randomizations B]
elseif ARG_script == "gaussian_equivariance_resamples.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_p = parse(Float64, ARGS[3])
    ARG_B = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n) and p=$(ARG_p)")
    
# Synthetic experiment: non-equivariance in mean
#      Args: [proportion p] [shift s] [sample size n]
elseif ARG_script == "gaussian_equivariance_sensitivity.jl"
    ARG_p = parse(Float64, ARGS[2])
    ARG_s = parse(Float64, ARGS[3])
    ARG_n = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with p=$(ARG_p), s=$(ARG_s), and n=$(ARG_n)")
    
# MNIST experiment
#      Args: [data augmentation? {true,false}] [include digit 9? {true,false}]
elseif ARG_script == "MNIST_conditional_invariance.jl"
    ARG_aug = parse(Bool, ARGS[2])
    ARG_9 = parse(Bool, ARGS[3])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with aug=$(ARG_aug) and keep_9=$(ARG_9)")
    
# Invariance experiment
#      Args: [sample size n] [proportion p]
elseif ARG_script == "invariance.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_p = parse(Float64, ARGS[3])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n) and p=$(ARG_p)")
    
# Other experiments with no arguments (LHC, Lorentz)
else
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script)")
end


# Run experiment
t = @elapsed include(dir_exp * ARG_script)
println("[$(Dates.format(now(),GV_DT))] Experiment $(ARG_script) completed in $(ceil(Int,t)) seconds\n")