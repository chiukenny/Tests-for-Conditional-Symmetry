## USAGE
## -----
##
## From the command line, execute the following command to run ./experiments/<script>.jl:
##     julia --threads 1 run_experiments.jl <script>.jl
##
## For the experiment ./experiments/equivariance_SO_gaussian_loc_seq.jl, run with two arguments p and s
##     julia --threads 1 run_experiments.jl equivariance_SO_gaussian_loc_seq.jl [p] [s]
##
## For the experiment ./experiments/equivariance_SO_gaussian_cov_B_seq.jl, run with two arguments n and p
##     julia --threads 1 run_experiments.jl equivariance_SO_gaussian_cov_B_seq.jl [n] [p]
##
## Edit the 'Experiment settings' and 'Paths' variables below as necessary


# Experiment settings
use_raw_data = true  # Start from raw data for LHC/TQT? Only need to do so if running for the first time

# Paths
dir_src = "./src/"               # Source code directory
dir_dat = "./data/"              # Raw data directory
dir_exp = "./experiments/"       # Experiment directory
dir_out = "./outputs/"           # Output directory
dir_out_dat = "./outputs/data/"  # Cleaned data directory


# -------------------


# For logging
using Dates
include(dir_src * "global_variables.jl") # Global variables

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

include(dir_src * "util.jl")                             # Shared functions
include(dir_src * "groups.jl")                           # Groups and related functions
include(dir_src * "test.jl")                             # Data structures and functions for tests
include(dir_src * "kernel.jl")                           # Kernel functions
include(dir_src * "resampler.jl")                        # Resampling functions
include(dir_src * "maximum_mean_discrepancy.jl")         # MMD test
include(dir_src * "baseline_test.jl")                    # Baseline test
include(dir_src * "conditional_randomization_test.jl")   # Conditional randomization test
include(dir_src * "aggregate_test.jl")                   # Aggregate tests
include(dir_src * "experiment_helpers.jl")               # Experiment helper functions


# Clean and save real data
if use_raw_data
    println("[$(Dates.format(now(),GV_DT))] Cleaning data")
    include(dir_dat * "LHC_data.jl")
    include(dir_dat * "TQT_data.jl")
end


# Run experiment(s)
ARG_script = ARGS[1]
if ARG_script=="gaussian_equivariance_covariance.jl" || ARG_script=="gaussian_nonequivariance_covariance.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_p = parse(Float64, ARGS[3])
    ARG_d = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n), p=$(ARG_p), and d=$(ARG_d)")
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
elseif ARG_script == "gaussian_equivariance_permutation.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_s = parse(Float64, ARGS[3])
    ARG_d = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n), s=$(ARG_s), and d=$(ARG_d)")
elseif ARG_script == "gaussian_equivariance_resamples.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_p = parse(Float64, ARGS[3])
    ARG_B = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n) and p=$(ARG_p)")
elseif ARG_script == "gaussian_equivariance_sensitivity.jl"
    ARG_p = parse(Float64, ARGS[2])
    ARG_s = parse(Float64, ARGS[3])
    ARG_n = parse(Int, ARGS[4])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with p=$(ARG_p), s=$(ARG_s), and n=$(ARG_n)")
elseif ARG_script == "MNIST_conditional_invariance.jl"
    ARG_aug = parse(Bool, ARGS[2])
    ARG_9 = parse(Bool, ARGS[3])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with aug=$(ARG_aug) and keep_9=$(ARG_9)")
elseif ARG_script == "invariance.jl"
    ARG_n = parse(Int, ARGS[2])
    ARG_p = parse(Float64, ARGS[3])
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script) with n=$(ARG_n) and p=$(ARG_p)")
else
    println("[$(Dates.format(now(),GV_DT))] Running $(ARG_script)")
end
t = @elapsed include(dir_exp * ARG_script)
println("[$(Dates.format(now(),GV_DT))] Experiment $(ARG_script) completed in $(ceil(Int,t)) seconds\n")