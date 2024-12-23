## USAGE
## -----
##
## From the command line, execute the following command to prepare data for
## the application demos and experiments:
##     julia preprocess_data.jl { LHC_DEMO | TQT_DEMO | LHC | TQT | ALL}
##
## Requirements:
##     - LHC demo/experiment: 'events_anomalydetection_v2.features.h5' in dir_dat from https://zenodo.org/record/6466204
##     - Lorentz demo/experiment: 'test.h5' in dir_dat from https://zenodo.org/record/2603256
##
## Edit the 'Experiment settings' and 'Paths' variables below as necessary


# Paths
dir_src = "./src/"               # Source code directory
dir_dat = "./data/"              # Raw data directory
dir_out = "./outputs/"           # Output directory
dir_out_dat = "./outputs/data/"  # Cleaned data directory


using Dates
include(dir_src * "global_variables.jl") # Global variables
println("[$(Dates.format(now(),GV_DT))] Loading packages and modules")
using StatsBase
using Random
using HDF5
import H5Zblosc


if ARGS[1]=="LHC_DEMO" || ARGS[1]=="ALL"
    # Create the data used in the LHC demo
    Random.seed!(1)

    # Read the data
    fid = h5open(dir_dat*"events_anomalydetection_v2.features.h5", "r")
    LHC = read(fid["df/block0_values"])
    n_LHC = size(LHC)[2]

    # Save the data
    n = 100
    inds = sample(1:n_LHC, n, replace=false)
    LHC_dat = LHC[[1,2,8,9], inds]
    fid2 = h5open(dir_out_dat*"demo_LHC.h5", "w")
    write(fid2, "data", LHC_dat)

    close(fid)
    close(fid2)
    println("[$(Dates.format(now(),GV_DT))] Created LHC demo data")
end

if ARGS[1]=="LHC" || ARGS[1]=="ALL"
    # Create the data used in the LHC experiments
    Random.seed!(1)

    # Read and process the data
    fid = h5open(dir_dat*"events_anomalydetection_v2.features.h5", "r")
    LHC = read(fid["df/block0_values"])
    n_LHC = size(LHC)[2]
    LHC_dat = LHC[[1,2,8,9], :]

    # Save the data
    fid2 = h5open(dir_out_dat*"LHC.h5", "w")
    write(fid2, "data", LHC_dat)

    close(fid)
    close(fid2)
    println("[$(Dates.format(now(),GV_DT))] Created LHC data")
end

if ARGS[1]=="TQT_DEMO" || ARGS[1]=="ALL"
    # Create the data used in the Lorentz demo
    Random.seed!(1)

    # Read the data
    fid = h5open(dir_dat*"test.h5", "r")
    n_TQT = read(fid["table/table"]["NROWS"])
    raw_TQT = fid["table/table"]

    # Extract the 4-momentum of the two leading constituents of a sample of jets
    n = 100
    inds = sample(1:n_TQT, n, replace=false)
    n_consts = 2
    d = 4 * n_consts
    TQT = Matrix{Float64}(undef, d, n)
    for i in 1:n
        TQT[:,i] = raw_TQT[inds[i]][:values_block_0][1:d]
    end

    # Save the data
    fid2 = h5open(dir_out_dat*"demo_TQT.h5", "w")
    write(fid2, "data", TQT)

    close(fid)
    close(fid2)
    println("[$(Dates.format(now(),GV_DT))] Created Lorentz demo data")
end

if ARGS[1]=="TQT" || ARGS[1]=="ALL"
    # Create the data used in the Lorentz experiments
    Random.seed!(1)

    # Read the data
    fid = h5open(dir_dat*"test.h5", "r")
    n_TQT = read(fid["table/table"]["NROWS"])
    raw_TQT = fid["table/table"]

    # Extract the 4-momentum of the two leading constituents in each jet
    n_consts = 2
    d = 4 * n_consts
    TQT = Matrix{Float64}(undef, d, n_TQT)
    for i in 1:n_TQT
        TQT[:,i] = raw_TQT[i][:values_block_0][1:d]
    end

    # Save the data
    fid2 = h5open(dir_out_dat*"TQT.h5", "w")
    write(fid2, "data", TQT)

    close(fid)
    close(fid2)
    println("[$(Dates.format(now(),GV_DT))] Created Lorentz data")
end