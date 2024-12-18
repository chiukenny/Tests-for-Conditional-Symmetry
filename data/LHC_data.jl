## Requires 'events_anomalydetection_v2.features.h5' saved in './data/'
## The dataset can be downloaded from the following link:
## https://zenodo.org/record/6466204


# Seed for reproducibility
Random.seed!(1)

# Read and save data
fid = h5open(dir_dat*"events_anomalydetection_v2.features.h5", "r")
LHC = read(fid["df/block0_values"])
n_LHC = size(LHC)[2]
LHC_dat = LHC[[1,2,8,9], :]

# Save the data
fid2 = h5open(dir_out_dat*"LHC.h5", "w")
write(fid2, "data", LHC_dat)

close(fid)
close(fid2)