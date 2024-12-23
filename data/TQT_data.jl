## Requires 'test.h5' saved in './data/'
## The dataset can be downloaded from the following link:
## https://zenodo.org/record/2603256

Random.seed!(1)

# Read the data
fid = h5open(dir_dat*"test.h5", "r")
n_TQT = read(fid["table/table"]["NROWS"])
raw_TQT = fid["table/table"]
close(fid)

# Extract the 4-momentum of the two leading constituents in each jet
n_consts = 2
dx = 4 * n_consts
TQT = zeros(dx, n_TQT)
for i in 1:n_TQT
    TQT[1:(dx-1),i] = raw_TQT[i][:values_block_0][1:(dx-1)]
end

# Save the data
fid2 = h5open(dir_out_dat*"TQT.h5", "w")
write(fid2, "data", TQT)
close(fid2)