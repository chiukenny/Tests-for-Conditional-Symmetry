## LHC experiment: tests for equivariance
## --------------------------------------

Random.seed!(seed)

# Set experimental parameters
n = 100
d = 2

# Create group object
G = Group(f_transform=rotate_2D, f_inv_transform=inv_rotate_2D, f_transform_Y=rotate_2D, f_inv_transform_Y=inv_rotate_2D,
          f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate_2D)

# Create resampler object
RS = EquivariantResampler(B, G)

# Initialize tests
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]

# Read in data
fid = h5open(dir_out_dat*"LHC.h5", "r")
LHC = copy(read(fid["data"]))
n_LHC = size(LHC, 2)
close(fid)

# Initialize data sampling function
function sample_data(data::Data, H1::Bool=false, lhc::AbstractMatrix{Float64}=LHC, n_lhc::Integer=n_LHC)
    xy = lhc[:, sample(1:n_lhc,data.n,replace=false)]
    copy!(data.x, @views xy[1:2,:])
    copy!(data.y, @views xy[3:4,:])
    if H1
        data.y = data.y[:,randperm(data.n)]
    end
end
sample_data_H1 = data -> sample_data(data,true)

# Run experiment
output_name = dir_out * "LHC_equiv_rot_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep_H0", tests, dx=d, dy=d, f_sample_data=sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1", tests, dx=d, dy=d, f_sample_data=sample_data_H1, seed=seed))
push!(results, run_tests(output_file, "reuse_H0", tests, dx=d, dy=d, f_sample_data=sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1", tests, dx=d, dy=d, f_sample_data=sample_data_H1, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)