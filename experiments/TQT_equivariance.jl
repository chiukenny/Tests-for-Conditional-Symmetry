## Testing for Lorentz equivariance in top quark data with two leading constituents


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
Random.seed!(seed)


# Read in data
# Context: https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data
fid = h5open(dir_out_dat*"TQT.h5", "r")
TQT = read(fid["data"])
close(fid)
n_TQT = size(TQT, 2)


# Experiment parameters
n = 100


# Group and resampling parameters
G = Group(f_transform=lorentz_transform, f_transform_Y=lorentz_transform, f_inv_transform_Y=lorentz_inv_transform,
          f_max_inv=max_inv_lorentz, f_rep_inv=rep_inv_lorentz)

RS = EquivariantResampler(B, G)


# Data generation parameters
function sample_data(data::Data, H1::Bool=false, tqt::AbstractMatrix{Float64}=TQT, n_tqt::Integer=n_TQT)
    # Subsample data
    inds = sample(1:n_tqt, data.n, replace=false)
    copy!(data.x, @views tqt[1:4,inds])
    if H1
        data.y = tqt[5:8, inds[randperm(data.n)]]
    else
        copy!(data.y, @views tqt[5:8,inds])
    end
end
sample_data_H1 = data -> sample_data(data, true)

# Run experiment
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]
output_name = dir_out * "TQT_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep_H0", tests, N=N, n=n, dx=4, dy=4, f_sample_data=sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1", tests, N=N, n=n, dx=4, dy=4, f_sample_data=sample_data_H1, seed=seed))
push!(results, run_tests(output_file, "reuse_H0", tests, N=N, n=n, dx=4, dy=4, f_sample_data=sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1", tests, N=N, n=n, dx=4, dy=4, f_sample_data=sample_data_H1, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)