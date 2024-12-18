## Testing for invariance in LHC data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
Random.seed!(seed)


# Read in data
fid = h5open(dir_out_dat*"LHC.h5", "r")
LHC = copy(read(fid["data"]))
close(fid)
n_LHC = size(LHC, 2)


# Experiment parameters
n = 25
d = 4


# Group and resampling parameters
G = Group(f_sample=rand_θ1_θ2, f_transform=rotate_θ1_θ2, f_max_inv=max_inv_θ1_θ2)

rand_θ1_θ2_dep = () -> rand_θ1_θ2(false)
max_inv_θ1_θ2_dep = (x) -> max_inv_θ1_θ2(x, false)
G_ind = Group(f_sample=rand_θ1_θ2_dep, f_transform=rotate_θ1_θ2, f_max_inv=max_inv_θ1_θ2_dep)

rand_rotation_SO4 = () -> rand_rotation(4)
G_SO4 = Group(f_sample=rand_rotation_SO4, f_transform=rotate_d, f_max_inv=max_inv_rotate)

RS_G = HaarTransformer(B, G)
RS_G_ind = HaarTransformer(B, G_ind)
RS_G_SO4 = HaarTransformer(B, G_SO4)


# Data generation parameters
function LHC_sample_data(data::Data, lhc::AbstractMatrix{Float64}=LHC, n_lhc::Integer=n_LHC)
    # Subsample data
    data.x = lhc[:, sample(1:n_lhc,data.n,replace=false)]
end


# Output name
output_name = dir_out * "LHC_invariance_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")


# Run experiment
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS_G,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS_G,:x)))
    FUSE(GV_RN_FS, MMD(G,RS_G,:x))
    SK(GV_RN_SK, MMD(G,RS_G,:x))
]
tests_ind = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G_ind,RS_G_ind,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G_ind,RS_G_ind,:x)))
    FUSE(GV_RN_FS, MMD(G_ind,RS_G_ind,:x))
    SK(GV_RN_SK, MMD(G_ind,RS_G_ind,:x))
]
tests_SO4 = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G_SO4,RS_G_SO4,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G_SO4,RS_G_SO4,:x)))
    FUSE(GV_RN_FS, MMD(G_SO4,RS_G_SO4,:x))
    SK(GV_RN_SK, MMD(G_SO4,RS_G_SO4,:x))
]
results = []
push!(results, run_tests(output_file, "indep_H0", tests, N=N, n=n, f_sample_data=LHC_sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1_ind", tests_ind, N=N, n=n, f_sample_data=LHC_sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1_SO4", tests_SO4, N=N, n=n, f_sample_data=LHC_sample_data, seed=seed))
push!(results, run_tests(output_file, "reuse_H0", tests, N=N, n=n, f_sample_data=LHC_sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1_ind", tests_ind, N=N, n=n, f_sample_data=LHC_sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1_SO4", tests_SO4, N=N, n=n, f_sample_data=LHC_sample_data, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)