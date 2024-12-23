## LHC experiment: tests for joint invariance
## ------------------------------------------

Random.seed!(seed)

# Set experimental parameters
n = 25

# Create group objects
G = Group(f_sample=rand_θ1_θ2, f_transform=rotate_θ1_θ2, f_max_inv=max_inv_θ1_θ2)

rand_θ1_θ2_dep = () -> rand_θ1_θ2(false)
max_inv_θ1_θ2_dep = (x) -> max_inv_θ1_θ2(x,false)
G_ind = Group(f_sample=rand_θ1_θ2_dep, f_transform=rotate_θ1_θ2, f_max_inv=max_inv_θ1_θ2_dep)

rand_rotation_SO4 = () -> rand_rotation(4)
G_SO4 = Group(f_sample=rand_rotation_SO4, f_transform=rotate_d, f_max_inv=max_inv_rotate)

# Create resampler objects
RS = HaarTransformer(B, G)
RS_ind = HaarTransformer(B, G_ind)
RS_SO4 = HaarTransformer(B, G_SO4)

# Initialize tests
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS,:x)))
    FUSE(GV_RN_FS, MMD(G,RS,:x))
    SK(GV_RN_SK, MMD(G,RS,:x))
]
tests_ind = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G_ind,RS_ind,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G_ind,RS_ind,:x)))
    FUSE(GV_RN_FS, MMD(G_ind,RS_ind,:x))
    SK(GV_RN_SK, MMD(G_ind,RS_ind,:x))
]
tests_SO4 = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G_SO4,RS_SO4,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G_SO4,RS_SO4,:x)))
    FUSE(GV_RN_FS, MMD(G_SO4,RS_SO4,:x))
    SK(GV_RN_SK, MMD(G_SO4,RS_SO4,:x))
]

# Read in data
fid = h5open(dir_out_dat*"LHC.h5", "r")
LHC = copy(read(fid["data"]))
n_LHC = size(LHC, 2)
close(fid)

# Initialize data sampling function
function sample_data(data::Data, lhc::AbstractMatrix{Float64}=LHC, n_lhc::Integer=n_LHC)
    data.x = lhc[:, sample(1:n_lhc,data.n,replace=false)]
end

# Run experiment
output_name = dir_out * "LHC_invariance_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep_H0", tests, f_sample_data=sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1_ind", tests_ind, f_sample_data=sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1_SO4", tests_SO4, f_sample_data=sample_data, seed=seed))
push!(results, run_tests(output_file, "reuse_H0", tests, f_sample_data=sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1_ind", tests_ind, f_sample_data=sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1_SO4", tests_SO4, f_sample_data=sample_data, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)