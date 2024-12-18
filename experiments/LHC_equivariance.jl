## Testing for equivariance in LHC data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
Random.seed!(seed)


# Read in data
fid = h5open(dir_out_dat*"LHC.h5", "r")
LHC = copy(read(fid["data"]))
close(fid)
n_LHC = size(LHC, 2)


# Experiment parameters
n = 100
d = 2


# Group and resampling parameters
G = Group(f_sample=rand_θ, f_transform=rotate_2D, f_inv_transform=inv_rotate_2D, f_transform_Y=rotate_2D, f_inv_transform_Y=inv_rotate_2D,
          f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate_2D)
G_trans = Group(f_transform=translate, f_inv_transform=inv_translate, f_transform_Y=translate, f_inv_transform_Y=inv_translate,
          f_max_inv=max_inv_translate, f_rep_inv=rep_inv_translate)

RS = EquivariantResampler(B, G)


# Data generation
function LHC_sample_data(data::Data, H1::Bool=false, lhc::AbstractMatrix{Float64}=LHC, n_lhc::Integer=n_LHC; pref=true)
    # Subsample data
    if pref
        @views @inbounds @threads for i in 1:data.n
            while true
                j = sample(1:n_lhc)
                θ = rep_inv_rotate_2D(lhc[1:2,j])
                if rand(Bernoulli(abs(θ-π)/π))
                    copy!(data.x[:,i], lhc[1:2,j])
                    copy!(data.y[:,i], lhc[3:4,j])
                    break
                end
            end
        end
    else
        xy = lhc[:, sample(1:n_lhc,data.n,replace=false)]
        copy!(data.x, @views xy[1:2,:])
        copy!(data.y, @views xy[3:4,:])
    end
    if H1
        data.y = data.y[:,randperm(data.n)]
    end
end
sample_data_H1 = data -> LHC_sample_data(data, true)


# Run experiment
println("[$(Dates.format(now(),GV_DT))] Running experiment")

# SO(2)-equivariance
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]
output_name = dir_out * "LHC_equiv_rot_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep_H0", tests, N=N, n=n, dx=d, dy=d, f_sample_data=LHC_sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1", tests, N=N, n=n, dx=d, dy=d, f_sample_data=sample_data_H1, seed=seed))
push!(results, run_tests(output_file, "reuse_H0", tests, N=N, n=n, dx=d, dy=d, f_sample_data=LHC_sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1", tests, N=N, n=n, dx=d, dy=d, f_sample_data=sample_data_H1, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)


# Translation-equivariance
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G_trans,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G_trans,RS)))
    FUSE(GV_RN_FS, CR(G_trans,RS))
    SK(GV_RN_SK, CR(G_trans,RS))
]
output_name = dir_out * "LHC_equiv_trans_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep_H0", tests, N=N, n=n, dx=d, dy=d, f_sample_data=LHC_sample_data, seed=seed))
push!(results, run_tests(output_file, "indep_H1", tests, N=N, n=n, dx=d, dy=d, f_sample_data=sample_data_H1, seed=seed))
push!(results, run_tests(output_file, "reuse_H0", tests, N=N, n=n, dx=d, dy=d, f_sample_data=LHC_sample_data, seed=seed, same_ref=true))
push!(results, run_tests(output_file, "reuse_H1", tests, N=N, n=n, dx=d, dy=d, f_sample_data=sample_data_H1, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)