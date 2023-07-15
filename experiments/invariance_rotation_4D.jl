## Testing for SO(4) invariance in R^4 data


# Set the seed for reproducibility
Random.seed!(1)


# Experiment parameters
N = 1000
n = 200
d = 4
α = 0.05


# Resampling parameters
B = 200
RS_G = Resampler(B=B, f_sampler=transform_sampler)
RS_sub = Resampler(B=B, f_sampler=subsampler)


# Group parameters
M = 2
rand_rotation_d = () -> rand_rotation(d)
GS = GroupSampler(M, f_sample=rand_rotation_d, f_transform=rotate_d)


# Data generation parameters
σ = 1

μ0 = zeros(d)
P0 = MvNormal(μ0, σ)
H0_data = n -> sample_data(n, P0)

μ1 = zeros(d)
μ1[1] = 0.4
P1 = MvNormal(μ1, σ)
H1_data = n -> sample_data(n, P1)


# Output name
output_name = dir_out * "invariance_rotation_d$(d)_N$(N)_n$(n)_M$(M)_B$(B)"
output_file = open(output_name*".txt", "w")


# Run experiment
tests = [
    Transform2S(GV_2S, MMD(GS=GS,RS=RS_sub))
    MMD(GV_MMD, GS=GS, RS=RS_G)
    NMMD(GV_NMMD, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
    CW(GV_CW, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
]

results = []
push!(results, run_tests(output_file, "H0", tests, f_sample_data=H0_data, f_sample_tr_data=H0_data, N=N, n=n, α=α))
push!(results, run_tests(output_file, "H1", tests, f_sample_data=H1_data, f_sample_tr_data=H1_data, N=N, n=n, α=α))
estimate_power(output_file, "H1", tests, f_sample_data=H1_data, f_sample_tr_data=H1_data, N=N, n=n, α=α)
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)