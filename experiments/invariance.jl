## Testing for invariance in LHC data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
Random.seed!(seed)


# Experiment parameters
d = 2


# Group and resampling parameters
G = Group(f_sample=rand_θ, f_transform=rotate_2D, f_max_inv=max_inv_rotate)
RS = HaarTransformer(B, G)


# Data generation parameters
Pz = Normal(1, 1)
function sample_data(data::Data, n::Integer=ARG_n, p::Float64=ARG_p, d::Integer=d, Pz::Distribution=Pz, G::Group=G)
    x = zeros(d, n)
    x[1,:] = rand(Pz, n)
    g = Vector{Float64}(undef, n)
    @threads for i in 1:n
        g[i] = rand_θ()
        while g[i] > p
            g[i] = rand_θ()
        end
    end
    data.x = transform_each(G, x, g)
end


# Run experiment
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS,:x)))
    FUSE(GV_RN_FS, MMD(G,RS,:x))
    SK(GV_RN_SK, MMD(G,RS,:x))
]
output_name = dir_out * "invariance_N$(N)_n$(ARG_n)_p$(ARG_p)_B$(B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, N=N, n=ARG_n, f_sample_data=sample_data, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, N=N, n=ARG_n, f_sample_data=sample_data, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)