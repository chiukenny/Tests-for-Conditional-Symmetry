## Invariance experiment
## ---------------------

Random.seed!(seed)

# Create group object
G = Group(f_sample=rand_θ, f_transform=rotate_2D, f_max_inv=max_inv_rotate)

# Create resampler object
RS = HaarTransformer(B, G)

# Initialize tests
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS,:x)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS,:x)))
    FUSE(GV_RN_FS, MMD(G,RS,:x))
    SK(GV_RN_SK, MMD(G,RS,:x))
]

# Initialize data sampling function
function sample_data(data::Data, n::Integer=ARG_n, p::Float64=ARG_p, d::Integer=2, Pz::Distribution=Normal(1,1), G::Group=G)
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
output_name = dir_out * "invariance_N$(N)_n$(ARG_n)_p$(ARG_p)_B$(B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, n=ARG_n, f_sample_data=sample_data, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, n=ARG_n, f_sample_data=sample_data, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)