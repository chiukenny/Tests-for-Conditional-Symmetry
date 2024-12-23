## Synthetic experiment: permutation
## ---------------------------------

Random.seed!(seed)

# Create group object
G = Group(f_transform=permute, f_inv_transform=inv_permute, f_transform_Y=permute, f_inv_transform_Y=inv_permute,
          f_max_inv=max_inv_permute, f_rep_inv=rep_inv_permute)

# Create resampler object
RS = EquivariantResampler(B, G)

# Initialize tests
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]

# Initialize data sampling function
Σx = fill(0.75, ARG_d, ARG_d)
Σx[diagind(Σx)] .= 1
Px = MvNormal(zeros(ARG_d), Σx)
function sample_data(data::Data, s::Float64, d::Integer, Px::Distribution)
    P = Normal(0, 1)
    rand!(P, data.y)
    rand!(Px, data.x)
    if s > 0
        u = s * unit_vector(d)
        @views @inbounds @threads for i in 1:data.n
            data.y[:,i] += findmax(data.x[:,i])[2]==1 ? data.x[:,i]+u : data.x[:,i]
        end
    else
        data.y += data.x
    end
end
f_sample_data = data -> sample_data(data,ARG_s,ARG_d,Px)

# Run experiment
output_name = dir_out * "gauss_perm_S$(ARG_d)_N$(N)_B$(B)_n$(ARG_n)_s$(ARG_s)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=f_sample_data, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=f_sample_data, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)