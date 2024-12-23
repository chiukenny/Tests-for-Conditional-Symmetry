## Synthetic experiment: non-equivariance in mean
## ----------------------------------------------

Random.seed!(seed)

# Create group object
G = Group(f_transform=rotate_d, f_inv_transform=inv_rotate_d, f_transform_Y=rotate_d, f_inv_transform_Y=inv_rotate_d,
          f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate)

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
d = 3
Σx = fill(0.75, d, d)
Σx[diagind(Σx)] .= 1
Px = MvNormal(zeros(d), Σx)
function sample_data(data::Data, p::Float64, s::Float64, d::Integer, Px::Distribution)
    P = Normal(0, 1)
    rand!(P, data.y)
    rand!(Px, data.x)
    s_vec = fill(s, d)
    @views @inbounds @threads for i in 1:data.n
        θ = rep_inv_rotate_2D(data.x[1:2,i])
        data.y[:,i] += θ < p ? data.x[:,i]+s_vec : data.x[:,i]
    end
end
f_sample_data = data -> sample_data(data,ARG_p,ARG_s,d,Px)

# Run experiment
output_name = dir_out * "gauss_sens_SO$(d)_N$(N)_B$(B)_p$(ARG_p)_s$(ARG_s)_n$(ARG_n)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=f_sample_data, n=ARG_n, dx=d, dy=d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=f_sample_data, n=ARG_n, dx=d, dy=d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)