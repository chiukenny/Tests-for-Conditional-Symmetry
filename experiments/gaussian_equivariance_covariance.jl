## Synthetic experiment: changing covariance
## -----------------------------------------

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
Σx = fill(0.75, ARG_d, ARG_d)
inds_diag = diagind(Σx)
Σx[inds_diag] .= 1
Px = MvNormal(zeros(ARG_d), Σx)
Σy = fill(min(0.99,ARG_p), ARG_d, ARG_d)
Σy[inds_diag] .= 1
sqrtΣy = cholesky(Σy).L
function sample_data(data::Data, Px::Distribution, sqrtΣy::AbstractMatrix{Float64})
    P = Normal(0, 1)
    mul!(data.y, sqrtΣy, rand!(P,data.x))  # data.x used here to avoid mem alloc
    rand!(Px, data.x)
    data.y += data.x
end
f_sample_data = data -> sample_data(data,Px,sqrtΣy)

# Run experiment
output_name = dir_out * "gauss_cov_SO$(ARG_d)_N$(N)_B$(B)_n$(ARG_n)_p$(ARG_p)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=f_sample_data, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=f_sample_data, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)