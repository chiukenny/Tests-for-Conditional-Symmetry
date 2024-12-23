## Synthetic experiment: number of randomizations
## ----------------------------------------------

Random.seed!(seed)

# Create group object
G = Group(f_transform=rotate_d, f_inv_transform=inv_rotate_d, f_transform_Y=rotate_d, f_inv_transform_Y=inv_rotate_d,
          f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate)

# Create resampler object
RS = EquivariantResampler(ARG_B, G)

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
inds_diag = diagind(Σx)
Σx[inds_diag] .= 1
Px = MvNormal(zeros(d), Σx)
Σy = fill(ARG_p, d, d)
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
output_name = dir_out * "gauss_resamples_SO$(d)_N$(N)_n$(ARG_n)_p$(ARG_p)_B$(ARG_B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=f_sample_data, n=ARG_n, dx=d, dy=d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=f_sample_data, n=ARG_n, dx=d, dy=d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)