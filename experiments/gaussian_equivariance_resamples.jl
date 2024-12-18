## Testing for equivariance in R^d data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
Random.seed!(seed)


# Experiment parameters
d = 3


# Group and resampling parameters
rand_rotation_d = () -> rand_rotation(d)
G = Group(f_sample=rand_rotation_d, f_transform=rotate_d, f_inv_transform=inv_rotate_d,
          f_transform_Y=rotate_d, f_inv_transform_Y=inv_rotate_d, f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate)

RS = EquivariantResampler(ARG_B, G)


# Data generation
Σx = fill(0.75, d, d)
inds_diag = diagind(Σx)
Σx[inds_diag] .= 1
Px = MvNormal(zeros(d), Σx)

Σy = fill(ARG_p, d, d)
Σy[inds_diag] .= 1
sqrtΣy = cholesky(Σy).L

function sample_data(data::Data, Px::Distribution, sqrtΣy::AbstractMatrix{Float64})
    P = Normal(0, 1)
    mul!(data.y, sqrtΣy, rand!(P,data.x))  # data.x here is just to avoid mem alloc
    rand!(Px, data.x)
    data.y += data.x
end
f_sample_data = data -> sample_data(data, Px, sqrtΣy)


# Run experiment
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]
output_name = dir_out * "gauss_resamples_SO$(d)_N$(N)_n$(ARG_n)_p$(ARG_p)_B$(ARG_B)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=f_sample_data, N=N, n=ARG_n, dx=d, dy=d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=f_sample_data, N=N, n=ARG_n, dx=d, dy=d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)