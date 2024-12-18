## Testing for equivariance in R^d data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
Random.seed!(seed)


# Experiment parameters
d = 3


# Group and resampling parameters
rand_rotation_d = () -> rand_rotation(d)
G = Group(f_sample=rand_rotation_d, f_transform=rotate_d, f_inv_transform=inv_rotate_d,
          f_transform_Y=rotate_d, f_inv_transform_Y=inv_rotate_d, f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate)

RS = EquivariantResampler(B, G)


# Data generation
Σx = fill(0.75, d, d)
Σx[diagind(Σx)] .= 1
Px = MvNormal(zeros(d), Σx)

# p=0:equivariant; p=π:not equivariant
# s=0:equivariant; s=1:not equivariant
function sample_xy(data::Data, p::Float64, s::Float64, d::Integer, Px::Distribution)
    P = Normal(0, 1)
    rand!(P, data.y)
    rand!(Px, data.x)
    s_vec = fill(s, d)
    @views @inbounds @threads for i in 1:data.n
        θ = rep_inv_rotate_2D(data.x[1:2,i])
        data.y[:,i] += θ < p ? data.x[:,i] + s_vec : data.x[:,i]
    end
end
sample_data = data -> sample_xy(data, ARG_p, ARG_s, d, Px)


# Run experiment
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]
output_name = dir_out * "gauss_sens_SO$(d)_N$(N)_B$(B)_p$(ARG_p)_s$(ARG_s)_n$(ARG_n)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=sample_data, N=N, n=ARG_n, dx=d, dy=d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=sample_data, N=N, n=ARG_n, dx=d, dy=d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)