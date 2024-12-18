## Testing for equivariance in R^d data


# Load experiment specific functions
# ----------------------------------

mutable struct TruthResampler <: AbstractResampler
    B::Int64                         # Number of resamples
    G::Group                         # Group to test on
    sqrtΣy::AbstractMatrix{Float64}  # Sqrt Y covariance matrix
    # Internal use
    data::Data  # Source data for reuse
    function TruthResampler(B::Int64, G::Group, sqrtΣy::AbstractMatrix{Float64}; data::Data=Data(0))
        return new(B, G, sqrtΣy, data)
    end
end
function initialize(RS::TruthResampler, data::Data)
    # Precompute the τY and maximal invariant distributions
    initialize(data, RS.G, :τy)
    initialize(data, RS.G, :M)
    RS.data = data
end
function resample(RS::TruthResampler, d::Int=ARG_d, n::Int=ARG_n)
    P = Normal(0, 1)
    z = @views rand(P,d,n) .+ RS.data.M
    RS_y = similar(RS.data.τy)
    @views @inbounds @threads for i in 1:n
        RS_y[:,i] = rotate_d(RS.data.τy[:,i], rep_inv_rotate(z[:,i]))
    end
    RS_data = Data(n, y=RS_y)
    return RS_data
end


# Start experiment
# ----------------

# Set the seed to ensure that simulated data are at least consistent irrespective of threads
Random.seed!(seed)


# Group parameters
G = Group(f_transform=rotate_d, f_inv_transform=inv_rotate_d, f_transform_Y=rotate_d, f_inv_transform_Y=inv_rotate_d,
          f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate)


# Data generation
Pr = Chi(1)

# p=0:equivariant; p=1:not equivariant
Σy = fill(min(0.99,ARG_p), ARG_d, ARG_d)
Σy[diagind(Σy)] .= 1
sqrtΣy = cholesky(Σy).L

function sample_data(data::Data, Pr::Distribution, sqrtΣy::AbstractMatrix{Float64}, d::Int=ARG_d)
    P = Normal(0, 1)
    mul!(data.y, sqrtΣy, rand!(P,data.x))  # data.x here is just to avoid mem alloc
    @views rand!(Pr, data.x[1,:])
    data.x[2:end,:] .= 0
    z = @views rand(P,d,data.n) .+ data.x[1:1,:]
    @views @inbounds @threads for i in 1:data.n
        data.x[:,i] = rotate_d(data.x[:,i], rep_inv_rotate(z[:,i]))
    end
    data.y += data.x 
end
f_sample_data = data -> sample_data(data, Pr, sqrtΣy)


# Resampling parameters
RS = ARG_exp=="truth" ? TruthResampler(B,G,sqrtΣy) : EquivariantResampler(B,G)


# Run experiment
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]
prefix = ARG_exp=="none" ? "" : "_$(ARG_exp)"
output_name = dir_out * "gauss_truth$(prefix)_N$(N)_B$(B)_n$(ARG_n)_p$(ARG_p)_d$(ARG_d)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=f_sample_data, N=N, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=f_sample_data, N=N, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)