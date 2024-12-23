## Synthetic experiment: approximate versus exact conditional sampling
## -------------------------------------------------------------------


# Create resampler that uses exact conditional resampling
# -------------------------------------------------------

mutable struct TruthResampler <: AbstractResampler
    B::Int64  # Number of resamples
    G::Group  # Group to test on
    # Internal use
    data::Data  # Source data for reuse
    function TruthResampler(B::Int64, G::Group; data::Data=Data(0))
        return new(B, G, data)
    end
end

function initialize(RS::TruthResampler, data::Data)
    # Precompute the τY and maximal invariant distributions
    initialize(data, RS.G, :τy)
    initialize(data, RS.G, :M)
    RS.data = data
end

function resample(RS::TruthResampler, d::Int=ARG_d, n::Int=ARG_n)
    # Randomize via exact conditional resampling
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

Random.seed!(seed)

# Create group object
G = Group(f_transform=rotate_d, f_inv_transform=inv_rotate_d, f_transform_Y=rotate_d, f_inv_transform_Y=inv_rotate_d,
          f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate)

# Create resampler object
RS = ARG_exp=="truth" ? TruthResampler(B,G) : EquivariantResampler(B,G)

# Initialize tests
tests = [
    FUSE(GV_BS_FS, Baseline(MMD2S(G,RS)))
    SK(GV_BS_SK, Baseline(MMD2S(G,RS)))
    FUSE(GV_RN_FS, CR(G,RS))
    SK(GV_RN_SK, CR(G,RS))
]

# Initialize data sampling function
Σy = fill(min(0.99,ARG_p), ARG_d, ARG_d)
Σy[diagind(Σy)] .= 1
sqrtΣy = cholesky(Σy).L
function sample_data(data::Data, sqrtΣy::AbstractMatrix{Float64}, d::Int=ARG_d)
    P = Normal(0, 1)
    mul!(data.y, sqrtΣy, rand!(P,data.x))  # data.x used here to avoid mem alloc
    @views rand!(Chi(1), data.x[1,:])
    data.x[2:end,:] .= 0
    z = @views rand(P,d,data.n) .+ data.x[1:1,:]
    @views @inbounds @threads for i in 1:data.n
        data.x[:,i] = rotate_d(data.x[:,i], rep_inv_rotate(z[:,i]))
    end
    data.y += data.x 
end
f_sample_data = data -> sample_data(data,sqrtΣy)

# Run experiment
prefix = ARG_exp=="none" ? "" : "_$(ARG_exp)"
output_name = dir_out * "gauss_truth$(prefix)_N$(N)_B$(B)_n$(ARG_n)_p$(ARG_p)_d$(ARG_d)"
output_file = open(output_name*".txt", "w")
results = []
push!(results, run_tests(output_file, "indep", tests, f_sample_data=f_sample_data, N=N, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed))
push!(results, run_tests(output_file, "reuse", tests, f_sample_data=f_sample_data, N=N, n=ARG_n, dx=ARG_d, dy=ARG_d, seed=seed, same_ref=true))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)