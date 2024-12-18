## Resampling methods for tests that require resampling/bootstrapping


using Base.Threads
using Random
using TriangularIndices
using StatsBase
include("util.jl")
include("groups.jl")
include("kernel.jl")


# Shared functions
# ----------------

abstract type AbstractResampler end

# Overload this as necessary
function initialize(RS::AbstractResampler, data::Data) end

# Estimates the p-value given the test statistic
function estimate_pvalue(test::AbstractTest, test_stat::Float64, same_ref::Bool)
    RS = get_resampler(test)
    B = RS.B
    bvals = Vector{Bool}(undef, B)
    @inbounds @threads for b in eachindex(bvals)
        bdata = resample(RS)
        bvals[b] = (same_ref ? null_test_statistic(test,bdata) : test_statistic(test,bdata)) > test_stat
    end
    return (1+sum(bvals)) / (1+B)
end


# Permutation resampling
# ----------------------

mutable struct Permuter <: AbstractResampler
    B::UInt16     # Number of resamples
    paired::Bool  # Paired samples?
    # Internal use
    data::Data  # Stored data
    function Permuter(B::Integer, paired::Bool=false, data::Data=Data(0))
        return new(B, paired, data)
    end
end

# Initializes the permuter
function initialize(RS::Permuter, data::Data)
    RS.data = data
end

# Permutes the sample
function resample(RS::Permuter)
    n = RS.data.n
    RS_data = Data(n)
    x = RS.data.x
    if RS.paired
        n2 = Int(n / 2)
        # Swap pairs randomly
        swaps = rand(Bernoulli(0.5), n2)
        RS_data.x = similar(x)
        @views @inbounds @threads for i in 1:n2
            RS_data.x[:,i] = swaps[i] ? x[:,n2+i] : x[:,i]
            RS_data.x[:,n2+i] = swaps[i] ? x[:,i] : x[:,n2+i]
        end
    else
        RS_data.x = x[:, randperm(n)]
    end
    return RS_data
end


# Haar transformation resampling (for invariance)
# -----------------------------------------------

mutable struct HaarTransformer <: AbstractResampler
    B::UInt16  # Number of resamples
    G::Group   # Group
    # Internal use
    data::Data  # Stored data
    function HaarTransformer(B::Integer, G::Group, data::Data=Data(0))
        return new(B, G, data)
    end
end

# Initializes the Haar transformer
function initialize(RS::HaarTransformer, data::Data)
    RS.data = data
end

# Applies a random transformation to each point in the sample
function resample(RS::HaarTransformer)
    return Data(RS.data.n, x=transform_all(RS.G,RS.data.x))
end


# Conditional randomization resampling (for equivariance)
# -------------------------------------------------------

mutable struct EquivariantResampler <: AbstractResampler
    B::UInt16  # Number of resamples
    G::Group   # Group
    # Internal parameters
    data::Data  # Source data
    function EquivariantResampler(B::Integer, G::Group; data=Data(0))
        return new(B, G, data)
    end
end

# Initializes the conditional randomization procedure
function initialize(RS::EquivariantResampler, data::Data)
    RS.data = data
    
    # Precompute conditional probabilities
    initialize(data, RS.G, :M)
    initialize(data, RS.G, :probs)
    
    # Precompute the G and Y distributions
    initialize(data, RS.G, :g)
    initialize(data, RS.G, :τy)
end

# Resamples via conditional randomization
function resample(RS::EquivariantResampler)
    data = RS.data
    y = similar(data.y)
    inds = 1:data.n
    @views @inbounds @threads for i in inds
        # Sample with probability proportional to orbit representative kernel distance
        j = sample(inds, data.probs[i])
        y[:,i] = RS.G.f_transform_Y(data.τy[:,i] , data.g[j])
    end
    RS_data = Data(data.n, y=y)
    return RS_data
end