## Implementation of our conditional randomization test for equivariance


using Distributions
using LinearAlgebra
using StatsBase
using Base.Threads
include("util.jl")
include("kernel.jl")
include("test.jl")


mutable struct CR <: AbstractTest
    name::String           # Test name for outputs
    G::Group               # Group being tested
    RS::AbstractResampler  # Resampling method
    K::AbstractKernel      # Kernel matrix
    # Internal use
    RS_data::Data          # Pre-sampled data shared across MMD computations
    mmd1::Vector{Float64}  # MMD contributions from pre-sampled y
    function CR(G::Group, RS::AbstractResampler; K=GaussianKernel(), RS_data=Data(0), mmd1=Vector{Float64}(undef,0))
        return new("", G, RS, K, RS_data, mmd1)
    end
    function CR(name::String, G::Group, RS::AbstractResampler; K=GaussianKernel(), RS_data=Data(0), mmd1=Vector{Float64}(undef,0))
        return new(name, G, RS, K, RS_data, mmd1)
    end
end


# Initializes the test
function initialize(test::CR, data::Data)
    # Initialize the resampler
    initialize(test.RS, data)
    
    # Precomputes the second sample and its MMD contribution for reuse across randomizations
    test.RS_data = resample(test.RS)
    K = test.K
    test.mmd1 = [ compute_Umean_K_1_x(K, test.RS_data.y) ]
    return data
end


# Computes the MMD test statistic for equivariance
function test_statistic(test::CR, data::Data)
    # Generate reference data
    RS = get_resampler(test)
    RS_data = resample(RS)
    
    # Compute the MMD
    mmds = zeros(MVector{3})
    @inbounds @threads for m in 1:3
        K = test.K
        if m == 1
            mmds[1] = compute_Umean_K_1_x(K, data.y)
        elseif m == 2
            mmds[2] = compute_Umean_K_1_x(K, RS_data.y)
        else
            mmds[3] = -2*compute_Vmean_K_x(K, data.y, RS_data.y)
        end
    end
    return sum(mmds)
end
function null_test_statistic(test::CR, data::Data)
    # Compute the MMD
    mmds = zeros(MVector{2})
    @inbounds @threads for m in 1:2
        K = test.K
        mmds[m] = m==1 ? compute_Umean_K_1_x(K,data.y) : -2*compute_Vmean_K_x(K,data.y,test.RS_data.y)
    end
    return sum(mmds) + test.mmd1[1]
end


# Computes the MMD test statistic for equivariance for a set of kernels and bandwidths
function test_statistic(test::CR, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    # Generate reference data
    RS = get_resampler(test)
    RS_data = resample(RS)
    
    n = data.n
    n_k = length(kernels[1][2])
    mmds = Vector{Float64}(undef, length(kernels)*n_k)
    dists = Matrix{Any}(undef, n, n)
    RS_dists = Matrix{Any}(undef, n, n)
    cross_dists = Matrix{Any}(undef, n, n)
    for (j,(k,ks)) in collect(enumerate(kernels))
        # Compute MMDs for a set of bandwidths for one type of kernel
        ind = (j-1) * n_k
        compute_distances!(dists, k, data.y)
        compute_distances!(RS_dists, k, RS_data.y)
        compute_distances!(cross_dists, k, data.y, RS_data.y)
        @inbounds @threads for i in 1:n_k
            mmds[ind+i] = compute_Umean_K_1_d(ks[i],dists) + compute_Umean_K_1_d(ks[i],RS_dists) - 2*compute_Vmean_K_d(ks[i],cross_dists)
        end
    end
    return mmds
end
function null_test_statistic(test::CR, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    n = data.n
    n_k = length(kernels[1][2])
    mmds = Vector{Float64}(undef, length(kernels)*n_k)
    dists = Matrix{Any}(undef, n, n)
    cross_dists = Matrix{Any}(undef, n, n)
    for (j,(k,ks)) in collect(enumerate(kernels))
        # Compute MMDs for a set of bandwidths for one type of kernel
        ind = (j-1) * n_k
        compute_distances!(dists, k, data.y)
        compute_distances!(cross_dists, k, data.y, test.RS_data.y)
        @inbounds @threads for i in 1:n_k
            mmds[ind+i] = compute_Umean_K_1_d(ks[i],dists) + test.mmd1[ind+i] - 2*compute_Vmean_K_d(ks[i],cross_dists)
        end
    end
    return mmds
end