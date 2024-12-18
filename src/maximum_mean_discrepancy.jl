## Implementation of MMD-based tests


using Distributions
using Base.Threads
include("util.jl")
include("groups.jl")
include("test.jl")
include("kernel.jl")
include("resampler.jl")


abstract type AbstractMMD <: AbstractTest end


# Standard two-sample MMD
# -----------------------

mutable struct MMD2S <: AbstractMMD
    name::String           # Test name for outputs
    G::Group               # Group being tested
    RS::AbstractResampler  # Resampling method
    test_on::Symbol        # One of {:x, :y}
    K::AbstractKernel      # Kernel matrix
    # Internal use
    n1::UInt16             # Size of first sample
    function MMD2S(G::Group, RS::AbstractResampler, test_on::Symbol=:y; K=GaussianKernel(), n1=0)
        return new("", G, RS, test_on, K, n1)
    end
    function MMD2S(name::String, G::Group, RS::AbstractResampler, test_on::Symbol=:y; K=GaussianKernel(), n1=0)
        return new(name, G, RS, test_on, K, n1)
    end
end


# Initializes the test
function initialize(test::MMD2S, data::Data)
    # Initialize the resampler
    initialize(test.RS, data)
    
    # Make sure sample size is set properly
    if test.n1 == 0
        test.n1 = ceil(UInt16, data.n/2)
    elseif test.n1 > data.n
        error("Sample size n1 incorrectly set for MMD2S")
    end
    return data
end


# Computes the standard two-sample MMD (Gretton, 2012)
function test_statistic(test::MMD2S, data::Data)
    n1 = test.n1
    mmds = zeros(MVector{3})
    x = data.x
    @views @inbounds @threads for m in 1:3
        if m == 1
            mmds[1] = compute_Umean_K_1_x(test.K, x[:,1:n1])
        elseif m == 2
            mmds[2] = compute_Umean_K_1_x(test.K, x[:,(n1+1):end])
        else
            mmds[3] = -2 * compute_Vmean_K_x(test.K, x[:,1:n1], x[:,(n1+1):end])
        end
    end
    return sum(mmds)
end
function null_test_statistic(test::MMD2S, data::Data)
    return test_statistic(test, data)
end


# Computes the standard two-sample MMD for a set of kernels and bandwidths
function test_statistic(test::MMD2S, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    n1 = test.n1
    n2 = data.n - n1    
    x = data.x

    n_k = length(kernels[1][2])
    mmds = Vector{Float64}(undef, length(kernels)*n_k)
    x1_d = Matrix{Any}(undef, n1, n1)
    x2_d = Matrix{Any}(undef, n2, n2)
    x1_x2_d = Matrix{Any}(undef, n1, n2)
    @views @inbounds for (j,(k,ks)) in collect(enumerate(kernels))
        compute_distances!(x1_d, k, x[:,1:n1])
        compute_distances!(x2_d, k, x[:,(n1+1):end])
        compute_distances!(x1_x2_d, k, x[:,1:n1], x[:,(n1+1):end])
        
        # Compute MMDs for a set of bandwidths for one type of kernel
        ind = (j-1) * n_k
        @threads for i in 1:n_k
            mmds[ind+i] = compute_Umean_K_1_d(ks[i],x1_d) + compute_Umean_K_1_d(ks[i],x2_d) - 2*compute_Vmean_K_d(ks[i],x1_x2_d)
        end
    end
    return mmds
end
function null_test_statistic(test::MMD2S, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    return test_statistic(test, data, kernels)
end


# One-sample MMD for invariance
# -----------------------------

mutable struct MMD <: AbstractMMD
    name::String            # Test name for outputs
    G::Group                # Group being tested
    RS::AbstractResampler   # Resampling method
    test_on::Symbol         # One of {:x, :y}
    K::AbstractKernel       # Kernel matrix
    # Internal use
    g_actions::Matrix{Any}  # Pre-sampled group actions
    function MMD(G::Group, RS::AbstractResampler, test_on::Symbol=:y; K=GaussianKernel(), g_actions=Matrix{Any}(undef,0,0))
        return new("", G, RS, test_on, K, g_actions)
    end
    function MMD(name::String, G::Group, RS::AbstractResampler, test_on::Symbol=:y; K=GaussianKernel(), g_actions=Matrix{Any}(undef,0,0))
        return new(name, G, RS, test_on, K, g_actions)
    end
end


# Initializes the MMD test for invariance
function initialize(test::MMD, data::Data)
    # Initialize the resampler
    initialize(test.RS, data)
    
    # Pre-sample group actions for reuse across resamples
    g_actions = Matrix{Any}(undef, data.n, 2)
    @inbounds @threads for i in eachindex(g_actions)
        g_actions[i] = test.G.f_sample()
    end
    test.g_actions = g_actions
    return data
end


# Computes the MMD test statistic for invariance
function test_statistic(test::MMD, data::Data)
    dat = test.test_on==:x ? data.x : data.y
    
    # Pre-compute one set of the transformed observations
    gdat = transform_all(test.G, dat)
    
    # Compute the test statistic
    mmds = zeros(MVector{3})
    @views @inbounds @threads for m in 1:3
        if m == 1
            mmds[1] = compute_Umean_K_1_x(test.K, dat)
        elseif m == 2
            mmds[2] = compute_Umean_K_2_x(test.K, gdat, transform_all(test.G, dat))
        else
            mmds[3] = -2 * compute_Vmean_K_x(test.K, dat, gdat)
        end
    end
    return sum(mmds)
end
function null_test_statistic(test::MMD, data::Data)
    dat = test.test_on==:x ? data.x : data.y
    
    # Pre-compute one set of the transformed observations
    gdat = @views transform_each(test.G, dat, test.g_actions[:,1])
    
    # Compute the test statistic
    mmds = zeros(MVector{3})
    @views @inbounds @threads for m in 1:3
        if m == 1
            mmds[1] = compute_Umean_K_1_x(test.K, dat)
        elseif m == 2
            mmds[2] = compute_Umean_K_2_x(test.K, gdat, transform_each(test.G,dat,test.g_actions[:,2]))
        else
            mmds[3] = -2 * compute_Vmean_K_x(test.K, dat, gdat)
        end
    end
    return sum(mmds)
end


# Computes the MMD test for invariance for a set of kernels and bandwidths
function test_statistic(test::MMD, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    dat = test.test_on==:x ? data.x : data.y
    d, n = size(dat)
    
    # Pre-compute transformed data for reuse across kernels
    gdat = Array{Float64}(undef, d, n, 2)
    @views @inbounds @threads for i in 1:2
        gdat[:,:,i] = transform_all(test.G, dat)
    end
    
    n_k = length(kernels[1][2])
    mmds = Vector{Float64}(undef, length(kernels)*n_k)
    dat_d = Matrix{Any}(undef, n, n)
    gdat_d = Matrix{Any}(undef, n, n)
    cross_d = Matrix{Any}(undef, n, n)
    @views @inbounds for (j,(k,ks)) in collect(enumerate(kernels))        
        # Precompute distances
        compute_distances!(dat_d, k, dat)
        compute_distances!(gdat_d, k, gdat[:,:,1], gdat[:,:,2])
        compute_distances!(cross_d, k, dat, gdat[:,:,1])
        
        # Compute MMDs for a set of bandwidths for one type of kernel
        ind = (j-1) * n_k
        @threads for i in 1:n_k
            mmds[ind+i] = compute_Umean_K_1_d(ks[i],dat_d) + compute_Umean_K_2_d(ks[i],gdat_d) - 2*compute_Vmean_K_d(ks[i],cross_d)
        end
    end
    return mmds
end
function null_test_statistic(test::MMD, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    dat = test.test_on==:x ? data.x : data.y
    d, n = size(dat)
    
    # Pre-compute transformed data for reuse across kernels
    gdat = Array{Float64}(undef, d, n, 2)
    @views @inbounds @threads for i in 1:2
        gdat[:,:,i] = transform_each(test.G, dat, test.g_actions[:,i])
    end
    
    n_k = length(kernels[1][2])
    mmds = Vector{Float64}(undef, length(kernels)*n_k)
    dat_d = Matrix{Any}(undef, n, n)
    gdat_d = Matrix{Any}(undef, n, n)
    cross_d = Matrix{Any}(undef, n, n)
    @views @inbounds for (j,(k,ks)) in collect(enumerate(kernels))        
        # Precompute distances
        compute_distances!(dat_d, k, dat)
        compute_distances!(gdat_d, k, gdat[:,:,1], gdat[:,:,2])
        compute_distances!(cross_d, k, dat, gdat[:,:,1])
        
        # Compute MMDs for a set of bandwidths for one type of kernel
        ind = (j-1) * n_k
        @threads for i in 1:n_k
            mmds[ind+i] = compute_Umean_K_1_d(ks[i],dat_d) + compute_Umean_K_2_d(ks[i],gdat_d) - 2*compute_Vmean_K_d(ks[i],cross_d)
        end
    end
    return mmds
end