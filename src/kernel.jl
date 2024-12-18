## Collection of kernel implementations that are used in the experiments
##
## The following kernel functions need to be implemented for each kernel:
##     - kernel
##         Inputs: kernel k, matrix x, matrix y OR kernel k, distance matrix d(x,y)
##         Output: kernel matrix k(x,y)

using Base.Threads
using LinearAlgebra
using TriangularIndices
include("util.jl")


# Shared functions
# ----------------

abstract type AbstractKernel end
abstract type AbstractProductKernel <: AbstractKernel end

# Overload these functions as necessary
# function initialize(k::AbstractKernel, param::Any) return end  # Called at the start of a test
function set_param(k::AbstractKernel, param::Any) return end   # Sets the parameter of a kernel


# Computes the kernel matrix K(x,x)
function kernel_mat(k::AbstractKernel, x::AbstractMatrix{Float64})
    n = size(x, 2)
    K = Matrix{Float64}(undef, n, n)
    K[1,1] = @views kernel(k, x[:,1], x[:,1])
    @views @inbounds @threads for (i,j) in UpperTriangularIndices(n)
        K[i,j] = i==j ? K[1,1] : K[j,i]=kernel(k,x[:,i],x[:,j])
    end
    return K
end
# Computes the kernel matrix K(x,y)
function kernel_mat(k::AbstractKernel, x::AbstractMatrix{Float64}, y::AbstractMatrix{Float64})
    n1 = size(x, 2)
    n2 = size(y, 2)
    K = Matrix{Float64}(undef, n1, n2)
    K_view = view(K, 1:n1, 1:n2)
    @views @inbounds @threads for i in eachindex(K_view)
        K[i] = kernel(k, x[:,i[1]], y[:,i[2]])
    end
    return K
end
# Compute the kernel matrix K(x,y) given distances d(x,y)
function kernel_mat_d(k::AbstractKernel, dists::AbstractMatrix{<:Any})
    n1, n2 = size(dists)
    K = similar(dists, Float64)
    @views @inbounds @threads for i in eachindex(K)
        K[i] = kernel(k, dists[i])
    end
    return K
end


# Compute the distances within a set of observations
function compute_distances(k::AbstractKernel, x::AbstractMatrix{Float64})
    n = size(x, 2)
    dists = Matrix{Any}(undef, n, n)
    dists[1,1] = @views k.f_dist(x[:,1], x[:,1])
    @views @inbounds @threads for (i,j) in UpperTriangularIndices(n)
        dists[i,j] = i==j ? dists[1,1] : dists[j,i]=k.f_dist(x[:,i],x[:,j])
    end
    return dists
end
function compute_distances!(dists::Matrix{Any}, k::AbstractKernel, x::AbstractMatrix{Float64})
    n = size(x, 2)
    dists[1,1] = @views k.f_dist(x[:,1], x[:,1])
    @views @inbounds @threads for (i,j) in UpperTriangularIndices(n)
        dists[i,j] = i==j ? dists[1,1] : dists[j,i]=k.f_dist(x[:,i],x[:,j])
    end
end
# Compute the distances between two sets of observations
function compute_distances(k::AbstractKernel, x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64})
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    dists = Matrix{Any}(undef, n1, n2)
    dists_view = view(dists, 1:n1, 1:n2)
    @views @inbounds @threads for i in eachindex(dists_view)
        dists[i] = k.f_dist(x1[:,i[1]], x2[:,i[2]])
    end
    return dists
end
function compute_distances!(dists::Matrix{Any}, k::AbstractKernel, x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64})
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    dists_view = view(dists, 1:n1, 1:n2)
    @views @inbounds @threads for i in eachindex(dists_view)
        dists[i] = k.f_dist(x1[:,i[1]], x2[:,i[2]])
    end
end


# Compute unbiased mean kernel distance
# Assumes n1 = n2
function compute_Umean_K_1_x(k::AbstractKernel, x::AbstractMatrix{Float64})
    n = size(x, 2)
    mmd = 0
    @views @inbounds for i in 2:n
        for j in 1:(i-1)
            mmd += 2 * kernel(k, x[:,i], x[:,j])
        end
    end
    return mmd / (n*(n-1))
end
function compute_Umean_K_2_x(k::AbstractKernel, x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64})
    n = size(x1, 2)
    mmd = 0
    @views @inbounds for i in 1:n
        for j in 1:n
            if i == j continue end
            mmd += kernel(k, x1[:,i], x2[:,j])
        end
    end
    return mmd / (n*(n-1))
end
function compute_Umean_K_1_xy(kx::AbstractKernel, x::AbstractMatrix{Float64}, ky::AbstractKernel, y::AbstractMatrix{Float64})
    n = size(x, 2)
    mmd = 0
    @views @inbounds for i in 2:n
        for j in 1:(i-1)
            mmd += 2 * kernel(kx,x[:,i],x[:,j]) * kernel(ky,y[:,i],y[:,j])
        end
    end
    return mmd / (n*(n-1))
end
function compute_Umean_K_1_d(k::AbstractKernel, d::AbstractMatrix{<:Any})
    n = size(d, 2)
    mmd = 0
    @views @inbounds for i in 2:n
        for j in 1:(i-1)
            mmd += 2*kernel(k, d[j,i])
        end
    end
    return mmd / (n*(n-1))
end
function compute_Umean_K_2_d(k::AbstractKernel, d::AbstractMatrix{<:Any})
    n = size(d, 2)
    mmd = 0
    @views @inbounds for i in 1:n
        for j in 1:n
            if i == j continue end
            mmd += kernel(k, d[j,i])
        end
    end
    return mmd / (n*(n-1))
end
function compute_Umean_K_1_2d(kx::AbstractKernel, xd::AbstractMatrix{<:Any}, ky::AbstractKernel, yd::AbstractMatrix{<:Any})
    n = size(xd, 2)
    mmd = 0
    @views @inbounds for i in 2:n
        for j in 1:(i-1)
            mmd += 2 * kernel(kx,xd[j,i]) * kernel(ky,yd[j,i])
        end
    end
    return mmd / (n*(n-1))
end


# Compute (potentially) biased mean kernel distance
function compute_Vmean_K_x(k::AbstractKernel, x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64})
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    mmd = 0
    @views @inbounds for i in 1:n1
        for j in 1:n2
            mmd += kernel(k, x1[:,i], x2[:,j])
        end
    end
    return mmd / (n1*n2)
end
function compute_Vmean_K_xy(kx::AbstractKernel, x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64},
                            ky::AbstractKernel, y1::AbstractMatrix{Float64}, y2::AbstractMatrix{Float64})
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    mmd = 0
    @views @inbounds for i in 1:n1
        for j in 1:n2
            mmd += kernel(kx,x1[:,i],x2[:,j]) * kernel(ky,y1[:,i],y2[:,j])
        end
    end
    return mmd / (n1*n2)
end
function compute_Vmean_K_d(k::AbstractKernel, d::AbstractMatrix{<:Any})
    n1, n2 = size(d)
    mmd = 0
    @views @inbounds for i in eachindex(d)
        mmd += kernel(k, d[i])
    end
    return mmd / (n1*n2)
end
function compute_Vmean_K_2d(kx::AbstractKernel, xd::AbstractMatrix{<:Any}, ky::AbstractKernel, yd::AbstractMatrix{<:Any})
    n1, n2 = size(xd)
    mmd = 0
    @views @inbounds for i in eachindex(xd)
        mmd += kernel(kx,xd[i]) * kernel(ky,yd[i])
    end
    return mmd / (n1*n2)
end


# Compute uniform bandwidths over discretized interval
function compute_uniform_bandwidths(k::AbstractKernel, x::AbstractMatrix{Float64}, n_k::Integer)
    f_dist = isa(k,AbstractProductKernel) ? (x1,x2)->k.f_dist(x1,x2)[1] : k.f_dist
    n = size(x, 2)
    dists = Vector{Float64}(undef, UInt32(n*(n-1)/2))
    @views @inbounds @threads for (i,j) in UpperTriangularIndices(n)
        if i==j continue end
        ind = UInt32( (j-1)*(j-2)/2 )
        dists[ind+i] = f_dist(x[:,i], x[:,j])
    end
    # Remove distances of 0
    dists = dists[dists .!= 0]
    sort!(dists)
    len = length(dists)
    return LinRange(dists[ceil(UInt32,0.05*len)]/2, dists[floor(UInt32,0.95*len)]*2, n_k)
end


# Implemented kernels
# -------------------

# Gaussian kernel
mutable struct GaussianKernel <: AbstractKernel
    f_dist::Function
    param::Float64
    function GaussianKernel(;f_dist=eucnorm, param=NaN)
        return new(f_dist, param)
    end
end
function set_param(k::GaussianKernel, param::Float64)
    k.param = param
end
function kernel(k::GaussianKernel, x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return exp(-eucnorm(x,y)^2 / k.param^2)
end
function kernel(k::GaussianKernel, dist::Float64)
    return exp(-dist^2 / k.param^2)
end


# Laplace kernel
mutable struct LaplaceKernel <: AbstractKernel
    f_dist::Function
    param::Float64
    function LaplaceKernel(;f_dist=ℓ1norm, param=NaN)
        return new(f_dist, param)
    end
end
function set_param(k::LaplaceKernel, param::Float64)
    k.param = param
end
function kernel(k::LaplaceKernel, x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return exp(-ℓ1norm(x,y) / k.param)
end
function kernel(k::LaplaceKernel, dist::Float64)
    return exp(-dist / k.param)
end


# Indicator (0-1) kernel
# https://upcommons.upc.edu/bitstream/handle/2099.1/17172/MarcoVillegas.pdf
mutable struct IndicatorKernel <: AbstractKernel
    f_dist::Function
    function IndicatorKernel(;f_dist=equals)
        return new(f_dist)
    end
end
function kernel(k::IndicatorKernel, x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return equals(x, y)
end
function kernel(k::IndicatorKernel, dist::Bool)
    return dist
end


# Gaussian-indicator product kernel for Euclidean x and discrete y
mutable struct GaussianIndicatorKernel <: AbstractProductKernel
    f_dist::Function
    param::Float64
    function GaussianIndicatorKernel(;f_dist=eucnorm_equals, param=NaN)
        return new(f_dist, param)
    end
end
function set_param(k::GaussianIndicatorKernel, param::Float64)
    k.param = param
end
function kernel(k::GaussianIndicatorKernel, x::AbstractVector{Float64}, y::AbstractVector{Float64})
    x_dist, y_dist = eucnorm_equals(x, y)
    return y_dist ? exp(-x_dist^2/k.param^2) : 0.
end
function kernel(k::GaussianIndicatorKernel, dist::Tuple{Float64,Bool})
    return dist[2] ? exp(-dist[1]/k.param^2) : 0.
end


# Laplace-indicator product kernel for Euclidean x and discrete y
mutable struct LaplaceIndicatorKernel <: AbstractProductKernel
    f_dist::Function
    param::Float64
    function LaplaceIndicatorKernel(;f_dist=ℓ1norm_equals, param=NaN)
        return new(f_dist, param)
    end
end
function set_param(k::LaplaceIndicatorKernel, param::Float64)
    k.param = param
end
function kernel(k::LaplaceIndicatorKernel, x::AbstractVector{Float64}, y::AbstractVector{Float64})
    x_dist, y_dist = ℓ1norm_equals(x, y)
    return y_dist ? exp(-x_dist/k.param) : 0.
end
function kernel(k::LaplaceIndicatorKernel, dist::Tuple{Float64,Bool})
    return dist[2] ? exp(-dist[1]/k.param) : 0.
end


# Information diffusion kernel for probability vectors
# https://proceedings.neurips.cc/paper_files/paper/2002/file/5938b4d054136e5d59ada6ec9c295d7a-Paper.pdf
mutable struct InformationDiffusionKernel <: AbstractKernel
    f_dist::Function
    param::Float64
    function InformationDiffusionKernel(;f_dist=geodesic, param=NaN)
        return new(f_dist, param)
    end
end
function set_param(k::InformationDiffusionKernel, param::Float64)
    k.param = param
end
function kernel(k::InformationDiffusionKernel, x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return exp(-geodesic(x,y)^2 / (4*k.param))
end
function kernel(k::InformationDiffusionKernel, dist::Float64)
    return exp(-dist^2 / (4*k.param))
end