## Collection of miscellaneous objects and functions shared across tests and experiments


using Random
using Distributions
using Distances
using TriangularIndices
using InvertedIndices
using StaticArrays
using Base.Threads


# General functions
# -----------------


# Generates a random integer
function randInt()
    return rand(1:9999999)
end


# Standardizes a dataset
function standardize(x::AbstractMatrix{Float64})
    return (x .- mean(x,dims=2)) ./ std(x,dims=2)
end


# Computes the median distance between points in a sample
function med_dist(x::AbstractMatrix{Float64}; max_n::Integer=100)
    n = size(x, 2)
    m = min(n, max_n)
    s = sample(1:n, m, replace=false)
    len = UInt16( m*(m-1)/2 )
    dists = Vector{Float64}(undef, len)
    @threads for (i,j) in UpperTriangularIndices(m)
        if i==j continue end
        ind = UInt16( (j-1)*(j-2)/2 )
        @inbounds dists[ind+i] = @views sqeuclidean(x[:,s[i]], x[:,s[j]])
    end
    return sqrt(median(dists))
end


# Redefine distance norms due to overloaded names
function eucnorm(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return euclidean(x, y)
end
function ℓ1norm(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return cityblock(x, y)
end
function equals(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return x == y
end
function eucnorm_equals(x1::AbstractVector{Float64}, x2::AbstractVector{Float64})
    return @views eucnorm(x1[1:end-1],x2[1:end-1]), x1[end]==x2[end]
end
function ℓ1norm_equals(x1::AbstractVector{Float64}, x2::AbstractVector{Float64})
    return @views ℓ1norm(x1[1:end-1],x2[1:end-1]), x1[end]==x2[end]
end
# Geodesic distance for probability vectors on simplex
function geodesic(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return 2 * acos( min(sum(sqrt,x.*y),1) )
end


# Creates a d-dimensional unit vector
function unit_vector(d::Integer)
    e = zeros(d)
    e[1] = 1
    return e
end


# Placeholder function for function-typed variables
function f_nothing(args...)
    error("f_nothing($(length(args)) args) called: function has not set")
end