## Data structures and functions for tests


using TriangularIndices
using LinearAlgebra
using StatsBase
using Base.Threads
include("./groups.jl")
include("./util.jl")


# Data
# ----

# Object for conveniently passing data
mutable struct Data
    n::UInt16           # Sample size
    x::Matrix{Float64}  # X data
    y::Matrix{Float64}  # Y data
    # Useful precomputed statistics
    #     Note: this assumes that a sample is only used to test a particular G
    ρx::Matrix{Float64}             # Saved orbit representatives for reuse
    g::Vector{Any}                  # Saved G distribution for reuse
    τy::Matrix{Float64}             # Saved τY data for reuse
    M::Matrix{Float64}              # Saved maximal invariant for reuse
    probs::Vector{AbstractWeights}  # Saved probability weights for reuse
    y_norm::Vector{Float64}         # Saved Y norms for reuse (Note: only works with rotations)
    images::AbstractArray{<:Any}    # Saved images for resampling
    function Data(n::Integer; x=Matrix{Float64}(undef,0,0), y=Matrix{Float64}(undef,0,0),
                              ρx=Matrix{Float64}(undef,0,0), g=[], τy=Matrix{Float64}(undef,0,0), M=Matrix{Float64}(undef,0,0),
                              probs=[], y_norm=Vector{Float64}(undef,0), images=Array{Float32}(undef,0,0,0))
        return new(n, x, y, ρx, g, τy, M, probs, y_norm, images)
    end
end

# Functions for lazy pre-computation of the statistics
function initialize(data::Data, G::Group, stat::Symbol)
    if stat==:M || stat==:probs
        initialize_M(data, G)
    end
    if stat==:g || stat==:τy
        initialize_g(data, G)
    end
    if stat == :τy
        initialize_τy(data, G)
    end
    if stat == :probs
        initialize_probs(data)
    end
    if stat == :y_norm
        initialize_y_norm(data)
    end
end
function initialize_M(data::Data, G::Group)
    if length(data.M) == 0
        data.M = G.f_max_inv(data.x)
    end
end
function initialize_g(data::Data, G::Group)
    if length(data.g) == 0
        data.g = Vector{Any}(undef, data.n)
        @views @inbounds @threads for i in 1:data.n
            data.g[i] = G.f_rep_inv(data.x[:,i])
        end
    end
end
function initialize_τy(data::Data, G::Group)
    if length(data.τy) == 0
        data.τy = similar(data.y)
        @views @inbounds @threads for i in 1:data.n
            data.τy[:,i] = G.f_inv_transform_Y(data.y[:,i], data.g[i])
        end
    end
end
function initialize_probs(data::Data)
    if length(data.probs) == 0
        M = data.M
        n = data.n
        dM = size(M, 1)
        # https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb
        σ2_inv = 1 ./ var(M,dims=2)
        H = (1 / ((4/(dM+2))^(1/(dM+4)) * n^(-1/(dM+4)))^2) * σ2_inv
        K_M = Matrix{Float64}(undef, n, n)
        @inbounds @threads for (i,j) in UpperTriangularIndices(Int(n))
            if i == j
                K_M[i,j] = 1.
            else
                M_diff = @views M[:,i] - M[:,j]
                K_M[i,j] = K_M[j,i] = exp(-dot(M_diff, H.*M_diff) / 2)
            end
        end
        data.probs = Vector{AbstractWeights}(undef, n)
        @inbounds @threads for i in 1:n
            data.probs[i] = Weights(K_M[:,i])  # No views might be better here
        end
    end
end
function initialize_y_norm(data::Data)
    if length(data.y_norm) == 0
        data.y_norm = vec(sqrt.(sum(abs2, data.y, dims=1)))
    end
end

# Clean the data object
function clean_data(data::Data)
    data.ρx = Matrix{Float64}(undef, 0, 0)
    data.g = []
    data.τy = Matrix{Float64}(undef, 0, 0)
    data.M = Matrix{Float64}(undef, 0, 0)
    data.probs = []
    data.y_norm = y_norm=Vector{Float64}(undef, 0)
    data.images = images=Array{Float32}(undef,0,0,0)
end


# Test functions
# --------------

abstract type AbstractTest end
abstract type AbstractAggregateTest <: AbstractTest end

# Object for standardizing outputs of tests
mutable struct TestSummary
    name::String             # Test name
    test_stat::Float64       # Test statistic value
    reject::Bool             # Result of test (1 reject, 0 not reject)
    pvalue::Float64          # p-value of test
    function TestSummary(name, test_stat, reject; pvalue=NaN)
        return new(name, test_stat, reject, pvalue)
    end
end

# Retrieves the innermost test for nested tests
function get_test(test::AbstractTest) return test end

# Determines which data to test on
function get_test_on(test::AbstractTest) return test.test_on end

# Retrieves the resampler
function get_resampler(test::AbstractTest) return test.RS end

# Initializes the test
function initialize(test::AbstractTest, data::Data) return data end

# Runs the test
function run_test(test::AbstractTest, data::Data, α::Float64, same_ref::Bool=false)
    # Initialize the test
    test_dat = initialize(test, data)
    
    # Compute the test statistic
    test_stat = same_ref ? null_test_statistic(test,test_dat) : test_statistic(test,test_dat)
    if isnan(test_stat)
        # Don't error silently
        error("Test statistic = NaN")
    end
    
    # Estimate the p-value and return results
    p = estimate_pvalue(test, test_stat, same_ref)
    return TestSummary(test.name, test_stat, p<=α, pvalue=p)
end 