## Implementations of the FUSE (Biggs, 2023) and supremum kernel distance (Carcamo, 2024) test statistics

include("util.jl")
include("groups.jl")
include("test.jl")
include("kernel.jl")
include("maximum_mean_discrepancy.jl")
include("conditional_randomization_test.jl")
include("baseline_test.jl")


# Aggregate test shared functions
# -------------------------------

# Retrieves the core test
function get_test(test::AbstractAggregateTest)
    return get_test(test.test)
end


# Retrieves the resampler
function get_resampler(test::AbstractAggregateTest)
    return get_resampler(test.test)
end


# Prepares kernels for use
#     Assumes kernel has a bandwidth parameter; otherwise no point in using an aggregate test
#     Product kernels are considered a single kernel; can be used as long as one of the kernels has a parameter
function prepare_kernels(test::AbstractAggregateTest, data::AbstractMatrix{Float64})
    test.ks = Vector{Tuple{AbstractKernel,Vector{<:AbstractKernel}}}(undef, length(test.kernels))
    for i in 1:length(test.kernels)
        k = test.kernels[i]
        ws = compute_uniform_bandwidths(k, data, test.n_k)
        ks = Vector{AbstractKernel}(undef, test.n_k)
        @inbounds @threads for j in eachindex(ks)
            ks[j] = deepcopy(k)
            set_param(ks[j], ws[j])
        end
        test.ks[i] = (k, ks)
    end
end


# Initializes the test
function initialize(test::AbstractAggregateTest, data::Data)
    if typeof(test) == FUSE
        test.λ = sqrt(data.n * (data.n-1))
    end
    return initialize(test, data, get_test(test))
end
# Initializes depending on the specific test
function initialize(test::AbstractAggregateTest, data::Data, core::Baseline)
    two_sample_data = initialize(core, data)
    prepare_kernels(test, two_sample_data.x)
    return two_sample_data
end
function initialize(test::AbstractAggregateTest, data::Data, core::AbstractMMD)
    if core.test_on==:y
        # Y marginal is assumed to be invariant in this case
        prepare_kernels(test, data.y)
    else
        # Test for invariance
        length(data.M)>0 ? prepare_kernels(test,data.M) : prepare_kernels(test,core.G.f_max_inv(data.x))
    end
    return initialize(core, data)
end
function initialize(test::AbstractAggregateTest, data::Data, core::CR)
    RS = core.RS
    initialize(RS, data)
    
    # Prepare kernels using tY
    if hasproperty(RS,:G)
        initialize(data, RS.G, :τy)
        prepare_kernels(test, data.τy)
    else
        # Should only be called for MNIST and experiments specifically testing this
        prepare_kernels(test, data.y)
    end
    
    # Precompute the second sample and its MMD contribution for reuse across randomizations
    core.RS_data = resample(RS)
    core.mmd1 = Vector{Float64}(undef, length(test.kernels)*test.n_k)
    RS_y_d = Matrix{Any}(undef, data.n, data.n)
    for (j,(k,ks)) in collect(enumerate(test.ks))
        compute_distances!(RS_y_d, k, core.RS_data.y)
        ind = (j-1) * test.n_k
        @inbounds @threads for i in 1:test.n_k
            core.mmd1[ind+i] = compute_Umean_K_1_d(ks[i], RS_y_d)
        end
    end
    return data
end


# Computes the test statistic
function test_statistic(test::AbstractAggregateTest, data::Data)
    return aggregate_MMD(test, test_statistic(get_test(test),data,test.ks))
end
function null_test_statistic(test::AbstractAggregateTest, data::Data)
    return aggregate_MMD(test, null_test_statistic(get_test(test),data,test.ks))
end


# MMD-FUSE (Biggs, 2023)
# ----------------------

mutable struct FUSE <: AbstractAggregateTest
    name::String                     # Test name for outputs
    test::AbstractTest               # Core test (only compatible with MMD, Baseline, and CR)
    n_k::UInt8                       # Number of bandwidths to fuse
    kernels::Vector{AbstractKernel}  # Kernels to fuse
    # Internal use
    λ::Float64                                       # Smoothing parameter
    ks::Vector{Tuple{AbstractKernel,Vector{<:Any}}}  # Flattened kernels
    function FUSE(name, test; n_k=10, kernels=[GaussianKernel(),LaplaceKernel()], λ=NaN, ks=[])
        return new(name, test, n_k, kernels, λ, ks)
    end
end

# Aggregates the test statistics via FUSE
function aggregate_MMD(test::FUSE, test_stats::AbstractVector{Float64})
    return log(sum(exp.(test.λ * test_stats))) / test.λ
end


# Supremum kernel distance (Carcamo, 2024)
# ----------------------------------------

mutable struct SK <: AbstractAggregateTest
    name::String                     # Test name for outputs
    test::AbstractTest               # Core test (only compatible with MMD, Baseline, and CR)
    n_k::UInt8                       # Number of bandwidths to fuse
    kernels::Vector{AbstractKernel}  # Kernels to fuse
    # Internal use
    ks::Vector{Tuple{AbstractKernel,Vector{<:Any}}}  # Flattened kernels
    function SK(name, test; n_k=10, kernels=[GaussianKernel(),LaplaceKernel()], ks=[])
        return new(name, test, n_k, kernels, ks)
    end
end

# Aggregates the test statistics via SK
function aggregate_MMD(test::SK, test_stats::AbstractVector{Float64})
    return maximum(test_stats)
end