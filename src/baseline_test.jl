## Implementation of the transformation two-sample (baseline) test for invariance


include("util.jl")
include("groups.jl")
include("test.jl")
include("kernel.jl")
include("maximum_mean_discrepancy.jl")


mutable struct Baseline <: AbstractTest
    name::String  # Test name for outputs
    test::MMD2S   # Core test
    RS::Permuter  # Permuter
    function Baseline(test::MMD2S; RS=Permuter(B))
        return new("", test, RS)
    end
    function Baseline(name::String, test::MMD2S; RS=Permuter(B))
        return new(name, test, RS)
    end
end

# Determines which data to test on
function get_test_on(test::Baseline)
    return get_test_on(test.test)
end


# Initializes the test
function initialize(test::Baseline, data::Data)
    # Get core test
    core = test.test
    
    # Initialize the core resampler
    RS = get_resampler(core)
    initialize(RS, data)
    
    # Generate the second sample
    if core.test_on == :x
        x = hcat(data.x, resample(RS).x)
    else
        x = hcat(data.y, resample(RS).y)
    end
    two_sample_data = Data(data.n*2, x=x)
    core.n1 = data.n
    
    # Initialize the permuter
    initialize(test.RS, two_sample_data)
    return two_sample_data
end


# Computes the test statistic
function test_statistic(test::Baseline, data::Data)
    return test_statistic(test.test, data)
end
function null_test_statistic(test::Baseline, data::Data)
    return null_test_statistic(test.test, data)
end
function test_statistic(test::Baseline, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    return test_statistic(test.test, data, kernels)
end
function null_test_statistic(test::Baseline, data::Data, kernels::Vector{Tuple{AbstractKernel,Vector{<:Any}}})
    return null_test_statistic(test.test, data, kernels)
end