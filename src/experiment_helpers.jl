## Helper functions for running experiments

using Distributions
using LinearAlgebra
using Random
using DataFrames
using Base.Threads
include("test.jl")


# Runs invariance/equivariance tests and saves p-values in CSV files
function run_tests(output_file::IOStream, exp_name::String, tests::AbstractVector{<:AbstractTest};
                   N::Integer=N, n::Integer=n, α::Float64=α, dx::Integer=0, dy::Integer=0, dz::Integer=0, same_ref::Bool=false,
                   f_sample_data::Function=f_nothing, seed::Integer=randInt())
    # Prepare output data frame
    n_tests = length(tests)
    results = DataFrame()
    p_dict = Dict()
    t_dict = Dict()
    r_dict = Dict()
    names_len = 0
    for test in tests
        names_len = max(names_len, length(test.name))
        col_p = Symbol(exp_name * "_" * test.name * "_p")
        col_t = Symbol(exp_name * "_" * test.name * "_time")
        col_r = Symbol(exp_name * "_" * test.name * "_rej")
        p_dict[test.name] = col_p
        t_dict[test.name] = col_t
        r_dict[test.name] = col_r
        results[!,col_p] = Vector{Float64}(undef, N)
        results[!,col_r] = Vector{Bool}(undef, N)
        results[!,col_t] = Vector{Float64}(undef, N)
    end
    
    # Initialize a data struct for reuse
    data = Data(n)
    if dx > 0
        data.x = Matrix{Float64}(undef, dx, n)
    end
    if dy > 0
        data.y = Matrix{Float64}(undef, dy, n)
    end
    if dz > 0
        data.M = Matrix{Float64}(undef, dz, n)
    end
    
    # Run independent simulations
    Random.seed!(seed)
    base_seed = randInt()
    for i in 1:N
        # Simulate data
        Random.seed!(base_seed + i)
        clean_data(data)
        f_sample_data(data)
        
        # Run test and save results to data frame
        for j in 1:n_tests
            test = tests[j]
            summary = @timed run_test(test, data, α, same_ref)
            results[i,p_dict[test.name]] = summary.value.pvalue
            results[i,r_dict[test.name]] = summary.value.reject
            results[i,t_dict[test.name]] = summary.time
        end
    end
    
    # Print aggregated results
    write(output_file, "Experiment \"$(exp_name)\": [rej.rate] ± [rej.std] (avg.time)\n")
    for test in tests
        test_name = lpad(test.name, names_len, " ")
        rej_rate = @views round(mean(results[:,r_dict[test.name]]), digits=5)
        rej_std = round(sqrt(rej_rate*(1-rej_rate)/N), digits=5)
        avg_time = @views round(mean(results[:,t_dict[test.name]]), digits=5)
        write(output_file, "$(test_name): $(rpad(rej_rate,7,'0')) ± $(rpad(rej_std,7,'0')) ($(rpad(avg_time,7,'0'))s)\n")
        # Remove rejection and time columns
        select!(results, Not([r_dict[test.name],t_dict[test.name]]))
    end
    write(output_file, "\n")
    return results
end