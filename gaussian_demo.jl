## DEMO: Test for SO(3)-equivariance on Gaussian data
## --------------------------------------------------
##
## From the command line, execute the following command to run this demo:
##     julia gaussian_demo.jl


include("src/util.jl")                             # Shared functions
include("src/groups.jl")                           # Groups and related functions
include("src/test.jl")                             # Data structures and functions for tests
include("src/kernel.jl")                           # Kernel functions
include("src/resampler.jl")                        # Resampling functions
include("src/conditional_randomization_test.jl")   # Conditional randomization test
include("src/aggregate_test.jl")                   # Aggregate tests


# 1. Create a SO(3) group object
# ------------------------------
#    The following functions need to be implemented for a group to test on:
#     - f_transform(_Y)
#         Inputs: data point x (x,y), action g
#         Output: transformed data point gx (x,gy)
#     - f_inv_transform(_Y)
#         Inputs: data point x (x,y), action g
#         Output: transformed data point g^{-1}x (x,g^{-1}y)
#     - f_max_inv
#         Inputs: data matrix X
#         Output: maximal invariant matrix M(X)
#     - f_rep_inv
#         Inputs: data point x
#         Output: action g such that g*ρ(x)=x
#    See /src/groups.jl for pre-implemented groups

G = Group(f_transform=rotate_d, f_inv_transform=inv_rotate_d, f_transform_Y=rotate_d, f_inv_transform_Y=inv_rotate_d,
          f_max_inv=max_inv_rotate, f_rep_inv=rep_inv_rotate)


# 2. Create a resampler object for performing conditional randomization
# ---------------------------------------------------------------------
#    See /src/resampler.jl for pre-implemented resamplers

B = 100  # Number of randomizations
RS = EquivariantResampler(B, G)


# 3. Initialize the conditional randomization test that uses the FUSE statistic
# -----------------------------------------------------------------------------

demo_test = FUSE("Demo test", CR(G,RS))


# 4. Prepare the data to test on
# ------------------------------
# Generate 200 samples (X,Y) from the distribution
#    - X ~ N(0,1)
#    - Y|X ~ N(X,1)

n = 200
d = 3
Px = MvNormal(zeros(d), 1)
x = rand(Px, n)
y = similar(x)
@views @inbounds for i in 1:n
    y[:,i] = rand( MvNormal(x[:,i],1) )
end
demo_data = Data(n, x=x, y=y)


# 5. Run the test
# ---------------

α = 0.05  # Significance level
test_summary = run_test(demo_test, demo_data, α)

println("Test name: " * test_summary.name)
println("Test statistic: " * string(test_summary.test_stat))
println("Test p-value: " * string(test_summary.pvalue))
println("Reject null hypothesis: " * string(test_summary.reject))