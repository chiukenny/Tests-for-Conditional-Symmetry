## DEMO: Test for Lorentz-equivariance on quark decay data
## -------------------------------------------------------
##
## From the command line, execute the following command to run this demo:
##     julia demo_TQT.jl


# Paths
dir_out_dat = "./outputs/data/"  # Path to demo_TQT.h5 (created by preprocess_data.jl)

include("src/util.jl")                             # Shared functions
include("src/groups.jl")                           # Groups and related functions
include("src/test.jl")                             # Data structures and functions for tests
include("src/kernel.jl")                           # Kernel functions
include("src/resampler.jl")                        # Resampling functions
include("src/conditional_randomization_test.jl")   # Conditional randomization test
include("src/aggregate_test.jl")                   # Aggregate tests

using HDF5
import H5Zblosc


# 1. Create a Lorentz group object
# --------------------------------
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

G = Group(f_transform=lorentz_transform, f_transform_Y=lorentz_transform, f_inv_transform_Y=lorentz_inv_transform,
          f_max_inv=max_inv_lorentz, f_rep_inv=rep_inv_lorentz)


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

fid = h5open(dir_out_dat*"demo_TQT.h5", "r")
TQT = copy(read(fid["data"]))
close(fid)

n = 100
demo_data = Data(n, x=TQT[1:4,:], y=TQT[5:8,:])


# 5. Run the test
# ---------------

α = 0.05  # Significance level
test_summary = run_test(demo_test, demo_data, α)

println("Test name: " * test_summary.name)
println("Test statistic: " * string(test_summary.test_stat))
println("Test p-value: " * string(test_summary.pvalue))
println("Reject null hypothesis: " * string(test_summary.reject))