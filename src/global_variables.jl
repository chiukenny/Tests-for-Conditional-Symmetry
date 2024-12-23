## Default and constant variables for convenience

const seed = 1  # Base seed for reproducibility


# Default experimental parameters
N = 1000  # Number of simulations
B = 100   # Number of resamples
n = 100   # Sample size

const α = 0.05  # Test size


# Experiment arguments
const GV_GAUSS_COV_n = 25:25:500
const GV_GAUSS_COV_p = 0:0.25:1
const GV_GAUSS_COV_d = [3, 6, 9]
const GV_GAUSS_TRUTH_n = 25:25:500
const GV_GAUSS_TRUTH_p = 0:0.25:1
const GV_GAUSS_TRUTH_d = [3, 4, 5]
const GV_GAUSS_PER_n = 25:25:500
const GV_GAUSS_PER_s = 0:0.5:2
const GV_GAUSS_PER_d = [3, 4, 5]
const GV_GAUSS_RES_n = 20:20:100
const GV_GAUSS_RES_p = [0, 0.99]
const GV_GAUSS_RES_B = 50:50:1000
const GV_GAUSS_SEN_n = 25:25:500
const GV_GAUSS_SEN_p = round.(π/4:π/4:π, digits=3)
const GV_GAUSS_SEN_s = 0.2:0.2:1
const GV_MNIST_aug = [false, true]
const GV_MNIST_9 = [false, true]
const GV_INV_n = 10:10:100
const GV_INV_p = round.(π/2:π/2:2π, digits=3)


# Date format for logging
const GV_DT = "yyyy-mm-dd HH:MM:SS"


# Default test names
const GV_BS = "BS"
const GV_MMD = "MMD"
const GV_CR = "CR"
const GV_FUSE = "FUSE"
const GV_SK = "SK"
const GV_BS_FS = "BS-FS"
const GV_BS_SK = "BS-SK"
const GV_RN_FS = "RN-FS"
const GV_RN_SK = "RN-SK"