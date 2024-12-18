## USAGE
## -----
##
## From the command line, execute the following command:
##     julia create_arg_file.jl

dir_src = "./src/"

# Load experiment arguments
include(dir_src * "global_variables.jl")


# gaussian_equivariance_covariance_*.jl
# -------------------------------------

output_file = open("gaussian_equivariance_covariance_args.txt", "w")
for n in GV_GAUSS_COV_n
    for p in GV_GAUSS_COV_p
        for d in GV_GAUSS_COV_d
            println(output_file, "$n,$p,$d")
        end
    end
end
close(output_file)


# gaussian_nonequivariance_covariance_*.jl
# -------------------------------------

output_file = open("gaussian_nonequivariance_covariance_args.txt", "w")
for n in GV_GAUSS_COV_n
    for p in GV_GAUSS_COV_p[2:end]
        for d in GV_GAUSS_COV_d
            println(output_file, "$n,$p,$d")
        end
    end
end
close(output_file)


# gaussian_equivariance_truth_*.jl
# -------------------------------------

output_file = open("gaussian_equivariance_truth_args.txt", "w")
for n in GV_GAUSS_TRUTH_n
    for p in GV_GAUSS_TRUTH_p
        for d in GV_GAUSS_TRUTH_d
            for exp in ["none","truth"]
                println(output_file, "$n,$p,$d,$exp")
            end
        end
    end
end
close(output_file)


# gaussian_equivariance_covariance_sensitivity.jl
# -----------------------------------------------

output_file = open("gaussian_equivariance_sensitivity_args.txt", "w")
for p in GV_GAUSS_SEN_p
    for s in GV_GAUSS_SEN_s
        for n in GV_GAUSS_SEN_n
            println(output_file, "$p,$s,$n")
        end
    end
end
close(output_file)


# gaussian_equivariance_resamples.jl
# --------------------------------

output_file = open("gaussian_equivariance_resamples_args.txt", "w")
for n in GV_GAUSS_RES_n
    for p in GV_GAUSS_RES_p
        for B in GV_GAUSS_RES_B
            println(output_file, "$n,$p,$B")
        end
    end
end
close(output_file)


# gaussian_equivariance_permutation.jl
# -------------------------------------

output_file = open("gaussian_equivariance_permutation_args.txt", "w")
for n in GV_GAUSS_PER_n
    for s in GV_GAUSS_PER_s
        for d in GV_GAUSS_PER_d
            println(output_file, "$n,$s,$d")
        end
    end
end
close(output_file)


# MNIST_conditional_invariance.jl
# -------------------------------------

output_file = open("MNIST_args.txt", "w")
for aug in GV_MNIST_aug
    for n in GV_MNIST_9
        println(output_file, "$(aug),$n")
    end
end
close(output_file)


# invariance.jl
# -------------------------------------

output_file = open("invariance_args.txt", "w")
for n in GV_INV_n
    for p in GV_INV_p
        println(output_file, "$n,$p")
    end
end
close(output_file)