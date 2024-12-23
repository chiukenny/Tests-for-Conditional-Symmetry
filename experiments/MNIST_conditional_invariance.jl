## MNIST experiment
## ----------------

using Flux
using MLDatasets, MLDataUtils
using Augmentor, ImageCore, MappedArrays

# Experiment parameters
n = 100
N_train = 20000    # Size of training set
N_batch = 32       # Training batch size
N_tests = 10       # Number of testing points during training
N_ep_per_test = 3  # Number of epochs between testing points


# Create MNIST specific functions
# -------------------------------

# Predicts the MNIST label probabilities using the provided model
function MNIST_predict(model::Chain, x::AbstractArray{<:Any})
    return softmax(model(ndims(x)==3 ? Flux.unsqueeze(x,3) : x))
end

# Measures the MNIST model test accuracy
function MNIST_accuracy(model::Chain, x_test::AbstractArray{<:Any}, y_test::AbstractVector{<:Integer})
    preds = map(x->x[1], vec(argmax(MNIST_predict(model,x_test),dims=1))) .- 1
    return mean(preds .== y_test)
end

# Create a MNIST resampler
mutable struct MNISTResampler <: AbstractResampler
    B::UInt16                # Number of resamples
    model::Chain             # Model
    aug::Augmentor.Pipeline  # Augmentor
    # Internal use
    data::Data  # Saved data for reuse
    function MNISTResampler(B::Integer, model::Chain, aug::Augmentor.Pipeline, data::Data=Data(0))
        return new(B, model, aug, data)
    end
end

function initialize(RS::MNISTResampler, data::Data)
    RS.data = data
end

function resample(RS::MNISTResampler, d::Integer=d)
    # Randomize by predicting on an augmented sample
    data = RS.data
    gx = Array{Float32}(undef, d, d, 1, data.n)
    augmentbatch!(gx, data.images, RS.aug)
    return Data(data.n, y=Float64.(MNIST_predict(RS.model,gx)))
end


# Start experiment
# ----------------

Random.seed!(seed)


# Preprocess data
println("[$(Dates.format(now(),GV_DT))] Loading data")
d = 28
dy = ARG_9 ? 10 : 9

# Create training set
x_train = Gray.(MNIST.traintensor(Float32, 1:N_train))
y_train = MNIST.trainlabels(1:N_train)
if !ARG_9
    # Remove 9's
    ind_9 = y_train .!= 9
    x_train = x_train[:,:,ind_9]
    y_train = y_train[ind_9]
end
n_train = length(y_train)
y_train = Flux.onehotbatch(y_train, 0:(dy-1))

# Create test set
x_test = Gray.(N_train < 60000 ? cat(MNIST.traintensor(Float32,(N_train+1):60000),MNIST.testtensor(Float32),dims=3) : 
                                 MNIST.testtensor(Float32))
y_test = N_train < 60000 ? vcat(MNIST.trainlabels((N_train+1):60000),MNIST.testlabels()) : MNIST.testlabels()
if !ARG_9
    # Remove 9's
    ind_9 = y_test .!= 9
    x_test = x_test[:,:,ind_9]
    y_test = y_test[ind_9]
end
n_test = length(y_test)
y_test_1h = Flux.onehotbatch(y_test, 0:(dy-1))


# Initialize LeNet model with default training settings
model = Chain(
    Conv((5,5),1 => 6, relu),
    MaxPool((2,2)),
    Conv((5,5),6 => 16, relu),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(256=>120,relu),
    Dense(120=>84, relu),
    Dense(84=>dy, sigmoid)
)
opt = Flux.Optimise.ADAM(0.001)

# Initialize augmentor
aug = Rotate(0:360) |> CropSize(d,d) |> SplitChannels() |> PermuteDims((2,3,1))

# Initialize data sampling function
function sample_data(model::Chain, data::Data, d::Integer=d, x_test::AbstractArray{<:Any}=x_test)
    inds = sample(1:nobs(x_test), data.n, replace=true)
    x = x_test[:,:,inds]
    data.y = MNIST_predict(model, x)
    data.M = Flux.flatten(x)
    data.images = reshape(Gray.(Float32.(data.M)), d, d, data.n)
end
f_sample_data = data -> sample_data(model,data)

# Create resampler object
RS = MNISTResampler(B, model, aug)

# Initialize tests
kernels = [InformationDiffusionKernel()]
tests = [
    FUSE(GV_FUSE, CR(ID_G,RS), kernels=kernels)
    SK(GV_SK, CR(ID_G,RS), kernels=kernels)
]


# Run experiment
output_name = dir_out * "MNIST_N$(N)_n$(n)_B$(B)_" * (ARG_aug ? "aug" : "naug") * "_" * (ARG_9 ? "w9" : "wo9")
output_file = open(output_name*".txt", "w")
results = []

# Test for conditional invariance prior to training
println("[$(Dates.format(now(),GV_DT))] Testing model (0/$(N_tests))")
push!(results, run_tests(output_file, "indep_e0", tests, dy=dy, f_sample_data=f_sample_data, seed=seed))
push!(results, run_tests(output_file, "reuse_e0", tests, dy=dy, f_sample_data=f_sample_data, seed=seed, same_ref=true))

# Train the model
println("[$(Dates.format(now(),GV_DT))] One epoch = $(n_train) training observations")

loss(x,y) = Flux.Losses.logitcrossentropy(model(x), y)
acc = Matrix{Float64}(undef, 2, N_tests)
ce = Matrix{Float64}(undef, 2, N_tests)

outbatch(x) = Array{Float32}(undef, (28, 28, 1, nobs(x)))
augmentbatch((x,y)) = (augmentbatch!(outbatch(x),x,aug), y)
batches = ARG_aug ? mappedarray(augmentbatch, batchview((x_train,y_train), maxsize=N_batch)) :
                    batchview(shuffleobs((Flux.unsqueeze(x_train,3),y_train)), maxsize=N_batch)

gx = Array{Float32}(undef, d, d, 1, n_test)
for e in 1:N_tests
    println("[$(Dates.format(now(),GV_DT))] Training model ($(e)/$(N_tests))")

    for j in 1:N_ep_per_test
        Flux.train!(loss, Flux.params(model), batches, opt)
    end
    
    # Measure model performance
    @inbounds acc[1,e] = MNIST_accuracy(model, x_test, y_test)
    @inbounds ce[1,e] = loss(Flux.unsqueeze(x_test,3), y_test_1h)
    augmentbatch!(gx, x_test, aug)
    @inbounds acc[2,e] = MNIST_accuracy(model, gx, y_test)
    @inbounds ce[2,e] = loss(gx, y_test_1h)

    # Test for conditional invariance
    println("[$(Dates.format(now(),GV_DT))] Testing model ($(e)/$(N_tests))")
    push!(results, run_tests(output_file, "indep_e$(e)", tests, dy=dy, f_sample_data=f_sample_data, seed=seed))
    push!(results, run_tests(output_file, "reuse_e$(e)", tests, dy=dy, f_sample_data=f_sample_data, seed=seed, same_ref=true))
end

# Output model performance metrics
println("\nAccuracy on unaug. data:")
for a in acc[1,:]
    println("\t$(a)")
end
println("Accuracy on aug. data:")
for a in acc[2,:]
    println("\t$(a)")
end
println("\nCross entropy loss on unaug. data:")
for c in ce[1,:]
    println("\t$(c)")
end
println("Cross entropy loss on aug. data:")
for c in ce[2,:]
    println("\t$(c)")
end

results_df = hcat(results...)
CSV.write(output_name*".csv", results_df)
close(output_file)