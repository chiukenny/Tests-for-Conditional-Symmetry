## Collection of group implementations that are tested in the experiments
##
## The following group functions may need to be implemented depending on the test:
##     - f_sample
##         Inputs: none
##         Output: action g
##     - f_transform(_Y)
##         Inputs: data point x (x,y), action g
##         Output: transformed data point gx (x,gy)
##     - f_inv_transform(_Y)
##         Inputs: data point x (x,y), action g
##         Output: transformed data point g^{-1}x (x,g^{-1}y)
##     - f_max_inv
##         Inputs: data matrix X
##         Output: maximal invariant matrix M(X)
##     - f_rep_inv
##         Inputs: data point x
##         Output: action g such that g*ρ(x)=x

using Base.Threads
using StaticArrays
using Random
using LinearAlgebra
using Rotations
using RandomMatrices
using InvertedIndices
using SliceMap
include("./util.jl")


struct Group
    f_sample::Function             # Samples a random group action
    f_transform::Function          # Transforms a data point by an action
    f_inv_transform::Function      # Transforms a data point by the inverse of an action
    f_transform_Y::Function        # Transforms the Y component of a data point by an action
    f_inv_transform_Y::Function    # Transforms the Y component of a data point by the inverse of an action
    f_max_inv::Function            # Applies a maximal invariant to all data points
    f_rep_inv::Function            # Computes the representative inversion
    function Group(;f_sample=f_nothing, f_transform=f_nothing, f_inv_transform=f_nothing, f_transform_Y=f_nothing,
                    f_inv_transform_Y=f_nothing, f_max_inv=f_nothing, f_rep_inv=f_nothing)
        return new(f_sample, f_transform, f_inv_transform, f_transform_Y, f_inv_transform_Y, f_max_inv, f_rep_inv)
    end
end


# Applies a single transformation to an entire dataset
# or a random transformation to each point if no transformation specified
function transform_all(G::Group, x::AbstractMatrix{Float64}, g::Any=nothing; f_transform=f_nothing)
    f_transform = f_transform==f_nothing ? G.f_transform : f_transform
    return isnothing(g) ? tmapcols(x->f_transform(x,G.f_sample()),x) : tmapcols(x->f_transform(x,g),x)
end


# Transforms each data point in a dataset given a set of predetermined transformations
function transform_each(G::Group, x::AbstractMatrix{Float64}, g::AbstractVector{<:Any}; f_transform=f_nothing)
    n = size(x, 2)
    transform = f_transform==f_nothing ? G.f_transform : f_transform
    gx = similar(x)
    @views @inbounds @threads for i in 1:n
        gx[:,i] = transform(x[:,i], g[i])
    end
    return gx
end


# Identity group
# --------------

# f_sample
function rand_identity()
    return 1
end
# f_transform
function identity(x::Any, g::Any)
    return x
end
# f_max_inv
function max_inv_identity(x::Any)
    return x
end
# f_rep_inv
function rep_inv_identity(x::Any)
    return 1
end
ID_G = Group(f_sample=rand_identity, f_transform=identity, f_max_inv=max_inv_identity, f_rep_inv=rep_inv_identity)


# Rotations
# ---------

## SO(2)
# f_sample
# Returns rotation angle in [0,2π]
function rand_θ()
    return rand() * 2π
end
# f_transform
function rotate_2D(x::AbstractVector{Float64}, θ::Float64)
    return RotMatrix(θ) * x
end
# f_inv_transform
function inv_rotate_2D(x::AbstractVector{Float64}, θ::Float64)
    return rotate_2D(x, -θ)
end
# f_rep_inv
# Computes rotation angle in [0,2π] that takes [norm(x) 0] -> x
function rep_inv_rotate_2D(x::AbstractVector{Float64})
    # https://www.mathworks.com/matlabcentral/answers/180131-how-can-i-find-the-angle-between-two-vectors-including-directional-information
    θ = atan(x[2], x[1])
    # Return counterclockwise angle
    return θ >= 0 ? θ : 2*π+θ
end
# f_max_inv
# function max_inv_rotate(x::AbstractMatrix{Float64})

## SO(d)
# f_sample
# Returns d-dimensional rotation matrix
function rand_rotation(d::Integer)
    return qr( rand(Normal(0,1),d,d) ).Q
end
# f_transform
function rotate_d(x::AbstractVector{Float64}, R::AbstractMatrix{Float64})
    return R * x
end
# f_inv_transform
function inv_rotate_d(x::AbstractVector{Float64}, R::AbstractMatrix{Float64})
    return R' * x
end
# f_max_inv
function max_inv_rotate(x::AbstractMatrix{Float64})
    return sqrt.(sum(abs2, x, dims=1))
end
# f_rep_inv
# Computes a rotation matrix that takes [norm(x) 0 ... 0] -> x
function rep_inv_rotate(x::AbstractVector{Float64})
    d = length(x)
    
    # https://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions
    uv = Matrix{Float64}(undef, d, 2)
    uv[1,1] = 1
    uv[2:end,1] .= 0
    uv[:,2] = x
    uv[1,2] = 0
    uv[:,2] /= @views sqrt(sum(abs2, uv[:,2]))
    
    θ = acos(x[1] / sqrt(sum(abs2,x)))
    Rxy = RotMatrix(θ)
    R1 = @views I(d) - uv[:,2]*uv[:,2]' + uv*Rxy*uv'
    R1[1,1] -= 1
    
    # Sample and apply a rotation from the stabilizer subgroup
    R2 = Matrix{Float64}(undef, d, d)
    R2[1,1] = 1
    R2[2:end,1] .= 0
    R2[1,2:end] .= 0
    R2[2:end,2:end] = rand_rotation(d-1)
    return R1 * R2
end

## SO(2) x SO(2)
# f_sample
# Returns a pair of angles; if paired=true, the angles are the same
function rand_θ1_θ2(paired::Bool=true)
    if !paired
        return SVector{2}(rand_θ(), rand_θ())
    end
    θ = rand_θ()
    return SVector{2}(θ, θ)
end
# f_transform
function rotate_θ1_θ2(x::AbstractVector{Float64}, θs::StaticVector{2,Float64})
    R = zeros(SizedMatrix{4,4})
    θ1 = θs[1]
    θ2 = θs[2]
    R[1:2,1:2] = RotMatrix(θ1)
    R[3:4,3:4] = RotMatrix(θ2)
    return R * x
end
# f_max_inv
function max_inv_θ1_θ2(x::AbstractMatrix{Float64}, paired::Bool=true)
    n = size(x, 2)
    max_x = Matrix{Float64}(undef, 4, n)
    @views begin
        max_x[1,:] = max_inv_rotate(x[1:2,:])
        max_x[2,:] .= 0
        if paired
            @threads for i in 1:n
                R = rep_inv_rotate(x[1:2,i])
                max_x[3:4,i] = R' * x[3:4,i]
            end
        else
            max_x[3,:] = max_inv_rotate(x[3:4,:])
            max_x[4,:] .= 0
        end
    end
    return max_x
end
# f_rep_inv
function rep_inv_θ1_θ2(x::AbstractVector{Float64}, paired::Bool=true)
    R = zeros(SizedMatrix{4,4})
    @views begin
        R[1:2,1:2] = rep_inv_rotate(x[1:2])
        if paired
            R[3:4,3:4] = R[1:2,1:2]
        else
            R[3:4,3:4] = rep_inv_rotate(x[3:4])
        end
    end
    return R'
end


# Exchangeability
# ---------------

# f_sample: randperm(Int)
# f_transform
function permute(x::AbstractVector{Float64}, P::Vector{<:Integer})
    return x[P]  # Return a copy so don't use @view
end
# f_inv_transform
function inv_permute(x::AbstractVector{Float64}, P::Vector{<:Integer})
    return x[invperm(P)]  # Return a copy so don't use @view
end
# f_max_inv
function max_inv_permute(x::AbstractMatrix{Float64})
    return tmapcols(c->sort(c), x)
end
# f_rep_inv
# Computes the permutation that takes [x_(1) ... x_(d)] -> x
function rep_inv_permute(x::AbstractVector{Float64})
    return invperm(sortperm(x))
end


# Translations
# ------------

# f_sample: not implemented
# f_transform
function translate(x::AbstractVector{Float64}, s::Vector{Float64})
    return x + s
end
# f_inv_transform
function inv_translate(x::AbstractVector{Float64}, s::Vector{Float64})
    return x - s
end
# f_max_inv
function max_inv_translate(x::AbstractMatrix{Float64})
    return zeros(size(x))
end
# f_rep_inv
function rep_inv_translate(x::AbstractVector{Float64})
    return x
end


# Lorentz
# -------
# http://hyperphysics.phy-astr.gsu.edu/hbase/Relativ/vec4.html

# f_sample: not implemented
# f_transform
function lorentz_transform(x::AbstractVector{Float64}, g::Tuple{Float64,AbstractMatrix{Float64}})
    β, R = g
    gx = similar(x)
    gx[1:2] = [x[1]-β*x[2], -β*x[1]+x[2]] / sqrt(1-β^2)
    gx[3:4] = x[3:4]
    gx[2:4] = @views R * gx[2:4]
    return gx
end
# f_inv_transform
function lorentz_inv_transform(x::AbstractVector{Float64}, g::Tuple{Float64,AbstractMatrix{Float64}})
    β, R = g
    gx = similar(x)
    gx[2:4] = @views R' * x[2:4]
    gx[1:2] = [x[1]+β*gx[2], β*x[1]+gx[2]] / sqrt(1-β^2)
    return gx
end
# f_max_inv
function max_inv_lorentz(x::AbstractMatrix{Float64})
    return @views SMatrix{1,4,Int8}(1,-1,-1,-1) * x.^2
end
# f_rep_inv
function rep_inv_lorentz(x::AbstractVector{Float64})
    # Take spatial coordinates [1,0,0] as orbit representative
    E = x[1]
    p = @views x[2:4]
    sum_p2 = sum(p.^2)
    Eρ = sqrt(E^2 - sum_p2 + 1)  # = sqrt(Q + 1)
    R = rep_inv_rotate(p)
    return ( (Eρ-E*sqrt(sum_p2))/(E^2+1), R )
end