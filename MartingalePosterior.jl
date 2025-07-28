
include("MondrianTrees.jl")
include("estimators.jl")

using InvertedIndices
using ThreadsX



# return a single sample from the martingale posterior
function mp_sample(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat,
    forest_x::MondrianForest, forest_y::MondrianForest,
    criterion::Function, N::Int, num_trees::Int; W::Union{Nothing, AbstractVecOrMat}=nothing
)
    n = length(y)
    y_full = Vector{eltype(y)}(undef, N)
    x_full = Matrix{eltype(x)}(undef, N, size(x, 2))
    z_full = Matrix{eltype(z)}(undef, N, size(z, 2))
    y_full[1:n] .= y
    x_full[1:n, :] .= x
    z_full[1:n, :] .= z

    W_full = isnothing(W) ? nothing : Matrix{eltype(W)}(undef, N, size(W, 2))
    if W_full !== nothing
        W_full[1:n, :] .= W
    end

    for i in (n+1):N
        new_idx = sample(1:(i-1), 1)[1]
        z_full[i, :] = z_full[new_idx, :]
        if W_full !== nothing
            W_full[i, :] = W_full[new_idx, :]
        end

        input_vec_x = isnothing(W) ? z_full[i, :] : [z_full[i, :]; W_full[i, :]]
        x_full[i, :] = [rand(predict(forest_x, input_vec_x))]
        extend!(forest_x, input_vec_x, x_full[i])

        input_vec_y = isnothing(W) ? x_full[i, :] : [x_full[i, :]; W_full[i, :]]
        y_full[i] = rand(predict(forest_y, input_vec_y))
        extend!(forest_y, input_vec_y, y_full[i])
    end

    result = isnothing(W) ? criterion(y_full, x_full, z_full) : criterion(y_full, x_full, z_full, W_full)
    return result
end


# implement the martingale posterior approach
# No need to add an intercept in x and z (will be done automatically)
function martingale_posterior(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat;
    W::Union{Nothing, AbstractVecOrMat} = nothing,
    criterion::Function = tsls, parallel::Bool = false,
    N::Int = 5 * length(y), B::Int = 100, num_trees::Int = 1
)
    # fit initial forests
    forest_input_x = isnothing(W) ? z[:,:] : [z W]
    forest_x = MondrianForest(x[:, 1], forest_input_x, 10, num_trees)

    forest_input_y = isnothing(W) ? x[:,:] : [x W]
    forest_y = MondrianForest(y, forest_input_y, 10, num_trees)

    # Run the Martingale posterior sampling
    if parallel
        results = ThreadsX.map(_ -> begin
            local_forest_x = deepcopy(forest_x) # copy the forest objects for thread-safety
            local_forest_y = deepcopy(forest_y) # otherwise the extended forests could be shared among threads 
            mp_sample(y, x, z, local_forest_x, local_forest_y, criterion, N, num_trees; W = W)
        end, 1:B)
    else
        results = map(_ -> mp_sample(y, x, z, forest_x, forest_y, criterion, N, num_trees; W = W), 1:B)
    end
    return results
end
