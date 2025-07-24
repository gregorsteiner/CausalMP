
include("MondrianTrees.jl")
include("estimators.jl")

using InvertedIndices


# auxiliary function that returns the indices for k-fold cross-validation
function kfold_indices(n::Int, k::Int; shuffle::Bool = true)
    @assert k > 1 "Number of folds k must be at least 2."
    @assert n >= k "Number of samples n must be at least equal to k."

    indices = collect(1:n)
    if shuffle
        shuffle!(indices)
    end
    base_size = div(n, k)
    remainder = rem(n, k)

    folds = Vector{Vector{Int}}(undef, k)
    start = 1
    for i in 1:k
        fold_size = base_size + (i <= remainder ? 1 : 0)
        folds[i] = indices[start:start + fold_size - 1]
        start += fold_size
    end

    return folds
end

# return a single sample from the martingale posterior
function mp_sample(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat,
    criterion::Function, N::Int, num_trees::Int; W::Union{Nothing, AbstractVecOrMat}=nothing
)
    n = length(y)
    y_full = Vector{eltype(y)}(undef, N)
    x_full = Matrix{eltype(x)}(undef, N, size(x, 2))
    z_full = Matrix{eltype(z)}(undef, N, size(z, 2))
    y_full[1:n], x_full[1:n, :], z_full[1:n, :] = y, x, z

    W_full = isnothing(W) ? nothing : Matrix{eltype(W)}(undef, N, size(W, 2))
    if W_full !== nothing
        W_full[1:n, :] = W
    end

    forest_input_x = isnothing(W) ? z[:,:] : [z W]
    forest_x = MondrianForest(x[:, 1], forest_input_x, 10, num_trees)

    forest_input_y = isnothing(W) ? x[:,:] : [x W]
    forest_y = MondrianForest(y, forest_input_y, 10, num_trees)

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
    W::Union{Nothing, AbstractVecOrMat}=nothing,
    criterion::Function = tsls,
    N::Int = 5 * length(y), B::Int = 100, num_trees::Int = 1
)
    results = map(_ -> mp_sample(y, x, z, criterion, N, num_trees; W = W), 1:B)
    return results
end
