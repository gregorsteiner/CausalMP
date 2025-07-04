# This file implements prediction based on Mondrian trees
# See e.g. Lakshminarayanan et. al. (2014, 2016)
# Some parts of the code are taken/adapted from https://github.com/WGUNDERWOOD/MondrianForests.jl

using LinearAlgebra, Distributions


struct MondrianBlock
    id::String # id of the node
    X::Matrix{Float64} # features in this node
    N::Vector{Int} # Indices of all training observations in this node
    split::Bool # true if split (otherwise this is a leaf node)
    δ::Union{Int, Nothing} # split dimension
    ξ::Union{Float64, Nothing} # split location
    τ::Union{Float64, Nothing} # split time
    L::Union{MondrianBlock, Nothing} # left child
    R::Union{MondrianBlock, Nothing} # right child
    parent::Union{String, Nothing} # id of parent node
end

struct MondrianTree
    non_leaf_nodes::Vector{String} # Vector of all nodes
    leaf_nodes::Vector{String} # Vector of only the leaf nodes
    δ::Vector{Int} # Vector of splitting dimensions (corresponding to the vector of non-leaf nodes)
    ξ::Vector{Float64} # Vector of splitting locations (corresponding to the vector of non-leaf nodes)
    τ::Vector{Float64} # Vector of splitting times (corresponding to the vector of non-leaf nodes)
    min_samples_split::Int # minimum number of splitting samples (only split if the number of samples at a node is greater or equal)
    N::Vector{Vector{Float64}} # Vector of Indices corresponding to each node
    X::Matrix{Float64} # Matrix of predictors
    y::Vector{Float64} # Vector of labels
end


function MondrianBlock(id::String, X::Matrix{Float64}, N::Vector{Int}, min_samples_split::Int, creation_time::Float64)
    n, d = size(X)
    if n >= min_samples_split
        lower, upper = map(f -> map(f, eachcol(X)), (minimum, maximum))
        size_cell = sum(upper .- lower)
        E = rand(Exponential(1 / size_cell))
        split_probabilities = collect(upper .- lower) ./ size_cell
        split_axis = rand(DiscreteNonParametric(1:d, split_probabilities))
        split_location = rand(Uniform(lower[split_axis], upper[split_axis]))
        split_bool = X[:, split_axis] .<= split_location
        N_left, N_right = (N[split_bool], N[.!split_bool])
        X_left, X_right = (X[split_bool, :], X[.!split_bool, :])
        block_left = MondrianBlock(id * "L", X_left, N_left, min_samples_split, creation_time + E)
        block_right = MondrianBlock(id * "R", X_right, N_right, min_samples_split, creation_time + E)
        block = MondrianBlock(id, X, N, true, split_axis, split_location, creation_time + E, block_left, block_right, ifelse(id == "", nothing, id[1:(end-1)]))
    else
        block = MondrianBlock(id, X, N, false, nothing, nothing, nothing, nothing, nothing, ifelse(id == "", nothing, id[1:(end-1)]))
    end
    return block
end



n = 100
X = rand(Normal(0, 1), n, 10)
res = MondrianBlock("", X, collect(1:n), 50, 0.0)



