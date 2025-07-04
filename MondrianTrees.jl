# This file implements prediction based on Mondrian trees
# See e.g. Lakshminarayanan et. al. (2014, 2016)
# Some parts of the code are adapted from https://github.com/WGUNDERWOOD/MondrianForests.jl

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
    nodes::Vector{String} # Vector of all nodes
    is_leaf::Vector{Bool} # Boolean vector indicating which nodes are leaf nodes
    N::Vector{Vector{Int}} # Vector of Indices corresponding to each node
    δ::Vector{Union{Int, Nothing}} # Vector of splitting dimensions (corresponding to the vector of non-leaf nodes)
    ξ::Vector{Union{Float64, Nothing}} # Vector of splitting locations (corresponding to the vector of non-leaf nodes)
    τ::Vector{Union{Float64, Nothing}} # Vector of splitting times (corresponding to the vector of non-leaf nodes)
    min_samples_split::Int # minimum number of splitting samples (only split if the number of samples at a node is greater or equal)
    X::Matrix{Float64} # Matrix of predictors
    y::Vector{Float64} # Vector of labels
end

# Construct a nested Mondrian block object
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


# Auxiliary functions to collect the data from the Mondrian blocks
function collect_data!(vec::Vector, node::MondrianBlock, field::Symbol)
    # Collect the data from this node
    push!(vec, getfield(node, field))

    # If there is a split recurse on left and right children
    if node.split
        collect_data!(vec, node.L, field)  # Left child
        collect_data!(vec, node.R, field)  # Right child
    end
end

function collect_data(node::MondrianBlock, field::Symbol)
    values = []
    collect_data!(values, node, field)
    return values
end

# Constructor for the tree structure
function MondrianTree(y::Vector{Float64}, X::Matrix{Float64}, min_samples_split::Int)
    n, d = size(X)
    block = MondrianBlock("", X, collect(1:n), min_samples_split, 0.0)
    values = map(f -> collect_data(block, f), [:id, :split, :N, :δ, :ξ, :τ])
    tree = MondrianTree(values[1], .!values[2], values[3], values[4], values[5], values[6], min_samples_split, X, y)
    return tree
end


n = 50
X = rand(Normal(0, 1), n, 10)
y = rand(Normal(0, 1), n)


tree = MondrianTree(y, X, 10)
