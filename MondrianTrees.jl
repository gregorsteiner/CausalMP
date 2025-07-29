# This file implements prediction based on Mondrian trees
# See e.g. Lakshminarayanan et. al. (2014, 2016)
# Some parts of the code are adapted from https://github.com/WGUNDERWOOD/MondrianForests.jl

using LinearAlgebra, Distributions


mutable struct MondrianBlock
    id::String # id of the node
    X::Matrix{Float64} # features in this node
    N::Vector{Int} # Indices of all training observations in this node
    split::Bool # true if split (otherwise this is a leaf node)
    δ::Union{Int, Nothing} # split dimension
    ξ::Union{Float64, Nothing} # split location
    τ::Float64 # split time
    L::Union{MondrianBlock, Nothing} # left child
    R::Union{MondrianBlock, Nothing} # right child
    parent::Union{String, Nothing} # id of parent node
end

mutable struct MondrianTree
    nodes::Vector{String} # Vector of all nodes
    is_leaf::Vector{Bool} # Boolean vector indicating which nodes are leaf nodes
    N::Vector{Vector{Int}} # Vector of Indices corresponding to each node
    δ::Vector{Union{Int, Nothing}} # Vector of splitting dimensions (nothing for leaf nodes)
    ξ::Vector{Union{Float64, Nothing}} # Vector of splitting locations (nothing for leaf nodes)
    τ::Vector{Float64} # Vector of splitting times (Inf for leaf nodes)
    min_samples_split::Int # minimum number of splitting samples (only split if the number of samples at a node is greater or equal)
    X::Matrix{Float64} # Matrix of predictors
    y::Vector{Float64} # Vector of labels
    block::MondrianBlock
end

# Construct a nested Mondrian block object
function MondrianBlock(id::String, X::Matrix{Float64}, N::Vector{Int}, min_samples_split::Int, creation_time::Float64)
    n, d = size(X)
    lower, upper = map(f -> map(f, eachcol(X)), (minimum, maximum))
    size_cell = sum(upper .- lower)
    if n >= min_samples_split && size_cell > 0.0
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
        block = MondrianBlock(id, X, N, false, nothing, nothing, Inf, nothing, nothing, ifelse(id == "", nothing, id[1:(end-1)]))
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
    tree = MondrianTree(values[1], .!values[2], values[3], values[4], values[5], values[6], min_samples_split, X, y, block)
    return tree
end


# auxiliary function to shift the ids when a new node is inserted
function update_ids!(block::MondrianBlock, new_id::String, parent_id::Union{String, Nothing}=nothing)
    # Update the ID and parent
    block.id = new_id
    block.parent = parent_id

    # Recursively update children if they exist
    if block.L !== nothing
        update_ids!(block.L, new_id * "L", new_id)
    end
    if block.R !== nothing
        update_ids!(block.R, new_id * "R", new_id)
    end

    return block
end

# extend an existing mondrian block by potentially inserting a new node
function extend_mondrian_block(block::MondrianBlock, x_new::Vector{Float64}, τ_parent::Float64, x_new_idx::Int, min_samples_split::Int)
    n, d = size(block.X)
    lower, upper = map(f -> map(f, eachcol(block.X)), (minimum, maximum))
    el, eu = (max.(lower - x_new, zeros(d)), max.(x_new - upper, zeros(d)))
    size_cell = sum(el + eu)
    E = rand(Exponential(1 / size_cell))
    if τ_parent + E < block.τ
        split_probabilities = collect(el + eu) ./ size_cell
        split_axis = rand(DiscreteNonParametric(1:d, split_probabilities))
        if x_new[split_axis] > upper[split_axis]
            split_location = rand(Uniform(upper[split_axis], x_new[split_axis]))
        else
            split_location = rand(Uniform(x_new[split_axis], lower[split_axis]))
        end

        if x_new[split_axis]  <= split_location
            block_left = MondrianBlock(block.id * "L", reshape(x_new, 1, :), [x_new_idx], min_samples_split, block.τ)
            block_right = deepcopy(block)
            update_ids!(block_right, block.id * "R", block.id)
        else
            block_left = deepcopy(block)
            update_ids!(block_left, block.id * "L", block.id)
            block_right = MondrianBlock(block.id * "R", reshape(x_new, 1, :), [x_new_idx], min_samples_split, block.τ)
        end
        block = MondrianBlock(block.id, [block.X; x_new'], push!(block.N, x_new_idx), true, split_axis, split_location, τ_parent + E, block_left, block_right, block.parent)
    else
        block.X = [block.X; x_new']
        push!(block.N, x_new_idx)
        if block.split
            if x_new[block.δ] <= block.ξ
                block.L = extend_mondrian_block(block.L, x_new, block.τ, x_new_idx, min_samples_split)
            else
                block.R = extend_mondrian_block(block.R, x_new, block.τ, x_new_idx, min_samples_split)
            end
        end
    end
    return block
end


# function to extend the tree
function extend!(tree::MondrianTree, x_new::Vector{Float64}, y_new::Float64)
    # generate a new block
    new_block = extend_mondrian_block(tree.block, x_new, Inf, Int(length(tree.y) + 1), tree.min_samples_split)

    # generate tree based on extended block
    values = map(f -> collect_data(new_block, f), [:id, :split, :N, :δ, :ξ, :τ])
    tree.nodes   = values[1]
    tree.is_leaf = .!values[2]
    tree.N       = values[3]
    tree.δ       = values[4]
    tree.ξ       = values[5]
    tree.τ       = values[6]
    tree.X       = [tree.X; x_new']
    tree.y       = [tree.y; y_new]
    tree.block   = new_block

    return tree
end

# predict function for MondrianTree objects
# we use the approximation using the empirical mean and variance for each node
# see Lakshminarayanan et. al. (2016, Appendix C)
function predict(tree::MondrianTree, x_new::Vector{Float64})
    n, d = size(tree.X)
    node = ""
    τ_parent = 0.0
    p_not_separated_yet = 1.0
    w, m, v = (Float64[], Float64[], Float64[]) # storage objects for the weights, mean, and std for each node on the path
    while true
        node_id = findfirst(node .== tree.nodes)
        Δ = tree.τ[node_id] - τ_parent
        lower, upper = map(f -> map(f, eachcol(tree.X[tree.N[node_id], :])), (minimum, maximum))
        η_x = sum(max.(x_new - upper, zeros(d)) + max.(lower - x_new, zeros(d)))
        p_x = 1 - exp(-Δ * η_x)
        if isnan(p_x) # if Δ = Inf but η = 0, we get p_x = NaN
            p_x = 0.0 # We set it to 0 in that case
        end

        y_node = tree.y[tree.N[node_id]] # get all the y-values in the node
        push!(m, mean(y_node))

        # if there is only one observation in a leaf node,
        # set the std to 0.0, i.e. the predictive becomes a point mass
        y_std = length(y_node) > 1 ? std(y_node) : 0.0
        push!(v, y_std)

        if tree.is_leaf[node_id]
            push!(w, 1 - sum(w))
            break
        else
            push!(w, p_x * p_not_separated_yet)
            p_not_separated_yet = (1 - p_x) * p_not_separated_yet
            τ_parent = tree.τ[node_id]
            if x_new[tree.δ[node_id]] <= tree.ξ[node_id]
                node = node * "L"
            else
                node = node * "R"
            end
        end
    end

    # Sometimes w does not exaclty sum to 1 (I assume this is due to small precision errors accumulating)
    # In that case, we divide by its sum
    w = clamp.(w, 0.0, 1.0)
    w ./= sum(w)

    # Return the posterior predictive (a mixture of Gaussians)
    return MixtureModel(map((μ, σ) -> Normal(μ, σ), m, v), w)
end


# Create forest object
# And extend the functions by iterating over every tree in the forest
mutable struct MondrianForest
    trees::Vector{MondrianTree} # a vector containing the individual trees
    num_trees::Int # the number of trees in the forest
end

function MondrianForest(y::Vector{Float64}, X::Matrix{Float64}, min_samples_split::Int, num_trees::Int)
    trees = [MondrianTree(y, X, min_samples_split) for _ in 1:num_trees]
    MondrianForest(trees, num_trees)
end

function predict(forest::MondrianForest, x_new)
    tree_preds = [predict(tree, x_new) for tree in forest.trees]
    return MixtureModel(tree_preds)
end

function extend!(forest::MondrianForest, x_new::Vector{Float64}, y_new::Float64)
    forest.trees = [extend!(tree, x_new, y_new) for tree in forest.trees]
    return forest
end
