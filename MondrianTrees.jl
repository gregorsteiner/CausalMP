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
    tree = MondrianTree(values[1], .!values[2], values[3], values[4], values[5], values[6], min_samples_split, X, y)
    return tree
end

# function to extend the tree
# We use a simplified version of the tree extension
# since we only consider new Xs that are exact replications of one of the previous Xs (Bayesian Bootstrap)
# Thus, we only add the new observations to all the nodes that they belong in, but do not introduce new nodes
# We could perhaps split leaf nodes if they then reach more observations than min_sample_split (TO-DO)
function extend!(tree::MondrianTree, x_new::Vector{Float64}, y_new::Float64)
    # find the index of the x observation that matches the new observation
    # we assume that at least one does, since we use a Bayesian Bootstrap to resample X
    idx_x = findfirst(row -> row == x_new, eachrow(tree.X))

    # add the new index to all nodes that contain the x observation
    N_updated = [ifelse(idx_x in N_it, push!(N_it, length(tree.y)+1), N_it) for N_it in tree.N]

    # return updated tree
    tree.N, tree.X, tree.y = (N_updated, [tree.X; x_new'], [tree.y; y_new])
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
    w, m, v = (Float64[], Float64[], Float64[]) # storage objects for the weights, mean, and sd for each node on the path
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
        # set the std to overall std
        y_std = ifelse(length(y_node) > 1, std(y_node; corrected = false), std(tree.y))
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
