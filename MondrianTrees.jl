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

struct MondrianTree
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

# extend the tree
function extend_mondrian_block(block::MondrianBlock, x_new::Vector{Float64}, τ_parent::Float64, x_new_idx::Int)
    n, d = size(block.X)
    lower, upper = map(f -> map(f, eachcol(block.X)), (minimum, maximum))
    el, eu = (max.(lower - x_new, zeros(d)), max.(x_new - upper, zeros(d)))
    size_cell = sum(el + eu)
    E = rand(Exponential(1 / size_cell))
    if τ_parent + E < block.τ
        split_probabilities = collect(el + eu) ./ size_cell
        split_axis = rand(DiscreteNonParametric(1:d, split_probabilities))
        split_location = ifelse(
            x_new[split_axis] > upper[split_axis],
            rand(Uniform(upper[split_axis], x_new[split_axis])), 
            rand(Uniform(x_new[split_axis], lower[split_axis]))
        )
        split_bool = block.X[:, split_axis] .<= split_location
        if x_new[split_axis]  <= split_location
            block_left = MondrianBlock(block.id * "L", x_new, [x_new_idx], min_samples_split, block.τ)
            block_right = MondrianBlock(block.id * "R", block.X, block.N, min_samples_split, block.τ)
        else
            block_left = MondrianBlock(block.id * "L", block.X, block.N, min_samples_split, block.τ)
            block_right = MondrianBlock(block.id * "R", x_new, [x_new_idx], min_samples_split, block.τ)
        end
        block = MondrianBlock(block.id, [block.X; new_x], push!(block.N, x_new_idx), true, split_axis, split_location, τ_parent + E, block_left, block_right, block.parent)
    else
        if block.split
            if x_new[block.δ] <= block.ξ
                block = extend_mondrian_block(block.L, x_new, block.τ, x_new_idx)
            else
                block = extend_mondrian_block(block.R, x_new, block.τ, x_new_idx)
            end
        else
            block = block
        end
    end
    return block
end

function extend(tree::MondrianTree, y_new::Float64, x_new::Vector{Float64})


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
        upper, lower = map(f -> map(f, eachcol(tree.X[tree.N[node_id], :])), (minimum, maximum))
        η_x = sum(max.(x_new - upper, zeros(d)) + max.(lower - x_new, zeros(d)))
        p_x = 1 - exp(-Δ * η_x)

        y_node = tree.y[tree.N[node_id]] # get all the y-values in the node
        push!(m, mean(y_node))
        push!(v, std(y_node; corrected = false))

        if tree.is_leaf[node_id]
            push!(w, (1 - p_x) * p_not_separated_yet)
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

    # Return the posterior predictive (mixture of Gaussians)
    # Note that w may not exactly sum to 1 (I suppose this is due to rounding errors)
    # So we divide by its sum
    return MixtureModel(map((μ, σ) -> Normal(μ, σ), m, v), w ./ sum(w))
    #return w
end




# test
function generate_data(n::Int, s::Real = 1, beta::Real = 1)
    alpha = 0.0
    gamma = 0.0
    delta = fill(s, 10)
    Sigma = [1.0 0.6; 0.6 1.0]

    mvnorm = MvNormal(zeros(2), 0.6 * Sigma)
    u = exp.(rand(mvnorm, n)')  # size (n, 2)

    z = rand(Uniform(0, 1), n, 10)
    x = gamma .+ z * delta .+ u[:, 1]
    y = alpha .+ beta * x .+ u[:, 2]

    return (y = y, x = x, z = z)
end


n = 100
y, x, z = generate_data(n)
tree = MondrianTree(y, x[:,:], 10)
x_new = [10.0]
res = predict(tree, x_new)
