# This file implements prediction based on Mondrian trees
# See e.g. Lakshminarayanan et. al. (2014, 2016)
# Some parts of the code are taken/adapted from https://github.com/WGUNDERWOOD/MondrianForests.jl

using LinearAlgebra, Distributions


struct MondrianTree
    id::String
    X::Matrix{Float64}
    min_samples_split::Int
    creation_time::Float64
    is_split::Bool
    split_axis::Union{Int,Nothing}
    split_location::Union{Float64,Nothing}
    tree_left::Union{MondrianTree,Nothing}
    tree_right::Union{MondrianTree,Nothing}
end



function mondrian_tree(id::String, X::Matrix{Float64}, min_samples_split::Int, creation_time::Float64)
    n, d = size(X)
    lower, upper = map(f -> map(f, eachcol(X)), (minimum, maximum))
    size_cell = sum(upper .- lower)
    E = rand(Exponential(1 / size_cell))
    if n >= min_samples_split
        split_probabilities = collect(upper .- lower) ./ size_cell
        split_axis = rand(DiscreteNonParametric(1:d, split_probabilities))
        split_location = rand(Uniform(lower[split_axis], upper[split_axis]))
        split_bool = X[:, split_axis] .<= split_location
        X_left, X_right = (X[split_bool, :], X[.!split_bool, :])
        tree_left = mondrian_tree(id * "L", X_left, min_samples_split, creation_time + E)
        tree_right = mondrian_tree(id * "R", X_right, min_samples_split, creation_time + E)
        tree = MondrianTree(id, X, min_samples_split, creation_time, true, split_axis,
                            split_location, tree_left, tree_right)
    else
        tree = MondrianTree(id, X, min_samples_split, creation_time, false,
                            nothing, nothing, nothing, nothing)
    end
    return tree
end


X = rand(Normal(0, 1), 100, 10)
mondrian_tree("", X, 10, 0.0)

