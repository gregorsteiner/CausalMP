
include("MondrianTrees.jl")
using RCall
using InvertedIndices

# Two-Stage Least Squares (TSLS)
add_intercept(x) = [ones(eltype(x), size(x, 1)) x] # auxiliary function to add column of ones to matrix x
project(X) = X * inv(X'X) * X' # projection onto the space spanned by X

function tsls(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat; intercept::Bool = true)
    if intercept
        X, Z = map(add_intercept, (x, z))
    else
        X, Z = x, z
    end
    P_Z = project(Z)
    β_hat = inv(X' * P_Z * X) * X' * P_Z * y
    return β_hat
end

# sisVIVE criterion function (see Kang et. al., 2016)
function sisvive(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat)
    n = length(y)
    #X, Z = map(add_intercept, (x, z))
    #ϵ = (I - project(X)) * y # estimate ϵ using the linear model y ~ 1 + x + ϵ (Is this valid?)
    #x_hat = project(Z) * x
    #λ = 3 * maximum(Z' * (I - project(x_hat) * ϵ))
    @rput y x z
    R"""
    res = sisVIVE::cv.sisVIVE(y, x, z, K = 5)
    """
    @rget res
    return [0.0; res[:beta]] # add 0.0 since sisvive does not return an intercept
end

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

# DDML IV criterion function
# This implements a double/debiased machine learning approach for IV estimation in a partially linear model
# see e.g. Chernozhukov et. al. (2018, 2024)
function ddml_iv_single_split(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, w::AbstractVecOrMat; k::Int = 5, min_samples_split::Int = 10, num_trees::Int = 5)
 
    # estimate partialled-out residuals in a cross-fitted way
    l_hat, r_hat, m_hat = (similar(y), similar(x), similar(z))
    folds = kfold_indices(length(y), k)
    for fold in folds
        # fit on all observations except the ones in fold
        l_fit = MondrianForest(y[Not(fold)], w[Not(fold), :], min_samples_split, num_trees)
        r_fit = MondrianForest(x[Not(fold), 1], w[Not(fold), :], min_samples_split, num_trees)
        m_fit = MondrianForest(z[Not(fold), 1], w[Not(fold), :], min_samples_split, num_trees)

        # predict the conditional mean functions for the observations in fold
        # the code below currently assumes that x and z are only one-dimensional
        # it might be useful to generalise this later
        for idx in fold
            l_hat[idx], r_hat[idx, 1], m_hat[idx, 1] = map(fit -> mean(predict(fit, w[idx, :])), (l_fit, r_fit, m_fit))
        end
    end

    # compute partialled-out residuals
    y_tilde, x_tilde, z_tilde = (y - l_hat, x - r_hat, z - m_hat)

    # return the estimate on the partialled-out data
    return tsls(y_tilde, x_tilde, z_tilde; intercept = false)[1]
end

function ddml_iv(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, w::AbstractVecOrMat; num_split::Int = 10, k::Int = 5, min_samples_split::Int = 10, num_trees::Int = 5)
    results = map(_ -> ddml_iv_single_split(y, x, z, w; k = k, min_samples_split = min_samples_split, num_trees = num_trees), 1:num_split)
    return median(results)
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
