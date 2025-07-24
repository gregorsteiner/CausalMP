
using RCall

# Some auxiliary functions
add_intercept(x) = [ones(eltype(x), size(x, 1)) x] # auxiliary function to add column of ones to matrix x
project(X) = X * inv(X'X) * X' # projection onto the space spanned by X

# Two-Stage Least Squares (TSLS)
function tsls(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, W::AbstractVecOrMat; intercept::Bool = true, ci::Bool = false, level::Float64 = 0.05)
    U, V = ([x W], [z W])
    if intercept
        U, V = map(add_intercept, (U, V))
    end
    P_V = project(V)
    β_hat = inv(U' * P_V * U) * U' * P_V * y

    if ci
        residuals = y - U * β_hat
        σ2_hat = sum(residuals.^2) / (size(U, 1) - size(U, 2))
        cov = σ2_hat * inv(U' * P_V * U)
        ci = [β_hat[j] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov[j, j]) for j in eachindex(β_hat)]
        return (beta_hat = β_hat, ci = ci)
    end 

    return β_hat
end

# define another method without covariates
function tsls(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat; intercept::Bool = true, ci::Bool = false, level::Float64 = 0.05)
    tsls(y, x, z, Matrix{eltype(x)}(undef, size(x, 1), 0); intercept = intercept, ci = ci, level = level)
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

# DDML IV
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