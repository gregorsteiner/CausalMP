
using LinearAlgebra, Distributions, Random
using RCall
#using GLM, DataFrames

# Some auxiliary functions
add_intercept(x) = [ones(eltype(x), size(x, 1)) x] # auxiliary function to add column of ones to matrix x
project(X) = X * inv(X'X) * X' # projection onto the space spanned by X

# OLS
function ols(y::AbstractVector, x::AbstractVecOrMat, W::AbstractVecOrMat; intercept::Bool = true, ci::Bool = false, level::Float64 = 0.05)
    U = [x W]
    if intercept
        U = add_intercept(U)
    end
    β_hat = U'U \ U'y

    if ci
        residuals = y - U * β_hat
        σ2_hat = sum(residuals.^2) / (size(U, 1) - size(U, 2))
        cov = σ2_hat * inv(U' * U)
        ci = [β_hat[j] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov[j, j]) for j in eachindex(β_hat)]
        return (beta_hat = β_hat, ci = ci)
    end 

    return β_hat
end

function ols(y::AbstractVector, x::AbstractVecOrMat; intercept::Bool = true)
    ols(y, x, Matrix{eltype(x)}(undef, size(x, 1), 0); intercept = intercept)
end

# outcome regression estimate for the ATE
function or_ate(y::AbstractVector, x::AbstractVecOrMat, W::AbstractVecOrMat; ci::Bool = false, level::Float64 = 0.05)
    res = ols(y, x, W; ci = ci, level = level)
    if ci
        return(ate_hat = res.beta_hat[2], ci = res.ci[2]) # the 2nd index is x (an intercept is always included)
    end
    return res[2]
end

# Two-Stage Least Squares (TSLS)
function tsls(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, W::AbstractVecOrMat; intercept::Bool = true, ci::Bool = false, level::Float64 = 0.05)
    U, V = ([x W], [z W])
    if intercept
        U, V = map(add_intercept, (U, V))
    end
    P_V = project(V)
    β_hat = (U' * P_V * U) \ (U' * P_V * y)

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


# Manual logistic regression
expit(x) = 1.0 / (1.0 + exp(-x))
function fit_logistic(x, w; max_iter=25, tol=1e-6)
    W = add_intercept(w)
    n, p = size(W)
    β = zeros(p)
    for i in 1:max_iter
        η = W * β
        p̂ = expit.(η)
        W_diag = Diagonal(p̂ .* (1 .- p̂))
        z = η + (x .- p̂) ./ (p̂ .* (1 .- p̂) .+ eps())  # Add eps to avoid divide by zero

        λ = 1/n # heuristic penalty parameter
        penalty = λ * diagm(0 => [0.0; ones(p-1)])
        XWX = W' * W_diag * W + penalty
        XWz = W' * W_diag * z + penalty * β
        β_new = XWX \ XWz
        if norm(β_new - β) < tol
            break
        end
        β = β_new
    end
    return β
end

# Predict probability for new data
function predict_logistic(β, w_new)
    η = dot([1.0; w_new], β) # intercept is automatically included, so we need to add a 1.0
    p_new = expit(η)
    p_new = clamp(p_new, 0.005, 0.995) # trim to avoid extreme propensity scores
    return p_new
end

# Inverse propensity weighting point estimator
function ipw_ate(y, x, w)
    n = length(y)

    # fit propensity score
    β = fit_logistic(x, w)
    e = [predict_logistic(β, w[i, :]) for i in 1:n]

    # estimate ATE
    ate = mean(x .* y ./ e - (1 .- x) .* y ./ (1 .- e))
    return ate
end


# Doubly robust (AIPW) point estimator for ATE
function aipw_ate(y, x, w)
    n = length(y)

    # fit propensity score model
    β_ps = fit_logistic(x, w)
    e = [predict_logistic(β_ps, w[i, :]) for i in 1:n]

    # fit outcome regression models
    U = [x add_intercept(w)]
    β_or = U'U \ U'y

    # Predict E[Y | X=1, W=w] and E[Y | X=0, W=w]
    μ1_hat = [dot([1.0; 1.0; w[i, :]], β_or) for i in 1:n]
    μ0_hat = [dot([0.0; 1.0; w[i, :]], β_or) for i in 1:n]

    # Calculate AIPW "pseudo-outcome" for each individual
    aipw = [μ1_hat[i] - μ0_hat[i] +
            (x[i] * (y[i] - μ1_hat[i]) / e[i]) -
            ((1 - x[i]) * (y[i] - μ0_hat[i]) / (1 - e[i]))
            for i in 1:n]

    # Return ATE estimate
    return mean(aipw)
end

# Bootstrapped CI for any ATE estimator
function bootstrap_ci(y, x, w; estimator = aipw_ate, n_boot = 1000)
    n = length(y)
    ates = zeros(n_boot)
    for b in 1:n_boot
        inds = sample(1:n, n, replace=true)
        y_b, x_b, w_b = y[inds], x[inds], w[inds, :]
        try
            ates[b] = estimator(y_b, x_b, w_b)
        catch error
            ates[b] = NaN
        end
    end
    ate_hat = estimator(y, x, w)

    ates = ates[.!isnan.(ates)]
    ci = quantile(ates, [0.025, 0.975])

    return (ate_hat = ate_hat, ci = ci)
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

# DDML IV
# This implements a double/debiased machine learning approach for IV estimation in a partially linear model
# see e.g. Chernozhukov et. al. (2018, 2024)
function ddml_single_split(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, w::AbstractVecOrMat; iv::Bool = true, k::Int = 5, min_samples_split::Int = 10, num_trees::Int = 5)
    # estimate partialled-out residuals in a cross-fitted way
    l_hat, r_hat = (similar(y), similar(x))
    if iv
        m_hat = similar(z)
    end
    folds = kfold_indices(length(y), k)
    for fold in folds
        # fit on all observations except the ones in fold
        l_fit = MondrianForest(y[Not(fold)], w[Not(fold), :], min_samples_split, num_trees)
        r_fit = MondrianForest(x[Not(fold), 1], w[Not(fold), :], min_samples_split, num_trees)
        if iv
            m_fit = MondrianForest(z[Not(fold), 1], w[Not(fold), :], min_samples_split, num_trees)
        end

        # predict the conditional mean functions for the observations in fold
        # the code below currently assumes that x and z are only one-dimensional
        # it might be useful to generalise this later
        for idx in fold
            if iv
                l_hat[idx], r_hat[idx, 1], m_hat[idx, 1] = map(fit -> mean(predict(fit, w[idx, :])), (l_fit, r_fit, m_fit))
            else
                l_hat[idx], r_hat[idx, 1] = map(fit -> mean(predict(fit, w[idx, :])), (l_fit, r_fit))
            end
        end
    end

    # compute partialled-out residuals
    y_tilde, x_tilde = (y - l_hat, x - r_hat)

    # return the estimate on the partialled-out data
    # this is the tsls estimator if iv = true, and the OLS estimator otherwise
    if iv
        z_tilde = z - m_hat
        return tsls(y_tilde, x_tilde, z_tilde; intercept = false)[1]
    else
        return ols(y_tilde, x_tilde; intercept = false)[1]
    end
end

function ddml(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, w::AbstractVecOrMat; iv::Bool = true, num_split::Int = 10, k::Int = 5, min_samples_split::Int = 10, num_trees::Int = 5)
    results = map(_ -> ddml_single_split(y, x, z, w; iv = iv, k = k, min_samples_split = min_samples_split, num_trees = num_trees), 1:num_split)
    return median(results)
end