
include("MondrianTrees.jl")
include("estimators.jl")

using InvertedIndices
using ThreadsX


# efficient influence function for the linear iv model
function eif_linear_iv(y, x, z, β)
    n = length(y)

    X, Z = map(add_intercept, (x, z))

    # estimate the ``fisher information'' on the previous n-1 observations
    fi_hat = sum([Z[j, :] * X[j, :]' for j in 1:n-1]) / (n-1)

    # compute the efficient influence function for the n-th observation
    eif = inv(fi_hat) * Z[n, :] * (y[n] - dot(X[n, :], β))
    return eif
end

# efficient influence function for the ATE
# this function assumes x is binary (i.e. x = 0, 1)
function eif_ate(y, x, w, θ)
    n = length(y)

    W = add_intercept(w)
    y_train, x_train, W_train = y[1:(n-1)], x[1:(n-1)], W[1:(n-1), :]
    U_train = [x_train W_train]

    # estimate outcome and propensity model
    β_or = inv(U_train'U_train) * U_train'y_train # outcome regression parameters
    m_0, m_1 = dot([0.0; W[n, :]], β_or), dot([1.0; W[n, :]], β_or)

    β_ps = fit_logistic(x_train, W_train) # propensity model parameters
    pi_ps = predict_logistic(β_ps, W[n, :]) # predicted propensity score

    # compute the influence function
    ϕ_0 = (1 - x[n]) * (y[n] - m_0) / (1-pi_ps) + m_0
    ϕ_1 = x[n] * (y[n] - m_1) / (pi_ps) + m_1
    eif = ϕ_1 - ϕ_0 - θ
    return eif
end

# return a single sample from the martingale posterior
# this method is for the ATE estimation (including covariates W) 
function mp_sample_ate(
    y::AbstractVector, x::AbstractVecOrMat, w::AbstractVecOrMat,
    β_init, eif::Function, N::Int
)
    n = length(y)
    y_full, x_full, w_full = Vector{eltype(y)}(undef, N), Matrix{eltype(x)}(undef, N, size(x, 2)), Matrix{eltype(w)}(undef, N, size(w, 2))
    y_full[1:n] .= y
    x_full[1:n, :] .= x
    w_full[1:n, :] .= w

    β = β_init
    for i in (n+1):N
        # predict new observatio
        new_idx = sample(1:(i-1), 1)[1]
        y_full[i], x_full[i, :], w_full[i, :] = y_full[new_idx], x_full[new_idx, :], w_full[new_idx, :]

        # update β estimate
        β = β + eif(y_full[1:i], x_full[1:i, :], w_full[1:i, :], β) / i
    end

    return β
end

# this method is for the ATE estimation
function mp_sample_iv(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat,
    β_init, eif::Function, N::Int
)
    n = length(y)
    y_full, x_full, z_full = Vector{eltype(y)}(undef, N), Matrix{eltype(x)}(undef, N, size(x, 2)), Matrix{eltype(z)}(undef, N, size(z, 2))
    y_full[1:n] .= y
    x_full[1:n, :] .= x
    z_full[1:n, :] .= z

    β = β_init
    for i in (n+1):N
        # predict new observatio
        new_idx = sample(1:(i-1), 1)[1]
        y_full[i], x_full[i, :], z_full[i, :] = y_full[new_idx], x_full[new_idx, :], z_full[new_idx, :]

        # update β estimate
        β = β + eif(y_full[1:i], x_full[1:i, :], z_full[1:i, :], β) / i
    end

    return β
end


# implement the martingale posterior approach
# No need to add an intercept in x, z, or W (will be done automatically)
function martingale_posterior(
    y::AbstractVector, x::AbstractVecOrMat;
    w::Union{Nothing, AbstractVecOrMat} = nothing, z::Union{Nothing, AbstractVecOrMat} = nothing,
    N::Int = 5 * length(y), B::Int = 100
)
    # check if instruments are provided
    type = isnothing(z) ? "ATE" : "IV"

    if type == "ATE"
        # initial estimate
        β_init = aipw_ate(y, x, w)

        # Run the Martingale posterior sampling
        results = ThreadsX.map(_ -> begin
                                        mp_sample_ate(y, x, w, β_init, eif_ate, N)
                                    end, 1:B)
    elseif type == "IV"
        # initial estimate
        β_init = tsls(y, x, z)

        # Run the Martingale posterior sampling
        results = ThreadsX.map(_ -> begin
                                        mp_sample_iv(y, x, z, β_init, eif_linear_iv, N)
                                    end, 1:B)
    end
    return results
end
