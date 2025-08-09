
include("MondrianTrees.jl")
include("estimators.jl")

using InvertedIndices
using ThreadsX


# efficient influence function for the linear iv model
function eif_linear_iv(y, x, z, β; intercept = true)
    n = length(y)

    if intercept
        X, Z = map(add_intercept, (x, z))
    else
        X, Z = x, z
    end

    # estimate the ``fisher information'' on the previous n-1 observations
    fi_hat = sum([Z[j, :] * X[j, :]' for j in 1:n-1]) / (n-1)

    # compute the efficient influence function for the n-th observation
    eif = inv(fi_hat) * Z[n, :] * (y[n] - dot(X[n, :], β))

    # In the scalar case return the scalar, otherwise return the vector
    return eif isa AbstractVector && length(eif) == 1 ? eif[1] : eif
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
    β_init, N::Int, ξ::Float64
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
        β = β + eif_ate(y_full[1:i], x_full[1:i, :], w_full[1:i, :], β) / (i^ξ)
    end

    return β
end

# method for IV estimation
function mp_sample_iv(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat,
    β_init, N::Int, ξ::Float64
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
        β = β + eif_linear_iv(y_full[1:i], x_full[1:i, :], z_full[1:i, :], β) / (i^ξ)
    end

    return β
end

# method for DDML IV estimation
function mp_sample_ddml_iv(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, w::AbstractVecOrMat,
    β_init, N::Int, ξ::Float64,
    l::MondrianForest, r::MondrianForest, m::MondrianForest
)
    n = length(y)
    y_full, x_full, z_full, w_full = Vector{eltype(y)}(undef, N), Matrix{eltype(x)}(undef, N, size(x, 2)), Matrix{eltype(z)}(undef, N, size(z, 2)), Matrix{eltype(w)}(undef, N, size(w, 2))
    y_full[1:n] .= y
    x_full[1:n, :] .= x
    z_full[1:n, :] .= z

    y_tilde, x_tilde, z_tilde = Vector{eltype(y)}(undef, N), Matrix{eltype(x)}(undef, N, size(x, 2)), Matrix{eltype(z)}(undef, N, size(z, 2))
    y_tilde[1:n] .= [y[j] - mean(predict(l, w[j, :])) for j in 1:n]
    x_tilde[1:n, :] .= [x[j, 1] - mean(predict(r, w[j, :])) for j in 1:n]
    z_tilde[1:n, :] .= [z[j, 1] - mean(predict(m, w[j, :])) for j in 1:n]

    β = β_init
    for i in (n+1):N
        # predict new observatio
        new_idx = sample(1:(i-1), 1)[1]
        y_full[i], x_full[i, :], z_full[i, :], w_full[i, :] = y_full[new_idx], x_full[new_idx, :], z_full[new_idx, :], w_full[new_idx, :]

        # partial out covariates and update forests
        y_tilde[i], x_tilde[i, 1], z_tilde[i, 1] = y_full[i] - mean(predict(l, w_full[i, :])), x_full[i, 1] - mean(predict(r, w_full[i, :])), z_full[i, 1] - mean(predict(m, w_full[i, :]))
        extend!(l, w_full[i, :], y_full[i])
        extend!(r, w_full[i, :], x_full[i, 1])
        extend!(m, w_full[i, :], z_full[i, 1])

        # update β estimate
        β = β + eif_linear_iv(y_tilde[1:i], x_tilde[1:i, :], z_tilde[1:i, :], β; intercept = false) / (i^ξ)
    end

    return β
end


# implement the martingale posterior approach
# No need to add an intercept in x, z, or W (will be done automatically)
function martingale_posterior(
    y::AbstractVector, x::AbstractVecOrMat;
    w::Union{Nothing, AbstractVecOrMat} = nothing, z::Union{Nothing, AbstractVecOrMat} = nothing,
    N::Int = 5 * length(y), B::Int = 100, ξ::Float64 = 1.0, num_trees::Int = 2
)
    # check if instruments are provided
    type = isnothing(z) ? "ATE" : "IV"
    if type == "IV" && !isnothing(w) # If both z and w are provided, we use DDML by default
        type = "DDML"
    end

    if type == "ATE"
        β_init = or_ate(y, x, w)

        results = ThreadsX.map(_ -> begin
                                        mp_sample_ate(y, x, w, β_init, N, ξ)
                                    end, 1:B)
    elseif type == "IV"
        β_init = tsls(y, x, z)

        results = ThreadsX.map(_ -> begin
                                        mp_sample_iv(y, x, z, β_init, N, ξ)
                                    end, 1:B)
    elseif type == "DDML"
        β_init = ddml(y, x, z, w)

        l, r, m = (MondrianForest(y, w, 10, num_trees), MondrianForest(x[:, 1], w, 10, num_trees), MondrianForest(z[:, 1], w, 10, num_trees))

        results = ThreadsX.map(_ -> begin
                                        local_l, local_r, local_m = map(x -> deepcopy(x), (l, r, m))
                                        mp_sample_ddml_iv(y, x, z, w, β_init, N, ξ, local_l, local_r, local_m)
                                    end, 1:B)
    end
    return results
end
