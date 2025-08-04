
include("MondrianTrees.jl")
include("estimators.jl")

using InvertedIndices
using ThreadsX


# efficient influence function for the linear iv model
function eif_linear_iv(y, x, w, z, β)
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
function eif_ate(y, x, w, z, θ)
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
    ϕ_0 = (x[n] == 0) * (y[n] - m_0) / (1-pi_ps) + m_0
    ϕ_1 = (x[n] == 1) * (y[n] - m_1) / (pi_ps) + m_1
    eif = ϕ_1 - ϕ_0 - θ
    return eif
end

# return a single sample from the martingale posterior
function mp_sample(
    y::AbstractVector, x::AbstractVecOrMat, β_init,
    eif::Function, N::Int;
    W::Union{Nothing, AbstractVecOrMat} = nothing, z::Union{Nothing, AbstractVecOrMat} = nothing
)
    n = length(y)
    y_full = Vector{eltype(y)}(undef, N)
    x_full = Matrix{eltype(x)}(undef, N, size(x, 2))
    y_full[1:n] .= y
    x_full[1:n, :] .= x

    W_full = isnothing(W) ? nothing : Matrix{eltype(W)}(undef, N, size(W, 2))
    z_full = isnothing(z) ? nothing : Matrix{eltype(z)}(undef, N, size(z, 2))
    if W_full !== nothing
        W_full[1:n, :] .= W
    end
    if z_full !== nothing
        z_full[1:n, :] .= z
    end
    

    β = β_init
    for i in (n+1):N
        # predict new observatio
        new_idx = sample(1:(i-1), 1)[1]
        y_full[i], x_full[i, :] = y_full[new_idx], x_full[new_idx, :]
        if W_full !== nothing
            W_full[i, :] = W_full[new_idx, :]
        end
        if z_full !== nothing
            z_full[i, :] = z_full[new_idx, :]
        end

        # update β estimate
        β = β + eif(y_full[1:i], x_full[1:i, :], W_full[1:i, :], z_full, β) / i
    end

    return β
end


# implement the martingale posterior approach
# No need to add an intercept in x and z (will be done automatically)
function martingale_posterior(
    y::AbstractVector, x::AbstractVecOrMat;
    W::Union{Nothing, AbstractVecOrMat} = nothing, z::Union{Nothing, AbstractVecOrMat} = nothing,
    estimator::Function = tsls, eif::Function = eif_linear_iv, 
    N::Int = 5 * length(y), B::Int = 100, parallel::Bool = false
)
    # fit initial forests
    #forest_input_x = isnothing(W) ? z[:,:] : [z W]
    #forest_x = MondrianForest(x[:, 1], forest_input_x, 10, num_trees)

    #forest_input_y = isnothing(W) ? x[:,:] : [x W]
    #forest_y = MondrianForest(y, forest_input_y, 10, num_trees)

    # initial estimate
    β_init = isnothing(z) ? estimator(y, x, W) : estimator(y, x, z, W)

    # Run the Martingale posterior sampling
    if parallel
        results = ThreadsX.map(_ -> begin
            #local_forest_x = deepcopy(forest_x) # copy the forest objects for thread-safety
            #local_forest_y = deepcopy(forest_y) # otherwise the extended forests could be shared among threads 
            mp_sample(y, x, β_init, eif, N; W = W, z = z)
        end, 1:B)
    else
        results = map(_ -> mp_sample(y, x, β_init, eif, N; W = W, z = z), 1:B)
    end
    return results
end
