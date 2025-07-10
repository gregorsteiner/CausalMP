
include("MondrianTrees.jl")

# Two-Stage Least Squares (TSLS)
add_intercept(x) = [ones(eltype(x), size(x, 1)) x] # auxiliary function to add column of ones to matrix x

function tsls(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat)
    X, Z = map(add_intercept, (x, z))
    P_Z = Z * inv(Z'Z) * Z'
    β_hat = inv(X' * P_Z * X) * X' * P_Z * y
    return β_hat
end

# return a single sample from the martingale posterior
function mp_sample(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, N)
    n = length(y)
    y_full, x_full, z_full = (Vector{eltype(y)}(undef, N), Matrix{eltype(x)}(undef, N, size(x, 2)), Matrix{eltype(z)}(undef, N, size(z, 2)))
    y_full[1:n], x_full[1:n, :], z_full[1:n,:] = (y, x, z)
    forest = MondrianForest(y, x[:,:], 10, 5)
    for i in (n+1):N
        new_idx = sample(1:(i-1), 1)[1]
        x_full[i, :], z_full[i, :] = (x_full[new_idx,:], z_full[new_idx,:])
        y_full[i] = rand(predict(forest, x_full[i, :]))
        extend!(forest, x_full[i, :], y_full[i])
    end
    return tsls(y_full, x_full, z_full)
end

# implement the martingale posterior approach
# No need to add an intercept in x and z (will be done automatically)
function martingale_posterior(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat;
    N = 5*length(y), B = 100
)  
    results = Matrix{Float64}(undef, size(x, 2) + 1, B) # each column is a posterior sample
    #Threads.@threads for i in 1:B
    for i in 1:B
        results[:, i] = mp_sample(y, x, z, N)
    end
    return results
end
