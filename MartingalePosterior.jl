
include("MondrianTrees.jl")
using RCall

# Two-Stage Least Squares (TSLS)
add_intercept(x) = [ones(eltype(x), size(x, 1)) x] # auxiliary function to add column of ones to matrix x
project(X) = X * inv(X'X) * X' # projection onto the space spanned by X

function tsls(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat)
    X, Z = map(add_intercept, (x, z))
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


# return a single sample from the martingale posterior
function mp_sample(y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat, criterion::Function, N::Int, num_trees::Int)
    n = length(y)
    y_full, x_full, z_full = (Vector{eltype(y)}(undef, N), Matrix{eltype(x)}(undef, N, size(x, 2)), Matrix{eltype(z)}(undef, N, size(z, 2)))
    y_full[1:n], x_full[1:n, :], z_full[1:n,:] = (y, x, z)
    forest = MondrianForest(y, [x z], 10, num_trees)
    for i in (n+1):N
        new_idx = sample(1:(i-1), 1)[1]
        x_full[i, :], z_full[i, :] = (x_full[new_idx,:], z_full[new_idx,:])
        y_full[i] = rand(predict(forest, [x_full[i, :]; z_full[i, :]]))
        extend!(forest, [x_full[i, :]; z_full[i, :]], y_full[i])
    end
    return criterion(y_full, x_full, z_full)
end

# implement the martingale posterior approach
# No need to add an intercept in x and z (will be done automatically)
function martingale_posterior(
    y::AbstractVector, x::AbstractVecOrMat, z::AbstractVecOrMat;
    criterion::Function = tsls,
    N::Int = 5*length(y), B::Int = 100, num_trees::Int = 1
)  
    results = Matrix{Float64}(undef, size(x, 2) + 1, B) # each column is a posterior sample
    #Threads.@threads for i in 1:B
    for i in 1:B
        results[:, i] = mp_sample(y, x, z, criterion, N, num_trees)
    end
    return results
end
