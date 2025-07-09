
include("MondrianTrees.jl")

#using Base.Threads # for parallelisation

# Data generating function (DGP from Conley et. al., 2008)
function generate_data(n::Int, s::Real = 1, beta::Real = 1)
    alpha = 0.0
    gamma = 0.0
    delta = fill(s, 10)
    Sigma = [1.0 0.6; 0.6 1.0]

    mvnorm = MvNormal(zeros(2), 0.6 * Sigma)
    #u = exp.(rand(mvnorm, n)')
    u = rand(mvnorm, n)'

    z = rand(Uniform(0, 1), n, 10)
    x = gamma .+ z * delta .+ u[:, 1]
    y = alpha .+ beta * x .+ u[:, 2]

    return (y = y, x = x, z = z)
end


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
    tree = MondrianTree(y, x[:,:], 10)
    for i in (n+1):N
        new_idx = sample(1:(i-1), 1)[1]
        x_full[i, :], z_full[i, :] = (x_full[new_idx,:], z_full[new_idx,:])
        y_full[i] = rand(predict(tree, x_full[i, :]))
        extend!(tree, x_full[i, :], y_full[i])
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




n, N = (100, 500)
y, x, z = generate_data(n, 1, 1)


y_full, x_full, z_full = (Vector{eltype(y)}(undef, N), Matrix{eltype(x)}(undef, N, size(x, 2)), Matrix{eltype(z)}(undef, N, size(z, 2)))
y_full[1:n], x_full[1:n, :], z_full[1:n,:] = (y, x, z)

forest = MondrianForest(y, x[:, :], 5, 20)
for i in (n+1):N
    new_idx = sample(1:(i-1), 1)[1]
    x_full[i, :], z_full[i, :] = (x_full[new_idx,:], z_full[new_idx,:])
    y_full[i] = rand(predict(forest, x_full[i, :]))
    extend!(forest, x_full[i, :], y_full[i])
end


(y[1], x[1, :])
predict(forest, x[1, :]) |> rand


using StatsPlots
scatter(x_full[1:n], y_full[1:n], label = "Original")
scatter!(x_full[(n+1):end], y_full[(n+1):end], label = "Imputed")

