
include("MondrianTrees.jl")
# test
function generate_data(n::Int, s::Real = 1, beta::Real = 1)
    alpha = 0.0
    gamma = 0.0
    delta = fill(s, 10)
    Sigma = [1.0 0.6; 0.6 1.0]

    mvnorm = MvNormal(zeros(2), 0.6 * Sigma)
    u = exp.(rand(mvnorm, n)')  # size (n, 2)

    z = rand(Uniform(0, 1), n, 10)
    x = gamma .+ z * delta .+ u[:, 1]
    y = alpha .+ beta * x .+ u[:, 2]

    return (y = y, x = x, z = z)
end


n = 500
y, x, z = generate_data(n)

N = 5000
y_full, x_full, z_full = (Vector{Float64}(undef, N), Matrix{Float64}(undef, N, size(x, 2)), Matrix{Float64}(undef, N, size(z, 2)))
y_full[1:n], x_full[1:n, :], z_full[1:n,:] = (y, x, z)

tree = MondrianTree(y, x[:,:], 10)
for i in (n+1):N
    new_idx = sample(1:(i-1), 1)[1]
    x_full[i, :], z_full[i, :] = (x_full[new_idx,:], z_full[new_idx,:])
    y_full[i] = rand(predict(tree, x_full[i, :]))
    extend!(tree, x_full[i, :], y_full[i])
end


using StatsPlots
density(y_full[(n+1):N], label = "Imputed")
density!(y_full[1:n], label = "Original")



