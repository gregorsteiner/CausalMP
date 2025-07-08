
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


n = 100
y, x, z = generate_data(n)
tree = MondrianTree(y, x[:,:], 10)
x_new = x[10:10]


res = predict(tree, x_new)
extend!(tree, x_new, rand(res))




block = MondrianBlock("", x[:,:], collect(1:n), 10, 0.0)
extend_mondrian_block(block, x_new, 0.0, n+1)
