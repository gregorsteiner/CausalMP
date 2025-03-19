

using Turing, DataFrames
using Plots

include("competing_methods.jl")
include("PostBayesTSLS.jl")

f_y(yy) = 1 + 1/3 * yy - 1/8 * yy^2

function gen_data(n, f_y; τ = 1, c = 1/2)
    Z = rand(MvNormal(zeros(10), I), n)'

    Σ = [1 c; c 1/2] ./ 5
    u = rand(MvTDist(4, [0, 0], Σ), n)'
    x = Z * [ones(5); zeros(5)] + u[:,2]
    y = f_y.(x) + u[:,1]

    return (y=y, x=x, z=Z)
end

n = 100
y, x, Z = gen_data(n, f_y)
X = [ones(length(x)) x x.^2]

PostBayesTSLS_marginal_likelihood(y, X, Z)

f(l) = PostBayesTSLS_marginal_likelihood(y, X, Z; λ = l)
plot(f, xlim = (0, 10000))

y_h, x_h, Z_h = gen_data(Int(n/10), f_y)
X_h = [ones(length(x_h)) x_h x_h.^2]

post_pred = PostBayesTSLS_posterior_predictive(y, X, Z, X_h, Z_h)
post_pred_shrunk = PostBayesTSLS_posterior_predictive(y, X, Z, X_h, Z_h; ω = 2)

map(d -> -logpdf(d, y_h), [post_pred, post_pred_shrunk])

