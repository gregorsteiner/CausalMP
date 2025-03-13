

using Turing, DataFrames
using CairoMakie

include("competing_methods.jl")
include("PostBayesTSLS.jl")


function gen_data(n, τ, p, s, c)
    Z = rand(MvNormal(zeros(p), I), n)'

    α = γ = 1
    δ = ones(p) .* 5/32 # chosen s.t. the first-stage R^2 is approximately 0.2
    β = [ones(s); zeros(p-s)]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ Z * β + u[:,1]

    return (y=y, x=x, Z=Z)
end

y, x, Z = gen_data(500, 1, 5, 0, 1/2)
X = [ones(length(x)) x]


tl = tune_learning_rate(y, X, Z)


res = PostBayesTSLS(y, X, Z; ω = tl[1])
