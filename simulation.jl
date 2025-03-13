

using Turing, DataFrames
using CairoMakie

include("competing_methods.jl")
include("PostBayesTSLS.jl")

using Pkg; Pkg.activate("../gIVBMA")
using gIVBMA


function gen_data(n, τ, p, s, c)
    Z = rand(MvNormal(zeros(p), I), n)'

    α = γ = 1
    δ = ones(p) .* 5/32 # chosen s.t. the first-stage R^2 is approximately 0.2
    β = [ones(s); zeros(p-s)]

    u = rand(MixtureModel(MvNormal[MvNormal([-2, -2], [1 c; c 1]), MvNormal([2, 2], [1 c; c 1])]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ Z * β + u[:,1]

    return (y=y, x=x, Z=Z)
end

n = 500
y, x, Z = gen_data(n, 1, 5, 0, 1/2)

tl = tune_learning_rate(y, [ones(n) x], Z)

res = PostBayesTSLS(y, [ones(n) x], Z; ω = tl[1])
res_givbma = givbma(y, x, Z)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, rbw(res_givbma)[1], color = :green)
lines!(ax, Normal(res.μ[2], res.Σ[2, 2]), color = :red)
fig


