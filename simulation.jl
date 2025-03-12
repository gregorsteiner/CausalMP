

using Turing, DataFrames
using CairoMakie

using Pkg; Pkg.activate("../gIVBMA")
using gIVBMA

include("competing_methods.jl")
include("PostBayesSisVIVE.jl")


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

y, x, Z = gen_data(500, 1, 5, 1, 1/2)
X = [ones(length(x)) x]

#ω_tuned, covg_tuned = tune_learning_rate(y, X, Z)

chain1 = sample(sisVIVE(y, X, Z, 1/10), NUTS(), 1000)
chain100 = sample(sisVIVE(y, X, Z, 1000), NUTS(), 1000)

cols = Makie.wong_colors()

fig = Figure()
ax = Axis(fig[1, 1])
density!(ax, chain1[:"β[2]"][:, 1], label = "Post Bayes sisVIVE (ω=1/4)")
density!(ax, chain100[:"β[2]"][:, 1], label = "Post Bayes sisVIVE (ω=100)")
vlines!(ax, [1], label = "True value", color = cols[3])
fig[1, 2] = Legend(fig, ax, "", framevisible = false)
fig




# check coverage in simulation
m = 100
covg = Vector{Bool}(undef, m)
for i in 1:m
    y, x, Z = gen_data(50, 1, 5, 1, 1/2)
    X = [ones(length(x)) x]
    m = IVdemo(y, X, Z)
    chain = sample(m, NUTS(), 1000; verbose = false)
    df = DataFrame(chain)
    ci = quantile(df."β[2]", [0.025, 0.975])
    covg[i] = (ci[1] < 1) && (ci[2] > 1)
end
mean(covg)

