

using Turing, DataFrames
using CairoMakie

using Pkg; Pkg.activate("../gIVBMA")
using gIVBMA

include("competing_methods.jl")

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


loss(y, X, Z, α, β) = 1/2 * (y - X*β - Z*α)' * Z * inv(Z'Z) * Z' * (y - X*β - Z*α)

@model function IVdemo(y, X, Z)
    β ~ MvNormal(zeros(size(X, 2)), I)

    λ ~ Exponential(1)
    α = zeros(size(Z, 2))
    for j in eachindex(α)
        α[j] ~ Laplace(0, 1 / λ)
    end

    Turing.@addlogprob! -loss(y, X, Z, α, β)
end


y, x, Z = gen_data(1000, 1, 5, 2, 1/2)
X = [ones(length(x)) x]

chain = sample(IVdemo(y, X, Z), NUTS(), 1000)
df = DataFrame(chain)

res_givbma = givbma(y, x, Z; g_prior = "hyper-g/n")


cols = Makie.wong_colors()

fig = Figure()
ax = Axis(fig[1, 1])
density!(ax, df."β[2]", label = "Post Bayes sisVIVE")
lines!(ax, rbw(res_givbma)[1], label = "gIVBMA", color = cols[2])
vlines!(ax, [1], label = "True value", color = cols[3])
vlines!(ax, tsls(y, X, Z).β_hat[2], label = "TSLS", color = cols[4])
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

