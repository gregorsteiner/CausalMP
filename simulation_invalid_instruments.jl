# This file implements an experiment where some of the instruments are invalid
using Distributions, LinearAlgebra, Random
using gIVBMA

include("MartingalePosterior.jl")

# data generating process
function gen_data(n, s; τ = 1, p = 10, c = 0.6)
    Z = rand(MvNormal(zeros(p), I), n)'

    α = γ = 1
    δ = ones(p) .* 5/32 # chosen s.t. the first-stage R^2 is approximately 0.2
    β = [ones(s); zeros(p-s)]

    u = rand(MixtureModel([MvNormal([-1, -1], [1 c; c 1]), MvNormal([1, 1], [1 c; c 1])]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ Z * β + u[:,1]

    return (y = y, x = x, Z = Z)
end


# Run for a single simulated dataset
Random.seed!(41)
y, x, z = gen_data(200, 3)

res_givbma = givbma(y, x, z)
res_mp = martingale_posterior(y, x, z; criterion = sisvive, B = 200)


p = plot(rbw(res_givbma), label = "gIVBMA", xlabel = "β", ylabel = "Posterior Density")
density!(res_mp[2, :], label = "MP sisVIVE (MF)")
vline!([1.0], linestyle = :dash, label = "True β")
savefig(p, "Invalid_Instruments_Example.pdf")







