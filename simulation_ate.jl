# This file implements a simple simulation experiment
using Distributions, LinearAlgebra, Random
using ProgressMeter
using JLD2

include("MartingalePosterior.jl")
include("estimators.jl")

# data generating function
expit(x) = 1 / (1 + exp(-x))

function generate_data(dist, n; c = 1/2, tau = 1.0)
    α = 0.0
    τ = tau
    β = [1.0, 1.0, 1.0, 0.0, 0.0] # outcome coefficients
    δ = [0.0, 1.0, 1.0, 1.0, 1.0] # propensity coefficients
    
    w = rand(Normal(0, 1), n, 5)
    x = [rand(Bernoulli(expit.(dot(w[i, :], δ)))) for i in 1:n]
    if dist == "Gaussian"
        u = rand(Normal(0, 1), n)
    elseif dist == "t"
        u = rand(TDist(2), n)
    end
    y = α .+ τ * x + w * β + u

    return (y = y, x = x, w = w)
end

y, x, w = generate_data("Gaussian", 500)
θ = or_ate(y, x, w)

eif_ate(y, x, w, nothing, θ)

res = martingale_posterior(y, x; W = w, estimator = or_ate, eif = eif_ate)
quantile(res, [0.025, 0.5, 0.975])

