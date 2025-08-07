# This file implements a simple simulation experiment
using Distributions, LinearAlgebra, Random
using ProgressMeter, LaTeXStrings

include("MartingalePosterior.jl")
include("estimators.jl")
include("simulation_aux.jl")

# data generating function
function generate_data(dist, n; tau = 1.0)
    α = 0.0
    τ = tau
    β = [1.0, 1.0, 1.0, 0.0, 0.0] # outcome coefficients
    δ = [0.0, 1.0, 1.0, 1.0, 1.0] # propensity coefficients
    
    w = rand(Normal(0, 1), n, 5)
    x = [rand(Bernoulli(expit(dot(w[i, :], δ)))) for i in 1:n]
    if dist == "Gaussian"
        u = rand(Normal(0, 1), n)
    elseif dist == "t"
        u = rand(TDist(2), n)
    end
    y = α .+ τ * x + w * β + u

    return (y = y, x = x, w = w)
end

# Wrapper function that runs the simulation
function run_simulation(dist::String, n::Int; M::Int = 100, N::Int = 4*n, B::Int = 100, true_value::Float64 = 1.0)
    # Preallocate arrays
    methods = [
        L"\text{MP}~(\xi = 1)",
        L"\text{MP}~(\xi = 2/3)",
        "OR",
        "IPW",
        "AIPW"
    ]
    errors = zeros(length(methods), M)
    coverage_flags = falses(length(methods), M)
    interval_lengths = zeros(length(methods), M)

    @showprogress for i in 1:M
        # Simulate data
        y, x, w = generate_data(dist, n; tau = true_value)

        # Get posterior samples
        mp_fit = martingale_posterior(y, x; w = w, N = N, B = B, ξ = 1.0) # Our Martingale posterior approach
        mp_fit_sl = martingale_posterior(y, x; w = w, N = N, B = B, ξ = 2/3) # Our Martingale posterior approach with a lower learning rate (ξ = 2/3)
        or_fit = or_ate(y, x, w; ci = true) # Outcome regression
        ipw_fit = bootstrap_ci(y, x, w; estimator = ipw_ate) # IPW with bootstrap CI
        aipw_fit = bootstrap_ci(y, x, w; estimator = aipw_ate) # AIPW with bootstrap CI

        # compute performance criteria
        errors[1, i], coverage_flags[1, i], interval_lengths[1, i] = performance_measures(mp_fit, true_value)
        errors[2, i], coverage_flags[2, i], interval_lengths[2, i] = performance_measures(mp_fit_sl, true_value)
        errors[3, i], coverage_flags[3, i], interval_lengths[3, i] = performance_measures(or_fit.ate_hat, or_fit.ci, true_value)
        errors[4, i], coverage_flags[4, i], interval_lengths[4, i] = performance_measures(ipw_fit.ate_hat, ipw_fit.ci, true_value)
        errors[5, i], coverage_flags[5, i], interval_lengths[5, i] = performance_measures(aipw_fit.ate_hat, aipw_fit.ci, true_value)
    end

    # Compute performance measures
    mae = median(abs.(errors); dims = 2)
    bias = median(errors; dims = 2)
    coverage = mean(coverage_flags; dims = 2)
    median_interval_length = median(interval_lengths; dims = 2)

    return (MAE = mae, Bias = bias, Coverage = coverage, MIL = median_interval_length, methods = methods, distribution = dist, n = n)
end

# run simulation
result = map(run_simulation, ["Gaussian", "t", "Gaussian", "t"], [50, 50, 250, 250])
print(result)


# print latex table with results
performance_table_latex(result)

