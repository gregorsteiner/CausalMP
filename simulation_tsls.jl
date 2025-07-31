# This file implements a simple simulation experiment
using Distributions, LinearAlgebra, Random
using ProgressMeter
using JLD2

include("MartingalePosterior.jl")
include("estimators.jl")

# data generating function
function generate_data(dist, n; c = 1/2, tau = 1.0)
    γ, α = (0.0, 0.0)
    δ, τ = (1.0, tau)
    
    z = rand(Uniform(0, 1), n)
    if dist == "Gaussian"
        u = rand(MvNormal([0.0, 0.0], [1.0 c; c 1.0]), n)'
    elseif dist == "t"
        u = rand(MvTDist(2, [0.0, 0.0], [1.0 c; c 1.0]), n)'
    end
    x = γ .+ z * δ + u[:,2]
    y = α .+ τ * x + u[:,1]

    return (y = y, x = x, z = z)
end


# function that computes the performance criteria
function performance_measures(posterior_sample, true_value)
    post_median = median(posterior_sample)
    lower, upper = quantile(posterior_sample, [0.025, 0.975])
    return (
        error = post_median - true_value,
        coverage_flag = lower <= true_value <= upper,
        interval_length = upper - lower
    )
end

function performance_measures(point_estimate, ci, true_value) # alternative method for the frequentist estimators
    return (
        error = point_estimate - true_value,
        coverage_flag = ci[1] <= true_value <= ci[2],
        interval_length = ci[2] - ci[1]
    )
end

# Wrapper function that runs the simulation
function run_simulation(dist::String; M::Int = 100, n::Int = 100, N::Int = 5*n, B::Int = 100, true_value::Float64 = 1.0)
    # Preallocate arrays
    methods = ["MP TSLS", "TSLS"]
    errors = zeros(length(methods), M)
    coverage_flags = falses(length(methods), M)
    interval_lengths = zeros(length(methods), M)

    @showprogress for i in 1:M
        # Simulate data
        y, x, z = generate_data(dist, n; tau = true_value)

        # Get posterior samples
        mp_fit = martingale_posterior(y, x, z; N = N, B = B, parallel = true)
        mp = getindex.(mp_fit, 2)
        tsls_fit = tsls(y, x, z; intercept = true, ci = true)

        # compute performance criteria
        errors[1, i], coverage_flags[1, i], interval_lengths[1, i] = performance_measures(mp, true_value)
        errors[2, i], coverage_flags[2, i], interval_lengths[2, i] = performance_measures(tsls_fit.beta_hat[2], tsls_fit.ci[2], true_value)
    end

    # Compute performance measures
    mae = median(abs.(errors); dims = 2)
    bias = median(errors; dims = 2)
    coverage = mean(coverage_flags; dims = 2)
    median_interval_length = median(interval_lengths; dims = 2)

    return (MAE = mae, Bias = bias, Coverage = coverage, MIL = median_interval_length, methods = methods)
end

result = map(run_simulation, ["Gaussian", "t"])
print(result)
save("Results_TSLS.jld2", Dict("Gaussian" => result[1], "t" => result[2]))
