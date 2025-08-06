# This file implements a simple simulation experiment
using Distributions, LinearAlgebra, Random
using ProgressMeter

include("MartingalePosterior.jl")
include("estimators.jl")
include("simulation_aux.jl")

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


# Wrapper function that runs the simulation
function run_simulation(dist::String, n::Int; M::Int = 100, N::Int = 5*n, B::Int = 100, true_value::Float64 = 1.0)
    # Preallocate arrays
    methods = ["MP IV", "TSLS"]
    errors = zeros(length(methods), M)
    coverage_flags = falses(length(methods), M)
    interval_lengths = zeros(length(methods), M)

    @showprogress for i in 1:M
        # Simulate data
        y, x, z = generate_data(dist, n; tau = true_value)

        # Get posterior samples
        mp_fit = martingale_posterior(y, x; z = z, N = N, B = B)
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

    return (MAE = mae, Bias = bias, Coverage = coverage, MIL = median_interval_length, methods = methods, distribution = dist, n = n)
end

# run simulation
result = map(run_simulation, ["Gaussian", "t", "Gaussian", "t"], [50, 50, 250, 250])
print(result)

# create table with results
performance_table_latex(result)
