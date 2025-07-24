# This file implements an experiment where some of the instruments are invalid
using Distributions, LinearAlgebra, Random
using gIVBMA
using Base.Threads # for parallelisation
using ProgressMeter
using JLD2

include("MartingalePosterior.jl")
include("estimators.jl")

# data generating process
function generate_data(s; n = 100, τ = 1, p = 5, c = 0.6)
    Z = rand(MvNormal(zeros(p), I), n)'

    α = γ = 1
    δ = ones(p) .* 5/32 # chosen s.t. the first-stage R^2 is approximately 0.2
    β = [ones(s); zeros(p-s)]

    u = rand(MvTDist(2, [0.0, 0.0], [1.0 c; c 1.0]), n)'
    #u = rand(MvNormal([0.0, 0.0], [1.0 c; c 1.0]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ Z * β + u[:,1]

    return (y = y, x = x, Z = Z)
end

# Bayesian IV with DP errors (see Conley et. al, 2008)
function iv_conley(y, x, z, w; B = 100)
    @rput y x z w B
    R"""
    fit = bayesm::rivDP(
        list(y = y, x = x, w = w, z = z),
        Mcmc = list(R = 2*B, keep = 2, nprint = 0),
    )
    betas = as.numeric(fit$betadraw)
    """
    @rget betas
    return betas
end

function iv_conley(y, x, z; B = 100)
    @rput y x z B
    R"""
    fit = bayesm::rivDP(
        list(y = y, x = x, z = z),
        Mcmc = list(R = 2*B, keep = 2, nprint = 0),
    )
    betas = as.numeric(fit$betadraw)
    """
    @rget betas
    return betas
end


# function that computes the performance criteria
function performance_measures(posterior_sample, true_value)
    post_median = median(posterior_sample)
    lower, upper = quantile(posterior_sample, [0.025, 0.975])
    return (
        absolute_error = abs( post_median - true_value),
        coverage_flag = lower <= true_value <= upper,
        interval_length = upper - lower
    )
end

function performance_measures(point_estimate, ci, true_value) # alternative method for the frequentist estimators
    return (
        absolute_error = abs(point_estimate - true_value),
        coverage_flag = ci[1] <= true_value <= ci[2],
        interval_length = ci[2] - ci[1]
    )
end

# Wrapper function that runs the simulation
function run_simulation(s::Int; M::Int = 100, n::Int = 100, N::Int = 5*n, B::Int = 100, true_value::Float64 = 1.0)
    # Preallocate arrays
    methods = ["MP sisVIVE", "gIVBMA", "Bayes IV (DP)", "Oracle Bayes IV (DP)", "Naive TSLS", "Oracle TSLS"]
    abs_errors = zeros(length(methods), M)
    coverage_flags = falses(length(methods), M)
    interval_lengths = zeros(length(methods), M)

    #Threads.@threads for i in 1:M
    @showprogress for i in 1:M
        # Simulate data
        y, x, z = generate_data(s; n = n, τ = true_value)

        # Get posterior samples
        mp_fit = martingale_posterior(
            y, x, z; W = z, N = N, B = B,
            criterion = (y, x, z, W) -> sisvive(y, x, z)
        )
        mp = getindex.(mp_fit, 2)
        givbma_fit = givbma(y, x, z; iter = 3000)
        ivdp = iv_conley(y, x, z; B = 2000)
        oracle_ivdp = iv_conley(y, x, z[:, (s+1):end], z[:, 1:s]; B = 2000)
        naive_tsls = tsls(y, x, z; intercept = true, ci = true)
        oracle_tsls = tsls(y, x, z[:, (s+1):end], z[:, 1:s]; intercept = true, ci = true)

        # compute performance criteria
        abs_errors[1, i], coverage_flags[1, i], interval_lengths[1, i] = performance_measures(mp, true_value)
        abs_errors[2, i], coverage_flags[2, i], interval_lengths[2, i] = performance_measures(rbw(givbma_fit)[1], true_value)
        abs_errors[3, i], coverage_flags[3, i], interval_lengths[3, i] = performance_measures(ivdp, true_value)
        abs_errors[4, i], coverage_flags[4, i], interval_lengths[4, i] = performance_measures(oracle_ivdp, true_value)
        abs_errors[5, i], coverage_flags[5, i], interval_lengths[5, i] = performance_measures(naive_tsls.beta_hat[2], naive_tsls.ci[2], true_value)
        abs_errors[6, i], coverage_flags[6, i], interval_lengths[6, i] = performance_measures(oracle_tsls.beta_hat[2], oracle_tsls.ci[2], true_value)
    end

    # Compute performance measures
    mae = median(abs_errors; dims = 2)
    coverage = mean(coverage_flags; dims = 2)
    mean_interval_length = mean(interval_lengths; dims = 2)

    return (MAE = mae, Coverage = coverage, MIL = mean_interval_length, s = s, methods = methods)
end

# Run simulation
ss = [2, 4] # 2 scenarios: 2 or 4 of the 5 available instruments are invalid
result = map(run_simulation, ss)
print(result)
save("Results_Invalid.jld2", Dict("s=2" => result[1], "s=4" => result[2]))

