

include("MartingalePosterior.jl")

using Base.Threads # for parallelisation
using ProgressMeter
using JLD2
using RCall

# Data generating function (DGP from Conley et. al., 2008)
function generate_data(n::Int, s::Real = 1, beta::Real = 1)
    alpha = 0.0
    gamma = 0.0
    delta = fill(s, 10)
    Sigma = [1.0 0.6; 0.6 1.0]

    mvnorm = MvNormal(zeros(2), 0.6 * Sigma)
    u = exp.(rand(mvnorm, n)')
    #u = rand(mvnorm, n)'

    z = rand(Uniform(0, 1), n, 10)
    x = gamma .+ z * delta .+ u[:, 1]
    y = alpha .+ beta * x .+ u[:, 2]

    return (y = y, x = x, z = z)
end

# Implement the method of Conley at. al. (2008)
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

function iv_rossi(y, x, z; B = 100)
    @rput y x z B
    R"""
    fit = bayesm::rivGibbs(
        list(y = y, x = x, w = matrix(1, ncol = 1, nrow = length(y)), z = cbind(1, z)),
        Mcmc = list(R = 2*B, keep = 2, nprint = 0),
      )
    betas = as.numeric(fit$betadraw)
    """
    @rget betas
    return betas
end


# function that computes the performance criteria
function performance_measures(posterior_sample::Vector{Float64}, true_value::Float64)
    post_median = median(posterior_sample)
    lower, upper = quantile(posterior_sample, [0.025, 0.975])
    return (
        absolute_error = abs( post_median - true_value),
        coverage_flag = lower <= true_value <= upper,
        interval_length = upper - lower
    )
end

# Wrapper function that runs the simulation
function run_simulation(s::Float64; M::Int = 100, n::Int = 100, B::Int = 100, true_value::Float64 = 1.0)
    # Preallocate arrays
    methods = ["Martingale Posterior (MF)", "Bayes IV", "Bayes IV (DP)"]
    abs_errors = zeros(length(methods), M)
    coverage_flags = falses(length(methods), M)
    interval_lengths = zeros(length(methods), M)

    #Threads.@threads for i in 1:M
    @showprogress for i in 1:M
        # Simulate data
        y, x, z = generate_data(n, s, true_value)

        # Get posterior samples
        mp_fit = martingale_posterior(y, x, z; B = B)
        mp = getindex.(mp_fit, 2)
        iv = iv_rossi(y, x, z; B = B)
        ivdp = iv_conley(y, x, z; B = B)

        # compute performance criteria
        abs_errors[1, i], coverage_flags[1, i], interval_lengths[1, i] = performance_measures(mp, true_value)
        abs_errors[2, i], coverage_flags[2, i], interval_lengths[2, i] = performance_measures(iv, true_value)
        abs_errors[3, i], coverage_flags[3, i], interval_lengths[3, i] = performance_measures(ivdp, true_value)
    end

    # Compute performance measures
    mae = median(abs_errors; dims = 2)
    coverage = mean(coverage_flags; dims = 2)
    median_interval_length = mean(interval_lengths; dims = 2)

    return (MAE = mae, Coverage = coverage, MIL = median_interval_length)
end

ss = [0.5, 1.0, 1.5]
result = map(run_simulation, ss)
print(result)
save("Results_Conley.jld2", Dict("s=0.5" => result[1], "s=1" => result[2], "s=1.5" => result[3]))

