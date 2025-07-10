

include("MartingalePosterior.jl")

#using Base.Threads # for parallelisation

# Data generating function (DGP from Conley et. al., 2008)
function generate_data(n::Int, s::Real = 1, beta::Real = 1)
    alpha = 0.0
    gamma = 0.0
    delta = fill(s, 10)
    Sigma = [1.0 0.6; 0.6 1.0]

    mvnorm = MvNormal(zeros(2), 0.6 * Sigma)
    #u = exp.(rand(mvnorm, n)')
    u = rand(mvnorm, n)'

    z = rand(Uniform(0, 1), n, 10)
    x = gamma .+ z * delta .+ u[:, 1]
    y = alpha .+ beta * x .+ u[:, 2]

    return (y = y, x = x, z = z)
end

function run_simulation(s::Float64; M::Int = 100, n::Int = 100, true_value::Float64 = 1.0)
    # Preallocate arrays
    abs_errors = zeros(M)
    coverage_flags = falses(M)
    interval_lengths = zeros(M)

    for i in 1:M
        # Simulate data
        y, x, z = generate_data(n, s, true_value)

        # Get posterior samples
        res = martingale_posterior(y, x, z)
        post_sample = res[2, :]

        # Compute posterior statistics
        post_median = median(post_sample)
        lower = quantile(post_sample, 0.025)
        upper = quantile(post_sample, 0.975)

        # Store metrics
        abs_errors[i] = abs(post_median - true_value)
        coverage_flags[i] = lower <= true_value <= upper
        interval_lengths[i] = upper - lower
    end

    # Compute performance measures
    mae = median(abs_errors)
    coverage = mean(coverage_flags)
    median_interval_length = median(interval_lengths)

    return mae, coverage, median_interval_length
end

ss = [0.5, 1.0, 1.5]
result = map(run_simulation, ss)

