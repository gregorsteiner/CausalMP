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


result = map(run_simulation, ["Gaussian", "t", "Gaussian", "t"], [50, 50, 250, 250])
print(result)

## create table with results
using Printf  # for @sprintf if preferred
function performance_table_latex(results)
    dists = unique(getfield.(results, :distribution))
    ns = unique(getfield.(results, :n))
    methods = results[1].methods
    nmethods = length(methods)
    perf = [:MAE, :Bias, :Coverage, :MIL]
    ncols = length(dists) * length(perf)

    # Start table: no vertical bars for booktabs style
    table = "\\begin{tabular}{ll" * repeat("c", ncols) * "}\n"
    
    # Toprule
    table *= "\\toprule\n"
    
    # Header row 1: distribution names with multicolumn for metrics
    table *= "Sample Size & Method "
    for d in dists
        table *= "& \\multicolumn{4}{c}{" * string(d) * "}"
    end
    table *= " \\\\\n"
    
    # Header row 2: metric names
    table *= " & "  # empty for sample size & method column
    for _ in dists
        for p in perf
            table *= "& " * string(p) * " "
        end
    end
    table *= " \\\\\n"
    
    # Midrule
    table *= "\\midrule\n"

    # Data rows
    for n in ns
        for i_m in 1:nmethods
            if i_m == 1
                # Multirow with sample size printed as n in mathmode
                table *= "\\multirow{$(nmethods)}{*}{\\(n = $(n)\\)} "
            else
                table *= " "  # empty space for following rows in sample size group
            end
            # Method name column
            table *= "& " * string(methods[i_m]) * " "
            for dist in dists
                idx = findfirst(x -> x.distribution == dist && x.n == n, results)
                if idx === nothing
                    error("Missing data for distribution=$dist and n=$n")
                end
                tup = results[idx]
                for p in perf
                    val = tup[p][i_m]
                    table *= "& " * @sprintf("%.3f", val) * " "
                end
            end
            table *= " \\\\\n"
        end
        table *= "\\midrule\n"
    end
    
    # Bottomrule
    table *= "\\bottomrule\n"
    table *= "\\end{tabular}"
    return println(table)
end


performance_table_latex(result)
