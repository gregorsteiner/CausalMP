### Auxiliary functions for the simulation experiments ###

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

# create latex table with simulation results
using Printf
function performance_table_latex(results)
    dists = unique(getfield.(results, :distribution))
    ns = unique(getfield.(results, :n))
    methods = results[1].methods
    nmethods = length(methods)
    perf = [:MAE, :Bias, :Coverage, :MIL]
    ncols_per_dist = length(perf)
    ncols = length(dists) * ncols_per_dist

    # Prepare storage for metrics
    metrics = Dict{Any, Dict{Symbol, Vector{Vector{Float64}}}}()
    for dist in dists
        metrics[dist] = Dict{Symbol, Vector{Vector{Float64}}}()
        for p in perf
            metrics[dist][p] = Vector{Vector{Float64}}()
        end
    end

    # Fill metrics by n, dist, p
    for n in ns
        for dist in dists
            idx = findfirst(x -> x.distribution == dist && x.n == n, results)
            if idx === nothing
                error("Missing data for distribution=$dist and n=$n")
            end
            tup = results[idx]
            for p in perf
                push!(metrics[dist][p], [tup[p][i_m] for i_m in 1:nmethods])
            end
        end
    end

    # Function to find best indices for a column of values:
    function best_indices(vals::Vector{Float64}, metric::Symbol)
        if metric == :Coverage
            dist_to_target = abs.(vals .- 0.95)
            minval = minimum(dist_to_target)
            findall(x -> isapprox(x, minval; atol=1e-8), dist_to_target)
        else
            minval = minimum(abs.(vals))
            findall(x -> isapprox(x, minval; atol=1e-8), vals)
        end
    end

    # Construct LaTeX tabular column format string with vertical lines
    # Columns: Sample Size (l), Method (l), vertical line, then for each dist: 4 ‘c’ columns, vertical line after each dist block
    col_format = "l" * "l|"  # Sample Size + Method columns, then vertical line
    for i in 1:length(dists)
        col_format *= "c"^ncols_per_dist
        # Add vertical line after each distribution block except maybe after last (optional)
        col_format *= "|"
    end

    # Replace the ^ operator with repetition...
    # Since Julia does not repeat strings with ^, do this:
    # Instead of "c"^4, do repeat("c",4)
    # So reconstruct col_format accordingly:
    col_format = "l" * "l|"  # start
    for i in 1:length(dists)
        col_format *= repeat("c", ncols_per_dist)
        if i < length(dists)
            col_format *= "|"
        end
    end

    # Begin building the table
    table = "\\begin{tabular}{" * col_format * "}\n"
    table *= "\\toprule\n"

    # Header Row 1: Sample Size & Method & | & multicolumn for distribution & vertical line after each distribution block
    table *= "Sample Size & Method "
    for (i, d) in enumerate(dists)
        table *= "& \\multicolumn{$(ncols_per_dist)}{c}{" * string(d) * "}"
    end
    table *= " \\\\\n"

    # Header Row 2: metric names
    table *= " & "
    for _ in dists
        for p in perf
            table *= "& " * string(p) * " "
        end
    end
    table *= " \\\\\n"
    table *= "\\midrule\n"

    # Data rows
    for (ni, n) in enumerate(ns)
        for i_m in 1:nmethods
            if i_m == 1
                table *= "\\multirow{$(nmethods)}{*}{\\(n = $(n)\\)} "
            else
                table *= " "
            end
            table *= "& " * string(methods[i_m]) * " "

            for dist in dists
                for p in perf
                    vals_n = metrics[dist][p][ni]
                    best_cols = best_indices(vals_n, p)

                    val = vals_n[i_m]
                    val_str = @sprintf("%.3f", val)
                    if i_m in best_cols
                        val_str = "\\textbf{$val_str}"
                    end

                    table *= "& $val_str "
                end
            end
            table *= " \\\\\n"
        end
        table *= "\\midrule\n"
    end

    table *= "\\bottomrule\n"
    table *= "\\end{tabular}"

    return println(table)
end
