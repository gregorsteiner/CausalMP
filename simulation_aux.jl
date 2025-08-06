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
