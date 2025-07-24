
# This file creates the plots to illustrate the simulation results

using JLD2
using Printf

## function that creates a table with the simulation results
function generate_latex_table_from_dict(data::Dict{String, Any})
    keys_sorted = sort(collect(keys(data)), by = x -> parse(Int, split(x, "=")[end]))
    metrics = [:MAE, :Coverage, :MIL]
    metric_labels = ["MAE", "Coverage", "MIL"]
    n_metrics = length(metrics)

    first_key = keys_sorted[1]
    methods = data[first_key].methods
    n_methods = length(methods)

    # Compute best values for bolding
    best_indices = Dict{String, Dict{Symbol, Int}}()
    for k in keys_sorted
        scenario = data[k]
        best_indices[k] = Dict{Symbol, Int}()
        for metric in metrics
            values = scenario[metric]
            if metric == :Coverage
                diffs = abs.(values .- 0.95)
                best_indices[k][metric] = LinearIndices(diffs)[argmin(diffs)]
            else
                best_indices[k][metric] = LinearIndices(values)[argmin(values)]
            end
        end
    end

    # Begin LaTeX
    latex = "\\begin{tabular}{l" * "c"^(n_metrics * length(keys_sorted)) * "}\n"
    latex *= "\\toprule\n"
    latex *= " & " * join(["\\multicolumn{$n_metrics}{c}{\$ $k \$}" for k in keys_sorted], " & ") * " \\\\\n"

    # Metric headers
    latex *= "Method"
    for _ in keys_sorted
        for label in metric_labels
            latex *= " & $label"
        end
    end
    latex *= " \\\\\n\\midrule\n"

    # Rows
    for i in 1:n_methods
        row = methods[i]
        for k in keys_sorted
            scenario = data[k]
            for metric in metrics
                val = scenario[metric][i]
                is_best = (i == best_indices[k][metric])
                formatted = @sprintf("%.3f", val)
                row *= is_best ? " & \\textbf{$formatted}" : " & $formatted"
            end
        end
        latex *= row * " \\\\\n"
    end

    latex *= "\\bottomrule\n\\end{tabular}"
    return println(latex)
end



## Plot results of the invalid instrument simulation
data_invalid = load("Results_Invalid.jld2")
generate_latex_table_from_dict(data_invalid)
