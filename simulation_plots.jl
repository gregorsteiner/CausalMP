

using JLD2
using Plots

## function that plots our simulation results
function plot_simulation_results(data, method_names)
    # Sort keys by s-value
    sorted_keys = sort(collect(keys(data)), by = k -> parse(Float64, split(k, "=")[2]))
    # Extract s values
    s_vals = [parse(Float64, split(k, "=")[2]) for k in sorted_keys]
    # Number of methods
    n_methods = length(data[sorted_keys[1]][1])

    # Extract measure data
    measures = ["MAE", "Coverage", "MIL"]
    measure_data = [zeros(n_methods, length(sorted_keys)) for _ in 1:3]

    for (j, k) in enumerate(sorted_keys)
        for i in 1:3
            measure_data[i][:, j] .= data[k][i]
        end
    end

    # Build the three subplots without legends
    subplots = []
    for (i, measure) in enumerate(measures)
        sp = plot(title = "", xlabel = "s", ylabel = measure, legend = false)
        for method in 1:n_methods
            plot!(sp, s_vals, measure_data[i][method, :], label = "", lw = 2, marker = :circle)
        end
        push!(subplots, sp)
    end

    # Create dummy plot to generate horizontal legend
    legend_plot = plot(legend = :bottom, legendcolumns = n_methods, grid = false, framestyle = :none, size = (800, 100))
    for method in 1:n_methods
        plot!(legend_plot, [NaN], [NaN], label = method_names[method], lw = 2, marker = :circle)
    end
    # Combine all in vertical layout: subplots + legend
    final_plot = plot(subplots..., legend_plot, layout = @layout [a b c; d{0.1h}])
    return final_plot
end


## Plot results of the Conley et al (2008) simulation ##
data_conley = load("Results_Conley.jld2")
plot_conley = plot_simulation_results(data_conley, ["MPIV (MF)", "Bayes IV", "Bayes IV (DP)"])
savefig(plot_conley, "Results_Conley.pdf")

## Plot results of the invalid instrument simulation
data_invalid = load("Results_Invalid.jld2")
plot_invalid = plot_simulation_results(data_invalid, ["MP sisVIVE", "gIVBMA"])
