

using JLD2
using Plots


## Plot results of the Conley et al (2008) simulation ##
data = load("Results_Conley.jld2")
method_names = ["MPIV (MF)", "Bayes IV", "Bayes IV (DP)"]

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

# Colors for methods
colors = [:blue, :red, :green]


# Build the three subplots without legends
subplots = []
for (i, measure) in enumerate(measures)
    sp = plot(title = "", xlabel = "s", ylabel = measure, legend = false)
    for method in 1:n_methods
        plot!(sp, s_vals, measure_data[i][method, :],
              label = "", color = colors[method], lw = 2, marker = :circle)
    end
    push!(subplots, sp)
end

# Create dummy plot just for the centered legend

# Create dummy plot to generate horizontal legend
legend_plot = plot(legend = :bottom, legendcolumns = 3, grid = false, framestyle = :none, size = (800, 100))
for method in 1:n_methods
    plot!(legend_plot, [NaN], [NaN], label = method_names[method],
          color = colors[method], lw = 2, marker = :circle)
end
# Combine all in vertical layout: subplots + legend
final_plot = plot(subplots..., legend_plot, layout = @layout [a b c; d{0.1h}])

savefig(final_plot, "Results_Conley.pdf")
