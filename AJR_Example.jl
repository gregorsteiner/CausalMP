
using CSV, DataFrames, Random
using gIVBMA
using StatsPlots

include("MartingalePosterior.jl")
include("estimators.jl")


# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y, x, z, W = (d.GDP, d.Exprop, d.logMort, Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]]))


# run analysis
N, B, num_trees = (300, 100, 5) # set the Martingale posterior parameters
Random.seed!(42)

mp_ddml = martingale_posterior(y, x; z = z, w = W, N = N, B = B, num_trees = num_trees)
mp_tsls = martingale_posterior(y, [x W]; z = [z W], N = N, B = B)


# plot posteriors results
post_plt = density(
    clamp.(extract_mp(mp_ddml), -0.5, 2.5),
    linewidth = 2,
    label = "MP DDML IV", xlabel = "Effect of institutions on output", ylabel = "Posterior Density"
)
density!(
    clamp.(extract_mp(mp_tsls; idx = 2), -0.5, 2.5),
    label = "MP IV", linewidth = 2
)
xlims!(0.0, 2.0)

savefig(post_plt, "AJR_Results.pdf")

# jellyfish plot
cols = palette(:default)
n = length(y)

data_ddml = vcat(mp_ddml...)'
plot_ddml = plot(n:N, data_ddml[n:N, :], color = cols[1], alpha=0.2, label = false)
plot!(n:N, data_ddml[n:N, 1], color = cols[1], alpha=0.2, label = "MP DDML IV")

data_tsls = vcat(map(x -> x[2:2, :], mp_tsls)...)'
plot_tsls = plot(n:N, data_tsls[n:N, :], color = cols[2], alpha=0.2, label = false)
plot!(n:N, data_tsls[n:N, 1], color = cols[2], alpha=0.2, label = "MP TSLS IV")


plot(plot_ddml, plot_tsls, ylabel = "Effect of institutions on output")
ylims!(-3.0, 5.0)