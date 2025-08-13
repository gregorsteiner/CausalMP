
using CSV, DataFrames, Random
using gIVBMA
using StatsPlots

include("MartingalePosterior.jl")
include("estimators.jl")


# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y, x, z, W = (d.GDP, d.Exprop, d.logMort, Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]]))


# run analysis
N, B, num_trees = (500, 1000, 5) # set the Martingale posterior parameters
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

plot(
    jellyfish_plot(mp_ddml, n; colour = cols[1], α = 0.2),
    jellyfish_plot(mp_tsls, n; idx = 2, colour = cols[2], α = 0.2),
    ylabel = "Effect of institutions on output"
)
ylims!(-3.0, 5.0)