
using CSV, DataFrames, Random
using gIVBMA
using StatsPlots

include("MartingalePosterior.jl")
include("estimators.jl")


# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y, x, z, W = (d.GDP, d.Exprop, d.logMort, Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]]))


# run analysis
N, B, num_trees = (200, 10, 1) # set the Martingale posterior parameters
Random.seed!(42)

mp_ddml = martingale_posterior(y, x; z = z, w = W, N = N, B = B, num_trees = num_trees)


# plot results
plt = density(
    mp_ddml_tsls,
    linewidth = 2,
    label = "MP DDML (TSLS)", xlabel = "Effect of institutions on output", ylabel = "Posterior Density"
)
density!(mp_ddml_ols, label = "MP DDML (OLS)", linewidth = 2)
density!(clamp.(getindex.(mp_tsls, 2), -5, 5), label = "MP TSLS", linewidth = 2)
plot!(rbw(givbma_fit), label = "gIVBMA", linewidth = 2)

xlims!(-1, 2.5)
savefig(plt, "AJR_Results.pdf")
