
using CSV, DataFrames, Random

include("MartingalePosterior.jl")


# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y, x, z, W = (d.GDP, d.Exprop, d.logMort, Matrix(d[:, ["Latitude", "Latitude2", "Africa", "Asia", "Namer", "Samer"]]))

# run analysis
Random.seed!(42)
mp_fit = martingale_posterior(y, x, z; W = W, criterion = ddml_iv, N = 200, B = 500, num_trees = 1)

# plot results
using StatsPlots
plt = density(
    mp_fit,
    linewidth = 2,
    label = "", xlabel = "Effect of institutions on output", ylabel = "Posterior Density"
)
savefig(plt, "AJR_Results.pdf")
