
using CSV, DataFrames, Random
using gIVBMA
using StatsPlots

include("MartingalePosterior.jl")
include("estimators.jl")


# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y, x, z, W = (d.GDP, d.Exprop, d.logMort, Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]]))


# run analysis
N, B, num_trees = (300, 1000, 5) # set the Martingale posterior parameters
Random.seed!(42)

mp_ddml = martingale_posterior(y, x; z = z, w = W, N = N, B = B, num_trees = num_trees)
mp_tsls = martingale_posterior(y, [x W]; z = [z W], N = N, B = B)


# plot results
plt = density(
    clamp.(mp_ddml, -0.5, 2.5),
    linewidth = 2,
    label = "MP DDML IV", xlabel = "Effect of institutions on output", ylabel = "Posterior Density"
)
density!(
    clamp.(getindex.(mp_tsls, 2), -0.5, 2.5),
    label = "MP IV", linewidth = 2
)
xlims!(0.0, 2.0)

savefig(plt, "AJR_Results.pdf")
