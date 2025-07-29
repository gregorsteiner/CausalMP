
using CSV, DataFrames, Random
using gIVBMA
using StatsPlots

include("MartingalePosterior.jl")
include("estimators.jl")


# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y, x, z, W = (d.GDP, d.Exprop, d.logMort, Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]]))


# run analysis
N, B, num_trees, parallel = (200, 500, 5, true) # set the Martingale posterior parameters
Random.seed!(42)

mp_tsls = martingale_posterior(y, x, z; W = W, N = N, B = B, num_trees = num_trees, parallel = parallel)

mp_ddml_tsls = martingale_posterior(y, x, z; W = W, criterion = (y, x, z, W) -> ddml(y, x, z, W; iv = true), N = N, B = B, num_trees = num_trees, parallel = parallel)
mp_ddml_ols = martingale_posterior(y, x, z; W = W, criterion = (y, x, z, W) -> ddml(y, x, z, W; iv = false), N = N, B = B, num_trees = num_trees, parallel = parallel)

givbma_fit = givbma(y, x, [z W]; iter = 10000, g_prior = "hyper-g/n")


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
