
using CSV, DataFrames, Random
using gIVBMA
using StatsPlots

include("MartingalePosterior.jl")
include("estimators.jl")


# load data
d = CSV.read("AJR_Data.csv", DataFrame)
y, x, z, W = (d.GDP, d.Exprop, d.logMort, Matrix(d[:, ["Latitude", "Africa", "Asia", "Namer", "Samer"]]))


# run analysis
Random.seed!(42)
mp_iv_fit = martingale_posterior(y, x, z; W = W, criterion = (y, x, z, W) -> ddml(y, x, z, W; iv = true), N = 200, B = 500, num_trees = 1, parallel = true)
mp_ols_fit = martingale_posterior(y, x, z; W = W, criterion = (y, x, z, W) -> ddml(y, x, z, W; iv = false), N = 200, B = 500, num_trees = 1, parallel = true)

givbma_fit = givbma(y, x, [z W]; iter = 10000, g_prior = "hyper-g/n")


# plot results
plt = density(
    mp_iv_fit,
    linewidth = 2,
    label = "MP DDML (IV)", xlabel = "Effect of institutions on output", ylabel = "Posterior Density"
)
density!(mp_ols_fit, label = "MP DDML (OLS)", linewidth = 2)
plot!(rbw(givbma_fit), label = "gIVBMA", linewidth = 2)
xlims!(-1.5, 3.5)
savefig(plt, "AJR_Results.pdf")
