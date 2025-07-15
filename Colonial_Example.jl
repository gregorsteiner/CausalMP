
using CSV, DataFrames, Random

include("MartingalePosterior.jl")


# load data
d = CSV.read("Colonial_Data.csv", DataFrame)
y, x, z, W = (d.logpgp95, d.avexpr, d.logem4, Matrix(d[:, ["lat_abst", "africa", "asia", "rich4"]]))

# run analysis
Random.seed!(42)
mp_fit = martingale_posterior(y, x, z; W = W, criterion = ddml_iv, B = 500)


using StatsPlots
plt = density(
    mp_fit[2, :],
    linewidth = 2,
    label = "", xlabel = "Effect of institutions on output", ylabel = "Posterior Density")
xlims!(0.25, 2)
savefig(plt, "Colonial_Example_Results.pdf")
