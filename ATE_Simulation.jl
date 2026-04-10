
# Implement a simulation for the ATE
using Distributions, Random, LinearAlgebra

# Data generatung-process
# inspired by Imbens & Menzel (2021, Section 6.3)
function generate_data(n)
    W = rand(Uniform(0, 1), n)
    Y_0 = rand(MvNormal(zeros(n), 0.25*I))
    Y_obs, X = Y_0, zeros(Int, n)
    for i in eachindex(Y_0)
        prob = (W[i] > 0.5) ? 0.6 : 0.4
        if rand(Uniform(0, 1)) < prob
            X[i] = 1
            Y_obs[i] = Y_0[i] + 1/2 * (1 + W[i])
        end
    end
    return Y_obs, X, W
end



include("CopulaMartingalePosterior.jl")


# plot for one dataset
Random.seed!(42)
y, x, w = generate_data(100)

# single learner
res = mp_density(y, [x w], 500, 100, w -> Normal(0, 1), 0.25:0.05:0.95, [0.8, 0.8])

using StatsPlots, LaTeXStrings

default(
    fontfamily="Computer Modern",
    titlefontsize=11, 
    guidefontsize=11, 
    tickfontsize=9, 
    legendfontsize=9,
    tick_direction=:out,
    frame=:axes, 
    grid=false,
    lw=1.5
)

w_eval = 0.75
xx = -1.0:0.02:2.0
xx_f, mu, lb, ub = calculate_posterior_stats(res.pdfs, xx, [0, w_eval])
plot(
    xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2,
    ylabel = "Estimated Density (with 95% CI)",
    label = "Y(0) | W = $(w_eval)"
)
xx_f, mu, lb, ub = calculate_posterior_stats(res.pdfs, xx, [1, w_eval])
plot!(
    xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2,
    label = "Y(1) | W = $(w_eval)"
)


