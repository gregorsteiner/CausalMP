
using CSV, DataFrames
using Roots

include("CopulaMartingalePosterior.jl")

d = CSV.read("ZnAcet.csv", DataFrame)

y, x = d.Duration, d.Zinc
#y_1, y_0 = d[d.Zinc .== 1, "Duration"], d[d.Zinc .== 0, "Duration"]


# fit Martingale posterior densities
res = mp_density(
    y, x[:, :],
    500, 200,
    w -> Exponential(5.0),
    0.8, [0.8]
)


# plots
using StatsPlots, Measures, LaTeXStrings
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

function calculate_posterior_stats(pdfs, x_grid, w)
    B = length(pdfs)
    nx = length(x_grid)
    
    # Pre-allocate a matrix: Rows = x-points, Cols = Simulations
    evaluations = zeros(nx, B)
    
    # Evaluate every PDF on the grid
    for b in 1:B
        for i in 1:nx
            evaluations[i, b] = pdfs[b](x_grid[i], w)
        end
    end

    valid_indices = [all(isfinite.(evaluations[i, :])) for i in 1:nx]
    filtered_x = x_grid[valid_indices]
    filtered_evals = evaluations[valid_indices, :]
    
    # Calculate statistics across the columns (the B simulations)
    post_mean = mean(filtered_evals, dims=2)[:]
    lower_95  = [quantile(filtered_evals[i, :], 0.025) for i in 1:size(filtered_evals, 1)]
    upper_95  = [quantile(filtered_evals[i, :], 0.975) for i in 1:size(filtered_evals, 1)]
    
    return filtered_x, post_mean, lower_95, upper_95
end

# plot interventional densities
xx = 0:0.1:20

xx_f, mu, lb, ub = calculate_posterior_stats(res.pdfs, xx, 1)
p = plot(xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2, 
    label=L"Y(1)",
    xlabel = "Duration (Days)", ylabel = "Density (with 95% CI)",
    color=1
    )


xx_f, mu, lb, ub = calculate_posterior_stats(res.pdfs, xx, 0)
plot!(p, xx_f, mu, 
      ribbon=(mu .- lb, ub .- mu), 
      fillalpha=0.2, 
      label=L"Y(0)", 
      color=2
    )

p


# plot quantile treatment effects
function get_quantile(F, q; x_min=0.0, x_max=200.0)
    target_func(x) = F(x) - q
    return find_zero(target_func, (x_min, x_max))
end

function quantile_te(q, F)
    q_1 = get_quantile(y -> F(y, 1), q)
    q_0 = get_quantile(y -> F(y, 0), q)
    return q_1 - q_0
end

qtes_01 = map(F -> quantile_te(0.1, F), res.cdfs)
qtes_05 = map(F -> quantile_te(0.5, F), res.cdfs)
qtes_09 = map(F -> quantile_te(0.9, F), res.cdfs)

p_qtes = density(
    qtes_01,
    label = "p = 0.1", ylabel = "Posterior Density",
    xlabel = "QTE(p)",
    legend = :topleft
)
density!(p_qtes, qtes_05, label = "p = 0.5")
density!(p_qtes, qtes_09, label = "p = 0.9")

# combine plot
p_final = plot(p, p_qtes, size = (600, 300), margins = 2mm)
savefig(p_final, "ZnAcetExample.pdf")

