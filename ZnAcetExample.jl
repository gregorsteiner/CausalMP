
using CSV, DataFrames
using Roots

include("CopulaMartingalePosterior.jl")

d = CSV.read("ZnAcet.csv", DataFrame)

y_1, y_0 = d[d.Zinc .== 1, "Duration"], d[d.Zinc .== 0, "Duration"]


# fit Martingale posterior densities
res_1 = mp_density(y_1, 500, 200, Exponential(5); rho_candidates = 0.0:0.01:0.8)
res_0 = mp_density(y_0, 500, 200, Exponential(5); rho_candidates = 0.0:0.01:0.8)

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

function calculate_posterior_stats(pdfs, x_grid)
    B = length(pdfs)
    nx = length(x_grid)
    
    # Pre-allocate a matrix: Rows = x-points, Cols = Simulations
    evaluations = zeros(nx, B)
    
    # Evaluate every PDF on the grid
    for b in 1:B
        for i in 1:nx
            evaluations[i, b] = pdfs[b](x_grid[i])
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

xx_f, mu, lb, ub = calculate_posterior_stats(res_1.pdfs, xx)
p = plot(xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2, 
    label=L"Y(1)",
    xlabel = "Duration (Days)", ylabel = "Density (with 95% CI)",
    color=1
    )


xx_f, mu, lb, ub = calculate_posterior_stats(res_0.pdfs, xx)
plot!(p, xx_f, mu, 
      ribbon=(mu .- lb, ub .- mu), 
      fillalpha=0.2, 
      label=L"Y(0)", 
      color=2
    )

p


# plot quantile treatment effects
function get_quantile(mc::MartingalePosteriorCDF, q; x_min=0.0, x_max=200.0)
    target_func(x) = mc(x) - q
    return find_zero(target_func, (x_min, x_max))
end

function quantile_te(q, F_1, F_0)
    q_1 = get_quantile(F_1, q)
    q_0 = get_quantile(F_0, q)
    return q_1 - q_0
end

qtes_01 = map((F1, F0) -> quantile_te(0.1, F1, F0), res_1.cdfs, res_0.cdfs)
qtes_05 = map((F1, F0) -> quantile_te(0.5, F1, F0), res_1.cdfs, res_0.cdfs)
qtes_09 = map((F1, F0) -> quantile_te(0.9, F1, F0), res_1.cdfs, res_0.cdfs)

p_qtes = density(
    qtes_05,
    label = "Median", ylabel = "Posterior Density",
    xlabel = "Quantile Treatment Effect",
    legend = :topleft
)
density!(p_qtes, qtes_01, label = "0.1-Quantile")
density!(p_qtes, qtes_09, label = "0.9-Quantile")

# combine plot
p_final = plot(p, p_qtes, size = (600, 300))


