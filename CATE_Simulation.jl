
# Implement a simulation for the ATE
using Distributions, Random, LinearAlgebra

# Data generatung-process
# inspired by Imbens & Menzel (2021, Section 6.3)
function generate_data(n)
    W = rand(Uniform(0, 1), n)
    Y_0 = rand(MvNormal(zeros(n), 0.5*I))
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
N, B = 500, 100
ρ_candidates, ρ_x = 0.25:0.05:0.95, [0.8, 0.8]
res = mp_density(y, [x w], N, B, w -> Normal(0, 1), ρ_candidates, ρ_x)


function marginalise_mp_pdf(y_grid, x_value, mp_results; qs = [0.025, 0.5, 0.975])
    B = length(mp_results.pdfs)
    N_total = size(mp_results.pdfs[1].W, 1)
    n_y = length(y_grid)
    
    marginal_matrix = zeros(B, n_y)
    
    for b in 1:B
        current_pdf_obj = mp_results.pdfs[b]
        # The empirical covariate distribution for this bootstrap
        W_sampled = current_pdf_obj.W 
        
        for (j, y) in enumerate(y_grid)
            sum_val = 0.0
            for i in 1:N_total
                # Evaluate the conditional PDF at y given covariate i
                sum_val += current_pdf_obj(y, [x_value; W_sampled[i, 2:end]])
            end
            marginal_matrix[b, j] = sum_val / N_total
        end
    end
    
    quantile_results = [quantile(marginal_samples[:, j], qs) for j in 1:n_y]
    return quantile_results
end


res_1 = marginalise_mp_pdf(-1.2:0.02:2.2, 0, res)



# dual learner
y_0, w_0 = y[x .== 0], w[x .== 0, :]
y_1, w_1 = y[x .== 1], w[x .== 1, :]

res_0 = mp_density(y_0, w_0, N, B, w -> Normal(0, 1), ρ_candidates, ρ_x)
res_1 = mp_density(y_1, w_1, N, B, w -> Normal(0, 1), ρ_candidates, ρ_x)


using StatsPlots, LaTeXStrings, Measures

default(
    fontfamily="Computer Modern",
    titlefontsize=11, 
    guidefontsize=11, 
    tickfontsize=9, 
    legendfontsize=7,
    tick_direction=:out,
    frame=:axes, 
    grid=false,
    lw=1.5,
    background_color_legend = :transparent,
    foreground_color_legend = nothing
)

w_eval = 0.75
xx = -1.2:0.02:2.2

# plot single learner
xx_f, mu, lb, ub = calculate_posterior_stats(res.pdfs, xx, [0, w_eval])
p_s = plot(
    xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2,
    ylabel = "Estimated Density (with 95% CI)",
    label = "Y(0) | W = $(w_eval)",
    title = "S-Learner"
)
xx_f, mu, lb, ub = calculate_posterior_stats(res.pdfs, xx, [1, w_eval])
plot!(
    p_s, xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2,
    label = "Y(1) | W = $(w_eval)"
)

# plot twin learner
xx_f, mu, lb, ub = calculate_posterior_stats(res_0.pdfs, xx, [w_eval])
p_d = plot(
    xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2,
    ylabel = "Estimated Density (with 95% CI)",
    label = "Y(0) | W = $(w_eval)",
    title = "T-Learner"
)
xx_f, mu, lb, ub = calculate_posterior_stats(res_1.pdfs, xx, [w_eval])
plot!(
    p_d, xx_f, mu, 
    ribbon=(mu .- lb, ub .- mu), 
    fillalpha=0.2,
    label = "Y(1) | W = $(w_eval)"
)


# combine plots
p_comb = plot(
    p_s, p_d,
    size = (600, 300), margins = 2mm,
    legend = :topleft
)



# plot CATEs
using QuadGK
function cate(w, pdf) 
    integrand(y, w) = y * (pdf(y, [1, w]) - pdf(y, [0, w]))
    val, error = quadgk(y -> integrand(y, w), -5.0, 5.0)
    return val
end
function cate(w, pdf_1, pdf_0) 
    integrand(y, w) = y * (pdf_1(y, [w]) - pdf_0(y, [w]))
    val, error = quadgk(y -> integrand(y, w), -5.0, 5.0)
    return val
end


# plot at each value of ww
ww = 0.0:0.1:1.0

cate_samples_single = zeros(length(ww), length(res.pdfs))
cate_samples_twin = zeros(length(ww), length(res_1.pdfs))
Threads.@threads for j in eachindex(res.pdfs)
    for i in eachindex(ww)
        cate_samples_single[i, j] = cate(ww[i], res.pdfs[j])
        cate_samples_twin[i, j] = cate(ww[i], res_1.pdfs[j], res_0.pdfs[j])
    end
end

cate_mean_s = vec(mean(cate_samples_single, dims=2))
cate_lb_s   = vec([quantile(row, 0.025) for row in eachrow(cate_samples_single)])
cate_ub_s   = vec([quantile(row, 0.975) for row in eachrow(cate_samples_single)])
p_cate = plot(
    ww, cate_mean_s, 
    ribbon = (cate_mean_s .- cate_lb_s, cate_ub_s .- cate_mean_s),
    fillalpha = 0.2,
    label = "S-learner",
    xlabel = "W", ylabel = "CATE (with 95% CI)"
)

cate_mean_t = vec(mean(cate_samples_twin, dims=2))
cate_lb_t   = vec([quantile(row, 0.025) for row in eachrow(cate_samples_twin)])
cate_ub_t   = vec([quantile(row, 0.975) for row in eachrow(cate_samples_twin)])
plot!(
    p_cate,
    ww, cate_mean_t, 
    ribbon = (cate_mean_t .- cate_lb_t, cate_ub_t .- cate_mean_t),
    fillalpha = 0.2,
    label = "T-learner",
)

plot!(
    p_cate,
    w -> (1+w)/2,
    label = "True CATE",
    linestyle=:dot, colour = :grey
)

l = @layout [
    grid(1, 2)  # Top row: 2 columns
    a           # Bottom row: 1 column (spans full width)
]
p_comb = plot(
    p_s, p_d, p_cate,
    layout = l,
    size = (700, 700), margins = 2mm,
    legend = :topleft
)


savefig(p_comb, "CATE_Simulation_Illustration.pdf")