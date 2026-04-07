
using DataFrames, StatFiles, Statistics, Distributions

# Read data
dat = DataFrame(load("BFP_replication_data.dta"))

# Delete observation 167
deleteat!(dat, 167)

# drop observations with missing maritalstatus
dat = dat[.!ismissing.(dat.maritalstatus), :]

# only keep female & single
d_femsing = dat[(dat.male .== 0) .& (dat.maritalstatus .== 0), :]

# replace compensation intervals by mean
function mean_from_string(s)
    if ismissing(s) return missing end
    # Find all numeric patterns (integers or floats)
    matches = [parse(Float64, m.match) for m in eachmatch(r"[0-9]+\.?[0-9]*", string(s))]
    return isempty(matches) ? missing : mean(matches)
end
d_femsing.compensation = [mean_from_string(x) for x in d_femsing.desiredcompensation]


# potential outcomes
# treatment A is private and B is public
y_1, y_0 = d_femsing[d_femsing.treatment .== "B", "compensation"], d_femsing[d_femsing.treatment .== "A", "compensation"]



# implement copula update (see Fong et al, 2023)
function conditional_gaussian_copula(u, v; ρ = 0.8)
    SN = Normal(0, 1)
    x = (quantile(SN, u) - ρ * quantile(SN, v)) / sqrt(1 - ρ^2)
    return cdf(SN, x)
end

function gaussian_copula_density(u, v; ρ = 0.8)
    BN = MvNormal([0.0, 0.0], [1.0 ρ; ρ 1.0])
    SN = Normal(0, 1)
    num = logpdf(BN, quantile(SN, [u, v]))
    den = logpdf(SN, quantile(SN, u)) + logpdf(SN, quantile(SN, v))
    return exp(num - den)
end

# PDF update logic: p_{i} = [1 - α + α * c(F_{i-1}, V)] * p_{i-1}
function step_pdf(curr_p, curr_F, V, α, ρ)
    return (1 - α + α * gaussian_copula_density(curr_F, V; ρ = ρ)) * curr_p
end

# CDF update logic: F_{i} = (1 - α) * F_{i-1} + α * C(F_{i-1}, V)
function step_cdf(curr_F, V, α, ρ)
    return (1 - α) * curr_F + α * conditional_gaussian_copula(curr_F, V; ρ = ρ)
end


struct MartingalePosteriorPDF
    V::Vector{Float64}   # Combined observed and simulated V_is
    α::Vector{Float64}   # Pre-calculated alpha sequence
    ρ::Float64
    p0::Function         # Initial PDF (e.g., Normal)
    F0::Function         # Initial CDF (e.g., Normal)
end

function (mp::MartingalePosteriorPDF)(x)
    p = mp.p0(x)
    F = mp.F0(x)
    
    @inbounds for i in eachindex(mp.V)
        # 1. Update p first using the CDF from the PREVIOUS step (F_{i-1})
        p = step_pdf(p, F, mp.V[i], mp.α[i], mp.ρ)
        
        # 2. Update F for the next step (F_i)
        F = step_cdf(F, mp.V[i], mp.α[i], mp.ρ)
    end
    
    return p
end


function fit_observed_data(y, ρ, α_seq)
    n = length(y)
    v_obs = zeros(n)
    log_score = 0.0
    
    p0 = x -> pdf(Normal(100, 10), x)
    F0 = x -> cdf(Normal(100, 10), x)

    for i in 1:n
        # Reset to base for each new observation point y[i]
        p_val = p0(y[i])
        F_val = F0(y[i])
        
        # Bring the values up to the current step i-1
        for j in 1:(i-1)
            p_val = step_pdf(p_val, F_val, v_obs[j], α_seq[j], ρ)
            F_val = step_cdf(F_val, v_obs[j], α_seq[j], ρ)
        end
        
        log_score += log(p_val)
        v_obs[i] = F_val 
    end
    
    return v_obs, log_score
end

function choose_optimal_rho(y, rho_candidates, α_seq)
    best_ρ = rho_candidates[1]
    max_log_score = -Inf
    best_v_obs = Float64[]

    for ρ in rho_candidates
        # Calculate log-score for this specific rho
        v_obs, log_score = fit_observed_data(y, ρ, α_seq)
        
        if log_score > max_log_score
            max_log_score = log_score
            best_ρ = ρ
            best_v_obs = v_obs
        end
    end
    
    return best_ρ, best_v_obs
end

function mp_density(y, N, B, P0; rho_candidates = 0.0:0.1:0.9)
    n = length(y)
    α_seq = [(2 - 1/i) * (1/(i+1)) for i in 1:N]
    
    best_ρ, v_obs = choose_optimal_rho(y, rho_candidates, α_seq)
    println("Optimal ρ found: ", best_ρ)

    p0 = x -> pdf(P0, x)
    F0 = x -> cdf(P0, x)

    # Simulation Phase: Only simulate new V_i by drawing form U(0, 1)
    pdfs = Vector{MartingalePosteriorPDF}(undef, B)
    
    for b in 1:B
        v_sim = rand(Uniform(0, 1), N - n)
        v_full = [v_obs; v_sim]
        
        pdfs[b] = MartingalePosteriorPDF(v_full, α_seq, best_ρ, p0, F0)
    end

    return pdfs
end

res_1 = mp_density(y_1, 500, 200, Normal(100, 10))
res_0 = mp_density(y_0, 500, 200, Normal(120, 20))

# plot
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


xx = 50:1:250
p = density(
    y_1, label = L"Y \mid X=1", linestyle = :dot,
    xlabel = "Desired compensation (1,000 USD)", ylabel = "Density"
)

xx_f, mu, lb, ub = calculate_posterior_stats(res_1, xx)
plot!(p, xx_f, mu, 
      ribbon=(mu .- lb, ub .- mu), 
      fillalpha=0.2, 
      label=L"Y(1)", 
      color=1
    )


density!(p, y_0, label = L"Y \mid X=0", linestyle = :dot, colour = 2)
xx_f, mu, lb, ub = calculate_posterior_stats(res_0, xx)
plot!(p, xx_f, mu, 
      ribbon=(mu .- lb, ub .- mu), 
      fillalpha=0.2, 
      label=L"Y(0)", 
      color=2
    )

p
