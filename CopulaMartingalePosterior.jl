
## This file implements a martingale posterior based on the copula update ## 
using Statistics, Distributions

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

# pdf object
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

# cdf object
struct MartingalePosteriorCDF
    V::Vector{Float64}
    α::Vector{Float64}
    ρ::Float64
    F0::Function
end

function (mc::MartingalePosteriorCDF)(x)
    F = mc.F0(x)
    @inbounds for i in eachindex(mc.V)
        F = step_cdf(F, mc.V[i], mc.α[i], mc.ρ)
    end
    return F
end


function fit_observed_data(y, ρ, α_seq, P0)
    n = length(y)
    v_obs = zeros(n)
    log_score = 0.0
    
    p0 = x -> pdf(P0, x)
    F0 = x -> cdf(P0, x)

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

function choose_optimal_rho(y, rho_candidates, α_seq, P0)
    best_ρ = rho_candidates[1]
    max_log_score = -Inf
    best_v_obs = Float64[]

    for ρ in rho_candidates
        # Calculate log-score for this specific rho
        v_obs, log_score = fit_observed_data(y, ρ, α_seq, P0)
        
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
    
    best_ρ, v_obs = choose_optimal_rho(y, rho_candidates, α_seq, P0)
    println("Optimal ρ found: ", best_ρ)

    p0 = x -> pdf(P0, x)
    F0 = x -> cdf(P0, x)

    pdfs = Vector{MartingalePosteriorPDF}(undef, B)
    cdfs = Vector{MartingalePosteriorCDF}(undef, B)
    
    for b in 1:B
        v_sim = rand(Uniform(0, 1), N - n)
        v_full = [v_obs; v_sim]
        
        pdfs[b] = MartingalePosteriorPDF(v_full, α_seq, best_ρ, p0, F0)
        cdfs[b] = MartingalePosteriorCDF(v_full, α_seq, best_ρ, F0)
    end

    return (pdfs = pdfs, cdfs = cdfs, best_ρ = best_ρ)
end