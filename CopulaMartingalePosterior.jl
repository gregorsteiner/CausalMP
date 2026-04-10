
## This file implements a martingale posterior based on the copula update ## 
using Statistics, Distributions
using Optim

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


# covariate dependent α sequence
# w_1, w_2 and ρ_x should be vectors of the same dimension
function alpha(w_1, w_2, α, ρ_x)
    Φ(x) = cdf(Normal(0, 1), x)
    copula_product = prod([gaussian_copula_density(Φ(w_1[j]), Φ(w_2[j]); ρ = ρ_x[j]) for j in eachindex(w_1)])
    return α * copula_product / (1-α+α*copula_product)
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
    W::Matrix{Float64}   # New covariate data
    α::Vector{Float64}   # Pre-calculated alpha sequence
    ρ::Float64           # ρ
    ρ_x::Vector{Float64} # vector of covariate ρs   
    p0::Function         # Initial PDF (e.g., Normal)
    F0::Function         # Initial CDF (e.g., Normal)
end

function (mp::MartingalePosteriorPDF)(y, w)
    p = mp.p0(y, w)
    F = mp.F0(y, w)
    
    @inbounds for i in eachindex(mp.V)
        # 1. Update p first using the CDF from the PREVIOUS step (F_{i-1})
        α_x = alpha(w, mp.W[i, :], mp.α[i], mp.ρ_x)
        p = step_pdf(p, F, mp.V[i], α_x, mp.ρ)
        
        # 2. Update F for the next step (F_i)
        F = step_cdf(F, mp.V[i], α_x, mp.ρ)
    end
    
    return p
end

# cdf object
struct MartingalePosteriorCDF
    V::Vector{Float64}
    W::Matrix{Float64} 
    α::Vector{Float64}
    ρ::Float64
    ρ_x::Vector{Float64}
    F0::Function
end

function (mc::MartingalePosteriorCDF)(y, w)
    F = mc.F0(y, w)
    @inbounds for i in eachindex(mc.V)
        α_x = alpha(w, mc.W[i, :], mc.α[i], mc.ρ_x)
        F = step_cdf(F, mc.V[i], α_x, mc.ρ)
    end
    return F
end


function fit_observed_data(y, W, ρ, ρ_x, P0)
    n = length(y)
    α_seq = [(2 - 1/i) * (1/(i+1)) for i in 1:n]
    v_obs = zeros(n)
    log_score = 0.0
    
    p0 = (y, w) -> pdf(P0(w), y)
    F0 = (y, w) -> cdf(P0(w), y)

    for i in 1:n
        # Reset to base for each new observation point y[i]
        p_val = p0(y[i], W[i, :])
        F_val = F0(y[i], W[i, :])
        
        # Bring the values up to the current step i-1
        for j in 1:(i-1)
            α_x = alpha(W[i, :], W[j, :], α_seq[j], ρ_x)
            p_val = step_pdf(p_val, F_val, v_obs[j], α_x, ρ)
            F_val = step_cdf(F_val, v_obs[j], α_x, ρ)
        end
        
        log_score += log(p_val)
        v_obs[i] = F_val 
    end
    
    return v_obs, -log_score
end

# generate Bayesian bootstrap sample for covariates
function bayes_bootstrap(W, N)
    n = size(W, 1)
    idx = collect(1:n)
    for _ in (n+1):N
        idx_new = sample(idx)
        push!(idx, idx_new)
    end
    return W[idx, :]
end

function mp_density(y, W, N, B, P0, ρ, ρ_x)
    n = length(y)
    α_seq = [(2 - 1/i) * (1/(i+1)) for i in 1:N]
    
    v_obs, lps = fit_observed_data(y, W, ρ, ρ_x, P0)

    p0 = (y, w) -> pdf(P0(w), y)
    F0 = (y, w) -> cdf(P0(w), y)

    pdfs = Vector{MartingalePosteriorPDF}(undef, B)
    cdfs = Vector{MartingalePosteriorCDF}(undef, B)
    
    for b in 1:B
        v_sim = rand(Uniform(0, 1), N - n)
        v_full = [v_obs; v_sim]
        
        W_full = bayes_bootstrap(W, N)
        pdfs[b] = MartingalePosteriorPDF(v_full, W_full, α_seq, ρ, ρ_x, p0, F0)
        cdfs[b] = MartingalePosteriorCDF(v_full, W_full, α_seq, ρ, ρ_x, F0)
    end

    return (pdfs = pdfs, cdfs = cdfs, lps = lps)
end