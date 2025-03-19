using LinearAlgebra, Distributions

# Return the posterior distribution under a Gaussian prior on β
function PostBayesTSLS_posterior(y, X, Z; ω = 1, g = max(size(X, 1), size(X, 2)^2), λ = 0)
    P_Z = Z * inv(Z'Z + λ * I) * Z'

    L = cholesky(Symmetric(X' * P_Z * X))
    Mean = ω*g / (ω*g + 1) * (L \ (X' * P_Z * y))
    Cov = g / (ω*g + 1) * (L \ I)
    return MvNormal(Mean, Symmetric(Cov))
end

function PostBayesTSLS_marginal_likelihood(y, X, Z; ω = 1, g = max(size(X, 1), size(X, 2)^2), λ = 0)
    k = size(X, 2)
    P_Z = Z * inv(Z'Z + λ * I) * Z'
    ml = -k/2 * log(g*ω + 1) - ω/2 * y'P_Z * ( I  - (ω*g + 1)/(ω*g) * X * inv(X' * P_Z * X) * X') * P_Z * y
    return ml
end

# This function implements the learning rate tuning procedure of Syring & Martin (2019, Biometrika)
function tune_learning_rate(y, X, Z; α = 0.05, B = 200, ϵ = 0.02, maxiters = 100)
    n = length(y)
    ω, t = ([1.0], 1)
    while true
        posterior_full = PostBayesTSLS(y, X, Z; ω = ω[t])
        est_tau = mean(posterior_full)[2]
        bool_covg = Vector{Bool}(undef, B)
        for i in eachindex(bool_covg)
            idx_boot = sample(1:n, n; replace = true)
            posterior_boot = PostBayesTSLS(y[idx_boot], X[idx_boot, :], Z[idx_boot, :]; ω = ω[t])
            ci = quantile(Normal(mean(posterior_boot)[2], sqrt(cov(posterior_boot)[2, 2])), [α/2, 1 - α/2])
            bool_covg[i] = ci[1] < est_tau < ci[2] 
        end
        covg = mean(bool_covg)
        
        if abs(covg - (1 - α)) < ϵ
            break
        end
        push!(ω, ω[t] + (t+1)^(-0.51) * (covg - (1 - α)))
        t += 1
        if t > maxiters
            break
        end
    end

    return (ω = ω[end], Iterations = t-1)
end
