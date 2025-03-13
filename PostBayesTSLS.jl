
# Return the posterior distribution under a Gaussian prior on β
function PostBayesTSLS(y, X, Z; ω = 1, Σ = Matrix(1.0 * I, size(X, 2), size(X, 2)))
    P_Z = Z * inv(Z'Z) * Z'

    Mean = inv(X' * P_Z * X + 1/ω * inv(Σ)) * X' * P_Z * y
    Cov = Symmetric(inv(ω * X' * P_Z * X + inv(Σ)))
    return MvNormal(Mean, Cov)
end

# This function implements the learning rate tuning procedure of Syring & Martin (2019, Biometrika)
function tune_learning_rate(y, X, Z; α = 0.05, B = 200, ϵ = 0.02, maxiters = 10)
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
