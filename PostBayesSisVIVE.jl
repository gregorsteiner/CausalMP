

# GMM loss
loss(y, X, Z, α, β) = (y - X*β - Z*α)' * Z * inv(Z'Z) * Z' * (y - X*β - Z*α)

# Define Turing model based on this loss
# the model depends on the learning rate ω
@model function sisVIVE(y, X, Z, ω)
    β ~ MvNormal(zeros(size(X, 2)), I)

    λ ~ Exponential(1)
    α = zeros(size(Z, 2))
    for j in eachindex(α)
        α[j] ~ Laplace(0, 1 / λ)
    end

    Turing.@addlogprob! -ω * loss(y, X, Z, α, β)
end

# This function implements the learning rate tuning procedure of Syring & Martin (2019, Biometrika)
function tune_learning_rate(y, X, Z; α = 0.05, B = 200, ϵ = 0.02, maxiters = 10)
    n = length(y)
    ω, t = ([1.0], 1)
    while true
        chain_full_sample = sample(sisVIVE(y, X, Z, ω[t]), NUTS(), 500; verbose = false)
        est_tau = median(chain_full_sample[:"β[2]"])
        bool_covg = Vector{Bool}(undef, B)
        for i in eachindex(bool_covg)
            idx_boot = sample(1:n, n; replace = true)
            chain_boot = sample(sisVIVE(y[idx_boot], X[idx_boot, :], Z[idx_boot, :], ω[t]), NUTS(), 100; verbose = false)
            ci = quantile(chain_boot[:"β[2]"], [α/2, 1 - α/2])
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

    return (ω = ω[end], Coverage = covg, Iterations = t-1)
end
