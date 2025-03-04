
using Distributions, LinearAlgebra

"""
    This function implements a TSLS estimator to compare our approach to.
"""
function tsls(y, X, Z; level = 0.05)
    n = length(y)
    l = size(x, 2)

    P_Z = Z * inv(Z'Z) * Z'
    
    β_hat = inv(X' * P_Z * X) * X' * P_Z * y

    residuals = y - X * β_hat
    σ2_hat = sum(residuals.^2) / n
    cov = σ2_hat * inv(X' * P_Z * X)
    ci = [β_hat[j] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov[j, j]) for j in eachindex(β_hat)]

    return (
        β_hat,
        CI = ci
    )
end

