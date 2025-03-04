
using Distributions, LinearAlgebra

function post_bayes_iv_gauss(y, X, Z, Σ_0)
    P_Z = Z * inv(Z'Z) * Z'

    Cov = inv(X' * P_Z * X + inv(Σ_0))
    Mean = Cov * X' * P_Z * y

    return MvNormal(Mean, Symmetric(Cov))
end