

function PostBayesTSLS(y, X, Z; ω = 1, Σ = Matrix(1.0 * I, size(X, 2), size(X, 2)))
    P_Z = Z * inv(Z'Z) * Z'

    Mean = inv(X' * P_Z * X + 1/ω * inv(Σ)) * X' * P_Z * y
    Cov = Symmetric(inv(ω * X' * P_Z * X + inv(Σ)))
    return MvNormal(Mean, Cov)
end
