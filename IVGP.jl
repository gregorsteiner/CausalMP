

se_kernel(x_1, x_2; σ2 = 1, l = 1) = σ2 * exp(-1/(2*l^2) * (x_1 - x_2)' * (x_1 - x_2))

function ivgp(y, X, Z, x_s; ω = 1, σ2 = 1, l = 1)
    n = length(y)
    P_Z = Z * inv(Z'Z) * Z'
    K, k_s = (Matrix{Float64}(undef, n, n), Vector{Float64}(undef, n))
    for i in eachindex(k_s)
        k_s[i] = se_kernel(X[i, :], x_s; σ2 = σ2, l = l)
        for j in eachindex(k_s)
            K[i, j] = se_kernel(X[i, :], X[j, :]; σ2 = σ2, l = l)
        end
    end
    k_ss = kernel(x_s, x_s; σ2 = σ2, l = l)

    L = cholesky(K + 1/ω * pinv(P_Z))
    α, v = (L' \ (L\y), L \ k_s)

    μ, σ2 = (k_s' * α, k_ss - v'v)
    σ2 = σ2 < 0 ? 0 : σ2
    res = Normal(μ, sqrt(σ2))
    return res
end

