
using LinearAlgebra, Random, Distributions

# Bayesian bootstrap
# D should be the data matrix
function bayes_bootstrap(D, N)
    n = size(D, 1)
    for i in (n+1):N
        idx = sample(1:(i-1))
        D = vcat(D, D[idx, :]')
    end
    return D
end

# compute the response type probabilities
function response_type_prob(x, z)
    p_A = mean(x[z .== 0])
    p_N = mean(1 .- x[z .== 1])
    p_C = 1 - p_A - p_N
    return [p_A, p_N, p_C]
end

# compute the LATE
function late(y, x, z)
    num = mean(y[z .== 1]) - mean(y[z .== 0])
    den = mean(x[z .== 1]) - mean(x[z .== 0])
    return num / den
end

# Martingale Posterior for the LATE and response-type probabilities
function mp_late(y, x, z; N = 1000, B = 100)
    probs = Matrix(undef, 3, B)
    lates = zeros(B)

    D_orig = [y x z]
    for b in 1:B
        D_pred = bayes_bootstrap(D_orig, N)
        y_pred, x_pred, z_pred = D_pred[:, 1], D_pred[:, 2], D_pred[:, 3]
        probs[:, b] = response_type_prob(x_pred, z_pred)
        lates[b] = late(y_pred, x_pred, z_pred)
    end
    return lates, probs
end

y = [1, 0, 1, 1]
x = [1, 0, 0, 1]
z = [1, 0, 1, 0]
res = mp_late(y, x, z; B = 500, N = 100)


using StatsPlots
density(res[1])
density(res[2])
