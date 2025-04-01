

using Turing, DataFrames
using StatsPlots

include("competing_methods.jl")
include("PostBayesTSLS.jl")

f_y(yy) = 1 + 1/3 * yy #- 1/8 * yy^2

function gen_data(n, f_y; c = 1/2)
    Z = rand(MvNormal(zeros(10), I), n)'

    Σ = [1 c; c 1/2] ./ 5
    u = rand(MvTDist(4, [0, 0], Σ), n)'
    x = Z * [ones(5); zeros(5)] + u[:,2]
    y = f_y.(x) + u[:,1]

    return (y=y, x=x, z=Z)
end

# auxiliary function to extract the marginal posteriors for each component
marginals(posterior) = map((μ, σ) -> Normal(μ, σ), posterior.μ, sqrt.(diag(posterior.Σ)))


n, m = (50, 500)
covg = Matrix{Bool}(undef, 2, m)
for i in 1:m
    y, x, Z = gen_data(n, f_y)
    X = [ones(length(x)) x]

    fit_1 = PostBayesTSLS_posterior(y, X, Z)

    ω_sm = tune_learning_rate(y, X, Z)[1]
    fit_sm = PostBayesTSLS_posterior(y, X, Z; ω = ω_sm)

    ci_1 = quantile(marginals(fit_1)[2], [0.025, 0.975])
    covg[1, i] = ci_1[1] < 1/3 < ci_1[2]

    ci_sm = quantile(marginals(fit_sm)[2], [0.025, 0.975])
    covg[2, i] = ci_sm[1] < 1/3 < ci_sm[2]
end


mean(covg, dims = 2)

