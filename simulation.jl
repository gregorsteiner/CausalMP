


using CairoMakie

include("post_bayes_iv.jl")
include("competing_methods.jl")

function gen_instr_coeff(p, c_M)
    res = zeros(p)
    for i in 1:p
        if i <= p/2
            res[i] = c_M * (1 - i/(p/2 + 1))^4
        end
    end
    return res
end

function gen_data(n = 100, c_M = 3/8, τ = 0.1, p = 20, c = 1/2)
    Z = rand(MvNormal(zeros(p), I), n)'

    α, γ = (1, 1)
    δ = gen_instr_coeff(p, c_M)

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ u[:,1]

    return (y=y, x=x, Z=[ones(n) Z])
end


y, x, Z = gen_data(5000, 1)
X = [ones(length(x)) x]


res = post_bayes_iv_gauss(y, X, Z, [1.0 0.0; 0.0 1.0])
res_tsls = tsls(y, X, Z)

plot(Normal(res.μ[2], sqrt(res.Σ[2, 2])))
res_tsls.CI[2]
