

using Turing, DataFrames
using CairoMakie

include("competing_methods.jl")
include("PostBayesTSLS.jl")

using Pkg; Pkg.activate("../gIVBMA")
using gIVBMA


f_x(xx) = sin(xx)
f_y(yy) = cos(4*yy) + 1/3 * yy - 1/8 * yy^2 #1/10 + 1/4 * yy - 1/2 * yy^2 + - 1/5 * yy^3 + 1/3 * yy^4 - 1/25 * yy^5

function gen_data(n, f_x, f_y; τ = 1, c = 1/2)
    z = rand(Uniform(0, 2*π), n)
    Σ = [1 c; c 1/2] ./ 5
    u = rand(MvTDist(4, [0, 0], Σ), n)'
    x = f_x.(z) + u[:,2]
    y = f_y.(x) + u[:,1]

    return (y=y, x=x, z=z)
end

n = 10000
y, x, z = gen_data(n, f_x, f_y)

f_basis(x) = [x^i for i in 0:5]
X = reduce(hcat, f_basis.(x))'
Z = reduce(hcat, f_basis.(z))'

Σ_0 = inv(X' * Z * inv(Z'Z) * Z' * X)
res = PostBayesTSLS(y, X, Z; ω = 1/2, Σ = Σ_0)

f_estimated(x, β) = f_basis(x)' * β


xx = minimum(x):0.001:maximum(x)
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, x, y)
lines!(ax, xx, f_y.(xx), color = :red)
lines!(ax, xx, f_estimated.(xx, Ref(mean(res))), color = :green)
fig


