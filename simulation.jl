

using Turing, DataFrames
using CairoMakie

include("competing_methods.jl")
include("PostBayesTSLS.jl")
include("IVGP.jl")


f_x(xx) = 1.0 + xx
f_y(yy) = cos(4*yy) + 1/3 * yy - 1/8 * yy^2 #1/10 + 1/4 * yy - 1/2 * yy^2 + - 1/5 * yy^3 + 1/3 * yy^4 - 1/25 * yy^5

function gen_data(n, f_x, f_y; τ = 1, c = 1/2)
    z = rand(Uniform(0, 2*π), n)
    Σ = [1 c; c 1/2] ./ 5
    u = rand(MvTDist(4, [0, 0], Σ), n)'
    x = f_x.(z) + u[:,2]
    y = f_y.(x) + u[:,1]

    return (y=y, x=x, z=z)
end

n = 50
y, x, z = gen_data(n, f_x, f_y)

fig = Figure()
ax = Axis(fig[1,1])
xx = minimum(x):0.05:maximum(x)
fits = map(x_star -> ivgp(y, x, [ones(n) z], [x_star]; ω = 1, l = 0.1, σ2 = 1), xx)

scatter!(ax, x, y, label = "Observations")
lines!(ax, xx, f_y.(xx), label = "True Counterfactual", color = :red)
lines!(ax, xx, map(mean, fits), color = :green, label = "IVGP Prediction")
band!(ax, xx, map(fit -> quantile(fit, 0.025), fits), map(fit -> quantile(fit, 0.975), fits), color = (:green, 0.3))
axislegend(ax)

fig
