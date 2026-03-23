
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
    Threads.@threads for b in 1:B
        D_pred = bayes_bootstrap(D_orig, N)
        y_pred, x_pred, z_pred = D_pred[:, 1], D_pred[:, 2], D_pred[:, 3]
        probs[:, b] = response_type_prob(x_pred, z_pred)
        lates[b] = late(y_pred, x_pred, z_pred)
    end
    return lates, probs
end


# Create the Sommer & Zeger Vitamin A study dataset
using DataFrames
data_counts = [
    (0, 0, 0, 74),      # Assigned: No, Received: No, Outcome: Died
    (0, 0, 1, 11514),   # Assigned: No, Received: No, Outcome: Survived
    (1, 0, 0, 34),      # Assigned: Yes, Received: No, Outcome: Died
    (1, 0, 1, 2385),    # Assigned: Yes, Received: No, Outcome: Survived
    (1, 1, 0, 12),      # Assigned: Yes, Received: Yes, Outcome: Died
    (1, 1, 1, 9663)     # Assigned: Yes, Received: Yes, Outcome: Survived
]

df = DataFrame(Z = Int[], X = Int[], Y = Int[])
for (z, x, y, count) in data_counts
    append!(df, DataFrame(Z = fill(z, count), X = fill(x, count), Y = fill(y, count)))
end

res = mp_late(df.Y, df.X, df.Z; B = 200, N = 50_000)


using StatsPlots, Measures, LaTeXStrings
default(
    fontfamily="Computer Modern",
    titlefontsize=11, 
    guidefontsize=11, 
    tickfontsize=9, 
    legendfontsize=9,
    tick_direction=:out,
    frame=:axes, 
    grid=false,
    lw=1.5
)

p1 = density(
    res[1] * 1000, # plot rate per 1000 individuals
    xlabel = "LATE (per 1000)", label = "",
    fill = true, fillalpha = 0.2
)

p2 = density(
    res[2][3, :],
    xlabel = "Proportion of Compliers",
    label = "",
    fill = true, fillalpha = 0.2
)


final_plot = plot(
    p1, p2,
    size = (600, 300),
    dpi = 300,
    margins = 2mm
)
savefig(final_plot, "Sommer_Zeger_Results.pdf")
