
using LinearAlgebra, Random, Statistics, StatsBase

"""
Martingale Posterior for LATE and response-type probabilities.
Returns:
- lates: Matrix of size (N-n) x B
- probs: 3D Array of size 3 x (N-n) x B
"""
function mp_late(y, x, z; N = 1000, B = 100)
    n_orig = length(y)
    num_steps = N - n_orig
    
    # Rows = steps (n+1 to N), Columns = bootstrap iterations
    all_lates = Matrix{Float64}(undef, num_steps, B)
    all_probs = Array{Float64, 3}(undef, 3, num_steps, B)

    # Compute initial summary statistics    
    μy1_init, μy0_init = mean(y[z .== 1]), mean(y[z .== 0])
    μx1_init, μx0_init = mean(x[z .== 1]), mean(x[z .== 0])
    n1_init, n0_init = sum(z .== 1), sum(z .== 0)

    Threads.@threads for b in 1:B
        # Initialise local summary statistics
        n1, n0 = n1_init, n0_init
        μy1, μy0 = μy1_init, μy0_init
        μx1, μx0 = μx1_init, μx0_init

        # Initialise Counts: Each original observation starts with count = 1
        counts = ones(Int, n_orig)

        # Recursive Loop from n+1 to N
        for s in 1:num_steps
            # Step size: total observations = n_orig + s - 1
            # Sample an original observation index proportional to its current 'weight'
            j = sample(1:n_orig, Weights(counts))
            counts[j] += 1
            
            # Update the specific group the sampled observation belongs to
            if z[j] == 1
                n1 += 1
                μy1 += (y[j] - μy1) / n1
                μx1 += (x[j] - μx1) / n1
            else
                n0 += 1
                μy0 += (y[j] - μy0) / n0
                μx0 += (x[j] - μx0) / n0
            end
            
            # Compute Estimates at current step 's'
            p_A = μx0
            p_N = 1.0 - μx1
            p_C = 1.0 - p_A - p_N
            
            all_probs[:, s, b] = [p_A, p_N, p_C]
            all_lates[s, b] = (μy1 - μy0) / (μx1 - μx0)
        end
    end
    
    return all_lates, all_probs
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

# run Martingale posterior
Random.seed!(42)
B, N = 500, 100_000
res = mp_late(df.Y, df.X, df.Z; B = B, N = N)


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
    res[1][end, :] * 1000, # plot rate per 1000 individuals
    xlabel = "LATE (per 1000)", label = "",
    fill = true, fillalpha = 0.2
)

p2 = density(
    res[2][3, end, :],
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
