
using DataFrames, StatFiles, Statistics, Distributions

# Read data
dat = DataFrame(load("BFP_replication_data.dta"))

# Delete observation 167
deleteat!(dat, 167)

# drop observations with missing maritalstatus
dat = dat[.!ismissing.(dat.maritalstatus), :]

# only keep female & single
d_femsing = dat[(dat.male .== 0) .& (dat.maritalstatus .== 0), :]

# replace compensation intervals by mean
function mean_from_string(s)
    if ismissing(s) return missing end
    # Find all numeric patterns (integers or floats)
    matches = [parse(Float64, m.match) for m in eachmatch(r"[0-9]+\.?[0-9]*", string(s))]
    return isempty(matches) ? missing : mean(matches)
end
d_femsing.compensation = [mean_from_string(x) for x in d_femsing.desiredcompensation]


# potential outcomes
# treatment A is private and B is public
y_1, y_0 = d_femsing[d_femsing.treatment .== "B", "compensation"], d_femsing[d_femsing.treatment .== "A", "compensation"]


# plot
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

density(y_1, label = "Y(1)", xlabel = "Desired compensation (1,000 USD)", ylabel = "Density")
density!(y_0, label = "Y(0)")


# implement copula update
function conditional_gaussian_copula(u, v; ρ = 0.8)
    SN = Normal(0, 1)
    x = (quantile(SN, u) - ρ * quantile(SN, v)) / sqrt(1 - ρ^2)
    return cdf(SN, x)
end

function update_cdf(prev_cdf::Function, V, α)
    return function(y)
        U = prev_cdf(y) 
        return (1-α)*U + α*conditional_gaussian_copula(U, V)
    end
end


function run_simulation(y_1, N)
    n_1 = length(y_1)
    current_ecdf = y -> cdf(Normal(100, 10), y)

    for i in 1:N
        α_i = (2 - 1/i) * (1/(i+1))
        V_i = rand(Uniform(0, 1))
        current_ecdf = update_cdf(current_ecdf, V_i, α_i)
    end
    
    return current_ecdf
end

# compute empirical and imputed cdfs
orig_ecdf(y) = mean(y_1 .<= y)
final_ecdf = run_simulation(y_1, 500)

xx = 50:1:200
plot(xx, orig_ecdf.(xx), label = "Empirical")
plot!(xx, final_ecdf.(xx), label = "Imputed")
plot!(xx, map(x -> cdf(Normal(100, 10), x), xx), label = "Starting")
