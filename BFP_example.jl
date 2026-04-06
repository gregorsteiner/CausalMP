
using DataFrames, StatFiles, CategoricalArrays, Chain, Statistics

# 1. Read data
# load() returns a Table, so we wrap it in DataFrame()
dat = DataFrame(load("BFP_replication_data.dta"))

# 2. Delete observation 167
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
y_1, y_0 = d_femsing[d_femsing.treatment .== "Public", "compensation"], d_femsing[d_femsing.treatment .== "Private", "compensation"]


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
