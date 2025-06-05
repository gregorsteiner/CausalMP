
# include the methods
source("MartingalePosteriorGMM.R")

# functions to generate the data
expit = function(x){1 / (1 + exp(-x))}


gen_data = function(n = 50, rho = 1/2){
  u = MASS::mvrnorm(n, c(0, 0), matrix(c(1, rho, rho, 1), ncol = 2))
  
  z = rbinom(n, 1, 1/2)
  x = rbinom(n, 1, expit(-1/2 + z + u[, 1]))
  y = -1/2 + 1 * x + u[, 2]
  
  return(
    cbind(
      y = y,
      x = x, 
      z = z
    )
  )
}


# Run analysis
d = gen_data()
lm(d[, 1] ~ d[, 2]) |> summary()


N = 1000
for (i in 1:N) {
  d = one_step_update(d)
  d
}
d





