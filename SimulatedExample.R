
# include the methods
source("MartingalePosteriorGMM.R")

# functions to generate the data
expit = function(x){1 / (1 + exp(-x))}


gen_data = function(n = 50, rho = 1/2){
  #u = MASS::mvrnorm(n, c(0, 0), matrix(c(1, rho, rho, 1), ncol = 2))
  u = mvtnorm::rmvt(n, df = 3, sigma = matrix(c(1, rho, rho, 1), ncol = 2))
  
  z = rbinom(n, 1, 1/2)
  x = rbinom(n, 1, expit(-1/2 + z + u[, 1]))
  y = -1/2 + 1 * x + u[, 2]
  
  # centre y
  y = (y - mean(y))
  
  return(
    cbind(
      y = y,
      x = x, 
      z = z
    )
  )
}


# Run analysis
set.seed(41)
d = gen_data(n = 500)


N = 2500
B = 500
res = martingale_posterior(d, B = B, N = N)
beta = do.call(cbind, lapply(res, "[[", 1))


# compare with regular Bayesian IV
res_rossi = rivGibbs(
  list(y = d[, 1], x = d[, 2], w = matrix(rep(1, nrow(d)), ncol = 1), z = cbind(rep(1, nrow(d)), d[ , 3])),
  Mcmc = list(R = B, nprint = 0),
)
beta_rossi = as.numeric(res_rossi$betadraw)

library(ggplot2)
# Combine the data for plotting
df_plot <- dplyr::bind_rows(
  data.frame(beta = beta[2, ], method = "Martingale Posterior"),
  data.frame(beta = beta_rossi, method = "Bayesian IV (Rossi)")
)


# Create the density plot
p = ggplot(df_plot, aes(x = beta, fill = method, color = method)) +
  geom_density(alpha = 0.4, adjust = 1.5) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.5) +
  annotate("text", x = 1, y = 0.1, label = "True β = 1", vjust = -0.5, hjust = 1.1, size = 4) +
  theme_minimal() +
  labs(
    title = "",
    x = expression(beta),
    y = "Posterior Density",
    fill = "Method",
    color = "Method"
  ) +
  theme(
    text = element_text(size = 14),
    plot.title = element_text(hjust = 0.5)
)

ggsave("mp_example_posterior.pdf", plot = p, width = 8, height = 4)



