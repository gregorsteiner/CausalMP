
# preliminaries
source("MartingalePosteriorGMM.R") # include the methods
library(parallel) # parallel package for parallelisation

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
set.seed(13)
d = gen_data(n = 500)

# tsls(d[, 1], d[, 2], d[, 3])
# ols(d[, 1], d[,2])$coef

N = 5000 # N > n
B = 500

res_naive = martingale_posterior(d[, 1], d[, 2], B = B, N = N, type = "LM")
res_gmm = martingale_posterior(d[, 1], d[, 2], z = d[, 3], B = B, N = N, type = "LM")


# compare with regular Bayesian IV
res_rossi = bayesm::rivGibbs(
  list(y = d[, 1], x = d[, 2], w = matrix(rep(1, nrow(d)), ncol = 1), z = cbind(rep(1, nrow(d)), d[ , 3])),
  Mcmc = list(R = 5*B, nprint = 0),
)
beta_rossi = as.numeric(res_rossi$betadraw)

library(ggplot2)
# Combine the data for plotting
df_plot <- rbind(
  data.frame(beta = do.call(cbind, lapply(res_naive, "[[", 1))[2, ], method = "Naive Martingale Posterior"),
  data.frame(beta = do.call(cbind, lapply(res_gmm, "[[", 1))[2, ], method = "GMM Martingale Posterior"),
  data.frame(beta = beta_rossi, method = "Bayesian IV (Rossi)")
)


# Create the density plot
p = ggplot(df_plot, aes(x = beta, fill = method, color = method)) +
  geom_density(alpha = 0.4, adjust = 1.5) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.5) +
  annotate("text", x = 1, y = 0.05, label = "True β = 1", vjust = -0.5, hjust = 1.1, size = 4) +
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
  ) +
  coord_cartesian(xlim = c(-1.5, 2.8))
p

ggsave("mp_example_posterior.pdf", plot = p, width = 8, height = 4)



