
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
d = gen_data(n = 100)

# tsls(d[, 1], d[, 2], d[, 3])
# ols(d[, 1], d[,2])$coef

y = d[, 1]
W = d[, 2:3]
PYpar <- PYcalibrate(Ek = 3, n = 500, discount = 0.25)
prior <- list(strength = PYpar$strength, discount = PYpar$discount)

grid_y <- seq(-7, 7, length.out = 100)
grid_x <- rbind(
  c(0, 0), c(1, 0), c(0, 1), c(1, 1)
)
mcmc <- list(niter = 2000, nburn = 1000)
output <- list(grid_x = grid_x, grid_y = grid_y, out_type = "FULL", out_param = TRUE)

res = PYregression(y = y, x = W, prior = prior, mcmc = mcmc, output = output)

df_plot = data.frame(grid_y, apply(res$density, c(1, 2), mean))
colnames(df_plot) = c("Grid", "00", "10", "01", "11")
df_plot = tidyr::pivot_longer(
  df_plot,
  cols = 2:5, values_to = "y", names_to = "Group"
)

library(ggplot2)
ggplot(df_plot) +
  geom_line(aes(x = Grid, y = y, colour = Group), linewidth = 0.8) +
  theme_bw()



res$beta[[1]]

ddp_predictive_sequence = function(i, )


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



