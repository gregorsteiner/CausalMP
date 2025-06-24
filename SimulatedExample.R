
# preliminaries
source("MartingalePosteriorGMM.R") # include the methods

# functions to generate the data
expit = function(x){1 / (1 + exp(-x))}

gen_data = function(n = 50, rho = 1/2){
  z = rbinom(n, 1, 1/2)
  x = rbinom(n, 1, expit(-1/2 + z + rnorm(n)))
  
  # sample errors from a mixture model with centres 1 and -1 depending on x
  u = rnorm(n, ifelse(x == 1, 1, -1), 1)
  
  y = -1/2 + 1 * x + u
  
  return(
    cbind(
      y = y,
      x = x, 
      z = z
    )
  )
}

# Run analysis
set.seed(14)
d = gen_data(n = 100)

N = 2000
B = 500

res_naive_ddp = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "DDP", endogeneity = FALSE)
res_gmm_ddp = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "DDP")
res_naive_lm = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "LM", endogeneity = FALSE)
res_gmm_lm = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "LM")

# compare with regular Bayesian IV
res_rossi = bayesm::rivGibbs(
  list(y = d[, 1], x = d[, 2], w = matrix(rep(1, nrow(d)), ncol = 1), z = cbind(rep(1, nrow(d)), d[ , 3])),
  Mcmc = list(R = B, nprint = 0),
)
beta_rossi = as.numeric(res_rossi$betadraw)

library(ggplot2)
# Combine the data for plotting
df_plot <- rbind(
  data.frame(beta = sapply(res_naive_ddp, "[[", 2), method = "Naive Martingale Posterior (DDP)"),
  data.frame(beta = sapply(res_gmm_ddp, "[[", 2), method = "GMM Martingale Posterior (DDP)"),
  data.frame(beta = sapply(res_naive_lm, "[[", 2), method = "Naive Martingale Posterior (LM)"),
  data.frame(beta = sapply(res_gmm_lm, "[[", 2), method = "GMM Martingale Posterior (LM)"),
  data.frame(beta = beta_rossi, method = "Bayesian IV (Rossi)")
)

# trim values
df_plot = df_plot[abs(df_plot$beta) < 10, ]

# Create the density plot
p = ggplot(df_plot, aes(x = beta, colour = method, fill = method)) +
  geom_density(alpha = 0.4) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.5) +
  #annotate("text", x = 1, y = 0.05, label = "True β = 1", vjust = -0.5, hjust = 1.1, size = 4) +
  #facet_wrap(~ method)+
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
  coord_cartesian(xlim = c(-5, 6))
p

ggsave("mp_example_posterior.pdf", plot = p, width = 8, height = 4)


# alternative plot
par(mfrow = c(2, 3))
lapply(unique(df_plot$method), function(x){
  y = df_plot[df_plot$method == x, "beta"]
  y = y[y > -10 & y < 10]
  plot(density(y), main = x)
})



