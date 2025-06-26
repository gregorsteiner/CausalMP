
# preliminaries
source("MartingalePosteriorGMM.R") # include the methods

# functions to generate the data
expit = function(x){1 / (1 + exp(-x))}

gen_data = function(n = 50, rho = 3/4){
  z = rbinom(n, 1, 1/2)
  x = rbinom(n, 1, expit(-1/2 + z))
  
  # sample errors from a t mixture model with centres 1 and -1
  shift = rbind(c(-1, -1), c(1, 1))[sample(c(1, 2), n, replace = TRUE), ]
  u = mvtnorm::rmvt(n, df = 2.5, sigma = matrix(c(1, rho, rho, 1), ncol = 2)) + shift
  
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
set.seed(12)
d = gen_data(n = 500)

N = 5000
B = 500

ols(d[, 1], d[, 2])$coef
tsls(d[, 1], d[, 2], d[, 3])

res_naive_ddp = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "DDP", endogeneity = FALSE)
res_gmm_ddp = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "DDP")
res_naive_lm = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "LM", endogeneity = FALSE)
res_gmm_lm = martingale_posterior(d[, 1], d[, 2], d[, 3], B = B, N = N, type = "LM")

# compare with regular Bayesian IV (Rossi et al, 2005) and DP IV (Conley et al, 2008)
res_rossi = bayesm::rivGibbs(
  list(y = d[, 1], x = d[, 2], w = matrix(rep(1, nrow(d)), ncol = 1), z = cbind(rep(1, nrow(d)), d[ , 3])),
  Mcmc = list(R = 2*B, keep = 2, nprint = 0),
)
res_conley = bayesm::rivDP(
  list(y = d[, 1], x = d[, 2], z = d[ , 3, drop = FALSE]),
  Mcmc = list(R = 2*B, keep = 2, nprint = 0),
)


library(ggplot2)
# Combine the data for plotting
df_plot <- rbind(
  data.frame(beta = sapply(res_naive_ddp, "[[", 2), method = "Naive Martingale Posterior (DDP)"),
  data.frame(beta = sapply(res_gmm_ddp, "[[", 2), method = "GMM Martingale Posterior (DDP)"),
  data.frame(beta = sapply(res_naive_lm, "[[", 2), method = "Naive Martingale Posterior (LM)"),
  data.frame(beta = sapply(res_gmm_lm, "[[", 2), method = "GMM Martingale Posterior (LM)"),
  data.frame(beta = as.numeric(res_rossi$betadraw), method = "Bayesian IV (Rossi et al)"),
  data.frame(beta = as.numeric(res_conley$betadraw), method = "Bayesian DP IV (Conley et al)")
)


# Create the density plot
p = ggplot(df_plot, aes(x = beta)) +
  geom_density(alpha = 0.4, colour = "blue", fill = "blue") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.5) +
  #annotate("text", x = 1, y = 0.05, label = "True β = 1", vjust = -0.5, hjust = 1.1, size = 4) +
  facet_wrap(~ method) +
  theme_bw() +
  labs(
    title = "",
    x = expression(beta),
    y = "Posterior Density",
    fill = "Method",
    color = "Method"
  ) +
  theme(
    text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5)
  ) +
  coord_cartesian(xlim = c(0.65, 1.45))
p

ggsave("mp_example_posterior.pdf", plot = p, width = 8, height = 4)


# alternative plot
par(mfrow = c(2, 3))
lapply(unique(df_plot$method), function(x){
  y = df_plot[df_plot$method == x, "beta"]
  y = y[abs(y) < 10]
  plot(density(y), main = x)
})



