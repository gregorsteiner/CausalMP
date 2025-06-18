
# preliminaries
source("MartingalePosteriorGMM.R") # include the methods
library(parallel) # parallel package for parallelisation
library(BNPmix) # for dependent Dirichlet prior models

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


ddp_predictive_sequence = function(idx, N, X, fit){
  if(!inherits(fit, "BNPdens")) stop("`fit` needs to be a BNPdens object.")
  n = ncol(fit$clust)
  alpha = 1 # strength parameter of the DP
  clust = numeric(N); clust[1:n] = fit$clust[idx, ] # the cluster allocation for the i-th posterior sample
  beta = fit$beta[[idx]]
  sigma2 = fit$sigma2[[idx]][, 1]
  
  y = numeric(N); y[1:n] = fit$data[, 1]
  k = ncol(beta) # number of columns of design matrix
  
  #prior hyperprarameters
  a = 2; b = var(y[1:n]) # prior shape and scale of inverse Gamma Base measure on sigma
  mu = c(mean(y[1:n]), rep(0, k-1)) # prior mean of Normal base measure
  Sigma = diag(100, k, k) # prior covariance of Normal base measure
  
  for (i in (n+1):N) {
    if(runif(1) < (alpha / (alpha + n))){ # with this probability sample from the base measure
      beta_new = MASS::mvrnorm(1, mu, Sigma)
      sigma2_new = 1 / rgamma(1, a, b)
      beta = rbind(beta, beta_new)
      sigma2 = c(sigma2, sigma2_new)
      clust[i] = max(clust) + 1 # new cluster
      y[i] = rnorm(1, c(1, X[i, ]) %*% beta_new, sqrt(sigma2_new))
      
    } else{ # else sample from the atoms
      new_idx = sample(clust[1:(i-1)], size = 1)
      clust[i] = new_idx
      y[i] = rnorm(1, c(1, X[i, ]) %*% beta[new_idx+1, ], sqrt(sigma2[new_idx+1]))
    }
  }
  
  # return predictive sequence
  return(y)
  
}

N = 1000
X_new = bayesian_bootstrap(W, N)
y_new = ddp_predictive_sequence(1, N, X_new, res)

df_plot = data.frame(
  y = y_new,
  group = ifelse(X_new[, 1] == 1, "Treatment", "Control"),
  type = c(rep("Original", n), rep("Imputed", N-n))
)

ggplot(df_plot, aes(x = y, colour = type, fill = type)) +
  geom_density(alpha = 0.8) +
  facet_wrap(~group) +
  theme_bw()


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



