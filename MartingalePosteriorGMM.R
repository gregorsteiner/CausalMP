

##### load packages #####
library(parallel) # parallel package for parallelisation
library(BNPmix)

##### Some auxiliary functions #####

# Utility to add intercept
add_intercept = function(X) {
  cbind(1, X)
}

# functions to fit OLS and TSLS
ols = function(y, x){
  n = length(y)
  X = add_intercept(x)
  X_t_X_inv = solve(t(X) %*% X)
  beta_hat = X_t_X_inv %*% t(X) %*% y
  sigma2 = t(y - X %*% beta_hat) %*% (y - X %*% beta_hat) / (n-ncol(X))
  return(list(
    coef = beta_hat,
    X_t_X_inv = X_t_X_inv,
    sigma2 = sigma2
  ))
}

tsls = function(y, x, z, ci = FALSE){
  Z = add_intercept(z)
  P_Z = Z %*% solve(t(Z) %*% Z) %*% t(Z)
  X = add_intercept(x)
  beta_hat = (solve(t(X) %*% P_Z %*% X, t(X) %*% P_Z %*% y))[, 1]
  
  if (ci){
    sigma2 = sum( (y - X %*% beta_hat)^2 ) / nrow(X)
    cov = sigma2 * solve(t(X) %*% P_Z %*% X)
    # this computes the CI only for the 2nd component of beta
    se = sqrt(cov[2, 2])
    ci_lower = beta_hat[2] - qnorm(0.975) * se
    ci_upper = beta_hat[2] + qnorm(0.975) * se
    return(list(
      coef = beta_hat,
      ci_lower = ci_lower,
      ci_upper = ci_upper
    ))
  }
  
  return(list(
    coef = beta_hat
  ))
}


##### Predictive update #####

# function predicting a new squence based on the Bayesian bootstrap
bayesian_bootstrap = function(x, N){
  n = nrow(x)
  x_full = matrix(NA, N, ncol(x))
  x_full[1:n, ] = x
  for (i in (n+1):N) {
    idx = sample(i-1, size = 1)
    x_full[i, ] = x_full[idx, ]
  }
  return(x_full)
}

# function generating a predictive sequence based on the dependent Dirichlet process (DDP)
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


# function generating a predictive sequence based on bayesian linear regression
bayes_lin_reg = function(y, X, N){
  n = length(y)
  y_full = numeric(N)
  y_full[1:n] = y
  
  for (i in (n+1):N) {
    fit_ols = ols(y_full[1:(i-1)], X[1:(i-1), ])
    X_new = matrix(c(1, X[i, ]), nrow = 1)
    
    y_full[i] = mvtnorm::rmvt(
      1, type = "shifted", df = (i-1) - ncol(X_new),
      sigma = fit_ols$sigma2 * (1 + X_new %*% fit_ols$X_t_X_inv %*% t(X_new)),
      delta = X_new %*% fit_ols$coef
    )
  }
  
  return(y_full)
}



##### Martingale posterior function #####
martingale_posterior = function(y, x, z, B = 100, N = 1000, type = "DDP", endogeneity = TRUE) {
  n = length(y)
  
  if(type == "DDP"){
    X = cbind(x)
    prior = list(strength = 1, discount = 0)
    grid_y = c(0)
    grid_x = matrix(0, ncol = ncol(X), nrow = 1)
    mcmc = list(niter = 1000 + B, nburn = 1000, print_message = FALSE)
    output = list(grid_x = grid_x, grid_y = grid_y, out_type = "FULL", out_param = TRUE)
    ddp_fit = PYregression(y = y, x = X, prior = prior, mcmc = mcmc, output = output)
  } else {ddp_fit = NULL}
  
  # initialise cluster for parallelisation
  cl = parallel::makeCluster(parallel::detectCores() - 4)
  parallel::clusterExport(cl, varlist = c("bayesian_bootstrap", "ddp_predictive_sequence", "bayes_lin_reg", "ols", "tsls", "add_intercept"))

  # get a posterior samplefor each predictive sequence (loop over 1:B -> B posterior samples)
  posterior = parallel::parLapply(cl, 1:B, function(j, y, x, z, N, ddp_fit, type, endogeneity) {
  #posterior = lapply(1:B, function(j, y, x, z, N, ddp_fit, type, endogeneity) {  
    # generate new X via Bayesian bootstrap
    x_full = bayesian_bootstrap(x, N)
    z_full = bayesian_bootstrap(z, N)
    if(type == "DDP"){
      y_full = ddp_predictive_sequence(j, N, x_full, ddp_fit)
    } else if(type == "LM"){
      y_full = bayes_lin_reg(y, x_full, N)
    }
    # compute the estimates
    if (endogeneity) {
      beta_est = tsls(y_full, x_full, z_full)$coef
    } else {
      beta_est = ols(y_full, x_full)$coef
    }
    
    return(beta_est)
  }, y, x, z, N, ddp_fit, type, endogeneity)
  
  # stop cluster
  parallel::stopCluster(cl)
  
  return(posterior)
}

  
  
  
  
  
  
  
  
