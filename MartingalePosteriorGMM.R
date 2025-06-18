
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

tsls = function(y, x, z){
  Z = add_intercept(z)
  P_Z = Z %*% solve(t(Z) %*% Z) %*% t(Z)
  X = add_intercept(x)
  beta_hat = solve(t(X) %*% P_Z %*% X, t(X) %*% P_Z %*% y)
  return(beta_hat)
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

# wrapper function that performs the update
one_step_update = function(y, x, z, type = "BB", endogeneity = TRUE) {
  n = length(y)
  
  if (type == "LM") {
    ## Use BB foe x and z and then predict y using a linear model
    idx = sample(n, 1) # sample BB index
    if (endogeneity) {
      XZ = cbind(x, z, x * z)
      x_new = XZ[idx, 1]; z_new = XZ[idx, 2]
      y_new = bayes_lin_reg(y, XZ, XZ[idx,])
    } else {
      x_new = x[idx]; z_new = NA
      y_new = bayes_lin_reg(y, x, x_new)
    }
    return(c(
      y = y_new,
      x = x_new,
      z = z_new
    ))
  }
}


##### Martingale posterior function #####
martingale_posterior = function(y, x, z = NULL, B = 100, N = 1000, type = "BB") {
  # if z is provided use GMM, otherwise use basic least squares
  if(is.null(z)){
    endogeneity = FALSE
    data = cbind(y, x, rep(NA, length(y)))
  } else{
    endogeneity = TRUE
    data = cbind(y, x, z)
  }
  
  # initialise cluster for parallelisation
  cl = parallel::makeCluster(parallel::detectCores() - 2)
  parallel::clusterExport(cl, varlist = c("one_step_update", "bayes_lin_reg", "ols", "tsls", "add_intercept"))

  # get a posterior samplefor each predictive sequence (loop over 1:B -> B posterior samples)
  posterior = parallel::parLapply(cl, 1:B, function(j, data, N, type, endogeneity) {
  #posterior = lapply(1:B, function(j, data, N, type, endogeneity) {  
    n = nrow(data)
    data_full = rbind(data, matrix(NA, nrow = N-n, ncol = 3))
    for (i in (n+1):N) {
      data_full[i, ] = one_step_update(
        data_full[1:(i-1), 1],
        data_full[1:(i-1), 2],
        data_full[1:(i-1), 3],
        type = type,
        endogeneity = endogeneity
      )
    }
    
    if (endogeneity) {
      beta_est = tsls(data_full[, 1], data_full[, 2], data_full[, 3])
    } else {
      beta_est = ols(data_full[, 1], data_full[, 2])$coef
    }
    
    return(list(beta = beta_est))
  }, data, N, type, endogeneity)
  
  # stop cluster and return posterior
  parallel::stopCluster(cl)
  return(posterior)
}

  
  
  
  
  
  
  
  
