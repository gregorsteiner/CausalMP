

# Utility to add intercept
add_intercept = function(X) {
  cbind(1, X)
}

# functions to fit OLS and TSLS
ols = function(y, x){
  n = length(y)
  X = add_intercept(x)
  beta_hat = solve(t(X) %*% X) %*% t(X) %*% y
  sigma = sqrt(t(y - X %*% beta_hat) %*% (y - X %*% beta_hat) / (n-ncol(X)))
  return(list(
    coef = beta_hat,
    sigma = sigma
  ))
}

tsls = function(y, x, z){
  Z = add_intercept(z)
  P_Z = Z %*% solve(t(Z) %*% Z) %*% t(Z)
  X = add_intercept(x)
  beta_hat = solve(t(X) %*% P_Z %*% X) %*% t(X) %*% P_Z %*% y
  return(beta_hat)
}

# function to perform the one-step predictive update
one_step_update = function(data, type = "BB", endogeneity = TRUE) {
  n = length(data$y)
  
  if (type == "BB") {
    idx = sample(n, 1) # sample BB index
    if (endogeneity) {
      return(list(
        y = c(data$y, data$y[idx]),
        x = c(data$x, data$x[idx]),
        z = c(data$z, data$z[idx])
      ))
    } else {
      return(list(
        y = c(data$y, data$y[idx]),
        x = c(data$x, data$x[idx]),
        z = NULL
      ))
    }
  } else if (type == "LM") {
    idx = sample(n, 1) # sample BB index
    if (endogeneity) {
      XZ = cbind(data$x, data$z)
      xz_new = XZ[idx, ]
      fit = ols(data$y, XZ)
      y_new = rnorm(1, c(1, xz_new) %*% fit$coef, fit$sigma)
      return(list(
        y = c(data$y, y_new),
        x = c(data$x, xz_new[1]),
        z = c(data$z, xz_new[2])
      ))
    } else {
      x_new = data$x[idx]
      fit = ols(data$y, data$x)
      y_new = rnorm(1, c(1, x_new) %*% fit$coef, fit$sigma)
      return(list(
        y = c(data$y, y_new),
        x = c(data$x, x_new),
        z = NULL
      ))
    }
  }
}


# General Martingale Posterior Sampler
martingale_posterior = function(y, x, z = NULL, B = 100, N = 1000, type = "BB") {
  # if z is provided use GMM, otherwise use basic least squares
  if(is.null(z)) endogeneity = FALSE else endogeneity = TRUE
  
  # initialise cluster for parallelisation
  cl = parallel::makeCluster(parallel::detectCores() - 2)
  parallel::clusterExport(cl, varlist = c("one_step_update", "ols", "tsls", "add_intercept"))
  
  # get a posterior estimate for each predictive sequence (loop over 1:b -> B posterior samples)
  posterior = parallel::parLapply(cl, 1:B, function(j, y, x, z, N, type, endogeneity) {
    data = list(y = y, x = x, z = z)
    for (i in 1:N) {
      data = one_step_update(data, type = type, endogeneity = endogeneity)
    }
    
    if (endogeneity) {
      beta_est = tsls(data$y, data$x, data$z)
    } else {
      beta_est = ols(data$y, data$x)$coef
    }
    
    return(list(beta = beta_est))
  }, y, x, z, N, type, endogeneity)
  
  # stop cluster and return posterior
  parallel::stopCluster(cl)
  return(posterior)
}

  
  
  
  
  
  
  
  
