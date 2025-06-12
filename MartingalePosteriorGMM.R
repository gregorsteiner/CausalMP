
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

# function implementing bayesian linear regression
bayes_lin_reg = function(y, X, X_new){
  fit_ols = ols(y, X)
  X_new = matrix(c(1, X_new), nrow = 1)
  y_new = mvtnorm::rmvt(
    1, type = "shifted", df = length(y) - ncol(X_new),
    sigma = fit_ols$sigma2 * (1 + X_new %*% fit_ols$X_t_X_inv %*% t(X_new)),
    delta = X_new %*% fit_ols$coef
  )
  return(y_new)
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

  
  
  
  
  
  
  
  
