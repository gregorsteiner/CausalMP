
one_step_update = function(d){
  # Draw x and z from a Bayesian bootstrap
  # Then update y | x, z using a linear model
  n = nrow(d)
  x_z_new = d[sample(n, size = 1), 2:3]
  X = cbind(rep(1, n), d[, 2:3])
  y = d[, 1]

  coefs_ols = solve(t(X) %*% X) %*% t(X) %*% y
  y_new = rnorm(
    1,
    c(1, x_z_new) %*% coefs_ols,
    sqrt(t(y - X %*% coefs_ols) %*% (y - X %*% coefs_ols) / (n-3))
    )
  return(rbind(d, c(y_new, x_z_new)))
  # Bayesian bootstrap
  #d_new = d[sample(n, size = 1),]
  #return(rbind(d, d_new))
}

est_parameter = function(d){
  n = nrow(d)
  z = cbind(rep(1, n), d[, 3])
  P_Z = z %*% solve(t(z) %*% z) %*% t(z)
  X = cbind(rep(1, n), d[, 2])
  beta_hat = solve(t(X) %*% P_Z %*% X) %*% t(X) %*% P_Z %*% d[, 1]
  return(beta_hat)
}

martingale_posterior = function(d, B = 100, N = 1000){
  posterior = lapply(1:B, function(x){
    # generate final dataset
    d <- Reduce(function(x, i) one_step_update(x), 1:N, init = d, accumulate = FALSE)
    
    # compute estimate of the parameter of interest
    return(list(
      "beta" = est_parameter(d),
      "data" = d
      ))
  })
  
  return(posterior)
}

  
  
  
  
  
  
  
  
