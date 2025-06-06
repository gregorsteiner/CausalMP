
# functions to fit OLS and TSLS
ols = function(y, x){
  n = length(y)
  X = cbind(rep(1, n), x)
  beta_hat = solve(t(X) %*% X) %*% t(X) %*% y
  sigma = sqrt(t(y - X %*% beta_hat) %*% (y - X %*% beta_hat) / (n-ncol(X)))
  return(list(
    coef = beta_hat,
    sigma = sigma
  ))
}

tsls = function(y, x, z){
  n = length(y)
  Z = cbind(rep(1, n), z)
  P_Z = Z %*% solve(t(Z) %*% Z) %*% t(Z)
  X = cbind(rep(1, n), x)
  beta_hat = solve(t(X) %*% P_Z %*% X) %*% t(X) %*% P_Z %*% y
  return(beta_hat)
}

# function to perform the one-step predictive update
one_step_update = function(data, type = "BB"){
  n = length(data$y)
  
  if(type == "BB"){
    new = cbind(data$y, data$x, data$z)[sample(n, size = 1), ]
    return(list(
      y = c(data$y, new[1]),
      x = c(data$x, new[2]),
      z = c(data$z, new[3])
    ))
    
  } else if(type == "LM") {
    x_z_new = cbind(data$x, data$z)[sample(n, size = 1), ]
    X = cbind(rep(1, n), data$x, data$z)
    
    ols_res = ols(data$y, cbind(data$x, data$z))
    y_new = rnorm(1, c(1, x_z_new) %*% ols_res$coef, ols_res$sigma)
    return(list(
      y = c(data$y, y_new),
      x = c(data$x, x_z_new[1]),
      z = c(data$z, x_z_new[2])
    ))
  }
  
}


# function implementing the netire martingale posterior procedure
martingale_posterior_gmm = function(y, x, z, B = 100, N = 1000, type = "BB"){
  posterior = lapply(1:B, function(j){
    d = list(y = y, x = x, z = z)
    # generate final dataset
    d = Reduce(function(d_int, i) one_step_update(d_int, type = type), 1:N, init = d, accumulate = FALSE)
    
    # compute estimate of the parameter of interest
    return(list(
      "beta" = tsls(d$y, d$x, d$z),
      "data" = d
      ))
  })
  
  return(posterior)
}


# functions implementing a naive martingale posterior (ignoring endogeneity)
one_step_update_naive = function(data, type = "BB"){
  n = length(data$y)
  if(type == "BB"){
    new = cbind(data$y, data$x)[sample(n, size = 1), ]
    return(list(
      y = c(data$y, new[1]),
      x = c(data$x, new[2]),
      z = c(data$z, new[3])
    ))
    
  } else if(type == "LM"){
    # Draw x and z from a Bayesian bootstrap
    # Then update y | x, z using a linear model
    x_new = data$x[sample(n, size = 1)]
    
    ols_res = ols(data$y, data$x)
    y_new = rnorm(1, c(1, x_new) %*% ols_res$coef, ols_res$sigma)
    return(list(
      y = c(data$y, y_new),
      x = c(data$x, x_new[1])
    ))
  }
}



martingale_posterior = function(y, x, B = 100, N = 1000, type = "BB"){
  posterior = lapply(1:B, function(j){
    d = list(y = y, x = x)
    # generate final dataset
    d = Reduce(function(d_int, i) one_step_update_naive(d_int, type = type), 1:N, init = d, accumulate = FALSE)
    
    # compute estimate of the parameter of interest
    return(list(
      "beta" = ols(d$y, d$x)$coef,
      "data" = d
    ))
  })
  
  return(posterior)
  
}

  
  
  
  
  
  
  
  
