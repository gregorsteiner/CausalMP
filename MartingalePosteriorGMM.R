
one_step_update = function(d){
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
}


