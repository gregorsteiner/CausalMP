##### This file implements the simulation setting proposed in Conley et al (2008) #####

source("MartingalePosteriorGMM.R") # load the Martingale posterior function
library(bayesm) # The bayesm package is needed for the competing methods

## function to generate the data ##
generate_data = function(n, s = 1, beta = 1){
  alpha = 0
  gamma = 0
  delta = rep(s, 10)
  Sigma = matrix(c(1, 0.6, 0.6, 1), ncol = 2)
  
  u = exp(MASS::mvrnorm(n, c(0, 0), 0.6 * Sigma))
  z = matrix(runif(10 * n), ncol = 10, nrow = n)
  x = gamma + z %*% delta + u[, 1]
  y = alpha + beta * x + u[, 2]
  return(list(y = y[, 1], x = x, z = z))
}

## functions to compute the performance criteria
mae = function(beta_estimates, true_beta = 1){
  return(median(abs(beta_estimates - true_beta)))
}
coverage = function(ci_lower, ci_upper, true_beta = 1){
  return(mean((ci_lower < true_beta) & (ci_upper > true_beta)))
}

## function that extracts the quantities of interest from the fit objects
compute_quantities = function(fit_list){
  # create list of the posterior samples
  post_list = list(
    sapply(fit_list$MP_DDP, "[[", 2),
    sapply(fit_list$MP_LM, "[[", 2),
    as.numeric(fit_list$Bayes_IV$betadraw),
    as.numeric(fit_list$Bayes_IV_DP$betadraw)
  )
  
  # return quantities of interest (point estimates and credible/confidence interval)
  return(list(
    point_estimates = c(sapply(post_list, median), fit_list$TSLS$coef[2]),
    ci_lower = c(sapply(post_list, quantile, 0.025), fit_list$TSLS$ci_lower),
    ci_upper = c(sapply(post_list, quantile, 0.975), fit_list$TSLS$ci_upper)
  ))
  
}

## wrapper function that runs the simulation
run_simulation = function(s, M = 100, n = 100, beta = 1, N = 500, B = 500){
  methods = c("MPIV (DDP)", "MPIV (LM)", "Bayes IV", "Bayes IV (DP)", "TSLS")
  
  # create storage objects
  point_estimates = matrix(NA, ncol = M, nrow = length(methods))
  ci_lower = matrix(NA, ncol = M, nrow = length(methods))
  ci_upper = matrix(NA, ncol = M, nrow = length(methods))
  
  # loop over M iterations
  pb <- txtProgressBar(min = 0, max = M, style = 3)
  for (j in 1:M) {
    # generate data
    d = generate_data(n, s = s)
    
    # fit models
    fit_list = list(
      "MP_DDP" = martingale_posterior(d$y, d$x, d$z, N = N, B = B, type = "DDP"),
      "MP_LM" = martingale_posterior(d$y, d$x, d$z, N = N, B = B, type = "LM"),
      "Bayes_IV" = bayesm::rivGibbs(
        list(y = d$y, x = d$x[, 1], w = matrix(1, ncol = 1, nrow = n), z = cbind(1, d$z)),
        Mcmc = list(R = 2*B, keep = 2, nprint = 0),
      ),
      "Bayes_IV_DP" = bayesm::rivDP(
        list(y = d$y, x = d$x[, 1], z = d$z),
        Mcmc = list(R = 2*B, keep = 2, nprint = 0),
      ),
      "TSLS" = tsls(d$y, d$x, d$z, ci = TRUE)
    )
    
    quantities = compute_quantities(fit_list)
    
    # extract quantities of interest
    point_estimates[, j] = quantities$point_estimates
    ci_lower[, j] = quantities$ci_lower
    ci_upper[, j] = quantities$ci_upper
    
    # Update the progress bar
    setTxtProgressBar(pb, j)
  }
  
  # compute performance measures
  coverage_results = numeric(length(methods))
  median_interval_length = numeric(length(methods))
  for (i in 1:length(methods)) {
    coverage_results[i] = coverage(ci_lower[i, ], ci_upper[i, ])
    median_interval_length[i] = median(ci_upper[i, ] - ci_lower[i, ])
  }
  
  res = cbind(
    "MAE" = apply(point_estimates, 1, mae),
    "Coverage" = coverage_results,
    "MIL" = median_interval_length
  )
  rownames(res) = methods
  return(res)
}

# Run the simulation
# We vary s between 0.5, 1, and 1.5 corresponding to weak, moderate and strong instruments
ss = c(0.5, 1, 1.5)
result = lapply(ss, run_simulation)
setNames(result, paste0("s = ", ss))
saveRDS(result, file = "Results_Conley.RDS")



