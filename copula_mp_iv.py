"""
Modified functions for IV-based density estimation with modified predictive resampling.

This module provides modified versions of predictive resampling functions specifically
for the mp_density_iv application. The key difference from standard predictive resampling
is that we resample (x, z) pairs together using Bayesian bootstrap, then update the
least squares estimate and compute new eta values before using them in the copula update.

Author: Modified from main_copula_regression_conditional.py
"""

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import time
from functools import partial
from jax import vmap
from jax.random import permutation, PRNGKey, split
from jax import random

# Import copula functions
from pr_copula.main_copula_regression_conditional import (
    fit_copula_cregression,
    predict_copula_cregression,
    check_convergence_pr_cregression
)
from pr_copula import copula_regression_functions as mvcr
from pr_copula import sample_copula_regression_functions as samp_mvcr


# Auxiliary Bayesian bootstrap function that returns indices for a bootstrap sample
def bayesian_bootstrap(n, T_fwd, rng):
    # Polya urn is exactly equivalent to Dirichlet(ones) weights
    # sampled once — this is the vectorised form
    counts = rng.dirichlet(np.ones(n))
    
    base_indices = np.arange(n)
    extra_indices = rng.choice(n, size=T_fwd, p=counts)
    
    indices = np.concatenate([base_indices, extra_indices])
    
    return indices


### Instrumental variable approach to binary outcome estimation for compliers ###
# Uses Bayesian bootstrap directly on (y, x, z) triplets to estimate probabilities
def mp_compliers(y, x, z, B_post, T_fwd, seed=42):    
    """
    Estimate P(Y(0)=1|C) and P(Y(1)=1|C) for compliers using Bayesian bootstrap.
    
    Parameters:
    -----------
    y : array-like
        Binary outcome variable (0/1)
    x : array-like
        Binary treatment variable (0/1)
    z : array-like
        Binary instrument variable (0/1)
    B_post : int
        Number of Bayesian bootstrap samples
    T_fwd : int
        Number of forward samples
    seed : int
        Random seed
    """
    # Convert to numpy arrays
    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    z_arr = np.asarray(z)
    
    n = len(y_arr)
    
    # Initialize random number generator
    rng = np.random.RandomState(seed)
    
    # Storage for posterior samples
    prob_y0_samples = np.zeros(B_post)
    prob_y1_samples = np.zeros(B_post)
    complier_prob = np.zeros(B_post)
    
    # Bayesian bootstrap loop
    for b in range(B_post):
        # Resample (y, x, z) according to the Bayesian bootstrap
        indices = bayesian_bootstrap(n, T_fwd, rng)
        y_boot = y_arr[indices]
        x_boot = x_arr[indices]
        z_boot = z_arr[indices]

        # type probabilities
        p_A = np.mean(x_boot[z_boot == 0])  # Prob treated when Z=0
        p_N = 1.0 - np.mean(x_boot[z_boot == 1])  # Prob untreated when Z=1
        p_C = (1 - p_A - p_N)

        # Compute conditional probabilities from bootstrap sample
        # P(Y=1|X=x, Z=z)
        mask_x0_z0 = (x_boot == 0) & (z_boot == 0)
        mask_x0_z1 = (x_boot == 0) & (z_boot == 1)
        mask_x1_z0 = (x_boot == 1) & (z_boot == 0)
        mask_x1_z1 = (x_boot == 1) & (z_boot == 1)
        
        # Compute probabilities
        p_y1_x0_z0_b = np.mean(y_boot[mask_x0_z0])
        p_y1_x0_z1_b = np.mean(y_boot[mask_x0_z1]) if p_N > 0 else 0
        p_y1_x1_z0_b = np.mean(y_boot[mask_x1_z0]) if p_A > 0 else 0
        p_y1_x1_z1_b = np.mean(y_boot[mask_x1_z1])
        
        prob_y0_b = ((p_N + p_C) / p_C) * p_y1_x0_z0_b - p_N / p_C * p_y1_x0_z1_b
        prob_y1_b = ((p_A + p_C) / p_C) * p_y1_x1_z1_b - p_A / p_C * p_y1_x1_z0_b
        
        # Ensure probabilities are in [0, 1]
        prob_y0_samples[b] = np.clip(prob_y0_b, 0, 1)
        prob_y1_samples[b] = np.clip(prob_y1_b, 0, 1)
        complier_prob[b] = np.clip(p_C, 0, 1)
    
    # Compute quantiles and means across posterior samples
    results = {
        'Control': prob_y0_samples,
        'Treatment': prob_y1_samples,
        'Complier': complier_prob
    }
    
    return results



### Modified Predictive Resampling for IV Approach ###

@partial(jax.jit, static_argnums=(9, 10))
def predictive_resample_single_loop_cregression_iv(
    key, logcdf_conditionals, logpdf_joints, y_orig, x_orig, z_orig, 
    x_test, rho, rho_x, n, T
):
    """
    Forward sampling with Bayesian bootstrap on (x, z) pairs for IV approach.
    
    Instead of just resampling x, we resample (x, z) pairs together,
    then recompute eta from the resampled (z, x) pairs using OLS.
    
    Parameters:
    -----------
    key : JAX random key
    logcdf_conditionals : array
        Log CDF conditionals from initial fitting
    logpdf_joints : array
        Log PDF joints from initial fitting
    y_orig : array
        Original outcome data (n,)
    x_orig : array
        Original treatment data (n, d_x)
    z_orig : array
        Original instrument data (n, d_z) - note: without intercept
    x_test : array
        Test covariates (n_test, 2) where columns are [x, eta]
    rho : float
        Copula correlation parameter for y
    rho_x : array
        Copula correlation parameters for x and eta
    n : int
        Original sample size
    T : int
        Length of forward sampling chain
        
    Returns:
    --------
    logcdf_conditionals : array
        Updated log CDF conditionals
    logpdf_joints : array
        Updated log PDF joints
    ind_new : array
        Indices of resampled observations
    """
    # Generate uniform random numbers for copula update
    key, subkey = random.split(key)
    a_rand = random.uniform(subkey, shape=(T, 1))
    
    # Draw random (y, x, z) triplets from Bayesian bootstrap
    key, subkey = random.split(key)
    w = random.dirichlet(subkey, jnp.ones(n))  # Dirichlet weights for BB
    key, subkey = random.split(key)
    ind_new = random.choice(key, a=jnp.arange(n), p=w, shape=(1, T))[0]
    
    # Resample the (y, x, z) triplets
    x_new = x_orig[ind_new]  # (T, d_x)
    z_new = z_orig[ind_new]  # (T, d_z)
    
    # Recompute OLS estimate on resampled data
    # beta = (Z'Z)^{-1} Z'X, where Z has intercept column
    Z_new_with_const = jnp.column_stack((jnp.ones(T), z_new))
    
    # Use least squares to solve Z_new * beta_new = x_new
    # We need to use numpy for this since jnp.linalg.lstsq may have issues
    # So we'll compute it directly: beta = (Z'Z)^{-1} Z'X
    ZZ = Z_new_with_const.T @ Z_new_with_const
    ZX = Z_new_with_const.T @ x_new
    beta_new = jnp.linalg.solve(ZZ, ZX)
    
    # Compute new eta values on resampled data
    x_pred_new = Z_new_with_const @ beta_new  # (T, d_x)
    eta_new = x_new - x_pred_new  # (T, d_x)
    
    # Combine original and resampled data with new eta values
    # Original data has original eta (not used in forward sampling, kept for consistency)
    # Resampled data has new eta computed from resampled z
    x_samp = jnp.concatenate((x_orig, x_new), axis=0)  # (n + T, d_x)
    eta_samp = jnp.concatenate((jnp.zeros_like(x_orig), eta_new), axis=0)  # (n + T, d_x)
    
    # Create the combined covariate array for copula update: [x, eta]
    X_samp = jnp.column_stack((x_samp, eta_samp))  # (n + T, 2*d_x) - but for regression case it's 2D
    
    # Append a_rand to vn structure for forward sampling
    # vT should have the same structure as the original v_n from fitting
    vT = jnp.concatenate((jnp.zeros((n, 1)), a_rand), axis=0)  # (n + T, 1)
    
    # Run forward loop using the copula regression update mechanism
    inputs = vT, logcdf_conditionals, logpdf_joints, X_samp, x_test, rho, rho_x
    rng = jnp.arange(n, n + T)
    outputs, rng = mvcr.update_ptest_single_scan(inputs, rng)
    _, logcdf_conditionals, logpdf_joints, *_ = outputs
    
    return logcdf_conditionals, logpdf_joints, ind_new


# Vmap over multiple test points, then over multiple seeds
# For each test point, logcdf_conditionals and logpdf_joints have different values (axis 0)
# Original data y_orig, x_orig, z_orig are constant across test points (None)
predictive_resample_loop_cregression_iv = jax.jit(
    vmap(predictive_resample_single_loop_cregression_iv, 
         (None, 0, 0, None, None, None, 0, None, None, None, None)),
    static_argnums=(9, 10)
)  # vmap across test points

# For multiple posterior samples, only the key differs (axis 0) 
# The vmapped function from above returns outputs that are already broadcast correctly
predictive_resample_loop_cregression_iv_B = jax.jit(
    vmap(predictive_resample_loop_cregression_iv, 
         (0, None, None, None, None, None, None, None, None, None, None)),
    static_argnums=(9, 10)
)  # vmap across B posterior samples


def predictive_resample_cregression_iv(
    copula_cregression_obj, y_orig, x_orig, z_orig, 
    y_test, x_test, B_postsamples, T_fwdsamples=5000, seed=100
):
    """
    Modified predictive resampling for IV approach.
    
    Performs Bayesian bootstrap on (y, x, z) triplets, recomputes eta,
    then does forward sampling for density estimation.
    
    Parameters:
    -----------
    copula_cregression_obj : namedtuple
        Fitted copula regression object
    y_orig : array
        Original outcome data (n,)
    x_orig : array
        Original treatment data (n,) or (n, d_x)
    z_orig : array
        Original instrument data (n, d_z) or (n, d_z) - without intercept
    y_test : array
        Test y values for density evaluation (n_test,)
    x_test : array
        Test covariates [x, eta] for density evaluation (n_test, 2)
    B_postsamples : int
        Number of posterior samples
    T_fwdsamples : int
        Length of each forward sampling chain
    seed : int
        Random seed
        
    Returns:
    --------
    logcdf_conditionals_pr : array
        Updated log CDF conditionals
    logpdf_joints_pr : array
        Updated log PDF joints
    ind_new_pr : array
        Indices of resampled observations across posterior samples
    """
    # Fit permutation averaged cdf/pdf
    logcdf_conditionals, logpdf_joints = predict_copula_cregression(
        copula_cregression_obj, y_test, x_test
    )
    
    # Initialize random seeds
    key = PRNGKey(seed)
    key, *subkey = split(key, B_postsamples + 1)
    subkey = jnp.array(subkey)
    
    # Convert original data to jnp arrays
    y_orig_jnp = jnp.array(y_orig)
    x_orig_jnp = jnp.array(x_orig)
    z_orig_jnp = jnp.array(z_orig)
    
    # Ensure 1D arrays are properly shaped
    if y_orig_jnp.ndim == 1:
        y_orig_jnp = y_orig_jnp.reshape(-1, 1)
    if x_orig_jnp.ndim == 1:
        x_orig_jnp = x_orig_jnp.reshape(-1, 1)
    if z_orig_jnp.ndim == 1:
        z_orig_jnp = z_orig_jnp.reshape(-1, 1)
    
    # Forward sample with IV modifications
    n = jnp.shape(copula_cregression_obj.vn_perm)[1]  # Get original data size
    print('Predictive resampling (IV-modified)...')
    start = time.time()
    
    logcdf_conditionals_pr, logpdf_joints_pr, ind_new_pr = predictive_resample_loop_cregression_iv_B(
        subkey, logcdf_conditionals, logpdf_joints, y_orig_jnp, x_orig_jnp, z_orig_jnp, 
        x_test, copula_cregression_obj.rho_opt, copula_cregression_obj.rho_x_opt, 
        n, T_fwdsamples
    )
    
    # The sampled indices are identical across test points for each predictive sequence,
    # so we take the first test-point slice.
    ind_new_pr = ind_new_pr[:, 0, :]
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready()  # for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end - start, 3)))
    
    return logcdf_conditionals_pr, logpdf_joints_pr, ind_new_pr


### Modified mp_density_iv function ###
def mp_density_iv(y, x, z, x_vals, y_grid, B_post, T_fwd, seed=42):
    """
    Estimate interventional distribution using IV control function approach with 
    modified predictive resampling.
    
    Key difference: In the predictive resampling step, we resample (y, x, z) triplets 
    together using Bayesian bootstrap, then recompute the OLS estimate and eta values
    before using them in the copula update.
    
    Parameters:
    -----------
    y : array-like
        Outcome variable
    x : array-like
        Treatment variable
    z : array-like or 2D array
        Instruments. If 1D, treated as single instrument. If 2D, each column is an instrument.
    x_vals : scalar or array-like
        Values of x at which to evaluate the interventional density
    y_grid : array-like
        Grid of y values for density evaluation
    B_post : int
        Number of posterior samples (predictive sequences)
    T_fwd : int
        Length of each predictive sequence
    seed : int
        Random seed
        
    Returns:
    --------
    results : dict
        Dictionary with keys 'x_{i}' for each x_val, containing 'mean', 'low', and 'high' 
        quantiles of the interventional density p(y(x))
    """
    # Convert inputs to arrays
    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    z_arr = np.asarray(z)
    
    # Handle 1D instruments and treatments
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(-1, 1)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    
    n = len(y_arr)
    
    # ========== FIRST-STAGE: Standard OLS estimation ==========
    # Estimate g(Z) = Z^T beta by least squares
    # Add intercept column
    Z_with_const = np.column_stack((np.ones(n), z_arr))
    
    # Solve X = Z * beta + eta using least squares
    # beta = (Z'Z)^{-1} Z'X
    beta_hat = np.linalg.lstsq(Z_with_const, x_arr, rcond=None)[0]
    
    # Recover residuals: eta = X - g(Z)
    x_pred = Z_with_const @ beta_hat
    eta = x_arr - x_pred
    
    print(f"First-stage estimation:")
    print(f"  Estimated coefficients: {beta_hat}")
    print(f"  Mean residual: {np.mean(eta):.6f}")
    print(f"  Std residual: {np.std(eta):.6f}")
    
    # ========== CONDITIONAL DENSITY ESTIMATION ==========
    # Fit conditional copula regression: p(y | x, eta)
    # Stack x and eta as covariates
    X_cov = np.column_stack((x_arr, eta))
    y_jnp = jnp.array(y_arr)
    X_cov_jnp = jnp.array(X_cov)
    
    # Fit conditional copula regression
    fit = fit_copula_cregression(y_jnp, X_cov_jnp, single_x_bandwidth=False, n_perm_optim=10)
    print(f"\nConditional density fit:")
    print(f"  Optimised rho: {fit.rho_opt}")
    print(f"  Optimised rho_x: {fit.rho_x_opt}")
    print(f"  Prequential log-likelihood: {fit.preq_loglik}")
    
    # Ensure x_vals is array-like
    if np.isscalar(x_vals):
        x_vals = np.array([x_vals])
    else:
        x_vals = np.asarray(x_vals)
    
    if x_vals.ndim == 1:
        x_vals = x_vals.reshape(-1, 1)
    
    n_x = len(x_vals)
    n_y = len(y_grid)
    n_eta = n  # Number of empirical eta values
    
    # ========== DENSITY EVALUATION ==========
    # Evaluate p(y | x, eta_i) for all combinations of x_vals, eta observations, and y_grid
    # Create grid: for each x_val and each observed eta_i, evaluate over y_grid
    y_target_list = []
    x_target_list = []
    eta_target_list = []
    
    for x_val in x_vals:
        for eta_val in eta:
            y_target_list.append(y_grid)
            # Handle both scalar and array cases
            x_scalar = x_val[0] if hasattr(x_val, '__len__') else x_val
            eta_scalar = eta_val[0] if hasattr(eta_val, '__len__') else eta_val
            x_target_list.append(np.full(n_y, x_scalar))
            eta_target_list.append(np.full(n_y, eta_scalar))
    
    y_target = jnp.array(np.concatenate(y_target_list))
    x_target = jnp.array(np.concatenate(x_target_list))
    eta_target = jnp.array(np.concatenate(eta_target_list))
    X_target = jnp.column_stack((x_target, eta_target))
    
    # Get predictive samples using MODIFIED resampling function
    # Pass original data (y, x, z) and fitted copula object
    _, logpdf_pr, _ = predictive_resample_cregression_iv(
        fit, y_arr, x_arr, z_arr,  # Original outcome, treatment, and instrument data
        y_target, X_target, B_post, T_fwd, seed=seed
    )
    
    logpdf_pr = jnp.squeeze(logpdf_pr)
    pdfs = jnp.exp(logpdf_pr)
    
    # Reshape: B_post x n_x x n_eta x n_y
    pdfs = pdfs.reshape(B_post, n_x, n_eta, n_y)
    
    # ========== INTEGRATION OVER EMPIRICAL ETA DISTRIBUTION ==========
    # Compute interventional density: p(y(x)) = (1/n) sum_i p(y | x, eta_i)
    # Average over empirical eta distribution
    results = {}
    for i in range(n_x):
        # Average over the empirical distribution of eta
        # Shape: B_post x n_y
        p_y_given_x = jnp.mean(pdfs[:, i, :, :], axis=1)
        
        results[f'x_{i}'] = {
            'mean': np.array(jnp.mean(p_y_given_x, axis=0)),
            'low': np.array(jnp.quantile(p_y_given_x, 0.025, axis=0)),
            'high': np.array(jnp.quantile(p_y_given_x, 0.975, axis=0))
        }
    
    return results
