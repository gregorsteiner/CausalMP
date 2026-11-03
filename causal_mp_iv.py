"""
Modified functions for IV-based density estimation with modified predictive resampling.

This module provides modified versions of predictive resampling functions specifically
for the mp_density_iv application. The key difference from standard predictive resampling
is that we resample (x, z) pairs together using Bayesian bootstrap, then update the
least squares estimate and compute new eta values before using them in the copula update.

It also provides mp_compliers for a binary experiment with non-compliance, which resamples
the latent compliance type directly (Section 4.1 of the paper).

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
# Predictive resampling on the latent compliance type C in {AT, NT, CP}: every imputed
# observation is generated from a model satisfying random assignment, exclusion and
# monotonicity, so P_i(Y(x)=1 | CP) = P_i(Y=1 | X=x, C=CP) is a valid probability for all i.
AT, NT, CP = 0, 1, 2
# outcome cells (X, C): 0 = (0, NT), 1 = (1, AT), 2 = (0, CP), 3 = (1, CP)
CELL_X0_NT, CELL_X1_AT, CELL_X0_CP, CELL_X1_CP = 0, 1, 2, 3


def _bernoulli_lik(y, q):
    return q if y == 1 else 1.0 - q


def _complier_posterior(y, p_cp, q_cp, p_other, q_other):
    # P(C = CP | X, Z, Y = y) for an ambiguous cell mixing compliers with one other type
    num = p_cp * _bernoulli_lik(y, q_cp)
    den = num + p_other * _bernoulli_lik(y, q_other)
    return num / den if den > 0 else p_cp / (p_cp + p_other)


def mp_compliers(y, x, z, B_post, T_fwd, n_checkpoints=50, seed=42):
    """
    Estimate P(Y(0)=1|CP) and P(Y(1)=1|CP) for compliers with binary (y, x, z).

    Initial predictive: the types of observed units with (X=0, Z=0) (CP or NT) and
    (X=1, Z=1) (CP or AT) are imputed independently in each sequence, conditional on the
    observed outcome via Bayes rule with the identified plug-in outcome probabilities.
    Forward step: draw Z from a Polya urn (Bayesian bootstrap), draw C independently from
    a Polya urn on the types initialised at n * p_hat_c, set X = X_C(Z), draw Y from a
    Polya urn within the (X, C) cell, and update all urns.
    Vectorised across B_post: only loops over T_fwd.

    Returns dict with keys 'Control', 'Treatment', 'Complier' (each shape
    (B_post, n_checkpoints)), 'checkpoints' (the forward step indices) and 'init'
    (plug-in type probabilities and imputation probabilities).
    """
    y_arr = np.asarray(y).astype(int)
    x_arr = np.asarray(x).astype(int)
    z_arr = np.asarray(z).astype(int)
    n = len(y_arr)

    rng = np.random.default_rng(seed)

    # counts n_xzy of observed units in each (x, z, y) cell
    cnt = np.zeros((2, 2, 2))
    np.add.at(cnt, (x_arr, z_arr, y_arr), 1)
    n_z = cnt.sum(axis=(0, 2))  # (n_z0, n_z1)

    def rate(xv, zv):
        tot = cnt[xv, zv].sum()
        return cnt[xv, zv, 1] / tot if tot > 0 else 0.5

    # ========== PLUG-IN TYPE AND OUTCOME PROBABILITIES ==========
    p_hat = np.zeros(3)
    p_hat[AT] = cnt[1, 0].sum() / n_z[0]
    p_hat[NT] = cnt[0, 1].sum() / n_z[1]
    p_hat[CP] = 1.0 - p_hat[AT] - p_hat[NT]
    if p_hat[CP] <= 0:
        raise ValueError(f"Estimated complier share is not positive ({p_hat[CP]:.4f}).")

    q_nt, q_at = rate(0, 1), rate(1, 0)
    q_cp0 = np.clip(((p_hat[NT] + p_hat[CP]) * rate(0, 0) - p_hat[NT] * q_nt) / p_hat[CP], 0, 1)
    q_cp1 = np.clip(((p_hat[AT] + p_hat[CP]) * rate(1, 1) - p_hat[AT] * q_at) / p_hat[CP], 0, 1)

    # imputation probabilities P(CP | X, Z, Y=y) for the ambiguous cells, indexed by y
    pi_00 = np.array([_complier_posterior(yv, p_hat[CP], q_cp0, p_hat[NT], q_nt) for yv in (0, 1)])
    pi_11 = np.array([_complier_posterior(yv, p_hat[CP], q_cp1, p_hat[AT], q_at) for yv in (0, 1)])

    # ========== INITIAL PREDICTIVE (per sequence) ==========
    # impute the number of compliers in each ambiguous (x, z, y) cell
    cp_00 = rng.binomial(cnt[0, 0].astype(int), pi_00, size=(B_post, 2)).astype(float)  # (B_post, y)
    cp_11 = rng.binomial(cnt[1, 1].astype(int), pi_11, size=(B_post, 2)).astype(float)

    # Polya urns on Y within each (X, C) cell: m = total count, s = count with Y = 1
    m = np.zeros((B_post, 4))
    s = np.zeros((B_post, 4))
    m[:, CELL_X0_CP] = cp_00.sum(axis=1)
    s[:, CELL_X0_CP] = cp_00[:, 1]
    m[:, CELL_X0_NT] = cnt[0, 1].sum() + cnt[0, 0].sum() - m[:, CELL_X0_CP]
    s[:, CELL_X0_NT] = cnt[0, 1, 1] + cnt[0, 0, 1] - s[:, CELL_X0_CP]
    m[:, CELL_X1_CP] = cp_11.sum(axis=1)
    s[:, CELL_X1_CP] = cp_11[:, 1]
    m[:, CELL_X1_AT] = cnt[1, 0].sum() + cnt[1, 1].sum() - m[:, CELL_X1_CP]
    s[:, CELL_X1_AT] = cnt[1, 0, 1] + cnt[1, 1, 1] - s[:, CELL_X1_CP]

    # an empty cell starts from a single pseudo-observation at its plug-in rate
    q_plugin = np.array([q_nt, q_at, q_cp0, q_cp1])
    empty = m == 0
    m[empty] = 1.0
    s[empty] = np.broadcast_to(q_plugin, m.shape)[empty]

    # Polya urns on Z (observed counts) and on the types (fractional counts n * p_hat)
    z_counts = np.tile(n_z, (B_post, 1))
    type_counts = np.tile(n * p_hat, (B_post, 1))

    checkpoints = np.unique(np.linspace(0, T_fwd - 1, n_checkpoints).astype(int))
    checkpoint_set = set(checkpoints)
    n_cp = len(checkpoints)

    ctrl_traj = np.zeros((B_post, n_cp))
    trt_traj = np.zeros((B_post, n_cp))
    comp_traj = np.zeros((B_post, n_cp))

    # ========== FORWARD SAMPLING ==========
    rows = np.arange(B_post)
    cp_idx = 0
    for j in range(T_fwd):
        tot = n + j
        u = rng.random((B_post, 3))

        # (1) instrument, (2) type independently of the instrument, (3) X = X_C(Z)
        zi = (u[:, 0] < z_counts[:, 1] / tot).astype(int)
        p_at = type_counts[:, AT] / tot
        p_nt = type_counts[:, NT] / tot
        ci = np.where(u[:, 1] < p_at, AT, np.where(u[:, 1] < p_at + p_nt, NT, CP))
        cell = np.where(ci == AT, CELL_X1_AT,
                        np.where(ci == NT, CELL_X0_NT,
                                 np.where(zi == 1, CELL_X1_CP, CELL_X0_CP)))

        # (4) outcome from the (X, C) cell, (5) update all urns
        yi = u[:, 2] < s[rows, cell] / m[rows, cell]
        z_counts[rows, zi] += 1
        type_counts[rows, ci] += 1
        m[rows, cell] += 1
        s[rows, cell] += yi

        if j not in checkpoint_set:
            continue

        ctrl_traj[:, cp_idx] = s[:, CELL_X0_CP] / m[:, CELL_X0_CP]
        trt_traj[:, cp_idx] = s[:, CELL_X1_CP] / m[:, CELL_X1_CP]
        comp_traj[:, cp_idx] = type_counts[:, CP] / (tot + 1)
        cp_idx += 1

    return {
        'Control': ctrl_traj,
        'Treatment': trt_traj,
        'Complier': comp_traj,
        'checkpoints': checkpoints,
        'init': {
            'p_hat': dict(zip(('AT', 'NT', 'CP'), p_hat)),
            'pi_x0z0': pi_00,  # P(CP | X=0, Z=0, Y=y) for y = 0, 1
            'pi_x1z1': pi_11,  # P(CP | X=1, Z=1, Y=y) for y = 0, 1
        },
    }


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
        Dictionary with keys 'x_{i}' for each x_val, containing 'median', 'low', and 'high'
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
