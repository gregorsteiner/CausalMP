
import jax.numpy as jnp
import pandas as pd
import numpy as np

#import copula functions
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression,predictive_resample_cregression,check_convergence_pr_cregression


# compute marginal density stats for Y(x) at given x_vals, averaging over W
def mp_density(y, x, w, x_vals, y_grid, B_post, T_fwd, seed=42):
    # stack treatment and covariates
    w = np.asarray(w)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    Z = np.column_stack((x, w))
    y_jnp, Z_jnp = jnp.array(y), jnp.array(Z)

    # fit conditional copula regression
    fit = fit_copula_cregression(y_jnp, Z_jnp, single_x_bandwidth = False, n_perm_optim = 10)
    print("Optimised rho: ", fit.rho_opt)
    print("Optimised rho_x: ", fit.rho_x_opt)
    print("Prequential log-likelihhod: ", fit.preq_loglik)

    # Ensure x_vals is array-like
    if np.isscalar(x_vals):
        x_vals = np.array([x_vals])
    else:
        x_vals = np.asarray(x_vals)

    # Build the test grid over y, all unique w rows, and all x values
    w_unique = np.unique(w, axis=0)
    w_unique_jnp = jnp.array(w_unique)
    n_w = len(w_unique_jnp)
    n_x = len(x_vals)
    n_y = len(y_grid)
    
    # Create grid with all combinations of x_vals, w_unique rows, and y_grid
    y_target = jnp.tile(jnp.array(y_grid), n_w * n_x)
    w_target = jnp.tile(jnp.repeat(w_unique_jnp, n_y, axis=0), (n_x, 1))
    x_target = jnp.repeat(jnp.array(x_vals), n_w * n_y)
    x_target = x_target[:, None]
    z_target = jnp.concatenate((x_target, w_target), axis=1)

    _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
        fit, Z, y_target, z_target, B_post, T_fwd, seed=seed
    )

    logpdf_pr = jnp.squeeze(logpdf_pr)
    pdfs = jnp.exp(logpdf_pr)
    pdfs = pdfs.reshape(B_post, n_x, n_w, n_y)

    sampled_w = Z_jnp[ind_new_pr, 1:]
    weights = jnp.all(sampled_w[:, :, None, :] == w_unique_jnp[None, None, :, :], axis=-1).sum(axis=1)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Compute marginal PDFs for each x value
    results = {}
    for i in range(n_x):
        marginal_pdfs = jnp.einsum('bw,bwy->by', weights, pdfs[:, i, :, :])
        results[f'x_{i}'] = {
            'mean': np.array(jnp.mean(marginal_pdfs, axis=0)),
            'low':  np.array(jnp.quantile(marginal_pdfs, 0.025, axis=0)),
            'high': np.array(jnp.quantile(marginal_pdfs, 0.975, axis=0))
        }
    
    return results


# estimate ATT counterfactual density p(y(0) | X=1) using the martingale posterior
# The control counterfactual is estimated by averaging p(y | X=0, W=w) over the covariate
# distribution in the treated group using resampled treated covariates.
def mp_density_att(y, x, w, y_grid, B_post, T_fwd, seed=42):
    w = np.asarray(w)
    if w.ndim == 1:
        w = w.reshape(-1, 1)

    x = np.asarray(x)
    if np.sum(x == 1) == 0:
        raise ValueError("mp_density_att requires at least one treated observation with x==1")
    if np.sum(x == 0) == 0:
        raise ValueError("mp_density_att requires at least one control observation with x==0")

    Z = np.column_stack((x, w))
    y_jnp, Z_jnp = jnp.array(y), jnp.array(Z)

    # fit conditional copula regression on the full sample
    fit = fit_copula_cregression(y_jnp, Z_jnp, single_x_bandwidth=False, n_perm_optim=10)
    print("Optimised rho: ", fit.rho_opt)
    print("Optimised rho_x: ", fit.rho_x_opt)
    print("Prequential log-likelihhod: ", fit.preq_loglik)

    # use the treated-group covariate distribution for the ATT target
    w_treated = w[x == 1]
    w_treated_unique = np.unique(w_treated, axis=0)
    w_treated_unique_jnp = jnp.array(w_treated_unique)

    n_w = len(w_treated_unique_jnp)
    n_y = len(y_grid)

    y_target = jnp.tile(jnp.array(y_grid), n_w)
    w_target = jnp.repeat(w_treated_unique_jnp, n_y, axis=0)
    x_target = jnp.zeros(n_w * n_y)[:, None]
    z_target = jnp.concatenate((x_target, w_target), axis=1)

    _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
        fit, Z, y_target, z_target, B_post, T_fwd, seed=seed
    )

    logpdf_pr = jnp.squeeze(logpdf_pr)
    pdfs = jnp.exp(logpdf_pr)
    pdfs = pdfs.reshape(B_post, 1, n_w, n_y)

    sampled_x = Z_jnp[ind_new_pr, 0]
    sampled_w = Z_jnp[ind_new_pr, 1:]

    treated_mask = sampled_x == 1
    matched = jnp.all(sampled_w[:, :, None, :] == w_treated_unique_jnp[None, None, :, :], axis=-1)
    weights = jnp.sum(matched & treated_mask[:, :, None], axis=1)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = jnp.where(weights_sum == 0, jnp.ones_like(weights) / n_w, weights / weights_sum)

    marginal_pdfs = jnp.einsum('bw,bwy->by', weights, pdfs[:, 0, :, :])
    return {
        'x_0': {
            'mean': np.array(jnp.mean(marginal_pdfs, axis=0)),
            'low':  np.array(jnp.quantile(marginal_pdfs, 0.025, axis=0)),
            'high': np.array(jnp.quantile(marginal_pdfs, 0.975, axis=0))
        }
    }


# compute marginal density stats for Y(x) at given x_vals, averaging over W
# Fits separate models for each x_val using only observations where x == x_val
def mp_density_t_learner(y, x, w, x_vals, y_grid, B_post, T_fwd, seed=42):
    # Ensure x_vals is array-like
    if np.isscalar(x_vals):
        x_vals = np.array([x_vals])
    else:
        x_vals = np.asarray(x_vals)

    results = {}
    for i, x_val in enumerate(x_vals):
        # Subset data for this x_val
        mask = x == x_val
        y_sub = y[mask]
        x_sub = x[mask]
        w_sub = w[mask]
        Z_sub = np.column_stack((x_sub, w_sub))
        y_sub_jnp, Z_sub_jnp = jnp.array(y_sub), jnp.array(Z_sub)

        # Fit conditional copula regression on subset
        fit = fit_copula_cregression(y_sub_jnp, Z_sub_jnp, single_x_bandwidth=False, n_perm_optim=10)
        print(f"Optimised rho for x={x_val}: ", fit.rho_opt)
        print(f"Optimised rho_x for x={x_val}: ", fit.rho_x_opt)
        print(f"Prequential log-likelihood for x={x_val}: ", fit.preq_loglik)

        # Unique w rows for this subset
        w_sub = np.asarray(w_sub)
        if w_sub.ndim == 1:
            w_sub = w_sub.reshape(-1, 1)
        w_sub_unique = np.unique(w_sub, axis=0)
        w_sub_unique_jnp = jnp.array(w_sub_unique)
        n_w = len(w_sub_unique_jnp)
        n_y = len(y_grid)

        # Build the test grid over y and unique w rows for this x_val
        y_target = jnp.tile(jnp.array(y_grid), n_w)
        w_target = jnp.repeat(w_sub_unique_jnp, n_y, axis=0)
        x_target = np.full(n_w * n_y, x_val)
        x_target = x_target[:, None]
        z_target = jnp.concatenate((x_target, w_target), axis=1)

        _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
            fit, Z_sub_jnp, y_target, z_target, B_post, T_fwd, seed=seed
        )

        logpdf_pr = jnp.squeeze(logpdf_pr)
        pdfs = jnp.exp(logpdf_pr)
        pdfs = pdfs.reshape(B_post, 1, n_w, n_y)

        sampled_w = Z_sub_jnp[ind_new_pr, 1:]
        weights = jnp.all(sampled_w[:, :, None, :] == w_sub_unique_jnp[None, None, :, :], axis=-1).sum(axis=1)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Compute marginal PDFs
        marginal_pdfs = jnp.einsum('bw,bwy->by', weights, pdfs[:, 0, :, :])
        results[f'x_{i}'] = {
            'mean': np.array(jnp.mean(marginal_pdfs, axis=0)),
            'low':  np.array(jnp.quantile(marginal_pdfs, 0.025, axis=0)),
            'high': np.array(jnp.quantile(marginal_pdfs, 0.975, axis=0))
        }
    
    return results

