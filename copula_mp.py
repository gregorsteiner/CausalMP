
import jax.numpy as jnp
import pandas as pd
import numpy as np

#import copula functions
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression,predictive_resample_cregression,check_convergence_pr_cregression


# compute marginal density stats for Y(x) at given x_vals, averaging over W
def mp_density(y, x, w, x_vals, y_grid, B_post, T_fwd, seed=42):
    # stack treatment and covariates
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

    # Build the test grid over y, all unique w values, and all x values
    w_unique_jnp = jnp.array(np.unique(w))
    n_w = len(w_unique_jnp)
    n_x = len(x_vals)
    n_y = len(y_grid)
    
    # Create grid with all combinations of x_vals, w_unique, and y_grid
    y_target = jnp.tile(jnp.array(y_grid), n_w * n_x)
    w_target = jnp.tile(jnp.repeat(w_unique_jnp, n_y), n_x)
    x_target = jnp.repeat(jnp.array(x_vals), n_w * n_y)
    z_target = jnp.column_stack((x_target, w_target))

    _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
        fit, Z, y_target, z_target, B_post, T_fwd, seed=seed
    )

    logpdf_pr = jnp.squeeze(logpdf_pr)
    pdfs = jnp.exp(logpdf_pr)
    pdfs = pdfs.reshape(B_post, n_x, n_w, n_y)

    sampled_w = Z_jnp[ind_new_pr, 1]
    weights = jnp.equal(sampled_w[:, :, None], w_unique_jnp[None, None, :]).sum(axis=1)
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

        # Unique w values for this subset
        w_sub_unique = np.unique(w_sub)
        w_sub_unique_jnp = jnp.array(w_sub_unique)
        n_w = len(w_sub_unique)
        n_y = len(y_grid)

        # Build the test grid over y and unique w values for this x_val
        y_target = jnp.tile(jnp.array(y_grid), n_w)
        w_target = jnp.repeat(w_sub_unique_jnp, n_y)
        x_target = jnp.full(n_w * n_y, x_val)
        z_target = jnp.column_stack((x_target, w_target))

        _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
            fit, Z_sub_jnp, y_target, z_target, B_post, T_fwd, seed=seed
        )

        logpdf_pr = jnp.squeeze(logpdf_pr)
        pdfs = jnp.exp(logpdf_pr)
        pdfs = pdfs.reshape(B_post, 1, n_w, n_y)

        sampled_w = Z_sub_jnp[ind_new_pr, 1]
        weights = jnp.equal(sampled_w[:, :, None], w_sub_unique_jnp[None, None, :]).sum(axis=1)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Compute marginal PDFs
        marginal_pdfs = jnp.einsum('bw,bwy->by', weights, pdfs[:, 0, :, :])
        results[f'x_{i}'] = {
            'mean': np.array(jnp.mean(marginal_pdfs, axis=0)),
            'low':  np.array(jnp.quantile(marginal_pdfs, 0.025, axis=0)),
            'high': np.array(jnp.quantile(marginal_pdfs, 0.975, axis=0))
        }
    
    return results

