
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


# Instrumental variable approach to interventional density estimation for compliers
# S-Learner approach: Fits single model on all data
def mp_compliers(y, x, z, y_grid, B_post, T_fwd, seed=42):    
    # Convert to jax arrays and stack x, z as covariates
    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    z_arr = np.asarray(z)
    
    # For S-Learner, use (x, z) as the predictor variables
    xz = np.column_stack((x_arr, z_arr))
    y_jnp = jnp.array(y_arr)
    xz_jnp = jnp.array(xz)
    
    # Fit conditional copula regression on full data
    fit = fit_copula_cregression(y_jnp, xz_jnp, single_x_bandwidth=False, n_perm_optim=10)
    print("Optimised rho: ", fit.rho_opt)
    print("Optimised rho_x: ", fit.rho_x_opt)
    print("Prequential log-likelihood: ", fit.preq_loglik)
    
    # Compute response-type probabilities from observed data
    # p_A = P(X=1|Z=0), p_N = P(X=0|Z=1), p_C = P(X=1|Z=1) - P(X=1|Z=0)
    p_A = np.mean(x_arr[z_arr == 0])  # Prob treated when Z=0
    p_N = 1.0 - np.mean(x_arr[z_arr == 1])  # Prob untreated when Z=1
    p_C = np.mean(x_arr[z_arr == 1]) - np.mean(x_arr[z_arr == 0])  # Complier proportion
    
    print(f"\nResponse-type probabilities:")
    print(f"  p_A (Always-takers): {p_A:.4f}")
    print(f"  p_N (Never-takers): {p_N:.4f}")
    print(f"  p_C (Compliers): {p_C:.4f}")
    
    if p_C <= 0:
        raise ValueError("Proportion of compliers is non-positive. Check data or assumptions.")
    
    n_y = len(y_grid)
    
    # Create grids for four conditional densities needed:
    # p(y|X=0, Z=0), p(y|X=0, Z=1), p(y|X=1, Z=0), p(y|X=1, Z=1)
    y_target_list = []
    xz_target_list = []
    n_grids = 4
    
    for x_val in [0, 1]:
        for z_val in [0, 1]:
            y_target_list.append(jnp.array(y_grid))
            xz_target_list.append(np.full((n_y, 2), [x_val, z_val]))
    
    y_target = jnp.concatenate(y_target_list)
    xz_target = jnp.array(np.vstack(xz_target_list))
    
    # Get predictive samples for all four conditional densities
    _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
        fit, xz_jnp, y_target, xz_target, B_post, T_fwd, seed=seed
    )
    
    logpdf_pr = jnp.squeeze(logpdf_pr)
    pdfs = jnp.exp(logpdf_pr)
    pdfs = pdfs.reshape(B_post, n_grids, n_y)
    
    # Extract the four conditional densities
    pdf_y0_z0 = pdfs[:, 0, :]  # p(y|X=0, Z=0)
    pdf_y0_z1 = pdfs[:, 1, :]  # p(y|X=0, Z=1)
    pdf_y1_z0 = pdfs[:, 2, :]  # p(y|X=1, Z=0)
    pdf_y1_z1 = pdfs[:, 3, :]  # p(y|X=1, Z=1)
    
    coef_z0_x0 = (p_N + p_C) / p_C  # Coefficient for p(y|X=0, Z=0)
    coef_z1_x0 = -p_N / p_C         # Coefficient for p(y|X=0, Z=1)
    
    coef_z1_x1 = (p_A + p_C) / p_C  # Coefficient for p(y|X=1, Z=1)
    coef_z0_x1 = -p_A / p_C         # Coefficient for p(y|X=1, Z=0)
    
    # Compute posterior distributions for interventional densities
    p_y0_given_C = coef_z0_x0 * pdf_y0_z0 + coef_z1_x0 * pdf_y0_z1
    p_y1_given_C = coef_z1_x1 * pdf_y1_z1 + coef_z0_x1 * pdf_y1_z0
    
    # Ensure non-negative densities (due to numerical issues)
    p_y0_given_C = jnp.maximum(p_y0_given_C, 0)
    p_y1_given_C = jnp.maximum(p_y1_given_C, 0)
    
    # Renormalize to ensure they integrate to 1
    # (assuming y_grid has uniform spacing)
    if len(y_grid) > 1:
        dy = y_grid[1] - y_grid[0]
    else:
        dy = 1.0
    
    integral_y0 = jnp.sum(p_y0_given_C, axis=1, keepdims=True) * dy
    integral_y1 = jnp.sum(p_y1_given_C, axis=1, keepdims=True) * dy
    
    p_y0_given_C = p_y0_given_C / (integral_y0 + 1e-10)
    p_y1_given_C = p_y1_given_C / (integral_y1 + 1e-10)
    
    # Compute quantiles across posterior samples
    results = {
        'y_0_given_C': {
            'mean': np.array(jnp.mean(p_y0_given_C, axis=0)),
            'low': np.array(jnp.quantile(p_y0_given_C, 0.025, axis=0)),
            'high': np.array(jnp.quantile(p_y0_given_C, 0.975, axis=0))
        },
        'y_1_given_C': {
            'mean': np.array(jnp.mean(p_y1_given_C, axis=0)),
            'low': np.array(jnp.quantile(p_y1_given_C, 0.025, axis=0)),
            'high': np.array(jnp.quantile(p_y1_given_C, 0.975, axis=0))
        }
    }
    
    return results