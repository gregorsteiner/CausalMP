
import jax.numpy as jnp
import pandas as pd
import numpy as np

#import copula functions
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression,predictive_resample_cregression,check_convergence_pr_cregression


# compute marginal density stats for Y(x) at given x, averaging over W
def mp_density(y, x, w, x_val, y_grid, B_post, T_fwd, seed=42):
    # stack treatment and covariates
    Z = np.column_stack((x, w))
    y_jnp, Z_jnp = jnp.array(y), jnp.array(Z)

    # fit conditional copula regression
    fit = fit_copula_cregression(y_jnp, Z_jnp, single_x_bandwidth = False, n_perm_optim = 10)
    print("Optimised rho: ", fit.rho_opt)
    print("Optimised rho_x: ", fit.rho_x_opt)
    print("Prequential log-likelihhod: ", fit.preq_loglik)


    # Build the test grid over y and all unique w values for fixed x_val
    w_unique_jnp = jnp.array(np.unique(w))
    n_w = len(w_unique_jnp)
    y_target = jnp.tile(jnp.array(y_grid), n_w)
    w_target = jnp.repeat(w_unique_jnp, len(y_grid))
    z_target = jnp.column_stack((jnp.repeat(float(x_val), y_target.shape[0]), w_target))

    _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
        fit, Z, y_target, z_target, B_post, T_fwd, seed=seed
    )

    logpdf_pr = jnp.squeeze(logpdf_pr)
    pdfs = jnp.exp(logpdf_pr)
    pdfs = pdfs.reshape(B_post, n_w, len(y_grid))

    sampled_w = Z_jnp[ind_new_pr, 1]
    weights = jnp.equal(sampled_w[:, :, None], w_unique_jnp[None, None, :]).sum(axis=1)
    weights = weights / weights.sum(axis=1, keepdims=True)

    marginal_pdfs = jnp.einsum('bw,bwy->by', weights, pdfs)

    return {
        'mean': np.array(jnp.mean(marginal_pdfs, axis=0)),
        'low':  np.array(jnp.quantile(marginal_pdfs, 0.025, axis=0)),
        'high': np.array(jnp.quantile(marginal_pdfs, 0.975, axis=0))
    }