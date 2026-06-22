
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
from jax.random import PRNGKey, split, dirichlet, choice, bernoulli
import pandas as pd
import numpy as np
import statsmodels.api as sm

#import copula functions
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression,predictive_resample_cregression,predictive_resample_cregression_presampled,check_convergence_pr_cregression


### Logistic regression score and Fisher information ###
# Score (gradient of the log-likelihood) summed over observations.
# w_aug may be a single row (p+1,) or a matrix (n, p+1); x is scalar or (n,) accordingly.
def logistic_score(beta, w_aug, x):
    p = jax.nn.sigmoid(w_aug @ beta)
    return jnp.sum((x - p)[..., None] * w_aug, axis=0) if w_aug.ndim == 2 else (x - p) * w_aug


# Fisher information matrix evaluated at beta, using the (fixed) augmented design matrix W_aug.
# Note that the information matrix of a *single* logistic regression observation,
# p(1-p) * w w^T, is rank one and therefore singular whenever there is more than one
# parameter. We therefore evaluate the information implied by the original design
# (which is full rank) at the current, recursively updated, parameter value.
def logistic_fisher_info(beta, W_aug):
    p = jax.nn.sigmoid(W_aug @ beta)
    weights = p * (1.0 - p)
    return (W_aug * weights[:, None]).T @ W_aug
### ###


# Recursively generate a length-T sequence of (x, w) pairs for a single predictive sequence:
#   - w is resampled from the original covariates via the Bayesian bootstrap (Dirichlet weights)
#   - x is drawn from a logistic model whose coefficients are updated at each step via the
#     natural gradient, using running score and Fisher-information accumulators that are
#     initialised from the original n observations and updated with each new draw.
#     This avoids the rank-1 singularity of a single-observation Fisher information matrix:
#     the accumulator F_t = F_{t-1} + p_t(1-p_t) w_t w_t^T is full rank from the start
#     (because F_0 from the original n observations is already full rank), and grows
#     richer with every additional draw.  The natural-gradient step is discounted by 1/(n+t).
def _logistic_nat_grad_sequence(key, w_pool, W_orig_aug, x_orig, beta_init, n, T):
    n_pool = jnp.shape(w_pool)[0]

    # Bayesian bootstrap resample of w
    key, subkey = split(key)
    bb_weights = dirichlet(subkey, jnp.ones(n_pool))
    key, subkey = split(key)
    ind_w = choice(subkey, a=jnp.arange(n_pool), p=bb_weights, shape=(T,))
    w_new = w_pool[ind_w]
    w_new_aug = jnp.concatenate((jnp.ones((T, 1)), w_new), axis=1)

    # Initialise accumulators from the original n observations evaluated at beta_init.
    # At the MLE, S_0 is ~0 by definition; we compute it explicitly so the recursion
    # is correct even if beta_init is not the exact MLE.
    S_0 = logistic_score(beta_init, W_orig_aug, x_orig)   # total score: (p+1,)
    F_0 = logistic_fisher_info(beta_init, W_orig_aug)      # total Fisher info: (p+1, p+1)

    def step(carry, inputs):
        beta, S, F, key = carry
        t, w_t = inputs

        key, subkey = split(key)
        p_t = jax.nn.sigmoid(w_t @ beta)
        x_t = bernoulli(subkey, p_t).astype(beta.dtype)

        # update running accumulators with the new observation
        S_new = S + logistic_score(beta, w_t, x_t)
        F_new = F + p_t * (1.0 - p_t) * jnp.outer(w_t, w_t)

        nat_grad = jnp.linalg.solve(F_new, S_new)
        eta_t = 1.0 / (n + t + 1.0)
        beta_new = beta + eta_t * nat_grad

        return (beta_new, S_new, F_new, key), x_t

    (beta_final, _, _, _), x_new = scan(
        step, (beta_init, S_0, F_0, key), (jnp.arange(T, dtype=beta_init.dtype), w_new_aug)
    )

    return x_new, w_new, beta_final


_logistic_nat_grad_sequence_B = vmap(_logistic_nat_grad_sequence, (0, None, None, None, None, None, None))


def fit_propensity_scores(Z, ind_new_pr):
    """Fit logistic regression of X on W for each resampled predictive sequence and
    predict propensity scores P(X=1|W) for all original training observations.

    Parameters
    ----------
    Z : np.ndarray, shape (n, 1 + p)
        Original data — first column is treatment x, remaining columns are covariates w.
    ind_new_pr : np.ndarray, shape (B_post, T)
        Resampled observation indices for each posterior sample.

    Returns
    -------
    prop_scores : np.ndarray, shape (B_post, n)
    """
    Z = np.asarray(Z)
    ind_new_pr = np.asarray(ind_new_pr)
    n = Z.shape[0]
    B_post = ind_new_pr.shape[0]
    w_orig = sm.add_constant(Z[:, 1:], has_constant='add')
    prop_scores = np.zeros((B_post, n))

    for b in range(B_post):
        Z_resamp = Z[ind_new_pr[b]]
        x_resamp = Z_resamp[:, 0]
        w_resamp = sm.add_constant(Z_resamp[:, 1:], has_constant='add')
        logit = sm.Logit(x_resamp, w_resamp).fit(disp=False)
        prop_scores[b] = logit.predict(w_orig)

    return prop_scores


### Internal helpers shared by the unified mp_density ###

def _print_fit(fit, label=None):
    suffix = "" if label is None else f" for {label}"
    print(f"Optimised rho{suffix}: ", fit.rho_opt)
    print(f"Optimised rho_x{suffix}: ", fit.rho_x_opt)
    print(f"Prequential log-likelihood{suffix}: ", fit.preq_loglik)


def _covariate_weights(sampled_w, w_unique, mask=None):
    """Empirical covariate distribution over the rows of ``w_unique`` implied by the
    resampled covariates ``sampled_w`` (shape (B, T, p)).  If ``mask`` (shape (B, T))
    is given, only the masked draws contribute (used for ATT weighting, where only the
    treated draws define the w | X=1 distribution).  Returns weights of shape (B, n_w);
    sequences with no contributing draw fall back to a uniform distribution.
    """
    matched = jnp.all(sampled_w[:, :, None, :] == w_unique[None, None, :, :], axis=-1)
    if mask is not None:
        matched = matched & mask[:, :, None]
    weights = jnp.sum(matched, axis=1)
    weights_sum = weights.sum(axis=1, keepdims=True)
    n_w = w_unique.shape[0]
    return jnp.where(weights_sum == 0, jnp.ones_like(weights) / n_w, weights / weights_sum)


def _summarize(marginal_pdfs):
    """marginal_pdfs: (B, n_x, n_y) -> dict keyed by integer index into x_vals."""
    results = {}
    n_x = marginal_pdfs.shape[1]
    for i in range(n_x):
        mp = marginal_pdfs[:, i, :]
        results[f'x_{i}'] = {
            'mean': np.array(jnp.mean(mp, axis=0)),
            'low':  np.array(jnp.quantile(mp, 0.025, axis=0)),
            'high': np.array(jnp.quantile(mp, 0.975, axis=0)),
        }
    return results


def _bb_covariate_weights(inv, n_w, B_post, seed):
    """Closed-form Bayesian-bootstrap posterior of the covariate distribution over a grid
    of ``n_w`` unique covariate rows.  ``inv`` (length n_pop) maps each observation in the
    covariate population to its row in the unique grid.  Rather than running a length-T
    forward resample and counting frequencies, the martingale posterior of a *marginal*
    with the empirical predictive is exactly Rubin's Bayesian bootstrap: a single
    Dirichlet(1, ..., 1) draw over the n_pop observations, aggregated to the grid.  Returns
    weights of shape (B_post, n_w).  The same draw is shared across treatment arms, so that
    the potential-outcome contrast inherits the correct (shared) covariate uncertainty.
    """
    inv = np.asarray(inv)
    n_pop = inv.shape[0]
    onehot = jax.nn.one_hot(jnp.array(inv), n_w)              # (n_pop, n_w)
    alpha = dirichlet(PRNGKey(seed), jnp.ones(n_pop), shape=(B_post,))  # (B_post, n_pop)
    return alpha @ onehot                                     # (B_post, n_w)


def _build_target_grid(x_vals, w_unique_jnp, y_grid):
    """Cartesian grid over (x_vals, rows of w_unique, y_grid) as a z_target / y_target pair."""
    n_x, n_w, n_y = len(x_vals), len(w_unique_jnp), len(y_grid)
    y_target = jnp.tile(jnp.array(y_grid), n_w * n_x)
    w_target = jnp.tile(jnp.repeat(w_unique_jnp, n_y, axis=0), (n_x, 1))
    x_target = jnp.repeat(jnp.array(x_vals, dtype=w_target.dtype), n_w * n_y)[:, None]
    z_target = jnp.concatenate((x_target, w_target), axis=1)
    return y_target, z_target


def _mp_density_s_learner(y, x, w, x_vals, y_grid, B_post, T_fwd,
                          x_update, weighting, seed):
    Z = np.column_stack((x, w))
    n = Z.shape[0]
    y_jnp, Z_jnp = jnp.array(y), jnp.array(Z)

    # fit conditional copula regression on the full sample (one model -> S-learner)
    fit = fit_copula_cregression(y_jnp, Z_jnp, single_x_bandwidth=False, n_perm_optim=10)
    _print_fit(fit)

    # covariates of the weighting population: full sample (ATE) or treated group (ATT)
    w_pop = w[x == 1] if weighting == "att" else w
    w_unique = np.unique(w_pop, axis=0)
    w_unique_jnp = jnp.array(w_unique)
    n_w, n_x, n_y = len(w_unique_jnp), len(x_vals), len(y_grid)

    y_target, z_target = _build_target_grid(x_vals, w_unique_jnp, y_grid)

    prop_scores = None
    ind_new_pr = None
    if x_update == "bb":
        # resample (x, w) jointly via the Bayesian bootstrap
        _, logpdf_pr, ind_new_pr = predictive_resample_cregression(
            fit, Z, y_target, z_target, B_post, T_fwd, seed=seed
        )
        sampled_x = Z_jnp[ind_new_pr, 0]
        sampled_w = Z_jnp[ind_new_pr, 1:]
    else:  # x_update == "logistic"
        # resample w via the Bayesian bootstrap, draw x from a recursively (natural-gradient)
        # updated logistic model; the outcome (copula) density is unchanged.
        W_aug = sm.add_constant(w, has_constant='add')
        logit_fit = sm.Logit(x, W_aug).fit(disp=False)
        beta_init = jnp.array(np.asarray(logit_fit.params))
        w_jnp = jnp.array(w)
        W_aug_jnp = jnp.array(np.asarray(W_aug))

        key = PRNGKey(seed)
        key, *subkey = split(key, B_post + 1)
        subkey = jnp.array(subkey)
        x_orig_jnp = jnp.array(np.asarray(x).astype(float))

        x_new_all, w_new_all, beta_final_all = _logistic_nat_grad_sequence_B(
            subkey, w_jnp, W_aug_jnp, x_orig_jnp, beta_init, float(n), T_fwd
        )

        Z_new = jnp.concatenate((x_new_all[:, :, None], w_new_all), axis=-1)
        Z_orig_tiled = jnp.tile(Z_jnp[None, :, :], (B_post, 1, 1))
        Z_samp = jnp.concatenate((Z_orig_tiled, Z_new), axis=1)  # (B_post, n + T_fwd, 1 + p)

        _, logpdf_pr = predictive_resample_cregression_presampled(
            fit, Z_samp, y_target, z_target, B_post, T_fwd, seed=seed
        )
        sampled_x = x_new_all
        sampled_w = w_new_all
        # propensity scores from the recursively updated logistic coefficients
        prop_scores = np.array(jax.nn.sigmoid(W_aug_jnp @ beta_final_all.T).T)

    logpdf_pr = jnp.squeeze(logpdf_pr)
    pdfs = jnp.exp(logpdf_pr).reshape(B_post, n_x, n_w, n_y)

    mask = (sampled_x == 1) if weighting == "att" else None
    weights = _covariate_weights(sampled_w, w_unique_jnp, mask)
    marginal_pdfs = jnp.einsum('bw,bxwy->bxy', weights, pdfs)

    results = _summarize(marginal_pdfs)

    # propensity scores are reported whenever they are well defined: the ATT case (via a
    # per-sequence logistic refit) or whenever x was generated from a logistic model.
    if weighting == "att" and x_update == "bb":
        prop_scores = fit_propensity_scores(Z, np.asarray(ind_new_pr))
    if prop_scores is not None:
        results['propensity_scores'] = np.asarray(prop_scores)

    return results


def _mp_density_t_learner(y, x, w, x_vals, y_grid, B_post, T_fwd, weighting, seed):
    y = np.asarray(y)
    n_y = len(y_grid)

    # The outcome model p(y | X=x, w) (per arm) and the covariate law P(w) are separate
    # parameters with independent posteriors; we draw each and combine via the integral
    #   p(y(x)) = sum_w P(w) p(y | X=x, w).
    # Both arms are integrated over a *common* covariate grid (full sample for ATE, treated
    # group for ATT) so each arm's conditional is evaluated even at w values it did not
    # observe, targeting the shared marginal P(w) rather than the arm-conditional P(w|X=x).
    w_pop = w[x == 1] if weighting == "att" else w
    w_unique, inv = np.unique(w_pop, axis=0, return_inverse=True)
    w_unique_jnp = jnp.array(w_unique)
    n_w = len(w_unique_jnp)

    # closed-form Bayesian-bootstrap posterior of P(w); a single draw shared across arms
    cov_weights = _bb_covariate_weights(inv, n_w, B_post, seed)  # (B_post, n_w)

    marginals = []
    for x_val in x_vals:
        mask_arm = x == x_val
        Z_sub_jnp = jnp.array(np.column_stack((x[mask_arm], w[mask_arm])))

        # fit a separate outcome model on this treatment arm
        fit = fit_copula_cregression(jnp.array(y[mask_arm]), Z_sub_jnp,
                                     single_x_bandwidth=False, n_perm_optim=10)
        _print_fit(fit, label=f"x={x_val}")

        y_target, z_target = _build_target_grid([x_val], w_unique_jnp, y_grid)
        _, logpdf_pr, _ = predictive_resample_cregression(
            fit, Z_sub_jnp, y_target, z_target, B_post, T_fwd, seed=seed
        )
        pdfs = jnp.exp(jnp.squeeze(logpdf_pr)).reshape(B_post, n_w, n_y)

        marginals.append(jnp.einsum('bw,bwy->by', cov_weights, pdfs))

    marginal_pdfs = jnp.stack(marginals, axis=1)  # (B, n_x, n_y)
    return _summarize(marginal_pdfs)


# Unified marginal counterfactual density estimator via the martingale posterior.
#
# Estimates the marginal densities of the potential outcomes Y(x) for each value in
# ``x_vals`` and returns posterior mean and 95% credible bands on the grid ``y_grid``.
#
# Three independent choices control the variant:
#   x_update : "bb"        -> resample (x, w) jointly via the Bayesian bootstrap (default)
#              "logistic"  -> resample w via the Bayesian bootstrap but draw x from a
#                             recursively (natural-gradient) updated logistic model
#   weighting: "ate"       -> integrate over the full-sample covariate distribution (default)
#              "att"       -> integrate over the treated (w | X=1) covariate distribution
#   learner  : "s"         -> a single outcome model fit on the full sample (default)
#              "t"         -> a separate outcome model fit on each treatment arm
#
# The first option of each is the default and reproduces the original ``mp_density``.
# ``x_update="logistic"`` and ``weighting="att"`` both assume a binary treatment with
# ``x_vals=(0, 1)``.  ``learner="t"`` is incompatible with ``x_update="logistic"`` (a
# logistic x-update has no meaning when treatment is fixed within each arm).
#
# Returns a dict with keys ``"x_0", ..., "x_{k-1}"`` (one per value of ``x_vals``), each a
# dict with "mean", "low", "high".  A "propensity_scores" entry (shape (B_post, n)) is
# added whenever propensity scores are well defined, i.e. for ``weighting="att"`` or
# ``x_update="logistic"``.
def mp_causal_density(y, x, w, y_grid, B_post, T_fwd, *,
                      x_vals=(0, 1), x_update="bb", weighting="ate", learner="s", seed=42):
    if x_update not in ("bb", "logistic"):
        raise ValueError("x_update must be 'bb' or 'logistic'")
    if weighting not in ("ate", "att"):
        raise ValueError("weighting must be 'ate' or 'att'")
    if learner not in ("s", "t"):
        raise ValueError("learner must be 's' or 't'")
    if learner == "t" and x_update == "logistic":
        raise ValueError(
            "learner='t' is incompatible with x_update='logistic': treatment is fixed "
            "within each arm, so an in-sequence logistic x-update is not defined."
        )

    w = np.asarray(w)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    x = np.asarray(x)

    if np.isscalar(x_vals):
        x_vals = np.array([x_vals])
    else:
        x_vals = np.asarray(x_vals)

    # the logistic x-update and the ATT weighting both assume a binary treatment
    if x_update == "logistic" or weighting == "att":
        x_levels = set(np.unique(x).tolist())
        if not x_levels.issubset({0, 1}):
            raise ValueError(
                f"x_update='logistic'/weighting='att' require binary x in {{0, 1}}; got levels {sorted(x_levels)}"
            )
        if set(np.asarray(x_vals).tolist()) != {0, 1}:
            raise ValueError(
                "x_update='logistic'/weighting='att' require x_vals=(0, 1)"
            )
        if np.sum(x == 1) == 0 or np.sum(x == 0) == 0:
            raise ValueError("both treated (x==1) and control (x==0) observations are required")

    if learner == "t":
        return _mp_density_t_learner(y, x, w, x_vals, y_grid, B_post, T_fwd, weighting, seed)
    return _mp_density_s_learner(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                 x_update, weighting, seed)
