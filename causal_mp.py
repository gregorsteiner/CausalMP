
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan, fori_loop
from jax.random import PRNGKey, split, dirichlet, choice, bernoulli
import pandas as pd
import numpy as np
import statsmodels.api as sm

#import copula functions
from pr_copula.main_copula_regression_conditional import fit_copula_cregression,predict_copula_cregression,predictive_resample_cregression,predictive_resample_cregression_presampled,check_convergence_pr_cregression
from pr_copula.copula_regression_functions import update_copula as _copula_update_reg, calc_logkxx as _calc_logkxx


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


# Shared recursion: coefficients are updated at each step via the natural gradient, using
# running score and Fisher-information accumulators that are initialised from the original
# n observations and updated with each new draw. This avoids the rank-1 singularity of a
# single-observation Fisher information matrix: the accumulator F_t = F_{t-1} + p_t(1-p_t) w_t w_t^T
# is full rank from the start (because F_0 from the original n observations is already full
# rank), and grows richer with every additional draw. The natural-gradient step is discounted
# by 1/(n+t).
# Given an *already presampled* augmented covariate sequence
# Z_new_aug (T, p+1), recursively natural-gradient-update a logistic model and draw a
# Bernoulli response at each step. Used both for the treatment natural-gradient update
# (w resampled internally, below) and for the zero-inflation indicator update (covariates
# presampled from the treatment/covariate resampling step, see mp_causal_density_zi).
def _logistic_nat_grad_step_presampled(key, Z_new_aug, Z_orig_aug, y_orig, beta_init, n, T):
    S_0 = logistic_score(beta_init, Z_orig_aug, y_orig)   # total score: (p+1,)
    F_0 = logistic_fisher_info(beta_init, Z_orig_aug)      # total Fisher info: (p+1, p+1)

    def step(carry, inputs):
        beta, S, F, key = carry
        t, z_t = inputs

        key, subkey = split(key)
        p_t = jax.nn.sigmoid(z_t @ beta)
        y_t = bernoulli(subkey, p_t).astype(beta.dtype)

        # update running accumulators with the new observation
        S_new = S + logistic_score(beta, z_t, y_t)
        F_new = F + p_t * (1.0 - p_t) * jnp.outer(z_t, z_t)

        nat_grad = jnp.linalg.solve(F_new, S_new)
        eta_t = 1.0 / (n + t + 1.0)
        beta_new = beta + eta_t * nat_grad

        return (beta_new, S_new, F_new, key), y_t

    (beta_final, _, _, _), y_new = scan(
        step, (beta_init, S_0, F_0, key), (jnp.arange(T, dtype=beta_init.dtype), Z_new_aug)
    )

    return y_new, beta_final


_logistic_nat_grad_step_presampled_B = vmap(
    _logistic_nat_grad_step_presampled, (0, 0, None, None, None, None, None)
)


def _logistic_nat_grad_sequence(key, w_pool, W_orig_aug, x_orig, beta_init, n, T):
    n_pool = jnp.shape(w_pool)[0]

    # Bayesian bootstrap resample of w
    key, subkey = split(key)
    bb_weights = dirichlet(subkey, jnp.ones(n_pool))
    key, subkey = split(key)
    ind_w = choice(subkey, a=jnp.arange(n_pool), p=bb_weights, shape=(T,))
    w_new = w_pool[ind_w]
    w_new_aug = jnp.concatenate((jnp.ones((T, 1)), w_new), axis=1)

    x_new, beta_final = _logistic_nat_grad_step_presampled(
        key, w_new_aug, W_orig_aug, x_orig, beta_init, n, T
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
            'mean':   np.array(jnp.mean(mp, axis=0)),
            'low':    np.array(jnp.quantile(mp, 0.025, axis=0)),
            'high':   np.array(jnp.quantile(mp, 0.975, axis=0)),
        }
    return results


def _summarize_conditional(cond_pdfs, w_cond):
    """Summarize the conditional interventional densities p(y | X=x, w) at the requested
    covariate rows.  ``cond_pdfs`` has shape (B, n_x, n_cond, n_y); returns a nested dict
    ``{'w_values': w_cond, 'x_0': {'w_0': {...}, ...}, ...}`` with posterior mean and 95%
    credible bands for each (treatment value, conditioning covariate) pair.
    """
    n_x, n_cond = cond_pdfs.shape[1], cond_pdfs.shape[2]
    out = {'w_values': np.asarray(w_cond)}
    for i in range(n_x):
        xi = {}
        for j in range(n_cond):
            s = cond_pdfs[:, i, j, :]
            xi[f'w_{j}'] = {
                'mean':   np.array(jnp.mean(s, axis=0)),
                'low':    np.array(jnp.quantile(s, 0.025, axis=0)),
                'high':   np.array(jnp.quantile(s, 0.975, axis=0)),
            }
        out[f'x_{i}'] = xi
    return out


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
                          x_update, weighting, seed, w_cond=None):
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

    # extend the evaluation grid with the conditioning covariate rows, kept in a separate
    # trailing block so they do not enter the marginalisation over the covariate law.
    n_cond = 0 if w_cond is None else len(w_cond)
    w_grid_jnp = (jnp.concatenate((w_unique_jnp, jnp.array(w_cond)), axis=0)
                  if n_cond else w_unique_jnp)
    n_w_tot = n_w + n_cond

    y_target, z_target = _build_target_grid(x_vals, w_grid_jnp, y_grid)

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
    pdfs = jnp.exp(logpdf_pr).reshape(B_post, n_x, n_w_tot, n_y)

    mask = (sampled_x == 1) if weighting == "att" else None
    weights = _covariate_weights(sampled_w, w_unique_jnp, mask)
    marginal_pdfs = jnp.einsum('bw,bxwy->bxy', weights, pdfs[:, :, :n_w, :])

    results = _summarize(marginal_pdfs)

    if n_cond:
        results['conditional'] = _summarize_conditional(pdfs[:, :, n_w:, :], w_cond)

    # propensity scores are reported whenever they are well defined: the ATT case (via a
    # per-sequence logistic refit) or whenever x was generated from a logistic model.
    if weighting == "att" and x_update == "bb":
        prop_scores = fit_propensity_scores(Z, np.asarray(ind_new_pr))
    if prop_scores is not None:
        results['propensity_scores'] = np.asarray(prop_scores)

    return results


def _mp_density_t_learner(y, x, w, x_vals, y_grid, B_post, T_fwd, weighting, seed, w_cond=None):
    y = np.asarray(y)
    n_y = len(y_grid)
    n_cond = 0 if w_cond is None else len(w_cond)
    w_cond_jnp = jnp.array(w_cond) if n_cond else None

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

    # extend each arm's grid with the conditioning rows, kept in a trailing block
    w_grid_jnp = (jnp.concatenate((w_unique_jnp, w_cond_jnp), axis=0)
                  if n_cond else w_unique_jnp)
    n_w_tot = n_w + n_cond

    marginals = []
    cond_blocks = []
    for x_val in x_vals:
        mask_arm = x == x_val
        Z_sub_jnp = jnp.array(w[mask_arm])

        # fit a separate outcome model on this treatment arm
        fit = fit_copula_cregression(jnp.array(y[mask_arm]), Z_sub_jnp,
                                     single_x_bandwidth=False, n_perm_optim=10)
        _print_fit(fit, label=f"x={x_val}")

        n_w_grid = len(w_grid_jnp)
        y_target = jnp.tile(jnp.array(y_grid), n_w_grid)
        z_target = jnp.repeat(w_grid_jnp, n_y, axis=0)
        _, logpdf_pr, _ = predictive_resample_cregression(
            fit, Z_sub_jnp, y_target, z_target, B_post, T_fwd, seed=seed
        )
        pdfs = jnp.exp(jnp.squeeze(logpdf_pr)).reshape(B_post, n_w_tot, n_y)

        marginals.append(jnp.einsum('bw,bwy->by', cov_weights, pdfs[:, :n_w, :]))
        if n_cond:
            cond_blocks.append(pdfs[:, n_w:, :])   # (B, n_cond, n_y) for this arm

    marginal_pdfs = jnp.stack(marginals, axis=1)  # (B, n_x, n_y)
    results = _summarize(marginal_pdfs)

    if n_cond:
        cond_pdfs = jnp.stack(cond_blocks, axis=1)  # (B, n_x, n_cond, n_y)
        results['conditional'] = _summarize_conditional(cond_pdfs, w_cond)
    return results


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
# dict with "median", "low", "high".  A "propensity_scores" entry (shape (B_post, n)) is
# added whenever propensity scores are well defined, i.e. for ``weighting="att"`` or
# ``x_update="logistic"``.
#
# ``w_cond`` (optional, shape (n_cond, p) or (p,)) additionally returns the *conditional*
# interventional densities p(y | X=x, w) at the given covariate rows, without marginalising
# over W, under the key ``"conditional"`` (nested as ``conditional[x_i][w_j]``).  These
# slices are a pure outcome-model object: they are unaffected by ``x_update`` and
# ``weighting`` (which only shape the marginalisation), and share the predictive-resampling
# draws of the marginal run.  With ``w_cond=None`` the marginal output is unchanged.
def mp_causal_density(y, x, w, y_grid, B_post, T_fwd, *,
                      x_vals=(0, 1), x_update="bb", weighting="ate", learner="s",
                      w_cond=None, seed=42):
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

    if w_cond is not None:
        w_cond = np.asarray(w_cond, dtype=float)
        if w_cond.ndim == 1:
            w_cond = w_cond.reshape(1, -1)
        if w_cond.shape[1] != w.shape[1]:
            raise ValueError(
                f"w_cond must have {w.shape[1]} columns to match w; got {w_cond.shape[1]}"
            )

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
        return _mp_density_t_learner(y, x, w, x_vals, y_grid, B_post, T_fwd, weighting, seed,
                                     w_cond=w_cond)
    return _mp_density_s_learner(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                 x_update, weighting, seed, w_cond=w_cond)


### Diagnostic functions ###

def _compute_w_map(x, w, w_unique, weighting):
    n = len(x)
    if weighting == "att":
        w_map_arr = np.full(n, -1, dtype=int)
        for i in range(n):
            matches = np.all(w[i:i+1] == w_unique, axis=1)
            if np.any(matches):
                w_map_arr[i] = np.argmax(matches)
        return jnp.array(w_map_arr)
    else:
        _, w_inv = np.unique(w, axis=0, return_inverse=True)
        return jnp.array(w_inv)


def _make_w_idx_and_mask(ind_new, w_map, Z_jnp, weighting):
    w_idx = w_map[ind_new]
    if weighting == "att":
        x_new = Z_jnp[ind_new, 0]
        w_msk = ((x_new == 1) & (w_idx >= 0)).astype(jnp.float32)
        w_idx = jnp.where(w_idx < 0, 0, w_idx)
    else:
        w_msk = jnp.ones(ind_new.shape[0])
    return w_idx, w_msk


def _mp_density_s_learner_diagnostic(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                     x_update, weighting, seed):
    Z = np.column_stack((x, w))
    n = Z.shape[0]
    y_jnp, Z_jnp = jnp.array(y), jnp.array(Z)

    fit = fit_copula_cregression(y_jnp, Z_jnp, single_x_bandwidth=False, n_perm_optim=10)
    _print_fit(fit)
    rho_opt = fit.rho_opt
    rho_x_opt = fit.rho_x_opt

    w_pop = w[x == 1] if weighting == "att" else w
    w_unique, w_inv = np.unique(w_pop, axis=0, return_inverse=True)
    w_unique_jnp = jnp.array(w_unique)
    n_w, n_x, n_y = len(w_unique), len(x_vals), len(y_grid)

    y_target, z_target = _build_target_grid(x_vals, w_unique_jnp, y_grid)

    logcdf_init, logpdf_init = predict_copula_cregression(fit, y_target, z_target)

    w_emp_counts = np.bincount(w_inv, minlength=n_w).astype(float)
    w_emp_weights = jnp.array(w_emp_counts / w_emp_counts.sum())

    flat_idx = np.arange(n_x * n_w * n_y).reshape(n_x, n_w, n_y)
    marginal_idx = jnp.array(flat_idx.transpose(0, 2, 1))  # (n_x, n_y, n_w)

    pdf_init_flat = jnp.exp(logpdf_init[:, -1])
    p_n_marginal = jnp.sum(pdf_init_flat[marginal_idx] * w_emp_weights[None, None, :], axis=-1)

    w_map = _compute_w_map(x, w, w_unique, weighting)

    prop_scores = None

    if x_update == "bb":
        key = PRNGKey(seed)
        key, *subkeys = split(key, B_post + 1)
        subkeys_arr = jnp.array(subkeys)

        def _single_diag(subkey):
            k1, k2, k3 = split(subkey, 3)
            bb_w = dirichlet(k1, jnp.ones(n))
            ind_new = choice(k2, a=jnp.arange(n), p=bb_w, shape=(T_fwd,))
            z_new = Z_jnp[ind_new]
            z_samp = jnp.concatenate((Z_jnp, z_new), axis=0)

            w_idx, w_msk = _make_w_idx_and_mask(ind_new, w_map, Z_jnp, weighting)
            a_rand = jax.random.uniform(k3, shape=(T_fwd, 1))

            counts_init = jnp.array(w_emp_counts)
            l1_init = jnp.zeros((T_fwd, n_x))

            def step(i, carry):
                logcdf, logpdf, l1_dists, counts = carry
                z_new_i = z_samp[n + i]
                logalpha = jnp.log(2.0 - 1.0 / (n + i + 1)) - jnp.log(n + i + 2.0)
                logk_xx = _calc_logkxx(z_target, z_new_i, rho_x_opt)
                logalphak_xx = logalpha + logk_xx
                log1alpha = jnp.log1p(-jnp.exp(logalpha))
                logalpha_x = logalphak_xx - jnp.logaddexp(log1alpha, logalphak_xx)
                u = jnp.exp(logcdf)
                v = a_rand[i]
                logcdf, logpdf = _copula_update_reg(logcdf, logpdf, u, v, logalpha_x, rho_opt)

                counts = counts.at[w_idx[i]].add(w_msk[i])
                total = counts.sum()
                weights = jnp.where(total > 0, counts / total, jnp.ones(n_w) / n_w)

                pdf_flat = jnp.exp(logpdf[:, -1])
                pdf_sel = pdf_flat[marginal_idx]
                marginal = jnp.sum(pdf_sel * weights[None, None, :], axis=-1)
                l1 = jnp.sum(jnp.abs(marginal - p_n_marginal), axis=-1)
                l1_dists = l1_dists.at[i].set(l1)

                return logcdf, logpdf, l1_dists, counts

            logcdf_f, logpdf_f, l1_dists, final_counts = fori_loop(
                0, T_fwd, step, (logcdf_init, logpdf_init, l1_init, counts_init))
            return logcdf_f, logpdf_f, l1_dists, final_counts, ind_new

        print('Diagnostic resampling...')
        logcdf_pr, logpdf_pr, l1_trajectory, final_counts_all, ind_new_pr = vmap(_single_diag)(subkeys_arr)

        sampled_x = Z_jnp[ind_new_pr, 0]
        sampled_w = Z_jnp[ind_new_pr, 1:]

    else:  # x_update == "logistic"
        W_aug = sm.add_constant(w, has_constant='add')
        logit_fit = sm.Logit(x, W_aug).fit(disp=False)
        beta_init = jnp.array(np.asarray(logit_fit.params))
        w_jnp = jnp.array(w)
        W_aug_jnp = jnp.array(np.asarray(W_aug))

        key = PRNGKey(seed)
        key, *subkeys = split(key, B_post + 1)
        subkeys_arr = jnp.array(subkeys)
        x_orig_jnp = jnp.array(np.asarray(x).astype(float))

        x_new_all, w_new_all, beta_final_all = _logistic_nat_grad_sequence_B(
            subkeys_arr, w_jnp, W_aug_jnp, x_orig_jnp, beta_init, float(n), T_fwd
        )

        Z_new = jnp.concatenate((x_new_all[:, :, None], w_new_all), axis=-1)
        Z_orig_tiled = jnp.tile(Z_jnp[None, :, :], (B_post, 1, 1))
        Z_samp_all = jnp.concatenate((Z_orig_tiled, Z_new), axis=1)

        matched = jnp.all(w_new_all[:, :, None, :] == w_unique_jnp[None, None, :, :], axis=-1)
        w_idx_all = jnp.argmax(matched, axis=-1)
        if weighting == "att":
            w_msk_all = (x_new_all == 1).astype(jnp.float32) * jnp.any(matched, axis=-1).astype(jnp.float32)
        else:
            w_msk_all = jnp.ones((B_post, T_fwd))

        key, *subkeys2 = split(key, B_post + 1)
        subkeys2_arr = jnp.array(subkeys2)

        def _single_diag_logistic(subkey, z_samp, w_idx, w_msk):
            a_rand = jax.random.uniform(subkey, shape=(T_fwd, 1))
            counts_init = jnp.array(w_emp_counts)
            l1_init = jnp.zeros((T_fwd, n_x))

            def step(i, carry):
                logcdf, logpdf, l1_dists, counts = carry
                z_new_i = z_samp[n + i]
                logalpha = jnp.log(2.0 - 1.0 / (n + i + 1)) - jnp.log(n + i + 2.0)
                logk_xx = _calc_logkxx(z_target, z_new_i, rho_x_opt)
                logalphak_xx = logalpha + logk_xx
                log1alpha = jnp.log1p(-jnp.exp(logalpha))
                logalpha_x = logalphak_xx - jnp.logaddexp(log1alpha, logalphak_xx)
                u = jnp.exp(logcdf)
                v = a_rand[i]
                logcdf, logpdf = _copula_update_reg(logcdf, logpdf, u, v, logalpha_x, rho_opt)

                counts = counts.at[w_idx[i]].add(w_msk[i])
                total = counts.sum()
                weights = jnp.where(total > 0, counts / total, jnp.ones(n_w) / n_w)

                pdf_flat = jnp.exp(logpdf[:, -1])
                pdf_sel = pdf_flat[marginal_idx]
                marginal = jnp.sum(pdf_sel * weights[None, None, :], axis=-1)
                l1 = jnp.sum(jnp.abs(marginal - p_n_marginal), axis=-1)
                l1_dists = l1_dists.at[i].set(l1)

                return logcdf, logpdf, l1_dists, counts

            logcdf_f, logpdf_f, l1_dists, final_counts = fori_loop(
                0, T_fwd, step, (logcdf_init, logpdf_init, l1_init, counts_init))
            return logcdf_f, logpdf_f, l1_dists, final_counts

        print('Diagnostic resampling...')
        logcdf_pr, logpdf_pr, l1_trajectory, final_counts_all = vmap(
            _single_diag_logistic)(subkeys2_arr, Z_samp_all, w_idx_all, w_msk_all)

        sampled_x = x_new_all
        sampled_w = w_new_all
        prop_scores = np.array(jax.nn.sigmoid(W_aug_jnp @ beta_final_all.T).T)

    logpdf_pr = jnp.squeeze(logpdf_pr)
    logcdf_pr = jnp.squeeze(logcdf_pr)
    pdfs = jnp.exp(logpdf_pr).reshape(B_post, n_x, n_w, n_y)
    cdfs = jnp.exp(logcdf_pr).reshape(B_post, n_x, n_w, n_y)

    final_weights = final_counts_all / final_counts_all.sum(axis=1, keepdims=True)
    marginal_pdfs = jnp.einsum('bw,bxwy->bxy', final_weights, pdfs)
    marginal_cdfs = jnp.einsum('bw,bxwy->bxy', final_weights, cdfs)

    results = _summarize(marginal_pdfs)
    results['marginal_pdfs'] = np.array(marginal_pdfs)
    results['marginal_cdfs'] = np.array(marginal_cdfs)
    l1_trajectory = jnp.concatenate([jnp.zeros((B_post, 1, n_x)), l1_trajectory], axis=1)
    results['l1_trajectory'] = np.array(l1_trajectory)

    if weighting == "att" and x_update == "bb":
        prop_scores = fit_propensity_scores(Z, np.asarray(ind_new_pr))
    if prop_scores is not None:
        results['propensity_scores'] = np.asarray(prop_scores)

    return results


def _mp_density_t_learner_diagnostic(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                     weighting, seed):
    y = np.asarray(y)
    n_y = len(y_grid)

    w_pop = w[x == 1] if weighting == "att" else w
    w_unique, w_inv = np.unique(w_pop, axis=0, return_inverse=True)
    w_unique_jnp = jnp.array(w_unique)
    n_w = len(w_unique_jnp)
    n_x = len(x_vals)

    w_emp_counts = np.bincount(w_inv, minlength=n_w).astype(float)
    w_emp_weights = jnp.array(w_emp_counts / w_emp_counts.sum())

    # Shared BB covariate sequence (explicit resampling, not Dirichlet)
    n_pop = len(w_pop)
    key = PRNGKey(seed)
    key, *subkeys_shared = split(key, B_post + 1)
    subkeys_shared = jnp.array(subkeys_shared)

    w_pop_jnp = jnp.array(w_pop)
    _, w_pop_inv = np.unique(w_pop, axis=0, return_inverse=True)
    w_pop_map = jnp.array(w_pop_inv)

    def _draw_shared_bb(subkey):
        k1, k2 = split(subkey)
        bb_w = dirichlet(k1, jnp.ones(n_pop))
        ind_shared = choice(k2, a=jnp.arange(n_pop), p=bb_w, shape=(T_fwd,))
        w_idx_shared = w_pop_map[ind_shared]
        return w_idx_shared

    shared_w_idx = vmap(_draw_shared_bb)(subkeys_shared)  # (B_post, T_fwd)

    marginals = []
    marginal_cdfs_arms = []
    l1_all_arms = []

    for arm_i, x_val in enumerate(x_vals):
        mask_arm = x == x_val
        y_arm = jnp.array(y[mask_arm])
        Z_sub_jnp = jnp.array(w[mask_arm])
        n_arm = int(mask_arm.sum())

        fit = fit_copula_cregression(y_arm, Z_sub_jnp,
                                     single_x_bandwidth=False, n_perm_optim=10)
        _print_fit(fit, label=f"x={x_val}")
        rho_opt = fit.rho_opt
        rho_x_opt = fit.rho_x_opt

        y_target = jnp.tile(jnp.array(y_grid), n_w)
        z_target = jnp.repeat(w_unique_jnp, n_y, axis=0)
        logcdf_init, logpdf_init = predict_copula_cregression(fit, y_target, z_target)

        flat_idx = np.arange(1 * n_w * n_y).reshape(1, n_w, n_y)
        marginal_idx = jnp.array(flat_idx.transpose(0, 2, 1))  # (1, n_y, n_w)

        pdf_init_flat = jnp.exp(logpdf_init[:, -1])
        p_n_marginal = jnp.sum(pdf_init_flat[marginal_idx] * w_emp_weights[None, None, :], axis=-1)

        key, *subkeys_arm = split(key, B_post + 1)
        subkeys_arm = jnp.array(subkeys_arm)

        def _single_diag_arm(subkey, w_idx_shared):
            k1, k2, k3 = split(subkey, 3)
            bb_w = dirichlet(k1, jnp.ones(n_arm))
            ind_new = choice(k2, a=jnp.arange(n_arm), p=bb_w, shape=(T_fwd,))
            z_new = Z_sub_jnp[ind_new]
            z_samp = jnp.concatenate((Z_sub_jnp, z_new), axis=0)

            a_rand = jax.random.uniform(k3, shape=(T_fwd, 1))
            counts_init = jnp.array(w_emp_counts)
            l1_init = jnp.zeros((T_fwd, 1))

            def step(i, carry):
                logcdf, logpdf, l1_dists, counts = carry
                z_new_i = z_samp[n_arm + i]
                logalpha = jnp.log(2.0 - 1.0 / (n_arm + i + 1)) - jnp.log(n_arm + i + 2.0)
                logk_xx = _calc_logkxx(z_target, z_new_i, rho_x_opt)
                logalphak_xx = logalpha + logk_xx
                log1alpha = jnp.log1p(-jnp.exp(logalpha))
                logalpha_x = logalphak_xx - jnp.logaddexp(log1alpha, logalphak_xx)
                u = jnp.exp(logcdf)
                v = a_rand[i]
                logcdf, logpdf = _copula_update_reg(logcdf, logpdf, u, v, logalpha_x, rho_opt)

                counts = counts.at[w_idx_shared[i]].add(1.0)
                weights = counts / counts.sum()

                pdf_flat = jnp.exp(logpdf[:, -1])
                pdf_sel = pdf_flat[marginal_idx]
                marginal = jnp.sum(pdf_sel * weights[None, None, :], axis=-1)
                l1 = jnp.sum(jnp.abs(marginal - p_n_marginal), axis=-1)
                l1_dists = l1_dists.at[i].set(l1)

                return logcdf, logpdf, l1_dists, counts

            logcdf_f, logpdf_f, l1_dists, final_counts = fori_loop(
                0, T_fwd, step, (logcdf_init, logpdf_init, l1_init, counts_init))
            return logcdf_f, logpdf_f, l1_dists, final_counts

        print(f'Diagnostic resampling for x={x_val}...')
        logcdf_pr, logpdf_pr, l1_arm, final_counts = vmap(_single_diag_arm)(subkeys_arm, shared_w_idx)

        pdfs = jnp.exp(jnp.squeeze(logpdf_pr)).reshape(B_post, n_w, n_y)
        cdfs = jnp.exp(jnp.squeeze(logcdf_pr)).reshape(B_post, n_w, n_y)
        final_weights = final_counts / final_counts.sum(axis=1, keepdims=True)
        marginals.append(jnp.einsum('bw,bwy->by', final_weights, pdfs))
        marginal_cdfs_arms.append(jnp.einsum('bw,bwy->by', final_weights, cdfs))
        l1_all_arms.append(l1_arm)

    marginal_pdfs = jnp.stack(marginals, axis=1)
    marginal_cdfs = jnp.stack(marginal_cdfs_arms, axis=1)
    results = _summarize(marginal_pdfs)
    results['marginal_pdfs'] = np.array(marginal_pdfs)
    results['marginal_cdfs'] = np.array(marginal_cdfs)

    l1_trajectory = jnp.concatenate(l1_all_arms, axis=-1)  # (B, T, n_x)
    l1_trajectory = jnp.concatenate([jnp.zeros((B_post, 1, n_x)), l1_trajectory], axis=1)
    results['l1_trajectory'] = np.array(l1_trajectory)
    return results


def mp_causal_density_diagnostic(y, x, w, y_grid, B_post, T_fwd, *,
                                 x_vals=(0, 1), x_update="bb", weighting="ate",
                                 learner="s", seed=42):
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
        return _mp_density_t_learner_diagnostic(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                                weighting, seed)
    return _mp_density_s_learner_diagnostic(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                            x_update, weighting, seed)
### ###


### Zero-inflated mixture: point mass at y0 plus a continuous copula-regression part ###
#
# Motivated by the Lalonde job-training data, where the outcome (re78) has a large point
# mass at zero (unemployed participants). The Gaussian-copula recursion works purely in
# CDF-value space and assumes an absolutely continuous marginal (see update_copula_single in
# pr_copula/copula_density_functions.py); feeding a point mass through it corrupts both the
# bandwidth fit and the stationarity of the forward-resampling walk. Here Y is modelled as a
# mixture: P(Y=y0|x,w) via a simple recursively (natural-gradient) updated logistic
# regression -- reusing the same recursion used for the treatment logistic update above --
# and Y|Y!=y0,x,w via the existing copula-regression treatment, fit only on the non-atom
# subsample. Since the response is never literally copied from history in this scheme (even
# under x_update="bb" the new y is generated from the copula's own evolving state via a
# fresh uniform quantile draw, not from the bootstrapped row's actual y -- see
# pr_copula/sample_copula_regression_functions.py), the atom/continuous decision at each
# forward step must also come from a live model rather than from the resampled row's
# original y.
#
# Only the S-learner is supported (matches the Lalonde ATT use case); T-learner support
# would need a separate zero-model and continuous model per treatment arm.

# Single forward-resampling loop (one posterior sequence) for the continuous submodel,
# gated by the zero-model draws is_zero_b. The martingale step-size schedule is keyed to
# j_cont, a running count of *continuous* draws seen so far (initialised at n_pos, the
# number of non-atom observations used to fit the continuous submodel): atom draws must not
# advance the continuous submodel's effective sample size. On atom steps the continuous
# cdf/pdf state is left exactly unchanged via jnp.where -- deliberately not relying on
# alpha_x -> 0 as a no-op, since logalpha_x is clipped away from -inf for numerical stability
# a few lines below and would otherwise leak a small update in on every atom step.
def _zi_forward_step_loop(subkey, z_samp_b, is_zero_b, rho_pos, rho_x_pos, z_target,
                          n_pos, T_fwd, logcdf_init, logpdf_init,
                          w_idx_b=None, w_msk_b=None, w_emp_counts=None, n_w=None,
                          p_n_marginal_pos=None, marginal_idx=None, diagnostic=False):
    a_rand = jax.random.uniform(subkey, shape=(T_fwd, 1))

    if diagnostic:
        n_x = marginal_idx.shape[0]
        counts_init = jnp.array(w_emp_counts)
        l1_init = jnp.zeros((T_fwd, n_x))
    else:
        counts_init = jnp.zeros(0)
        l1_init = jnp.zeros(0)

    def step(i, carry):
        logcdf, logpdf, j_cont, counts, l1_dists = carry
        z_new_i = z_samp_b[i]
        is_zero_i = is_zero_b[i]

        logalpha = jnp.log(2.0 - 1.0 / (j_cont + 1.0)) - jnp.log(j_cont + 2.0)
        logk_xx = _calc_logkxx(z_target, z_new_i, rho_x_pos)
        logalphak_xx = logalpha + logk_xx
        log1alpha = jnp.log1p(-jnp.exp(logalpha))
        logalpha_x = logalphak_xx - jnp.logaddexp(log1alpha, logalphak_xx)
        eps = 1e-4
        logalpha_x = jnp.clip(logalpha_x, jnp.log(eps), jnp.log(1 - eps))

        u = jnp.exp(logcdf)
        v = a_rand[i]
        logcdf_upd, logpdf_upd = _copula_update_reg(logcdf, logpdf, u, v, logalpha_x, rho_pos)

        is_zero_mask = is_zero_i > 0.5
        logcdf_new = jnp.where(is_zero_mask, logcdf, logcdf_upd)
        logpdf_new = jnp.where(is_zero_mask, logpdf, logpdf_upd)
        j_cont_new = j_cont + (1.0 - is_zero_i)

        if diagnostic:
            counts = counts.at[w_idx_b[i]].add(w_msk_b[i])
            total = counts.sum()
            weights = jnp.where(total > 0, counts / total, jnp.ones(n_w) / n_w)
            pdf_flat = jnp.exp(logpdf_new[:, -1])
            pdf_sel = pdf_flat[marginal_idx]
            marginal = jnp.sum(pdf_sel * weights[None, None, :], axis=-1)
            l1 = jnp.sum(jnp.abs(marginal - p_n_marginal_pos), axis=-1)
            l1_dists = l1_dists.at[i].set(l1)

        return logcdf_new, logpdf_new, j_cont_new, counts, l1_dists

    init_carry = (logcdf_init, logpdf_init, jnp.asarray(float(n_pos)), counts_init, l1_init)
    logcdf_f, logpdf_f, j_cont_f, final_counts, l1_dists = fori_loop(0, T_fwd, step, init_carry)
    return logpdf_f, final_counts, l1_dists


# Covariate-bucket index/mask for a resampled (x, w) sequence, used to accumulate the running
# ATT/ATE covariate weights inside the diagnostic loop. Mirrors the two branches already used
# in _mp_density_s_learner_diagnostic (bb: historical row indices via _make_w_idx_and_mask;
# logistic: match freshly-drawn w rows against the unique covariate grid).
def _zi_w_idx_mask(x_update, weighting, w_unique_jnp, w_map=None, ind_new_pr=None, Z_jnp=None,
                   sampled_x=None, sampled_w=None):
    if x_update == "bb":
        return _make_w_idx_and_mask(ind_new_pr, w_map, Z_jnp, weighting)
    matched = jnp.all(sampled_w[:, :, None, :] == w_unique_jnp[None, None, :, :], axis=-1)
    w_idx = jnp.argmax(matched, axis=-1)
    if weighting == "att":
        w_msk = (sampled_x == 1).astype(jnp.float32) * jnp.any(matched, axis=-1).astype(jnp.float32)
    else:
        w_msk = jnp.ones_like(sampled_x)
    return w_idx, w_msk


def _mp_density_zi_s_learner_impl(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                  x_update, weighting, y0, seed, diagnostic):
    Z = np.column_stack((x, w))
    n = Z.shape[0]
    y_np = np.asarray(y)
    is_zero_orig = (y_np == y0)
    n_pos = int((~is_zero_orig).sum())
    if n_pos == 0 or int(is_zero_orig.sum()) == 0:
        raise ValueError(f"y0={y0} must have both atom and non-atom observations present")
    print(f"Zero-atom rate at y0={y0}: {is_zero_orig.mean():.4f} ({int(is_zero_orig.sum())}/{n})")

    Z_jnp = jnp.array(Z)
    y_pos_jnp = jnp.array(y_np[~is_zero_orig])
    Z_pos_jnp = jnp.array(Z[~is_zero_orig])

    # continuous submodel: fit only on the non-atom subsample
    fit_pos = fit_copula_cregression(y_pos_jnp, Z_pos_jnp, single_x_bandwidth=False, n_perm_optim=10)
    _print_fit(fit_pos, label="continuous part (Y != y0)")

    # zero-atom submodel: simple logistic regression of 1(Y=y0) on (x, w), recursively
    # natural-gradient-updated during forward resampling using the SAME (x, w) draws as the
    # treatment/covariate resampling below
    Z_aug = sm.add_constant(Z, has_constant='add')
    zero_logit_fit = sm.Logit(is_zero_orig.astype(float), Z_aug).fit(disp=False)
    beta_zero_init = jnp.array(np.asarray(zero_logit_fit.params))
    Z_aug_jnp = jnp.array(np.asarray(Z_aug))
    is_zero_orig_jnp = jnp.array(is_zero_orig.astype(float))

    w_pop = w[x == 1] if weighting == "att" else w
    w_unique, w_pop_inv = np.unique(w_pop, axis=0, return_inverse=True)
    w_unique_jnp = jnp.array(w_unique)
    n_w, n_x, n_y = len(w_unique_jnp), len(x_vals), len(y_grid)

    y_target, z_target = _build_target_grid(x_vals, w_unique_jnp, y_grid)
    logcdf_init_pos, logpdf_init_pos = predict_copula_cregression(fit_pos, y_target, z_target)

    key = PRNGKey(seed)
    key, *subkeys_cov = split(key, B_post + 1)
    subkeys_cov = jnp.array(subkeys_cov)

    prop_scores = None
    ind_new_pr = None
    w_map = _compute_w_map(x, w, w_unique, weighting)

    if x_update == "bb":
        def _bb_resample(subkey):
            k1, k2 = split(subkey)
            bb_w = dirichlet(k1, jnp.ones(n))
            return choice(k2, a=jnp.arange(n), p=bb_w, shape=(T_fwd,))
        ind_new_pr = vmap(_bb_resample)(subkeys_cov)          # (B_post, T_fwd)
        sampled_x = Z_jnp[ind_new_pr, 0]
        sampled_w = Z_jnp[ind_new_pr, 1:]
    else:  # x_update == "logistic"
        W_aug = sm.add_constant(w, has_constant='add')
        logit_fit_x = sm.Logit(x, W_aug).fit(disp=False)
        beta_init_x = jnp.array(np.asarray(logit_fit_x.params))
        w_jnp = jnp.array(w)
        W_aug_jnp = jnp.array(np.asarray(W_aug))
        x_orig_jnp = jnp.array(np.asarray(x).astype(float))

        sampled_x, sampled_w, beta_final_x = _logistic_nat_grad_sequence_B(
            subkeys_cov, w_jnp, W_aug_jnp, x_orig_jnp, beta_init_x, float(n), T_fwd
        )
        prop_scores = np.array(jax.nn.sigmoid(W_aug_jnp @ beta_final_x.T).T)

    # zero-atom model: recursively updated along the SAME (x, w) sequence via natural gradient
    Z_new_aug = jnp.concatenate(
        (jnp.ones((B_post, T_fwd, 1)), sampled_x[:, :, None], sampled_w), axis=-1
    )
    key, *subkeys_zero = split(key, B_post + 1)
    subkeys_zero = jnp.array(subkeys_zero)
    is_zero_new, beta_zero_final = _logistic_nat_grad_step_presampled_B(
        subkeys_zero, Z_new_aug, Z_aug_jnp, is_zero_orig_jnp, beta_zero_init, float(n), T_fwd
    )  # (B_post, T_fwd)

    z_samp = jnp.concatenate((sampled_x[:, :, None], sampled_w), axis=-1)  # (B_post, T_fwd, 1+p)

    key, *subkeys_fwd = split(key, B_post + 1)
    subkeys_fwd = jnp.array(subkeys_fwd)

    if diagnostic:
        w_emp_counts = np.bincount(w_pop_inv, minlength=n_w).astype(float)
        w_emp_weights = jnp.array(w_emp_counts / w_emp_counts.sum())
        flat_idx = np.arange(n_x * n_w * n_y).reshape(n_x, n_w, n_y)
        marginal_idx = jnp.array(flat_idx.transpose(0, 2, 1))
        pdf_init_flat = jnp.exp(logpdf_init_pos[:, -1])
        p_n_marginal_pos = jnp.sum(pdf_init_flat[marginal_idx] * w_emp_weights[None, None, :], axis=-1)

        w_idx, w_msk = _zi_w_idx_mask(x_update, weighting, w_unique_jnp, w_map=w_map,
                                      ind_new_pr=ind_new_pr, Z_jnp=Z_jnp,
                                      sampled_x=sampled_x, sampled_w=sampled_w)

        def _single(subkey, z_samp_b, is_zero_b, w_idx_b, w_msk_b):
            return _zi_forward_step_loop(
                subkey, z_samp_b, is_zero_b, fit_pos.rho_opt, fit_pos.rho_x_opt, z_target,
                n_pos, T_fwd, logcdf_init_pos, logpdf_init_pos,
                w_idx_b, w_msk_b, w_emp_counts, n_w, p_n_marginal_pos, marginal_idx,
                diagnostic=True,
            )
        logpdf_pr, final_counts_all, l1_trajectory = vmap(_single)(
            subkeys_fwd, z_samp, is_zero_new, w_idx, w_msk
        )
        final_weights = final_counts_all / final_counts_all.sum(axis=1, keepdims=True)
    else:
        def _single(subkey, z_samp_b, is_zero_b):
            return _zi_forward_step_loop(
                subkey, z_samp_b, is_zero_b, fit_pos.rho_opt, fit_pos.rho_x_opt, z_target,
                n_pos, T_fwd, logcdf_init_pos, logpdf_init_pos, diagnostic=False,
            )
        logpdf_pr, _, _ = vmap(_single)(subkeys_fwd, z_samp, is_zero_new)

        mask = (sampled_x == 1) if weighting == "att" else None
        final_weights = _covariate_weights(sampled_w, w_unique_jnp, mask)

    pdfs_pos = jnp.exp(jnp.squeeze(logpdf_pr)).reshape(B_post, n_x, n_w, n_y)
    marginal_pdfs_pos = jnp.einsum('bw,bxwy->bxy', final_weights, pdfs_pos)
    results = _summarize(marginal_pdfs_pos)
    # raw per-draw continuous density (integrates to 1 over y, i.e. NOT yet scaled by (1 - p0)),
    # kept alongside the summarized version so callers can combine it with the raw per-draw
    # p0_marginal below (e.g. to compute a posterior of E[Y(x)] or the ATT) without having to
    # redo the marginalisation over w.
    results['marginal_pdfs_pos'] = np.array(marginal_pdfs_pos)

    # zero-atom posterior P(Y=y0 | X=x_val), marginalised over w with the same weights used
    # for the continuous part
    x_rep = jnp.repeat(jnp.array(x_vals, dtype=w_unique_jnp.dtype), n_w)
    w_rep = jnp.tile(w_unique_jnp, (n_x, 1))
    z_xw_aug = jnp.concatenate((jnp.ones((n_x * n_w, 1)), x_rep[:, None], w_rep), axis=1)
    p0_grid = jax.nn.sigmoid(z_xw_aug @ beta_zero_final.T).T.reshape(B_post, n_x, n_w)
    p0_marginal = jnp.einsum('bw,bxw->bx', final_weights, p0_grid)
    p0_results = {}
    for i in range(n_x):
        p0_i = p0_marginal[:, i]
        p0_results[f'x_{i}'] = {
            'mean': float(jnp.mean(p0_i)),
            'low':  float(jnp.quantile(p0_i, 0.025)),
            'high': float(jnp.quantile(p0_i, 0.975)),
        }
    results['p0'] = p0_results
    results['p0_marginal'] = np.array(p0_marginal)  # (B_post, n_x), raw per-draw P(Y=y0|X=x_val)

    if diagnostic:
        l1_trajectory = jnp.concatenate([jnp.zeros((B_post, 1, n_x)), l1_trajectory], axis=1)
        results['l1_trajectory'] = np.array(l1_trajectory)

    if weighting == "att" and x_update == "bb":
        prop_scores = fit_propensity_scores(Z, np.asarray(ind_new_pr))
    if prop_scores is not None:
        results['propensity_scores'] = np.asarray(prop_scores)

    return results


def _validate_zi_args(x, x_vals, x_update, weighting, caller):
    if x_update not in ("bb", "logistic"):
        raise ValueError("x_update must be 'bb' or 'logistic'")
    if weighting not in ("ate", "att"):
        raise ValueError("weighting must be 'ate' or 'att'")
    x_levels = set(np.unique(x).tolist())
    if not x_levels.issubset({0, 1}):
        raise ValueError(f"{caller} requires binary x in {{0, 1}}; got levels {sorted(x_levels)}")
    if set(np.asarray(x_vals).tolist()) != {0, 1}:
        raise ValueError(f"{caller} requires x_vals=(0, 1)")
    if np.sum(x == 1) == 0 or np.sum(x == 0) == 0:
        raise ValueError("both treated (x==1) and control (x==0) observations are required")


# Zero-inflated counterpart of mp_causal_density (S-learner only): models Y as a point mass
# at y0 plus a continuous copula-regression part for Y != y0, so that a large point mass
# (e.g. Y=0 for unemployed participants in the Lalonde data) does not corrupt the continuous
# copula's bandwidth fit or forward-resampling stationarity.
#
# Returns the same "x_i" continuous-density entries as mp_causal_density (marginalised over
# the non-atom part only, i.e. these integrate to 1 over y, NOT (1 - p0)), plus a "p0" entry
# with the posterior of P(Y=y0 | X=x_val) for each x_val. Combining the atom and the
# continuous part into a single mixture density for plotting is left to the caller. Also
# returns raw (non-summarized, per posterior draw) "marginal_pdfs_pos" (B_post, n_x, n_y) and
# "p0_marginal" (B_post, n_x) arrays, so the caller can combine them into a posterior of the
# mixture mean E[Y(x)] = (1 - p0) * E[Y(x) | Y(x) != y0] + p0 * y0 (e.g. for an ATT posterior)
# without redoing the marginalisation over w.
def mp_causal_density_zi(y, x, w, y_grid, B_post, T_fwd, *,
                         x_vals=(0, 1), x_update="bb", weighting="ate", y0=0.0, seed=42):
    w = np.asarray(w)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    x = np.asarray(x)
    x_vals = np.asarray(x_vals) if not np.isscalar(x_vals) else np.array([x_vals])
    _validate_zi_args(x, x_vals, x_update, weighting, "mp_causal_density_zi")

    return _mp_density_zi_s_learner_impl(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                        x_update, weighting, y0, seed, diagnostic=False)


# Diagnostic twin of mp_causal_density_zi: additionally tracks an L1 convergence trajectory
# of the CONTINUOUS marginal density against its initial (t=0) fit, at every forward step.
# The atom probability P(Y=y0|x) is deliberately NOT included in this L1 metric -- the goal
# is to check whether excluding zeros from the continuous copula recursion resolves the
# divergence seen previously, before tackling a combined point-mass + continuous L1 metric.
def mp_causal_density_zi_diagnostic(y, x, w, y_grid, B_post, T_fwd, *,
                                    x_vals=(0, 1), x_update="bb", weighting="ate", y0=0.0, seed=42):
    w = np.asarray(w)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    x = np.asarray(x)
    x_vals = np.asarray(x_vals) if not np.isscalar(x_vals) else np.array([x_vals])
    _validate_zi_args(x, x_vals, x_update, weighting, "mp_causal_density_zi_diagnostic")

    return _mp_density_zi_s_learner_impl(y, x, w, x_vals, y_grid, B_post, T_fwd,
                                        x_update, weighting, y0, seed, diagnostic=True)
### ###
