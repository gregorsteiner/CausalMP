"""Benchmark: SPQR-DS of Xu et al. (2022), arXiv:2211.01591, called from Python via rpy2.

SPQR-DS estimates the marginal counterfactual densities p(Y(1)) and p(Y(0)) in three steps
(Algorithm 1 of the paper):
  1. propensity score pi(X) by BART probit (R package BayesTree), keeping N_pi posterior draws;
  2. for each PS draw j, the conditional density f(y | T, S) with the double balancing score
     S = (pi^(j)(X), X) by SPQR (R package SPQR): a mixture of K M-spline densities on [0, 1]
     whose weights come from a one-hidden-layer tanh network in (T, S), fitted by NUTS under
     the Gaussian scale mixture (GSM) prior with a = b = 0.01 for all precision hyper-priors;
  3. f_t(y) = sum_i u_i f(y | t, S_i^(j)) with u ~ Dirichlet(1, ..., 1) (Bayesian bootstrap),
     one u per posterior draw of the network weights, shared by both arms.
The posterior draws are pooled over all PS draws.

Deviations from the paper, to keep a repeated-data study feasible:
  - K (n_knots) and V1 (n_hidden) are fixed, not selected by WAIC over K in {8,10,12},
    V1 in {5,8,10};
  - shorter chains by default (1500 iterations, 500 warmup, thin 10 -> 100 draws per PS draw;
    the paper uses 3000 / 1000 / 10);
  - the outcome is mapped to [0, 1] by min-max scaling of a range padded by ``pad`` times the
    sample range on each side (the paper uses plain min-max), so the density is not forced to
    zero at the sample extremes. Outside the padded range the density is zero.
Covariates are min-max scaled to [0, 1], as in the paper.

Requires R with the packages `BayesTree` and `SPQR` (archived from CRAN; install with
``remotes::install_github("stevengxu/SPQR")``, which needs Rtools on Windows), and `rpy2`.
"""

import numpy as np

# importing npcausal_benchmark locates R (sets R_HOME/PATH) before rpy2 starts
from simulated_examples.npcausal_benchmark import _coerce, _r_conv
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402

# R routine: steps 1-3 above, entirely in R so the (n x grid x draws) conditional-density
# array never crosses into Python. Returns an (n_grid x n_draws x 2) array of the marginal
# densities f_0, f_1 on the original outcome scale, one column per pooled posterior draw.
_SPQR_DS_R = """
function(y, a, x, ygrid, n_knots, n_hidden, n_ps, iter, warmup, thin, pad, hyp, seed) {
    set.seed(seed)
    n <- length(y)
    keep <- 100L
    # BART probit: nskip burn-in, then every keep-th of n_ps * keep draws (paper: 500/100/5)
    invisible(capture.output(
        ps <- BayesTree::bart(x.train = x, y.train = a, x.test = matrix(0, 0, ncol(x)),
                              ndpost = n_ps * keep, nskip = 500L, keepevery = keep,
                              verbose = FALSE)))
    pis <- pnorm(matrix(ps$yhat.train, ncol = n))       # n_ps x n

    xs <- apply(x, 2, function(c) (c - min(c)) / (max(c) - min(c)))
    lo <- min(y) - pad * diff(range(y)); hi <- max(y) + pad * diff(range(y)); span <- hi - lo
    ys <- (y - lo) / span
    tg <- (ygrid - lo) / span
    inside <- which(tg >= 0 & tg <= 1)

    nsave <- floor((iter - warmup) / thin)
    dens <- array(0, c(length(ygrid), n_ps * nsave, 2))
    for (j in seq_len(n_ps)) {
        pij <- pis[j, ]
        fit <- SPQR::SPQR(X = cbind(a, pij, xs), Y = ys, n.knots = n_knots, n.hidden = n_hidden,
                          activation = "tanh", method = "MCMC", prior = "GSM", hyperpar = hyp,
                          control = list(iter = iter, warmup = warmup, thin = thin),
                          normalize = FALSE, verbose = FALSE, seed = seed + j)
        # Bayesian-bootstrap weights: one Dirichlet(1,...,1) vector per draw (n x nsave)
        U <- matrix(rexp(n * nsave), n)
        U <- sweep(U, 2, colSums(U), "/")
        cols <- (j - 1) * nsave + seq_len(nsave)
        for (t in 0:1) {
            P <- predict(fit, X = cbind(t, pij, xs), Y = tg[inside], type = "PDF", getAll = TRUE)
            P <- array(P, c(n, length(inside), nsave))   # guard against dropped dimensions
            for (l in seq_len(nsave))
                dens[inside, cols[l], t + 1] <- crossprod(U[, l], P[, , l]) / span
        }
    }
    dens
}
"""
_spqr_ds_r = ro.r(_SPQR_DS_R)
# SPQR imports torch, whose .onLoad asks interactively to download libtorch when it is
# missing (the rpy2 session counts as interactive). The MCMC path does not use libtorch,
# so switch the prompt off; otherwise an unattended run blocks on it.
ro.r('Sys.setenv(TORCH_INSTALL = "0")')


def spqr_ds_draws(y, a, x, y_grid, n_knots=10, n_hidden=8, n_ps=5, iter=1500, warmup=500,
                  thin=10, pad=0.1, hyper=0.01, seed=42):
    """Posterior draws of the SPQR-DS marginal counterfactual densities on ``y_grid``.

    ``a`` is the binary treatment, ``x`` the covariate matrix. Returns an array of shape
    ``(n_grid, n_ps * (iter - warmup) // thin, 2)``, with the last axis (f_0, f_1).
    """
    y, a, x, y_grid = _coerce(y, a, x, y_grid)
    hyp = ro.ListVector({k: float(hyper) for k in ("a_sigma", "b_sigma", "a_lambda", "b_lambda")})
    with localconverter(_r_conv):
        D = _spqr_ds_r(y, a, x, y_grid, int(n_knots), int(n_hidden), int(n_ps), int(iter),
                       int(warmup), int(thin), float(pad), hyp, int(seed))
    return np.asarray(D)


def spqr_ds_counterfactual(y, a, x, y_grid, **kwargs):
    """SPQR-DS estimates of the marginal counterfactual densities on ``y_grid``.

    Returns ``{'x_0': {...}, 'x_1': {...}}`` with the posterior mean ('mean') and a pointwise
    95% credible band ('low'/'high'), the same shape as the other estimators. Keyword
    arguments are passed to :func:`spqr_ds_draws`.
    """
    D = spqr_ds_draws(y, a, x, y_grid, **kwargs)

    def _entry(d):
        low, high = np.quantile(d, [0.025, 0.975], axis=1)
        return {"mean": d.mean(axis=1), "low": low, "high": high}

    return {"x_0": _entry(D[:, :, 0]), "x_1": _entry(D[:, :, 1])}
