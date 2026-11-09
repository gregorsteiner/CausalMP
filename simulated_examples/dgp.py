"""Data-generating processes and true densities for the simulated examples."""

import numpy as np
from scipy.stats import norm


# ── Simulation 1 (J = 2), based on Xu et al. (2022), arXiv:2211.01591 ────────────────────
# Two scenarios share the (right-skewed) control outcome Y(0) but differ in Y(1):
#   scenario 1 (simple): Y(1) is unimodal and skewed to the opposite side of the control,
#                        so the two potential-outcome distributions are broadly similar;
#   scenario 2 (paper) : Y(1) is a bimodal mixture of normals.

def _expit(a):
    return 1.0 / (1.0 + np.exp(-a))


def _Z(w, k):
    # Z_k = expit(0.8 * sum_j w_j + 0.1 * sum_j |w_j|^k), per covariate row
    w = np.asarray(w, dtype=float)
    axis = w.ndim - 1
    return _expit(0.8 * w.sum(axis=axis) + 0.1 * (np.abs(w) ** k).sum(axis=axis))


def _eps_pdf(e):
    # right-skewed control error: eps ~ 0.75 N+(0, 0.9^2) + 0.25 N-(0, 0.3^2)
    pos = 0.75 * 2.0 * norm.pdf(e, 0.0, 0.9) * (e >= 0)
    neg = 0.25 * 2.0 * norm.pdf(e, 0.0, 0.3) * (e < 0)
    return pos + neg


def _sample_eps0(n):
    # right-skewed (control) half-normal mixture
    pos = np.random.uniform(size=n) < 0.75
    return np.where(pos, np.abs(np.random.normal(0.0, 0.9, n)), -np.abs(np.random.normal(0.0, 0.3, n)))


def _sample_eps1(n):
    # mirror of eps0 -> left-skewed (scenario-1 treated error)
    pos = np.random.uniform(size=n) < 0.75
    return np.where(pos, -np.abs(np.random.normal(0.0, 0.9, n)), np.abs(np.random.normal(0.0, 0.3, n)))


def generate_data_sim1(n, scenario=2, J=2):
    # draws from the global NumPy RNG -- set np.random.seed(...) once to control it
    X = np.random.uniform(-2, 2, size=(n, 5))
    ps = _expit((4.0 / J) * X[:, :J].sum(axis=1))
    Tr = (np.random.uniform(size=n) < ps).astype(int)
    Z1, Z2 = _Z(X, 1), _Z(X, 2)

    # Y(0): -2.3 + Z1 + Z1^2 + eps0  (right-skewed; shared by both scenarios)
    y0 = -2.3 + Z1 + Z1 ** 2 + _sample_eps0(n)

    if scenario == 1:
        # Y(1): unimodal, skewed to the opposite side, mild location shift -> similar to Y(0)
        y1 = -1.3 + Z2 + Z2 ** 2 + _sample_eps1(n)
    else:
        # Y(1): bimodal mixture of normals with means depending on Z2 (paper)
        comp1 = np.random.uniform(size=n) < 0.7
        y1 = np.where(comp1, np.random.normal(-2.5 + 5 * Z2, 0.35), np.random.normal(2.5 - 5 * Z2, 0.35))

    Y_obs = np.where(Tr == 1, y1, y0)
    return Y_obs, Tr, X


# True *conditional* interventional densities p(y | X=t, W=w).  w may be a single covariate
# row (-> (n_y,)) or a batch of rows (-> (n_rows, n_y)).  Y(0) is shared across scenarios.
def true_y0_cond_density_sim1(y_grid, w):
    m = np.atleast_1d(-2.3 + _Z(w, 1) + _Z(w, 1) ** 2)
    dens = _eps_pdf(np.asarray(y_grid)[None, :] - m[:, None])
    return dens[0] if np.ndim(_Z(w, 1)) == 0 else dens


def true_y1_cond_density_sim1(y_grid, w, scenario=2):
    y = np.asarray(y_grid)[None, :]
    if scenario == 1:
        m = np.atleast_1d(-1.3 + _Z(w, 2) + _Z(w, 2) ** 2)
        dens = _eps_pdf(m[:, None] - y)            # mirrored -> left-skewed
    else:
        z2 = np.atleast_1d(_Z(w, 2))
        dens = (0.7 * norm.pdf(y, (-2.5 + 5 * z2)[:, None], 0.35)
                + 0.3 * norm.pdf(y, (2.5 - 5 * z2)[:, None], 0.35))
    return dens[0] if np.ndim(_Z(w, 2)) == 0 else dens


# True marginal counterfactual densities: average the conditional over X ~ U(-2,2)^5.
# A large, fixed Monte Carlo sample from a private RNG keeps this deterministic ground
# truth reproducible and independent of the global RNG used for data generation.
def true_y0_density_sim1(y_grid, M=200000):
    X = np.random.default_rng(1).uniform(-2, 2, size=(M, 5))
    return true_y0_cond_density_sim1(y_grid, X).mean(axis=0)


def true_y1_density_sim1(y_grid, scenario=2, M=200000):
    X = np.random.default_rng(2).uniform(-2, 2, size=(M, 5))
    return true_y1_cond_density_sim1(y_grid, X, scenario).mean(axis=0)


def true_quantiles_sim1(probs, outcome, scenario=2):
    """True quantiles of Y(1) (outcome=1) or Y(0) (outcome=0), by inverting the numerical CDF
    of the true marginal density on a fine grid."""
    # 401 points keep the (M x grid) Monte Carlo average in memory; the CDF is smooth enough
    # that linear interpolation on this grid is accurate to ~1e-3
    grid = np.linspace(-6.0, 6.0, 401)
    M = 50000
    dens = true_y1_density_sim1(grid, scenario, M=M) if outcome == 1 else true_y0_density_sim1(grid, M=M)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(np.asarray(probs, dtype=float), cdf, grid)


# ── IV with continuous treatment ──────────────────────────────────────────────────────────
# Z, eta ~ N(0, 1);  X = pi Z + eta;  Y = f(X) + eps
# eps = c1 eta + c2 (eta^2 - 1) + nu  ->  nonlinear control function E[eps | eta]
# c1 drives the mean gap between p(y(x)) and p(y | X = x), c2 the skew; raising either
# shrinks s_nu, so c1^2 + 2 c2^2 < 1 is the ceiling.
IV_ALPHA = 0.0
IV_PI    = 1.0
IV_C1    = 0.7
IV_C2    = 0.35
IV_SNU2  = 1.0 - IV_C1 ** 2 - 2 * IV_C2 ** 2     # chosen so that Var(eps) = 1
IV_SNU   = np.sqrt(IV_SNU2)


def iv_f(x):
    return IV_ALPHA + 2.0 * np.tanh(1.5 * np.asarray(x, dtype=float))


def generate_iv_dgp(n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(n)
    eta = rng.standard_normal(n)
    nu  = rng.normal(0.0, IV_SNU, n)
    eps = IV_C1 * eta + IV_C2 * (eta ** 2 - 1.0) + nu
    X   = IV_PI * Z + eta
    return iv_f(X) + eps, X, Z


# Y | x, eta is Gaussian and both eta laws below are Gaussian, so each density is a
# one-dimensional integral over eta.
_ETA_QUAD = np.linspace(-8.0, 8.0, 4001)


def _iv_density(y_grid, x, eta_mean, eta_sd):
    w = norm.pdf(_ETA_QUAD, eta_mean, eta_sd)
    w = w / np.trapezoid(w, _ETA_QUAD)
    m = iv_f(x) + IV_C1 * _ETA_QUAD + IV_C2 * (_ETA_QUAD ** 2 - 1.0)
    k = norm.pdf(np.asarray(y_grid, dtype=float)[:, None], m[None, :], IV_SNU)
    return np.trapezoid(k * w[None, :], _ETA_QUAD, axis=1)


def true_iv_interventional(y_grid, x):
    """p(y(x)): eta ~ N(0, 1)."""
    return _iv_density(y_grid, x, 0.0, 1.0)


def true_iv_observational(y_grid, x):
    """p(y | X = x): eta | X = x ~ N(x / (pi^2 + 1), 1 - 1 / (pi^2 + 1))."""
    v = 1.0 - 1.0 / (IV_PI ** 2 + 1.0)
    return _iv_density(y_grid, x, x / (IV_PI ** 2 + 1.0), np.sqrt(v))
