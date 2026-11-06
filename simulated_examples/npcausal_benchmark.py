"""Benchmark: npcausal::cdensity (Kennedy, Balakrishnan and Wasserman) called from Python via rpy2.

cdensity is a projection/series estimator of the *marginal* counterfactual densities
p(Y(1)) and p(Y(0)) (random-forest nuisances + cross-fitting). It returns R closures g1(k, t)
and g0(k, t) evaluated on the standardised scale t = (y - ymin)/(ymax - ymin); the density
on the original scale is g(k, t)/(ymax - ymin) (exactly the package's own makeplot logic).
The number of cosine-basis terms k is chosen per arm by the package's own cross-validation
routine cv.cdensity (see cdensity_select_k below).

Requires R with the packages `npcausal` and `ranger`, and the Python package `rpy2`.
"""

import glob
import os

import numpy as np


def _configure_r():
    """Locate R so rpy2 can find it, if R is not on PATH.

    R's own package DLLs (e.g. stats.dll -> Rblas.dll) are loaded via LoadLibrary, which
    searches the process PATH, so on Windows bin\\x64 must be prepended to PATH --
    os.add_dll_directory alone is not enough. Set R_HOME yourself if R is installed elsewhere.
    """
    if "R_HOME" not in os.environ:
        r_cands = sorted(glob.glob(r"C:\Program Files\R\R-*"))
        if r_cands:
            os.environ["R_HOME"] = r_cands[-1]
    for r_bin in (os.path.join(os.environ.get("R_HOME", ""), "bin", "x64"),
                  os.path.join(os.environ.get("R_HOME", ""), "bin")):
        if os.path.isdir(r_bin):
            if r_bin not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = r_bin + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(r_bin)   # let the R shared libraries load on Windows
            except (OSError, AttributeError):
                pass


_configure_r()

import rpy2.robjects as ro  # noqa: E402  (must come after _configure_r)
from rpy2.robjects import numpy2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402

_r_conv = ro.default_converter + numpy2ri.converter

# R routine: fit cdensity, evaluate g1/g0 (at per-arm dimensions k1/k0) and the pointwise
# influence-function SE on the passed grid, returning an (n_grid x 4) matrix of columns
# [g1, g0, se1, se0] on the original outcome scale.  bfun/sefun mirror the package's makeplot.
_CDENSITY_R = """
function(y, a, x, ygrid, k1, k0, kmax, nsplits, seed) {
    set.seed(seed)
    fit  <- npcausal::cdensity(y, a, data.frame(x), makeplot = FALSE, l2 = FALSE,
                               progress_updates = FALSE, nsplits = nsplits, kmax = kmax)
    ymin <- min(y); ymax <- max(y); span <- ymax - ymin
    t    <- (ygrid - ymin) / span
    n    <- length(y)
    bfun <- function(t, kk) sqrt(2) * cos(kk * pi * t)
    b    <- Vectorize(bfun, vectorize.args = "kk")
    sefun <- function(A, k) {
        # drop=FALSE / explicit matrix() keep the k=1 case (cov of a single column) valid,
        # which CV may select for a simple arm.
        Ak <- as.matrix(A[, 1:k, drop = FALSE])
        Bm <- matrix(b(t, 1:k), nrow = length(t))
        sqrt(diag(Bm %*% cov(Ak) %*% t(Bm)) / n) / span
    }
    cbind(fit$g1(k1, t) / span, fit$g0(k0, t) / span,
          sefun(fit$ifvals$a1, k1), sefun(fit$ifvals$a0, k0))
}
"""
_cdensity_r = ro.r(_CDENSITY_R)

# R routine: run the package's own cross-validation (cv.cdensity) and return the selected
# basis dimension for each arm, c(k1, k0).  cv.cdensity only *plots* a pseudo-L2 risk curve
# per arm (Y^1 then Y^0) and returns nothing, so we shadow the graphics primitives in a
# private environment to capture the plotted risk vectors and take their argmin.
_CVCDENSITY_R = """
function(y, a, x, kmax, gridlen, nsplits, seed) {
    set.seed(seed)
    risk <- new.env(); risk$store <- list()
    e <- new.env(parent = environment(npcausal::cv.cdensity))
    e$plot     <- function(x, y, ...) { risk$store[[length(risk$store) + 1]] <- as.numeric(y); invisible() }
    e$points   <- function(...) invisible()
    e$lines    <- function(...) invisible()
    e$segments <- function(...) invisible()
    e$par      <- function(...) invisible()
    f <- npcausal::cv.cdensity
    environment(f) <- e
    invisible(f(y, a, data.frame(x), kmax = kmax, gridlen = gridlen,
                nsplits = nsplits, progress_updates = FALSE))
    c(which.min(risk$store[[1]]), which.min(risk$store[[2]]))   # (k1 for Y^1, k0 for Y^0)
}
"""
_cvcdensity_r = ro.r(_CVCDENSITY_R)


def _coerce(y, a, x, y_grid=None):
    y = np.asarray(y, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y_grid is None:
        return y, a, x
    return y, a, x, np.asarray(y_grid, dtype=float).ravel()


def cdensity_select_k(y, a, x, kmax=8, gridlen=20, nsplits=2, seed=42):
    """Cross-validated basis dimension for cdensity, per arm, via npcausal::cv.cdensity.

    Returns ``(k1, k0)`` -- the dimensions minimising the package's pseudo-L2 risk for the
    Y(1) and Y(0) arms respectively (each in 1..kmax).
    """
    y, a, x = _coerce(y, a, x)
    with localconverter(_r_conv):
        kk = np.asarray(_cvcdensity_r(y, a, x, int(kmax), int(gridlen), int(nsplits), int(seed)))
    return int(kk[0]), int(kk[1])


def cdensity_counterfactual(y, a, x, y_grid, k1=5, k0=5, kmax=None, nsplits=2, seed=42):
    """npcausal::cdensity estimates of the marginal counterfactual densities on ``y_grid``.

    ``a`` is the binary treatment and ``x`` the covariate matrix.  ``k1``/``k0`` are the
    per-arm basis dimensions for Y(1)/Y(0) (e.g. from :func:`cdensity_select_k`).  Returns a
    dict shaped like our other estimators, ``{'x_0': {...}, 'x_1': {...}}`` with 'mean' and a
    pointwise 95% band ('low'/'high') from cdensity's influence-function standard errors.
    """
    y, a, x, y_grid = _coerce(y, a, x, y_grid)
    if kmax is None:
        kmax = max(int(k1), int(k0))
    with localconverter(_r_conv):
        M = np.asarray(_cdensity_r(y, a, x, y_grid,
                                   int(k1), int(k0), int(kmax), int(nsplits), int(seed)))
    g1v, g0v, se1, se0 = M[:, 0], M[:, 1], M[:, 2], M[:, 3]

    def _entry(m, se):
        return {"mean": m, "low": np.clip(m - 1.96 * se, 0.0, None), "high": m + 1.96 * se}

    return {"x_0": _entry(g0v, se0), "x_1": _entry(g1v, se1)}
