"""Simulation 1 (J = 2), repeated datasets: L1 error and pointwise coverage.

Monte Carlo study on the DGP of simulated_examples/illustration.py (Xu et al., 2022): two
scenarios that share Y(0) and differ in Y(1), at sample sizes n in NS. For each replicate we
draw a fresh dataset and fit
  - S-Learner + Bayesian bootstrap,
  - T-Learner + Bayesian bootstrap
    (both on unstandardised y: the outcomes are already near unit scale, and a pilot run
    with standardise=True gave mostly larger L1 errors and wider bands),
  - L2 Proj: npcausal::cdensity of Kennedy, Balakrishnan and Wasserman, with the basis
    dimension chosen per arm and per dataset by cv.cdensity,
  - SPQR-DS: the double-balancing-score method of Xu et al. (2022) (BART-probit propensity
    score from BayesTree + SPQR conditional density by NUTS + Bayesian bootstrap); see
    simulated_examples/spqr_benchmark.py for the settings and the deviations from the paper,
and record, for each counterfactual marginal density p(y(x)),
  (i)  the L1 distance int |p_hat - p| between the truth and the posterior mean (our methods)
       or the point estimate (L2 Proj), by the trapezoid rule on L1_GRID;
  (ii) whether the pointwise 95% interval for the density covers the true density value at
       the true quantiles q_0.1, q_0.5, q_0.9 of Y(x), and the interval width there;
  (iii) the wall-clock runtime of each fit (time_s).

Produces Figures/Simulation_Study.pdf, Figures/Simulation_Study_summary.csv,
Figures/Simulation_Study_table.tex and the per-replicate Figures/Simulation_Study_raw.csv.
Requires R with npcausal, ranger, BayesTree and SPQR, and rpy2. Slow: use --n-datasets (and
--spqr-iter/--spqr-warmup) for a quick test run, e.g.
`python -m simulated_examples.simulation_study --n-datasets 2 --spqr-iter 300 --spqr-warmup 200`,
or --skip-spqr to leave SPQR-DS out.
"""

import argparse
import contextlib
import io
import os
import time


def limit_cpu(n_threads):
    """Keep this (long) run from taking over the machine: pin the process to ``n_threads``
    cores and lower its priority. Affinity covers every thread of the process, i.e. the JAX/XLA
    pool and the in-process R session (rpy2). Thread-count variables are set too, so BLAS/OpenMP
    and ranger do not oversubscribe the pinned cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "R_RANGER_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity(list(range(n_threads)))
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as e:   # psutil missing or affinity unsupported: run unrestricted
        print(f"could not limit CPU usage ({e})")


# applied at import, before JAX / R start their thread pools; override with SIM_THREADS
N_THREADS = int(os.environ.get("SIM_THREADS", max(1, (os.cpu_count() or 2) // 2)))
limit_cpu(N_THREADS)

import numpy as np
import pandas as pd

from simulated_examples.plot_style import set_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from causal_mp import mp_causal_density
from simulated_examples.dgp import (generate_data_sim1, true_y0_density_sim1, true_y1_density_sim1,
                                    true_quantiles_sim1)
from simulated_examples.npcausal_benchmark import cdensity_select_k, cdensity_counterfactual
from simulated_examples.spqr_benchmark import spqr_ds_counterfactual

NS = (100,)
SCENARIOS = (1, 2)
QUANTILES = (0.1, 0.5, 0.9)
N_DATASETS = 50
B_POST = 100
T_FWD = 1000
CDENSITY_KMAX = 8
# SPQR-DS: fixed K (n_knots) and V1 (n_hidden) instead of the paper's WAIC selection, and
# shorter chains than the paper (250 warmup, 500 post-warmup iterations thinned by 5 -> 100
# kept draws per PS draw, 5 PS draws)
SPQR_SETTINGS = dict(n_knots=10, n_hidden=8, n_ps=5, iter=750, warmup=250, thin=5)
# L1 grid (plus the six true-quantile points, added per scenario). Cost is linear in the grid
# size; spacing 0.3 keeps the trapezoid L1 within ~2% of a 451-point grid and holds
# > 0.9999 of the true mass in every case.
L1_GRID = np.linspace(-4.5, 4.5, 31)
OUTCOMES = (("x_0", "Y(0)"), ("x_1", "Y(1)"))

MP_METHODS = {
    "S-Learner + BB": dict(),
    "T-Learner + BB": dict(learner="t"),
}
CDENSITY_NAME = "L2 Proj"
SPQR_NAME = "SPQR-DS"
METHODS = (*MP_METHODS, CDENSITY_NAME, SPQR_NAME)
COLORS = {"S-Learner + BB": "#1f77b4", "T-Learner + BB": "#2ca02c", CDENSITY_NAME: "#9467bd",
          SPQR_NAME: "#d62728"}

RAW_CSV = "Figures/Simulation_Study_raw.csv"
SUMMARY_CSV = "Figures/Simulation_Study_summary.csv"
TABLE_TEX = "Figures/Simulation_Study_table.tex"
FIGURE = "Figures/Simulation_Study.pdf"


def truth(scenario):
    """Evaluation grid (L1_GRID plus the true quantile points) and true quantities on it."""
    q = {"x_0": true_quantiles_sim1(QUANTILES, 0),
         "x_1": true_quantiles_sim1(QUANTILES, 1, scenario)}
    grid = np.unique(np.concatenate([L1_GRID, q["x_0"], q["x_1"]]))
    dens = {"x_0": true_y0_density_sim1(grid), "x_1": true_y1_density_sim1(grid, scenario=scenario)}
    return {
        "grid": grid,
        "dens": dens,
        "l1_idx": np.searchsorted(grid, L1_GRID),
        "q_idx": {k: np.searchsorted(grid, q[k]) for k in q},
    }


def fit_cdensity(y, x, w, grid, seed):
    k1, k0 = cdensity_select_k(y, x, w, kmax=CDENSITY_KMAX, seed=seed)
    res = cdensity_counterfactual(y, x, w, grid, k1=k1, k0=k0, kmax=CDENSITY_KMAX, seed=seed)
    # the cosine-series estimate lives on [min y, max y] (outside it the basis just repeats
    # periodically), so the estimate and its band are set to zero outside the observed range
    outside = (grid < y.min()) | (grid > y.max())
    for k in res:
        for key in ("mean", "low", "high"):
            res[k][key] = np.where(outside, 0.0, res[k][key])
    return res, (k1, k0)


def evaluate(res, tr):
    """Rows of metrics (L1, coverage and width at each quantile) for one fit."""
    grid, l1_idx = tr["grid"], tr["l1_idx"]
    rows = []
    for k, lab in OUTCOMES:
        d, p = res[k], tr["dens"][k]
        row = {"outcome": lab,
               "l1": float(np.trapezoid(np.abs(d["mean"][l1_idx] - p[l1_idx]), grid[l1_idx]))}
        for tau, gi in zip(QUANTILES, tr["q_idx"][k]):
            row[f"cover_q{tau}"] = float(d["low"][gi] <= p[gi] <= d["high"][gi])
            row[f"width_q{tau}"] = float(d["high"][gi] - d["low"][gi])
        rows.append(row)
    return rows


def dataset_seed(n, s):
    return 100_000 * n + s


def run_experiment(ns, scenarios, n_datasets, t_fwd, spqr=SPQR_SETTINGS):
    """``spqr``: SPQR-DS settings (see SPQR_SETTINGS), or None to leave SPQR-DS out."""
    rows = []
    for n in ns:
        for sc in scenarios:
            tr = truth(sc)
            start = time.time()
            for s in range(n_datasets):
                # seed every replicate right before drawing it: the copula fit resets NumPy's
                # global RNG (np.random.seed(20) in fit_copula_cregression), so a single seed
                # before the loop would make every replicate after the first identical. The
                # per-replicate seed also gives both scenarios the same covariates, treatment
                # and Y(0) (only Y(1) differs).
                np.random.seed(dataset_seed(n, s))
                y, x, w = generate_data_sim1(n, scenario=sc, J=2)
                base = {"n": n, "scenario": sc, "replicate": s}
                # time_s: wall-clock seconds of the whole fit (both arms; for L2 Proj including
                # the cross-validation), repeated on the fit's Y(0) and Y(1) rows
                for m, kw in MP_METHODS.items():
                    t0 = time.perf_counter()
                    with contextlib.redirect_stdout(io.StringIO()):   # silence per-fit prints
                        res = mp_causal_density(y, x, w, tr["grid"], B_POST, t_fwd, seed=s, **kw)
                    el = time.perf_counter() - t0
                    rows += [{**base, "method": m, "time_s": el, **r} for r in evaluate(res, tr)]
                try:
                    t0 = time.perf_counter()
                    res, (k1, k0) = fit_cdensity(y, x, w, tr["grid"], seed=s)
                    el = time.perf_counter() - t0
                    rows += [{**base, "method": CDENSITY_NAME, "time_s": el, "k": kk, **r}
                             for kk, r in zip((k0, k1), evaluate(res, tr))]
                except Exception as e:   # record the failure rather than losing the whole run
                    print(f"  cdensity failed (n={n}, scenario={sc}, replicate={s}): {e}", flush=True)
                    rows += [{**base, "method": CDENSITY_NAME, "outcome": lab} for _, lab in OUTCOMES]
                if spqr is not None:
                    try:
                        t0 = time.perf_counter()
                        res = spqr_ds_counterfactual(y, x, w, tr["grid"], seed=s, **spqr)
                        el = time.perf_counter() - t0
                        rows += [{**base, "method": SPQR_NAME, "time_s": el, **r}
                                 for r in evaluate(res, tr)]
                    except Exception as e:
                        print(f"  SPQR-DS failed (n={n}, scenario={sc}, replicate={s}): {e}", flush=True)
                        rows += [{**base, "method": SPQR_NAME, "outcome": lab} for _, lab in OUTCOMES]
                if (s + 1) % 10 == 0 or s + 1 == n_datasets:
                    print(f"n={n}, scenario {sc}: {s + 1}/{n_datasets} datasets "
                          f"({time.time() - start:.0f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(RAW_CSV, index=False)   # checkpoint
    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_CSV, index=False)
    print(f"Saved {RAW_CSV}")
    return raw


def summarise(raw):
    keys = ["scenario", "n", "outcome", "method"]
    metric_cols = ["l1"] + [f"{p}_q{tau}" for p in ("cover", "width") for tau in QUANTILES]
    if "time_s" in raw:   # older raw CSVs have no runtimes
        metric_cols.append("time_s")
    g = raw.groupby(keys, sort=False)
    summary = g[metric_cols].mean()
    summary.insert(1, "l1_se", g["l1"].std() / np.sqrt(g["l1"].count()))
    summary.insert(0, "n_fits", g["l1"].count())
    summary = summary.reset_index()
    # only the methods present (e.g. a run with --skip-spqr, or an older raw CSV)
    summary["method"] = pd.Categorical(summary["method"],
                                       [m for m in METHODS if m in set(summary["method"])])
    summary = summary.sort_values(keys).reset_index(drop=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {SUMMARY_CSV}")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))

    # compact LaTeX table: L1 (SE) and coverage at the three quantiles
    tab = summary[keys].copy()
    tab["$L_1$ (SE)"] = [f"{a:.3f} ({b:.3f})" for a, b in zip(summary["l1"], summary["l1_se"])]
    for tau in QUANTILES:
        tab[f"Cov. $q_{{{tau}}}$"] = summary[f"cover_q{tau}"].map(lambda v: f"{v:.2f}")
    if "time_s" in summary:
        tab["Time (s)"] = summary["time_s"].map(lambda v: f"{v:.1f}")
    tab.to_latex(TABLE_TEX, index=False, escape=False)
    print(f"Saved {TABLE_TEX}")
    return summary


def plot(summary):
    set_style()
    ns = sorted(summary["n"].unique())
    scenarios = sorted(summary["scenario"].unique())
    fig, axes = plt.subplots(2, len(scenarios) * 2, figsize=(3 * len(scenarios) * 2, 5),
                             sharex=True, sharey="row", constrained_layout=True, squeeze=False)
    methods = list(summary["method"].cat.categories)
    offsets = dict(zip(methods, np.linspace(-0.15, 0.15, len(methods))))
    xpos = np.arange(len(ns))

    for j, (sc, (_, lab)) in enumerate([(sc, o) for sc in scenarios for o in OUTCOMES]):
        for m in methods:
            d = summary[(summary["scenario"] == sc) & (summary["outcome"] == lab)
                        & (summary["method"] == m)].set_index("n").reindex(ns)
            axes[0, j].errorbar(xpos + offsets[m], d["l1"], yerr=2 * d["l1_se"], fmt="o",
                                ms=4, capsize=2, color=COLORS[m])
            for tau, mk in zip(QUANTILES, ("v", "o", "^")):
                axes[1, j].plot(xpos + offsets[m], d[f"cover_q{tau}"], mk, ms=4, color=COLORS[m],
                                linestyle="none")
        axes[0, j].set_title(f"Scenario {sc}, {lab}", fontsize=11)
        axes[1, j].axhline(0.95, color="grey", linestyle=(0, (5, 3)), linewidth=1)
        axes[1, j].set_ylim(0, 1.05)
        axes[1, j].set_xticks(xpos, [f"n={n}" for n in ns])
        axes[0, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel(r"Mean $L_1$ error ($\pm$ 2 SE)")
    axes[1, 0].set_ylabel("Coverage at quantiles")

    handles = [Line2D([0], [0], color=COLORS[m], marker="o", linestyle="none", label=m) for m in methods]
    handles += [Line2D([0], [0], color="grey", marker=mk, linestyle="none", label=rf"$q_{{{tau}}}$")
                for tau, mk in zip(QUANTILES, ("v", "o", "^"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=10)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    print(f"Saved {FIGURE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS,
                        help=f"replicated datasets per (n, scenario) (default {N_DATASETS})")
    parser.add_argument("--n", type=int, nargs="+", default=list(NS), help="sample sizes")
    parser.add_argument("--scenario", type=int, nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--t-fwd", type=int, default=T_FWD, help="forward-sampling horizon")
    parser.add_argument("--spqr-iter", type=int, default=SPQR_SETTINGS["iter"],
                        help="SPQR-DS NUTS iterations per PS draw, including warmup")
    parser.add_argument("--spqr-warmup", type=int, default=SPQR_SETTINGS["warmup"],
                        help="SPQR-DS NUTS warmup iterations per PS draw")
    parser.add_argument("--skip-spqr", action="store_true", help="leave SPQR-DS out")
    parser.add_argument("--summarise-only", action="store_true",
                        help=f"skip the fits and rebuild the summary and figure from {RAW_CSV}")
    args = parser.parse_args()

    spqr = None if args.skip_spqr else {**SPQR_SETTINGS, "iter": args.spqr_iter,
                                         "warmup": args.spqr_warmup}
    raw = pd.read_csv(RAW_CSV) if args.summarise_only else \
        run_experiment(args.n, args.scenario, args.n_datasets, args.t_fwd, spqr)
    plot(summarise(raw))


if __name__ == "__main__":
    main()
