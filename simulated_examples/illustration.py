"""Simulation 1 (J = 2), single realisation: every method of the simulation study versus the truth.

Based on Simulation 1 with strong confounding (J = 2) from Xu et al. (2022, arXiv:2211.01591).
Each subject has 5 covariates W_j ~ U(-2, 2), of which the first two are confounders that drive
the propensity score X | W ~ Bern(expit(2 (W_1 + W_2))). The control outcome Y(0) is
right-skewed and shared across two scenarios that differ only in the treated outcome Y(1):
  - Scenario 1: Y(1) is unimodal and skewed to the opposite side of the control.
  - Scenario 2: Y(1) is a bimodal mixture of normals.

The dataset is replicate REPLICATE of simulated_examples/simulation_study.py (n = 100, same
seed), so this figure can be rebuilt without rerunning the Monte Carlo study. All settings
(posterior draws, forward horizon, SPQR-DS and L2 Proj settings) are taken from that script.
The top row shows the marginal counterfactual densities of MP-SL, MP-TL, L2 Proj and SPQR-DS;
the bottom row shows the L1 stability trajectories of the forward resampling (MP methods only).

Produces Figures/Simulation_Illustration.pdf. Requires R with npcausal, ranger, BayesTree and
SPQR, and rpy2.
"""

import contextlib
import io
import os

import numpy as np
import pandas as pd

from simulated_examples.simulation_study import (B_POST, T_FWD, SPQR_SETTINGS, MP_METHODS, OUTCOMES,
                                                 CDENSITY_NAME, SPQR_NAME, METHODS, COLORS, OLD_NAMES,
                                                 RAW_CSV, dataset_seed, fit_cdensity)
from simulated_examples.plot_style import set_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from causal_mp import mp_causal_density_diagnostic
from simulated_examples.dgp import generate_data_sim1, true_y0_density_sim1, true_y1_density_sim1
from simulated_examples.spqr_benchmark import spqr_ds_counterfactual

N = 100
# Replicate of the simulation study to show. Chosen from Figures/Simulation_Study_raw.csv:
# MP-SL has its lowest summed L1 error of all 50 replicates here and beats L2 Proj and SPQR-DS
# in all four (scenario, outcome) panels.
REPLICATE = 8
SCENARIOS = (1, 2)
FIGURE = "Figures/Simulation_Illustration.pdf"

ALPHA = 0.2
DASH = (0, (5, 3))


def report_study_l1():
    """Print this replicate's L1 errors as recorded by the simulation study."""
    if not os.path.exists(RAW_CSV):
        return
    raw = pd.read_csv(RAW_CSV)
    raw = raw.assign(method=raw["method"].replace(OLD_NAMES))
    r = raw[(raw["n"] == N) & (raw["replicate"] == REPLICATE)]
    print(f"L1 errors of replicate {REPLICATE} in the simulation study:")
    print(r.pivot_table(index="method", columns=["scenario", "outcome"], values="l1")
          .reindex(METHODS).round(3).to_string())


def fit_all():
    results = {}
    for sc in SCENARIOS:
        # same draw as the study: both scenarios share covariates, treatment and Y(0)
        np.random.seed(dataset_seed(N, REPLICATE))
        y, x, w = generate_data_sim1(N, scenario=sc, J=2)
        grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 50)

        fits = {}
        for m, kw in MP_METHODS.items():
            with contextlib.redirect_stdout(io.StringIO()):   # silence per-fit prints
                fits[m] = mp_causal_density_diagnostic(y, x, w, grid, B_POST, T_FWD,
                                                       seed=REPLICATE, **kw)
        fits[CDENSITY_NAME], (k1, k0) = fit_cdensity(y, x, w, grid, seed=REPLICATE)
        print(f"scenario {sc}: cv-selected L2 Proj dimensions -> Y(1) k={k1}, Y(0) k={k0}")
        fits[SPQR_NAME] = spqr_ds_counterfactual(y, x, w, grid, seed=REPLICATE, **SPQR_SETTINGS)

        results[sc] = {
            "grid": grid,
            "fits": fits,
            "true": {"x_0": true_y0_density_sim1(grid),
                     "x_1": true_y1_density_sim1(grid, scenario=sc)},
        }
    return results


def plot(results):
    set_style()
    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.05)

    for sf, sc in zip(subfigs, SCENARIOS):
        sf.suptitle(f"Scenario {sc}", fontsize=13)
        # taller density row, shorter L1 row
        axs = sf.subplots(2, 2, sharex="row", sharey="row", gridspec_kw={"height_ratios": [2.2, 1]})
        R = results[sc]
        grid = R["grid"]

        # top row: marginal densities with pointwise 95% bands
        for c, (xkey, lab) in enumerate(OUTCOMES):
            ax = axs[0, c]
            for m in METHODS:
                d = R["fits"][m][xkey]
                ax.fill_between(grid, d["low"], d["high"], color=COLORS[m], alpha=ALPHA)
                ax.plot(grid, d["mean"], color=COLORS[m], linewidth=2)
            ax.plot(grid, R["true"][xkey], color="black", linewidth=1.6, linestyle=DASH)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(lab, fontsize=11)
        axs[0, 0].set_ylabel("Density (with 95% CI)", fontsize=11)

        # bottom row: mean L1 stability trajectories of the forward resampling
        for c, (xkey, lab) in enumerate(OUTCOMES):
            ax = axs[1, c]
            xi = int(xkey.split("_")[1])
            for m in MP_METHODS:
                l1 = R["fits"][m]["l1_trajectory"][:, :, xi]   # (B, T+1)
                ax.plot(np.arange(l1.shape[1]), l1.mean(axis=0), color=COLORS[m], linewidth=2)
            ax.set_ylim(0, 2.5)
            ax.set_xlabel("Forward step", fontsize=11)
        axs[1, 0].set_ylabel(r"Mean $L_1$ distance", fontsize=11)

    handles = [Line2D([0], [0], color=COLORS[m], linewidth=2, label=m) for m in METHODS]
    handles.append(Line2D([0], [0], color="black", linewidth=1.6, linestyle=DASH, label="True density"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=10)

    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    print(f"Saved {FIGURE}")


if __name__ == "__main__":
    report_study_l1()
    plot(fit_all())
