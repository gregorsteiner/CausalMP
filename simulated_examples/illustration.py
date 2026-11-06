"""Simulation 1 (J = 2), single realisation: our estimators versus npcausal::cdensity and the truth.

Based on Simulation 1 with strong confounding (J = 2) from Xu et al. (2022, arXiv:2211.01591).
Each subject has 5 covariates W_j ~ U(-2, 2), of which the first two are confounders that drive
the propensity score X | W ~ Bern(expit(2 (W_1 + W_2))). The control outcome Y(0) is
right-skewed and shared across two scenarios that differ only in the treated outcome Y(1):
  - Scenario 1: Y(1) is unimodal and skewed to the opposite side of the control.
  - Scenario 2: Y(1) is a bimodal mixture of normals.

On one dataset (n = 100) we compare S-Learner + Bayesian bootstrap, S-Learner + logistic and
T-Learner + Bayesian bootstrap against the projection estimator cdensity of Kennedy,
Balakrishnan and Wasserman (R package npcausal). The top row shows the marginal counterfactual
densities; the bottom row shows the L1 stability trajectories of the forward resampling.

Produces Figures/Simulation_Illustration.pdf. Requires R with npcausal and ranger, and rpy2.
"""

import numpy as np

from simulated_examples.plot_style import set_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from causal_mp import mp_causal_density_diagnostic
from simulated_examples.dgp import generate_data_sim1, true_y0_density_sim1, true_y1_density_sim1
from simulated_examples.npcausal_benchmark import cdensity_select_k, cdensity_counterfactual

N = 100
SEED = 42
T_FWD, B_POST = 2000, 100
CDENSITY_KMAX = 8   # candidate basis dimensions 1..KMAX for cv.cdensity to choose from

ALPHA = 0.2
DASH = (0, (5, 3))
CDENSITY_COLOR = "#9467bd"   # npcausal::cdensity benchmark


def fit_all():
    results = {}
    for sc in (1, 2):
        # reset the seed so both scenarios share covariates, treatment and Y(0) (only Y(1) differs)
        np.random.seed(SEED)
        y_s, x_s, w_s = generate_data_sim1(N, scenario=sc, J=2)
        grid = np.linspace(y_s.min() - 0.5, y_s.max() + 0.5, 50)

        methods = [
            ("S-Learner + BB",       mp_causal_density_diagnostic(y_s, x_s, w_s, grid, B_POST, T_FWD),                      "#1f77b4"),
            ("S-Learner + Logistic", mp_causal_density_diagnostic(y_s, x_s, w_s, grid, B_POST, T_FWD, x_update="logistic"), "#ff7f0e"),
            ("T-Learner + BB",       mp_causal_density_diagnostic(y_s, x_s, w_s, grid, B_POST, T_FWD, learner="t"),         "#2ca02c"),
        ]

        # npcausal::cdensity benchmark (marginal densities only; no L1 analogue).  The per-arm
        # basis dimension is chosen by the package's own CV (cv.cdensity) for a fair comparison.
        k1, k0 = cdensity_select_k(y_s, x_s, w_s, kmax=CDENSITY_KMAX, seed=SEED)
        cdensity_res = cdensity_counterfactual(y_s, x_s, w_s, grid, k1=k1, k0=k0, kmax=CDENSITY_KMAX, seed=SEED)
        print(f"scenario {sc}: cv-selected L2 Proj dimensions -> Y(1) k={k1}, Y(0) k={k0}")

        results[sc] = {
            "grid": grid,
            "methods": methods,
            "cdensity": cdensity_res,
            "cdensity_k": (k1, k0),
            "true": {
                "marg": {"x_1": true_y1_density_sim1(grid, scenario=sc),
                         "x_0": true_y0_density_sim1(grid)},
            },
        }
    return results


def plot(results):
    set_style()
    cols = [("x_0", "Y(0)"), ("x_1", "Y(1)")]

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.05)

    l1_axes = []

    for sf, sc in zip(subfigs, (1, 2)):
        sf.suptitle(f"Scenario {sc}", fontsize=13)
        # taller density row, shorter L1 row
        axs = sf.subplots(2, 2, sharex='row', sharey='row', gridspec_kw={"height_ratios": [2.2, 1]})

        R = results[sc]
        grid = R["grid"]

        # top row: marginal densities with credible bands
        for c, (xkey, lab) in enumerate(cols):
            ax = axs[0, c]
            for name, res, color in R["methods"]:
                d = res[xkey]
                ax.fill_between(grid, d['low'], d['high'], color=color, alpha=ALPHA)
                ax.plot(grid, d['mean'], color=color, linewidth=2)
            # npcausal::cdensity benchmark (mean + pointwise 95% band)
            cd = R["cdensity"][xkey]
            ax.fill_between(grid, cd['low'], cd['high'], color=CDENSITY_COLOR, alpha=ALPHA)
            ax.plot(grid, cd['mean'], color=CDENSITY_COLOR, linewidth=2)
            ax.plot(grid, R["true"]["marg"][xkey], color="red", linewidth=1.6, linestyle=DASH)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(lab, fontsize=11)
        axs[0, 0].set_ylabel("Density (with 95% CI)", fontsize=11)

        # bottom row: mean L1 stability trajectories
        for c, (xkey, lab) in enumerate(cols):
            ax = axs[1, c]
            xi = int(xkey.split("_")[1])
            for name, res, color in R["methods"]:
                l1 = res["l1_trajectory"][:, :, xi]  # (B, T+1)
                steps = np.arange(l1.shape[1])
                ax.plot(steps, l1.mean(axis=0), color=color, linewidth=2)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Forward step", fontsize=11)
        axs[1, 0].set_ylabel(r"Mean $L_1$ distance", fontsize=11)
        l1_axes.extend([axs[1, 0], axs[1, 1]])

    # common y-limits across all L1 plots
    for ax in l1_axes:
        ax.set_ylim(0, 2.5)

    # shared legend below the whole figure
    legend_handles = [Line2D([0], [0], color=color, linewidth=2, label=name)
                      for name, _, color in results[1]["methods"]]
    legend_handles.append(Line2D([0], [0], color=CDENSITY_COLOR, linewidth=2, label="L2 Proj"))
    legend_handles.append(Line2D([0], [0], color="red", linewidth=1.6, linestyle=DASH, label="True density"))
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5, frameon=False, fontsize=10)

    fig.savefig('Figures/Simulation_Illustration.pdf', dpi=300, bbox_inches='tight')
    print("Saved Figures/Simulation_Illustration.pdf")


if __name__ == "__main__":
    plot(fit_all())
