"""The effect of zinc supplementation on the duration of the common cold.

Produces Figures/ZincExample.pdf and prints posterior summaries of the quantile treatment effects.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from real_examples.plot_style import set_style
import matplotlib.pyplot as plt

from causal_mp import mp_causal_density_diagnostic


def quantile_treatment_effects(cdfs, y_grid, quantiles):
    """cdfs: (B, 2, n_y) raw per-draw CDFs for x=0/1 on the same y_grid.
    Returns (B, len(quantiles)) posterior draws of Q_1(delta) - Q_0(delta)."""
    B, n_x, _ = cdfs.shape
    q = np.empty((B, n_x, len(quantiles)))
    for b in range(B):
        for i in range(n_x):
            q[b, i] = np.interp(quantiles, cdfs[b, i], y_grid)
    return q[:, 1, :] - q[:, 0, :]


def main():
    # load data
    d = pd.read_csv("Data/ZnAcet.csv")

    y = d.Duration.values
    x = d.Zinc.values
    w = np.ones(len(y))

    # compute standardisation parameters and standardise y
    mu_y = y.mean()
    sig_y = y.std()
    y_std = (y - mu_y) / sig_y

    # run analysis, grid on standardised scale
    T = 5000
    B = 1000

    y_grid_orig = np.linspace(0, 18, 50)
    y_grid_std = (y_grid_orig - mu_y) / sig_y

    # single learner approach
    res = mp_causal_density_diagnostic(y_std, x, w, y_grid_std, B, T)

    # twin learner approach
    res_t = mp_causal_density_diagnostic(y_std, x, w, y_grid_std, B, T, learner="t")

    # quantile treatment effects
    deltas = np.array([0.1, 0.5, 0.9])
    qte_s = quantile_treatment_effects(res['marginal_cdfs'], y_grid_std, deltas) * sig_y     # (B, 3)
    qte_t = quantile_treatment_effects(res_t['marginal_cdfs'], y_grid_std, deltas) * sig_y   # (B, 3)

    for label, qte in [("S-learner", qte_s), ("T-learner", qte_t)]:
        for k, delta in enumerate(deltas):
            print(f"{label} QTE(delta={delta}): mean={qte[:, k].mean():.2f}, "
                  f"95% CI=[{np.quantile(qte[:, k], 0.025):.2f}, {np.quantile(qte[:, k], 0.975):.2f}]")

    # figure
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(7, 4), sharey='row')

    # top row: counterfactual densities for each learner
    for ax, r, title in [(axes[0, 0], res, "S-Learner"), (axes[0, 1], res_t, "T-Learner")]:
        ax.plot(y_grid_orig, r['x_0']['mean'] / sig_y, label="Y(0)", color='tab:blue')
        ax.fill_between(y_grid_orig, r['x_0']['low'] / sig_y, r['x_0']['high'] / sig_y, color='tab:blue', alpha=0.2)
        ax.plot(y_grid_orig, r['x_1']['mean'] / sig_y, label="Y(1)", color='tab:orange')
        ax.fill_between(y_grid_orig, r['x_1']['low'] / sig_y, r['x_1']['high'] / sig_y, color='tab:orange', alpha=0.2)
        ax.set_xlabel("Duration (days)")
        ax.set_title(title)
        ax.legend(frameon=False)
    axes[0, 0].set_ylabel("Density")

    # bottom row: quantile treatment effect posteriors
    delta_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for ax, qte in [(axes[1, 0], qte_s), (axes[1, 1], qte_t)]:
        for k, delta in enumerate(deltas):
            kde = gaussian_kde(qte[:, k])
            grid = np.linspace(qte[:, k].min(), qte[:, k].max(), 200)
            ax.plot(grid, kde(grid), color=delta_colors[k], label=fr"$\delta={delta}$")
        ax.set_xlabel(r"$\delta$-quantile treatment effect")
        ax.legend(frameon=False, fontsize=9)
    axes[1, 0].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig('Figures/ZincExample.pdf', dpi=300, bbox_inches='tight')
    print("Saved Figures/ZincExample.pdf")


if __name__ == "__main__":
    main()
