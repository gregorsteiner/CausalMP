"""The effect of zinc supplementation on the duration of the common cold.

Produces Figures/ZincExample.pdf: counterfactual densities (top row) and the distribution of the
individual treatment effect Y(1) - Y(0) under a Gaussian copula with several correlations
(bottom row). Prints posterior summaries of the individual treatment effect distribution.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from real_examples.plot_style import set_style
import matplotlib.pyplot as plt

from causal_mp import mp_causal_density_diagnostic


def ite_draws(cdfs, y_grid, rho, z):
    """Individual treatment effect draws under a Gaussian copula with correlation rho.

    cdfs: (B, 2, n_y) per-draw CDFs of Y(0), Y(1) on y_grid.
    z: (B, M, 2) independent standard normals (reused across rho for common random numbers).
    Returns (B, M) draws of Y(1) - Y(0), one set of M joint draws per posterior sample.
    """
    B = cdfs.shape[0]
    u0 = norm.cdf(z[..., 0])
    u1 = norm.cdf(rho * z[..., 0] + np.sqrt(1 - rho**2) * z[..., 1])
    cdfs = np.maximum.accumulate(cdfs, axis=-1)
    ite = np.empty(u0.shape)
    for b in range(B):
        y0 = np.interp(u0[b], cdfs[b, 0], y_grid)
        y1 = np.interp(u1[b], cdfs[b, 1], y_grid)
        ite[b] = y1 - y0
    return ite


def summarise(draws):
    return draws.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)


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

    # distribution of individual treatment effects under a Gaussian copula
    rhos = [0.5, 0.75, 0.9]
    M = 2000
    z = np.random.default_rng(1).standard_normal((B, M, 2))
    bins = np.arange(-15, 10.25, 0.5)
    bin_mid = 0.5 * (bins[:-1] + bins[1:])

    ite_dens = {}  # (learner, rho) -> (B, n_bins) per-draw densities
    for label, r in [("S-learner", res), ("T-learner", res_t)]:
        for rho in rhos:
            ite = ite_draws(r['marginal_cdfs'], y_grid_orig, rho, z)
            ite_dens[label, rho] = np.stack([np.histogram(e, bins=bins, density=True)[0] for e in ite])
            for name, stat in [("mean ITE", ite.mean(axis=1)),
                               ("P(ITE < 0)", (ite < 0).mean(axis=1)),
                               ("SD ITE", ite.std(axis=1))]:
                m, lo, hi = summarise(stat)
                print(f"{label} rho={rho} {name}: mean={m:.2f}, 95% CI=[{lo:.2f}, {hi:.2f}]")

    # figure
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(7, 4), sharey='row')

    # top row: counterfactual densities for each learner
    for ax, r, title in [(axes[0, 0], res, "S-Learner"), (axes[0, 1], res_t, "T-Learner")]:
        ax.plot(y_grid_orig, r['x_0']['mean'] / sig_y, label="Control", color='tab:blue')
        ax.fill_between(y_grid_orig, r['x_0']['low'] / sig_y, r['x_0']['high'] / sig_y, color='tab:blue', alpha=0.2)
        ax.plot(y_grid_orig, r['x_1']['mean'] / sig_y, label="Treated", color='tab:orange')
        ax.fill_between(y_grid_orig, r['x_1']['low'] / sig_y, r['x_1']['high'] / sig_y, color='tab:orange', alpha=0.2)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xlabel("Duration (days)")
        ax.set_title(title)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_ylabel("Density")

    # bottom row: individual treatment effect distributions for each copula correlation
    rho_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for ax, label in [(axes[1, 0], "S-learner"), (axes[1, 1], "T-learner")]:
        for k, rho in enumerate(rhos):
            dens = ite_dens[label, rho]
            ax.plot(bin_mid, dens.mean(axis=0), color=rho_colors[k], label=fr"$\rho={rho}$")
            ax.fill_between(bin_mid, np.quantile(dens, 0.025, axis=0), np.quantile(dens, 0.975, axis=0),
                            color=rho_colors[k], alpha=0.15)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlim(-10, 5)
        ax.set_xticks([-10, -5, 0, 5])
        ax.set_xlabel("Individual treatment effect (days)")
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig('Figures/ZincExample.pdf', dpi=300, bbox_inches='tight')
    print("Saved Figures/ZincExample.pdf")


if __name__ == "__main__":
    main()
