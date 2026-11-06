"""The LaLonde job-training data: counterfactual earnings distributions among the treated.

Produces Figures/LalondeExample.pdf and prints posterior summaries of the ATT.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from real_examples.plot_style import set_style
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from causal_mp import mp_causal_density_zi_diagnostic

BINS = np.linspace(-2, 1.5, 26)
BIN_CENTRES = 0.5 * (BINS[:-1] + BINS[1:])
WIDTH = BINS[1] - BINS[0]


def plot_lalonde_panels(res, panel_axes, y_grid_orig, sig_y, x,
                        bins=BINS, bin_centres=BIN_CENTRES, width=WIDTH):
    ax_dens, ax_att, ax_l1, ax_ps = panel_axes

    # --- Counterfactual densities, with the point mass P(Y=0|x) on its own twin axis ---
    # (the point mass lives on a probability scale, not a density scale, and is typically much
    # larger in magnitude than the continuous density -- plotting both on the same axis makes
    # the continuous curve unreadable)
    ax_dens2 = ax_dens.twinx()
    ax_dens2.spines['right'].set_visible(True)  # the plot style disables it by default
    ax_dens2.spines['top'].set_visible(False)
    for xi, color, offset in [(0, "tab:blue", -400), (1, "tab:orange", 400)]:
        p0 = res['p0'][f'x_{xi}']
        scale = (1 - p0['mean']) / sig_y
        mean_d = res[f'x_{xi}']['mean'] * scale
        low_d  = res[f'x_{xi}']['low'] * scale
        high_d = res[f'x_{xi}']['high'] * scale

        ax_dens.plot(y_grid_orig, mean_d, color=color)
        ax_dens.fill_between(y_grid_orig, low_d, high_d, color=color, alpha=0.2)

        ax_dens2.errorbar([offset], [p0['mean']], yerr=[[p0['mean'] - p0['low']], [p0['high'] - p0['mean']]],
                          fmt='D', color=color, capsize=3, markersize=6,
                          markeredgecolor='black', markeredgewidth=0.5)
    ax_dens2.set_ylim(0, 1)

    ax_dens.set_xlabel("Real earnings in 1978")
    ax_dens.set_xticks([0, 20000])
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax_dens.yaxis.set_major_formatter(fmt)
    ax_dens.set_ylabel("Density")

    # --- ATT posterior ---
    # E[Y(x)] is a convex combination of the continuous part and the zero part: the zero part
    # contributes exactly 0 (since y0 = 0 here), but its probability p0 still matters by
    # shrinking the continuous contribution: E[Y(x)] = (1 - p0) * E[Y(x) | Y(x) != 0].
    pdfs_pos_orig = res['marginal_pdfs_pos'] / sig_y  # (B, n_x, n_y), raw per-draw, density in original units
    p0_marginal = res['p0_marginal']                  # (B, n_x), raw per-draw P(Y(x)=0)
    E_cont = np.trapezoid(y_grid_orig[None, None, :] * pdfs_pos_orig, y_grid_orig, axis=-1)  # (B, n_x)
    E_y = (1 - p0_marginal) * E_cont  # (B, n_x)
    att_samples = E_y[:, 1] - E_y[:, 0]

    kde = gaussian_kde(att_samples)
    att_grid = np.linspace(att_samples.min() - 200, att_samples.max() + 200, 300)
    ax_att.plot(att_grid, kde(att_grid), color='tab:blue')
    ax_att.set_xlabel("ATT")
    fmt2 = ticker.ScalarFormatter(useMathText=True)
    fmt2.set_powerlimits((0, 0))
    ax_att.yaxis.set_major_formatter(fmt2)
    ax_att.set_ylabel("Density")

    # --- L1 convergence (continuous part only) ---
    l1 = res['l1_trajectory']
    mean_l1 = l1.mean(axis=0)
    t_axis = np.arange(mean_l1.shape[0])
    ax_l1.plot(t_axis, mean_l1[:, 0], color='tab:blue')
    ax_l1.plot(t_axis, mean_l1[:, 1], color='tab:orange')
    ax_l1.set_ylim(bottom=0)
    ax_l1.set_xlabel("Forward step")
    ax_l1.set_ylabel(r"Mean $L_1$ distance")

    # --- Propensity score histogram ---
    def _hist_density(lo_samples, x_flag, B_post):
        out = np.zeros((B_post, len(bins) - 1))
        for b in range(B_post):
            vals = lo_samples[b, x == x_flag]
            counts, _ = np.histogram(vals, bins=bins, density=True)
            out[b] = counts
        return out

    ps = res['propensity_scores']
    log_odds = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / np.clip(1 - ps, 1e-6, 1 - 1e-6))
    B_post = ps.shape[0]

    hist_trt  = _hist_density(log_odds, 1, B_post)
    hist_ctrl = _hist_density(log_odds, 0, B_post)

    med_trt = np.mean(hist_trt,  axis=0)
    lo_trt  = np.quantile(hist_trt,  0.025, axis=0)
    hi_trt  = np.quantile(hist_trt,  0.975, axis=0)

    med_ctrl = np.mean(hist_ctrl, axis=0)
    lo_ctrl  = np.quantile(hist_ctrl, 0.025, axis=0)
    hi_ctrl  = np.quantile(hist_ctrl, 0.975, axis=0)

    ax_ps.axhline(0, color='black', linewidth=0.9)
    ax_ps.bar(bin_centres,  med_trt,  width=width * 0.95, color='tab:orange', alpha=0.7)
    # clip to guard against tiny negative round-off, which errorbar rejects outright
    ax_ps.errorbar(bin_centres, med_trt,
                   yerr=[np.maximum(med_trt - lo_trt, 0), np.maximum(hi_trt - med_trt, 0)],
                   fmt='none', color='tab:orange', capsize=3, linewidth=1.2, alpha=0.9)
    ax_ps.bar(bin_centres, -med_ctrl, width=width * 0.95, color='tab:blue', alpha=0.7)
    ax_ps.errorbar(bin_centres, -med_ctrl,
                   yerr=[np.maximum(hi_ctrl - med_ctrl, 0), np.maximum(med_ctrl - lo_ctrl, 0)],
                   fmt='none', color='tab:blue', capsize=3, linewidth=1.0, alpha=0.9)
    ax_ps.set_xlabel("Log-odds ratio")
    ax_ps.set_ylabel("Density")

    return att_samples, ax_dens2


def main():
    # load data
    d = pd.read_csv("Data/lalonde.csv")

    y = d.re78.values
    x = d.treat.values
    w = d[['age', 'educ', 'hisp', 'black', 'married', 'nodegr', 're74', 're75', 'u74', 'u75']].values

    print("Number of treated and control observations:", x.sum(), d.shape[0] - x.sum())

    # compute standardisation parameters and standardise y
    mu_y = y.mean()
    sig_y = y.std()
    y_std = (y - mu_y) / sig_y
    y0_std = (0 - mu_y) / sig_y  # standardised location of the point mass at re78 = 0

    w_std = (w - w.mean(axis=0)) / w.std(axis=0)

    # run analysis on standardised grid
    T = 5000
    B = 200

    y_grid_orig = np.linspace(0, 25000, 50)
    y_grid_std = (y_grid_orig - mu_y) / sig_y

    # Counterfactual outcomes among treated Y(0)|X = 1 and Y(1)|X = 1. re78 has a large point
    # mass at zero (unemployed participants), which corrupts the continuous copula recursion if
    # fed through it directly, so we use the zero-inflated mixture model: P(Y=0|x,w) via a
    # recursively updated logistic regression, plus a copula-regression model for Y|Y!=0,x,w
    # fit on the non-zero subsample.
    res_bb = mp_causal_density_zi_diagnostic(y_std, x, w_std, y_grid_std, B, T, weighting="att", y0=y0_std)

    # logistic regression as comparison
    res_log = mp_causal_density_zi_diagnostic(y_std, x, w_std, y_grid_std, B, T, weighting="att",
                                              x_update="logistic", y0=y0_std)

    # figure
    set_style()
    fig, axes = plt.subplots(2, 4, figsize=(9, 4))

    att_bb, ax2_bb = plot_lalonde_panels(res_bb, [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
                                         y_grid_orig, sig_y, x)
    ax2_bb.set_ylabel(r"$P(Y(x)=0)$")

    att_log, ax2_log = plot_lalonde_panels(res_log, [axes[0, 2], axes[0, 3], axes[1, 2], axes[1, 3]],
                                           y_grid_orig, sig_y, x)
    ax2_log.set_ylabel(r"$P(Y(x)=0)$")

    fig.text(0.25, 1.0, "Bayesian bootstrap", ha='center', va='bottom', fontsize=13)
    fig.text(0.75, 1.0, "Logistic regression", ha='center', va='bottom', fontsize=13)

    fig.legend(handles=[Patch(color='tab:blue', label='Control'),
                        Patch(color='tab:orange', label='Treated')],
               loc='lower center', ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.05))

    for label, att in [("Bayesian Bootstrap", att_bb), ("Logistic", att_log)]:
        print(f"{label}: ATT posterior mean: {att.mean():.0f}")
        print(f"{label}: ATT 95% CI: [{np.quantile(att, 0.025):.0f}, {np.quantile(att, 0.975):.0f}]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('Figures/LalondeExample.pdf', dpi=300, bbox_inches='tight')
    print("Saved Figures/LalondeExample.pdf")


if __name__ == "__main__":
    main()
