"""IV with a continuous treatment: interventional versus observational density.

A nonlinear instance of the general IV model

    Y = f(X) + eps,   X = pi Z + eta,   Z, eta ~ N(0, 1),

with f(x) = alpha + 2 tanh(1.5 x) and eps = c1 eta + c2 (eta^2 - 1) + nu, nu ~ N(0, s_nu^2).
The control function E[eps | eta] is nonlinear, so p(y(x)) is right-skewed and differs from
p(y | X = x) in shape rather than only in location (see simulated_examples/dgp.py).

Produces Figures/IV_Simulation.pdf. Note: mp_density_iv uses a lot of memory (it grows as
B_post * len(y_grid) * n * T_fwd).
"""

import numpy as np

from simulated_examples.plot_style import set_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from causal_mp import mp_causal_density
from causal_mp_iv import mp_density_iv
from simulated_examples.dgp import generate_iv_dgp, true_iv_interventional, true_iv_observational

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
ALPHA  = 0.18
DASH   = (0, (5, 3))


def main():
    # simulate data and fit martingale posteriors
    y, x, z = generate_iv_dgp(n=500, seed=42)
    T_fwd, B_post = 2000, 100

    x_val = -1.0

    # standardise y for the copula fit; densities are mapped back to the original scale below
    mean_y, sd_y = np.mean(y), np.std(y)
    y_std = (y - mean_y) / sd_y

    y_grid = np.linspace(-5.0, 1.0, 50)
    y_grid_std = (y_grid - mean_y) / sd_y

    # IV control-function estimate of p(y(x))
    res_int = mp_density_iv(y_std, x, z, x_val, y_grid_std, B_post, T_fwd)['x_0']

    # observational comparison p(y | X = x); w is constant, so nothing is conditioned on
    res_obs = mp_causal_density(y_std, x, np.ones(len(y)), y_grid_std, B_post, T_fwd, x_vals=x_val)['x_0']

    # figure
    set_style()
    fig, ax = plt.subplots(figsize=(5, 2.5))

    # estimated densities + credible bands (undoing the y standardisation)
    ax.fill_between(y_grid, res_int['low'] / sd_y, res_int['high'] / sd_y, color=BLUE,   alpha=ALPHA)
    ax.fill_between(y_grid, res_obs['low'] / sd_y, res_obs['high'] / sd_y, color=ORANGE, alpha=ALPHA)

    ax.plot(y_grid, res_int['mean'] / sd_y, color=BLUE,   linewidth=2)
    ax.plot(y_grid, res_obs['mean'] / sd_y, color=ORANGE, linewidth=2)

    # true densities (dashed)
    ax.plot(y_grid, true_iv_interventional(y_grid, x_val), color=BLUE,   linewidth=1.4, linestyle=DASH)
    ax.plot(y_grid, true_iv_observational(y_grid, x_val),  color=ORANGE, linewidth=1.4, linestyle=DASH)

    legend_handles = [
        Line2D([0], [0], color=BLUE,   label=rf"$Y({x_val:g})$"),
        Line2D([0], [0], color=ORANGE, label=rf"$Y \mid X = {x_val:g}$"),
        Line2D([0], [0], color="grey", linewidth=1.4, linestyle=DASH, label="True density"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=10)

    ax.set_xlabel(r"$Y$",    fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(y_grid[0], y_grid[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    plt.savefig('Figures/IV_Simulation.pdf', dpi=300, bbox_inches='tight')
    print("Saved Figures/IV_Simulation.pdf")


if __name__ == "__main__":
    main()
