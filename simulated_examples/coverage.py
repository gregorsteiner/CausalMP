"""Simulation 1 (J = 2), repeated datasets: pointwise coverage of the 95% credible intervals.

For each scenario and replicate we draw a fresh dataset, fit all three estimators on a coarse
fixed grid, and record (i) whether the true density lies in the pointwise 95% credible interval
and (ii) the interval length, at every grid point.

Produces Figures/Simulation_Results.pdf and Figures/Simulation_coverage_results.csv.
This is by far the slowest script (n_datasets x 2 scenarios x 3 estimators fits). Use
--n-datasets for a quick test run, e.g. `python -m simulated_examples.coverage --n-datasets 5`.
"""

import argparse
import contextlib
import io

import numpy as np
import pandas as pd

from simulated_examples.plot_style import set_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from causal_mp import mp_causal_density
from simulated_examples.dgp import generate_data_sim1, true_y0_density_sim1, true_y1_density_sim1

N = 100                                     # sample size per dataset
Y_GRID_EXP = np.linspace(-3.5, 3.5, 20)     # coarse, fixed evaluation grid
T_FWD_EXP, B_POST_EXP = 500, 100
NOMINAL = 0.95
SCENARIOS = (1, 2)

VARIANTS = {
    "S-Learner + BB":       dict(),
    "S-Learner + Logistic": dict(x_update="logistic"),
    "T-Learner + BB":       dict(learner="t"),
}
COLORS = {
    "S-Learner + BB":       "#1f77b4",
    "S-Learner + Logistic": "#ff7f0e",
    "T-Learner + BB":       "#2ca02c",
}


def run_experiment(n_datasets):
    n_grid = len(Y_GRID_EXP)
    coverage = {sc: {m: {} for m in VARIANTS} for sc in SCENARIOS}
    mean_len = {sc: {m: {} for m in VARIANTS} for sc in SCENARIOS}

    for sc in SCENARIOS:
        # true densities on the evaluation grid (Y(0) shared; Y(1) scenario-specific)
        true_exp = {"x_1": true_y1_density_sim1(Y_GRID_EXP, scenario=sc),
                    "x_0": true_y0_density_sim1(Y_GRID_EXP)}

        covered = {m: {k: np.zeros(n_grid) for k in ("x_0", "x_1")} for m in VARIANTS}
        length  = {m: {k: np.zeros(n_grid) for k in ("x_0", "x_1")} for m in VARIANTS}

        np.random.seed(0)   # same replicate datasets across scenarios (only Y(1) differs)
        for s in range(n_datasets):
            y_d, x_d, w_d = generate_data_sim1(N, scenario=sc, J=2)
            for m, kw in VARIANTS.items():
                with contextlib.redirect_stdout(io.StringIO()):   # silence per-fit prints
                    res = mp_causal_density(y_d, x_d, w_d, Y_GRID_EXP, B_POST_EXP, T_FWD_EXP, seed=s, **kw)
                for k in ("x_0", "x_1"):
                    d = res[k]
                    covered[m][k] += (true_exp[k] >= d["low"]) & (true_exp[k] <= d["high"])
                    length[m][k]  += d["high"] - d["low"]
            if (s + 1) % 10 == 0 or s + 1 == n_datasets:
                print(f"scenario {sc}: completed {s + 1}/{n_datasets} datasets", flush=True)

        for m in VARIANTS:
            for k in ("x_0", "x_1"):
                coverage[sc][m][k] = covered[m][k] / n_datasets
                mean_len[sc][m][k] = length[m][k] / n_datasets

    return coverage, mean_len


def save_table(coverage, mean_len):
    rows = []
    for sc in SCENARIOS:
        for m in VARIANTS:
            for k, lab in [("x_1", "Y(1)"), ("x_0", "Y(0)")]:
                for gi, g in enumerate(Y_GRID_EXP):
                    rows.append({"scenario": sc, "method": m, "outcome": lab, "y": round(float(g), 2),
                                 "coverage": coverage[sc][m][k][gi],
                                 "mean_length": mean_len[sc][m][k][gi]})
    coverage_df = pd.DataFrame(rows)
    coverage_df.to_csv("Figures/Simulation_coverage_results.csv", index=False)
    print("Saved Figures/Simulation_coverage_results.csv")

    # overall average across grid points, per scenario / method / outcome
    summary = (coverage_df.groupby(["scenario", "method", "outcome"])[["coverage", "mean_length"]]
               .mean().round(3))
    print("\nAverage over grid points:")
    print(summary)


def plot(coverage):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True, constrained_layout=True)

    for r, sc in enumerate(SCENARIOS):
        for c, (k, lab) in enumerate([("x_1", r"Y(1)"), ("x_0", r"Y(0)")]):
            ax = axes[r, c]
            for m in VARIANTS:
                ax.plot(Y_GRID_EXP, coverage[sc][m][k], marker="o", ms=4, color=COLORS[m])
            ax.axhline(NOMINAL, color="grey", linestyle=(0, (5, 3)), linewidth=1)
            ax.set_ylim(0, 1.05)
            if r == len(SCENARIOS) - 1:
                ax.set_xlabel(lab, fontsize=12)
        axes[r, 0].set_ylabel(f"Scenario {sc}\nCoverage", fontsize=11)

    legend_handles = [Line2D([0], [0], color=c, linewidth=2, marker="o", ms=4, label=m)
                      for m, c in COLORS.items()]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4, frameon=False, fontsize=10)

    plt.savefig('Figures/Simulation_Results.pdf', dpi=300, bbox_inches='tight')
    print("Saved Figures/Simulation_Results.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-datasets", type=int, default=200,
                        help="number of replicated datasets per scenario (paper: 200)")
    args = parser.parse_args()

    coverage, mean_len = run_experiment(args.n_datasets)
    save_table(coverage, mean_len)
    plot(coverage)


if __name__ == "__main__":
    main()
