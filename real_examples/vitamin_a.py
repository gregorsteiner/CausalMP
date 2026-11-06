"""A vitamin A supplementation trial with non-compliance (Sommer and Zeger, 1991).

Produces Figures/Sommer_Zeger_Results.pdf and prints credible intervals for the complier
proportion and the LATE.
"""

import numpy as np
from scipy.stats import gaussian_kde

from real_examples.plot_style import set_style
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from causal_mp_iv import mp_compliers


def load_data():
    """Expand the Sommer & Zeger contingency table into unit-level arrays (Z, X, Y)."""
    data_counts = [
        (0, 0, 0, 74),      # Assigned: No, Received: No, Outcome: Died
        (0, 0, 1, 11514),   # Assigned: No, Received: No, Outcome: Survived
        (1, 0, 0, 34),      # Assigned: Yes, Received: No, Outcome: Died
        (1, 0, 1, 2385),    # Assigned: Yes, Received: No, Outcome: Survived
        (1, 1, 0, 12),      # Assigned: Yes, Received: Yes, Outcome: Died
        (1, 1, 1, 9663),    # Assigned: Yes, Received: Yes, Outcome: Survived
    ]
    Z_data, X_data, Y_data = [], [], []
    for z, x, y, count in data_counts:
        Z_data.extend([z] * count)
        X_data.extend([x] * count)
        Y_data.extend([y] * count)
    return np.array(Z_data), np.array(X_data), np.array(Y_data)


def main():
    Z_data, X_data, Y_data = load_data()

    print(f"Total observations: {len(Y_data)}")
    print(f"Outcome distribution: {np.mean(Y_data):.4f}")
    print(f"Treatment (X=1) rate: {np.mean(X_data):.4f}")
    print(f"Instrument (Z=1) rate: {np.mean(Z_data):.4f}")
    print("\nSummary by Instrument Assignment:")
    print(f"  Z=0: P(X=1|Z=0) = {np.mean(X_data[Z_data == 0]):.4f}")
    print(f"  Z=1: P(X=1|Z=1) = {np.mean(X_data[Z_data == 1]):.4f}")

    # predictive resampling on the latent compliance type
    B_post = 1000  # number of posterior samples
    T_fwd = 100_000

    results_compliers = mp_compliers(Y_data, X_data, Z_data, B_post, T_fwd, n_checkpoints=100)

    # plug-in type probabilities (initialise the type urn) and imputation probabilities
    # P(CP | X, Z, Y=y) for the observed units with ambiguous type
    print("Plug-in type probabilities:",
          {k: round(float(v), 4) for k, v in results_compliers['init']['p_hat'].items()})
    print("P(CP | X=0, Z=0, Y=0), P(CP | X=0, Z=0, Y=1):", results_compliers['init']['pi_x0z0'].round(4))

    # complier proportion and LATE
    print("90% CI for the complier proportion: ",
          np.quantile(results_compliers['Complier'][:, -1], [0.05, 0.95]))
    print(
        "90% CI for the LATE (per 1000 individuals): ",
        1000 * np.quantile(results_compliers['Treatment'][:, -1] - results_compliers['Control'][:, -1], [0.05, 0.95])
    )

    # figure
    p_y0_samples = results_compliers['Control'][:, -1]
    p_y1_samples = results_compliers['Treatment'][:, -1]

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True, gridspec_kw={'wspace': 0.05})

    steps = results_compliers['checkpoints']

    # panel 1: all sequences
    # rasterise the many trajectory lines to keep the PDF small; axes and text stay vector
    ax = axes[0]
    for samples, color in [
        (results_compliers['Control'], 'tab:blue'),
        (results_compliers['Treatment'], 'tab:orange'),
    ]:
        for b in range(samples.shape[0]):
            ax.plot(steps, samples[b], color=color, alpha=0.2, linewidth=0.3, rasterized=True)
    ax.set_xlabel("Forward step")
    ax.set_ylabel("Survival probability")
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt)

    # panel 2: KDE of the limiting survival probabilities
    ax = axes[1]
    y_grid_kde = np.linspace(
        min(p_y0_samples.min(), p_y1_samples.min()),
        max(p_y0_samples.max(), p_y1_samples.max()),
        200,
    )
    for samples, color, label in [
        (p_y0_samples, 'tab:blue', 'Control'),
        (p_y1_samples, 'tab:orange', 'Treated'),
    ]:
        kde = gaussian_kde(samples)
        ax.plot(kde(y_grid_kde), y_grid_kde, color=color, label=label)
    ax.set_xlabel('Density')
    ax.legend(frameon=False, loc='lower right')

    plt.savefig("Figures/Sommer_Zeger_Results.pdf", dpi=300, bbox_inches='tight')
    print("Saved Figures/Sommer_Zeger_Results.pdf")


if __name__ == "__main__":
    main()
