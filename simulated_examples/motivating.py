"""Motivating example: two treatments with the same mean effect but very different distributions.

Produces Figures/Motivating_Example.pdf.
"""

import numpy as np
from scipy.stats import norm, lognorm

from simulated_examples.plot_style import set_style
import matplotlib.pyplot as plt


def main():
    # --- Left panel: polarisation (same mean, bimodal treated outcome) ---
    mu = 10          # the common mean
    sigma_0 = 2.0    # SD for control
    x = np.linspace(mu - 8, mu + 8, 1000)

    delta = 3
    sigma_sub = 1.2

    pdf_y0 = norm.pdf(x, mu, sigma_0)
    pdf_y1_pol = 0.5 * norm.pdf(x, mu - delta, sigma_sub) + 0.5 * norm.pdf(x, mu + delta, sigma_sub)

    # --- Right panel: lottery (same mean, more right-skewed treated outcome) ---
    target_mean = 10
    x_b = np.linspace(0, 35, 1000)  # longer tail for lottery

    # control: lower skewness
    s0 = 0.3
    mu0 = np.log(target_mean) - (s0**2 / 2)
    pdf_y0_b = lognorm.pdf(x_b, s0, scale=np.exp(mu0))
    # treatment: higher skewness
    s1 = 0.8
    mu1 = np.log(target_mean) - (s1**2 / 2)
    pdf_y1_b = lognorm.pdf(x_b, s1, scale=np.exp(mu1))

    # --- Plotting ---
    set_style(font_size=10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2), sharey=True)

    ax1.plot(x, pdf_y0, label='Control', lw=2)
    ax1.plot(x, pdf_y1_pol, label='Treatment', lw=2)
    ax1.set_xlabel("Outcome")
    ax1.set_ylabel("Density")

    ax2.plot(x_b, pdf_y0_b, label='Control', lw=2)
    ax2.plot(x_b, pdf_y1_b, label='Treatment', lw=2)
    ax2.set_xlabel("Outcome")
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("Figures/Motivating_Example.pdf", dpi=300, bbox_inches='tight')
    print("Saved Figures/Motivating_Example.pdf")


if __name__ == "__main__":
    main()
