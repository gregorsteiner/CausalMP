"""Shared matplotlib settings for all figures in the paper."""

import matplotlib

matplotlib.use("Agg")  # write figures to file only; never open a window

import matplotlib.pyplot as plt


def set_style(font_size=11):
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        font_size,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.linewidth":   0.8,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
    })
