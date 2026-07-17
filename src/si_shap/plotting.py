"""Visualizations for simulation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results):
    """Plot p-value histograms, excluding failed SI estimates."""
    alpha = results["alpha"]
    summary = results["summary"].set_index("method")
    colors = {
        "Random": "gray",
        "Naive SHAP": "blue",
        "Selective SHAP (AIS)": "green",
    }

    plt.figure(figsize=(12, 6))
    for method, p_values in results["p_values"].items():
        if p_values.size == 0:
            continue
        fpr = summary.loc[method, "fpr"]
        label_fpr = "unavailable" if not np.isfinite(fpr) else f"{fpr:.3f}"
        plt.hist(
            p_values,
            bins=20,
            range=(0, 1),
            alpha=0.45,
            color=colors[method],
            edgecolor="black",
            label=f"{method} (FPR: {label_fpr})",
        )
    plt.axvline(
        alpha,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"alpha={alpha}",
    )
    plt.title("Known-variance chi tests: random, naive, and selective inference")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
