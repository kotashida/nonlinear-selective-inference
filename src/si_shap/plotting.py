"""Visualizations for simulation results."""

from __future__ import annotations

import os
from collections.abc import Sequence

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .inference import _defensive_mixture_logpdf, _truncated_normal_logpdf
from .selection_regions import SelectionRegionResult


def plot_results(results):
    """Plot p-value histograms, excluding failed SI estimates."""
    alpha = results["alpha"]
    summary = results["summary"].set_index("method")
    colors = {
        "Random": "gray",
        "Unadjusted SHAP": "blue",
        "Selective SHAP (conditional MC)": "green",
        "Selective SHAP (approximate AIS)": "orange",
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
            color=colors.get(method, "green"),
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
    plt.title(
        "Known-variance chi tests: random, unadjusted, and selective inference"
    )
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_selection_regions(
    results: SelectionRegionResult | Sequence[SelectionRegionResult],
    output_path=None,
    *,
    dpi=180,
    show_proposal_components=False,
):
    """Plot one or more selection regions with the final AIS proposal density."""
    if isinstance(results, SelectionRegionResult):
        results = [results]
    else:
        results = list(results)
    if not results:
        raise ValueError("results must contain at least one data set.")
    if any(result.selection_probability <= 0.0 for result in results):
        raise ValueError("Every selection probability must be positive.")

    single_panel = len(results) == 1
    n_columns = 1 if single_panel else 3
    n_rows = int(np.ceil(len(results) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(11, 6) if single_panel else (18, 5.0 * n_rows),
        squeeze=False,
        sharex=not single_panel,
    )
    flat_axes = axes.ravel()

    for axis, result in zip(flat_axes, results):
        _plot_selection_region_on_axis(
            axis,
            result,
            show_proposal_components=show_proposal_components,
            detailed_labels=single_panel,
        )

    for axis in flat_axes[len(results) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(r"Test statistic $z$")

    if single_panel:
        flat_axes[0].set_xlim(0.0, results[0].z_max)
        flat_axes[0].legend(
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        figure.tight_layout()
    else:
        handles, labels = flat_axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=6,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.01),
        )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    if output_path is not None:
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return figure


def _plot_selection_region_on_axis(
    axis,
    result: SelectionRegionResult,
    *,
    show_proposal_components: bool,
    detailed_labels: bool,
):
    """Draw a selection region and its proposal densities on one axis."""
    probability_upper_bound = result.selection_probability_upper_bound
    if not np.isfinite(probability_upper_bound):
        probability_upper_bound = min(
            1.0,
            result.selection_probability + result.omitted_tail_probability,
        )
    z = np.linspace(0.0, result.z_max, 1600)
    null_density = stats.chi.pdf(z, df=result.rank)
    observed_proposal = np.exp(
        _truncated_normal_logpdf(
            z,
            result.t_obs,
            result.observed_proposal_sd,
        )
    )
    adapted_proposal = np.exp(
        _truncated_normal_logpdf(
            z,
            result.adapted_proposal_mean,
            result.adapted_proposal_sd,
        )
    )
    final_proposal = np.exp(
        _defensive_mixture_logpdf(
            z,
            result.rank,
            result.t_obs,
            result.adapted_proposal_mean,
            result.adapted_proposal_sd,
        )
    )

    in_region = np.zeros_like(z, dtype=bool)
    for interval_number, (left, right) in enumerate(result.intervals):
        in_region |= (z >= left) & (z <= right)
        axis.axvspan(
            left,
            right,
            color="tab:green",
            alpha=0.13,
            label=(
                "Feature-specific exact-set selection region"
                if result.selection_event == "exact_set"
                else f"Feature-specific {result.selection_event.replace('_', '-')} "
                "selection region"
            )
            if interval_number == 0
            else None,
        )

    conditional_density = np.where(
        in_region,
        null_density / result.selection_probability,
        np.nan,
    )
    axis.plot(
        z,
        null_density,
        color="0.35",
        linestyle="--",
        linewidth=1.8,
        label=rf"Null $\chi_{{{result.rank}}}$ density",
    )
    axis.plot(
        z,
        conditional_density,
        color="tab:blue",
        linewidth=2.4 if detailed_labels else 2.2,
        label="Conditional density over detected region",
    )
    axis.plot(
        z,
        final_proposal,
        color="tab:orange",
        linewidth=2.4 if detailed_labels else 2.2,
        label=r"Final proposal $q_{\mathrm{final}}$",
    )
    if show_proposal_components:
        axis.plot(
            z,
            observed_proposal,
            color="tab:purple",
            linestyle="-.",
            linewidth=1.5 if detailed_labels else 1.4,
            alpha=0.7,
            label=(
                (
                    rf"$q_{{\mathrm{{obs}}}}$ component "
                    rf"($\mu={result.t_obs:.3f}$, "
                    rf"$\sigma={result.observed_proposal_sd:.3f}$)"
                )
                if detailed_labels
                else r"$q_{\mathrm{obs}}$ component"
            ),
        )
        axis.plot(
            z,
            adapted_proposal,
            color="tab:brown",
            linestyle=":",
            linewidth=1.8 if detailed_labels else 1.6,
            alpha=0.7,
            label=(
                (
                    rf"$q_{{\mathrm{{adapt}}}}$ component "
                    rf"($\mu={result.adapted_proposal_mean:.3f}$, "
                    rf"$\sigma={result.adapted_proposal_sd:.3f}$)"
                )
                if detailed_labels
                else r"$q_{\mathrm{adapt}}$ component"
            ),
        )
    axis.axvline(
        result.t_obs,
        color="tab:red",
        linewidth=2.0,
        label=(
            rf"$T_{{\mathrm{{obs}}}}={result.t_obs:.3f}$"
            if detailed_labels
            else r"$T_{\mathrm{obs}}$"
        ),
    )
    axis.set_title(
        f"Data set {result.dataset_number} (seed={result.seed}) | "
        f"target $x_{{{result.selected_feature}}}$, position "
        f"{result.selection_position}/{result.k_select} | "
        f"{result.selection_event.replace('_', '-')}\n"
        f"df={result.rank}, diagnostic detected probability="
        f"{result.selection_probability:.4f}, upper bound="
        f"{probability_upper_bound:.4f}"
    )
    axis.set_ylabel("Density")
    axis.grid(True, linestyle=":", alpha=0.35)
