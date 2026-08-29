"""Create analysis plots for pooled same-target versus exact-set shards.

The input directory is expected to contain one directory per selection method,
with the pooled outputs produced by ``summarize_same_target_exact_set_shards``.
Plots and compact derived tables are written to ``pooled_summary/analysis_plots``
unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVENTS = ("same_target", "exact_set")
EVENT_LABELS = {"same_target": "Same target", "exact_set": "Exact set"}
EVENT_COLORS = {"same_target": "#2878B5", "exact_set": "#59A14F"}
METHOD_LABELS = {
    "shap": "SHAP",
    "mutual_information": "Mutual information",
    "marginal_screening": "Marginal screening",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--methods", nargs="+", default=("shap", "mutual_information")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _read_inputs(input_dir: Path, methods: list[str]):
    summaries = []
    null_frames = []
    target_frames = []
    diagnostic_frames = []
    for method in methods:
        pooled = input_dir / method / "pooled_summary"
        paths = {
            "summary": pooled / "comparison_summary.csv",
            "null": pooled / "pooled_null" / "p_value_results.csv",
            "targets": pooled / "pooled_power" / "target_results.csv",
            "diagnostics": pooled / "pooled_power" / "feature_results.csv",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing pooled inputs: " + ", ".join(missing))
        loaded = {name: pd.read_csv(path) for name, path in paths.items()}
        for name, frame in loaded.items():
            if frame.empty:
                raise ValueError(f"{paths[name]} is empty.")
            if "selection_event" in frame and set(frame["selection_event"]) != set(EVENTS):
                raise ValueError(f"{paths[name]} does not contain exactly {EVENTS}.")
            frame.insert(0, "selection_method", method)
        summaries.append(loaded["summary"])
        null_frames.append(loaded["null"])
        target_frames.append(loaded["targets"])
        diagnostic_frames.append(loaded["diagnostics"])
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(null_frames, ignore_index=True),
        pd.concat(target_frames, ignore_index=True),
        pd.concat(diagnostic_frames, ignore_index=True),
    )


def _style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linestyle=":", linewidth=0.8)
    axis.set_axisbelow(True)


def plot_headline_rates(summary: pd.DataFrame, methods: list[str], path: Path, dpi: int):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    panels = (
        (
            "null_rejection_rate",
            "null_ci_95_lower",
            "null_ci_95_upper",
            "Null rejection rate",
        ),
        (
            "conditional_power",
            "conditional_power_ci_95_lower",
            "conditional_power_ci_95_upper",
            "Conditional power",
        ),
    )
    x = np.arange(len(methods), dtype=float)
    width = 0.34
    for axis, (value_col, lower_col, upper_col, title) in zip(axes, panels):
        for event_index, event in enumerate(EVENTS):
            ordered = (
                summary[summary["selection_event"] == event]
                .set_index("selection_method")
                .loc[methods]
            )
            values = ordered[value_col].to_numpy(float)
            lower = ordered[lower_col].to_numpy(float)
            upper = ordered[upper_col].to_numpy(float)
            positions = x + (event_index - 0.5) * width
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=EVENT_COLORS[event],
                label=EVENT_LABELS[event],
                zorder=2,
            )
            axis.errorbar(
                positions,
                values,
                yerr=np.vstack((values - lower, upper - values)),
                fmt="none",
                color="#222222",
                capsize=4,
                linewidth=1.1,
                zorder=3,
            )
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        if value_col == "null_rejection_rate":
            axis.axhline(0.05, color="#C44E52", linestyle="--", linewidth=1.3)
            axis.text(
                len(methods) - 0.55,
                0.052,
                "nominal 5%",
                color="#9F3438",
                fontsize=8.5,
                ha="right",
                va="bottom",
            )
        axis.set_xticks(x, [_method_label(method) for method in methods])
        axis.set_ylabel("Rate (95% exact CI)")
        axis.set_title(title, loc="left", fontsize=11.5)
        axis.set_ylim(0, max(0.12, axis.get_ylim()[1] * 1.13))
        _style_axis(axis)
    axes[1].legend(frameon=False, loc="upper left")
    figure.suptitle("Conditioning-event comparison across selection methods", fontsize=14)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_null_calibration(null: pd.DataFrame, methods: list[str], path: Path, dpi: int):
    figure, axes = plt.subplots(
        len(methods), 2, figsize=(11.5, 4.1 * len(methods)), squeeze=False,
        constrained_layout=True,
    )
    bin_edges = np.linspace(0, 1, 11)
    alpha_grid = np.linspace(0, 0.20, 101)
    for row, method in enumerate(methods):
        frame = null[null["selection_method"] == method]
        for event in EVENTS:
            values = frame.loc[frame["selection_event"] == event, "p_value"].to_numpy(float)
            axes[row, 0].hist(
                values,
                bins=bin_edges,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=EVENT_COLORS[event],
                label=EVENT_LABELS[event],
            )
            rejection_curve = np.array([(values < alpha).mean() for alpha in alpha_grid])
            axes[row, 1].plot(
                alpha_grid,
                rejection_curve,
                color=EVENT_COLORS[event],
                linewidth=2.0,
                label=EVENT_LABELS[event],
            )
        axes[row, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        axes[row, 0].set(
            title=f"{_method_label(method)}: null p-value distribution",
            xlabel="Selective p-value",
            ylabel="Density",
            xlim=(0, 1),
        )
        axes[row, 1].plot(
            alpha_grid, alpha_grid, color="#333333", linestyle="--", linewidth=1.0,
            label="Nominal",
        )
        axes[row, 1].set(
            title=f"{_method_label(method)}: rejection curve",
            xlabel="Nominal level",
            ylabel="Empirical rejection rate",
            xlim=(0, 0.20),
            ylim=(0, 0.20),
        )
        axes[row, 1].set_aspect("equal", adjustable="box")
        for axis in axes[row]:
            _style_axis(axis)
        axes[row, 0].legend(frameon=False)
        axes[row, 1].legend(frameon=False)
    figure.suptitle("Null calibration diagnostics (1,000 paired iterations per method)", fontsize=14)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_paired_p_values(null: pd.DataFrame, methods: list[str], path: Path, dpi: int):
    figure, axes = plt.subplots(
        1, len(methods), figsize=(5.4 * len(methods), 5.0), squeeze=False,
        constrained_layout=True,
    )
    for column, method in enumerate(methods):
        axis = axes[0, column]
        frame = null[null["selection_method"] == method]
        paired = frame.pivot(index="iteration", columns="selection_event", values="p_value")
        axis.scatter(
            paired["same_target"],
            paired["exact_set"],
            s=15,
            alpha=0.35,
            color="#4C78A8",
            edgecolors="none",
            rasterized=True,
        )
        axis.plot([0, 1], [0, 1], color="#222222", linestyle="--", linewidth=1.0)
        axis.axvline(0.05, color="#C44E52", linestyle=":", linewidth=1.2)
        axis.axhline(0.05, color="#C44E52", linestyle=":", linewidth=1.2)
        correlation = paired.corr(method="spearman").iloc[0, 1]
        axis.text(
            0.04,
            0.95,
            f"Spearman r = {correlation:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        axis.set(
            title=_method_label(method),
            xlabel="Same-target p-value",
            ylabel="Exact-set p-value",
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
        )
        axis.set_aspect("equal", adjustable="box")
        _style_axis(axis)
    figure.suptitle("Paired null p-values by conditioning event", fontsize=14)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _paired_outcome_counts(
    frame: pd.DataFrame, methods: list[str], value_column: str, analysis: str
) -> pd.DataFrame:
    rows = []
    for method in methods:
        method_frame = frame[frame["selection_method"] == method]
        paired = method_frame.pivot(
            index="iteration", columns="selection_event", values=value_column
        ).astype(bool)
        same = paired["same_target"]
        exact = paired["exact_set"]
        counts = {
            "Neither": int((~same & ~exact).sum()),
            "Same target only": int((same & ~exact).sum()),
            "Exact set only": int((~same & exact).sum()),
            "Both": int((same & exact).sum()),
        }
        for outcome, count in counts.items():
            rows.append(
                {
                    "analysis": analysis,
                    "selection_method": method,
                    "outcome": outcome,
                    "count": count,
                    "proportion": count / len(paired),
                }
            )
    return pd.DataFrame(rows)


def plot_paired_outcomes(
    outcome_counts: pd.DataFrame, methods: list[str], path: Path, dpi: int
):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    outcome_order = ("Same target only", "Exact set only", "Both")
    colors = (EVENT_COLORS["same_target"], EVENT_COLORS["exact_set"], "#7B61A8")
    x = np.arange(len(methods), dtype=float)
    width = 0.24
    for axis, analysis, title in zip(
        axes,
        ("Null rejections", "Successful detections"),
        ("Paired null rejection outcomes", "Paired successful-detection outcomes"),
    ):
        for outcome_index, (outcome, color) in enumerate(zip(outcome_order, colors)):
            ordered = (
                outcome_counts[
                    (outcome_counts["analysis"] == analysis)
                    & (outcome_counts["outcome"] == outcome)
                ]
                .set_index("selection_method")
                .loc[methods]
            )
            values = ordered["count"].to_numpy(int)
            positions = x + (outcome_index - 1) * width
            bars = axis.bar(
                positions, values, color=color, label=outcome, width=width,
            )
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.6,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        axis.set_xticks(x, [_method_label(method) for method in methods])
        axis.set_ylim(0, max(axis.get_ylim()[1] * 1.10, 5))
        axis.set_ylabel("Number of paired iterations")
        axis.set_title(title, loc="left", fontsize=11.5)
        _style_axis(axis)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Agreement and discordance between conditioning events", fontsize=14)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_mc_diagnostics(
    diagnostics: pd.DataFrame, methods: list[str], path: Path, dpi: int
):
    metrics = (
        ("selection_probability_estimate", "Selection probability"),
        ("denominator_ess", "Denominator ESS"),
        ("tail_ess", "Tail ESS"),
        ("mc_se", "Monte Carlo SE"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    positions = []
    tick_positions = []
    tick_labels = []
    for method_index, method in enumerate(methods):
        for event_index, event in enumerate(EVENTS):
            positions.append(method_index * 3 + event_index)
        tick_positions.append(method_index * 3 + 0.5)
        tick_labels.append(_method_label(method))
    for axis, (column, title) in zip(axes.flat, metrics):
        values = []
        colors = []
        for method in methods:
            for event in EVENTS:
                subset = diagnostics.loc[
                    (diagnostics["selection_method"] == method)
                    & (diagnostics["selection_event"] == event),
                    column,
                ].dropna().to_numpy(float)
                values.append(subset)
                colors.append(EVENT_COLORS[event])
        artists = axis.boxplot(
            values,
            positions=positions,
            widths=0.72,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )
        for box, color in zip(artists["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.82)
            box.set_edgecolor("#555555")
        axis.set_ylim(bottom=0)
        axis.set_xticks(tick_positions, tick_labels)
        axis.set_title(title, loc="left", fontsize=11.5)
        _style_axis(axis)
    handles = [
        plt.Line2D([0], [0], color=EVENT_COLORS[event], linewidth=8, label=EVENT_LABELS[event])
        for event in EVENTS
    ]
    axes[0, 1].legend(handles=handles, frameon=False)
    figure.suptitle("Conditional Monte Carlo diagnostics (power experiment)", fontsize=14)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _shard_summary(
    null: pd.DataFrame, targets: pd.DataFrame, methods: list[str]
) -> pd.DataFrame:
    null_work = null.assign(rejected=null["p_value"] < 0.05)
    null_summary = (
        null_work.groupby(["selection_method", "shard", "selection_event"], sort=False)
        .agg(rate=("rejected", "mean"), numerator=("rejected", "sum"), denominator=("rejected", "size"))
        .reset_index()
        .assign(analysis="Null rejection rate")
    )
    signal = targets[targets["target_is_signal"].astype(bool)].copy()
    power_summary = (
        signal.groupby(["selection_method", "shard", "selection_event"], sort=False)
        .agg(
            rate=("successful_detection", "mean"),
            numerator=("successful_detection", "sum"),
            denominator=("successful_detection", "size"),
        )
        .reset_index()
        .assign(analysis="Conditional power")
    )
    result = pd.concat([null_summary, power_summary], ignore_index=True)
    result["shard_index"] = result["shard"].str.removeprefix("shard_").astype(int)
    order = pd.CategoricalDtype(methods, ordered=True)
    result["selection_method"] = result["selection_method"].astype(order)
    return result.sort_values(["analysis", "selection_method", "selection_event", "shard_index"])


def plot_shard_stability(
    shard_summary: pd.DataFrame, methods: list[str], path: Path, dpi: int
):
    figure, axes = plt.subplots(
        len(methods), 2, figsize=(11.5, 3.8 * len(methods)), squeeze=False,
        constrained_layout=True, sharex=True,
    )
    analyses = ("Null rejection rate", "Conditional power")
    for row, method in enumerate(methods):
        for column, analysis in enumerate(analyses):
            axis = axes[row, column]
            frame = shard_summary[
                (shard_summary["selection_method"] == method)
                & (shard_summary["analysis"] == analysis)
            ]
            for event in EVENTS:
                event_frame = frame[frame["selection_event"] == event]
                axis.plot(
                    event_frame["shard_index"],
                    event_frame["rate"],
                    marker="o",
                    markersize=4.5,
                    linewidth=1.5,
                    color=EVENT_COLORS[event],
                    label=EVENT_LABELS[event],
                )
            if analysis == "Null rejection rate":
                axis.axhline(0.05, color="#C44E52", linestyle="--", linewidth=1.0)
            axis.set_title(f"{_method_label(method)}: {analysis.lower()}", loc="left", fontsize=11.5)
            axis.set_ylabel("Rate")
            axis.set_xticks(sorted(frame["shard_index"].unique()))
            axis.set_xlabel("Shard index (100 iterations per shard)")
            axis.set_ylim(bottom=0)
            _style_axis(axis)
        axes[row, 1].legend(frameon=False)
    figure.suptitle("Shard-to-shard stability", fontsize=14)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    methods = list(args.methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("--methods must contain unique method names.")
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "pooled_summary" / "analysis_plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, null, targets, diagnostics = _read_inputs(input_dir, methods)
    null_outcomes = _paired_outcome_counts(
        null.assign(rejected=null["p_value"] < 0.05), methods, "rejected", "Null rejections"
    )
    power_outcomes = _paired_outcome_counts(
        targets, methods, "successful_detection", "Successful detections"
    )
    outcome_counts = pd.concat([null_outcomes, power_outcomes], ignore_index=True)
    shard_summary = _shard_summary(null, targets, methods)

    plot_headline_rates(summary, methods, output_dir / "01_headline_rates.png", args.dpi)
    plot_null_calibration(null, methods, output_dir / "02_null_calibration.png", args.dpi)
    plot_paired_p_values(null, methods, output_dir / "03_paired_null_p_values.png", args.dpi)
    plot_paired_outcomes(outcome_counts, methods, output_dir / "04_paired_outcomes.png", args.dpi)
    plot_mc_diagnostics(diagnostics, methods, output_dir / "05_mc_diagnostics.png", args.dpi)
    plot_shard_stability(shard_summary, methods, output_dir / "06_shard_stability.png", args.dpi)

    outcome_counts.to_csv(output_dir / "paired_outcome_counts.csv", index=False)
    shard_summary.to_csv(output_dir / "shard_summary.csv", index=False)
    print(f"Saved six analysis plots and two derived tables to {output_dir}")
    return summary, outcome_counts, shard_summary


if __name__ == "__main__":
    main()
