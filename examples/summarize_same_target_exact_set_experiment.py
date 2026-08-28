"""Summarize paired same-target versus exact-set simulation outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EVENTS = ("same_target", "exact_set")
LABELS = {"same_target": "same target", "exact_set": "exact set"}
COLORS = {"same_target": "#1f77b4", "exact_set": "#59a14f"}


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else stats.beta.ppf(
        0.025, successes, trials - successes + 1
    )
    upper = 1.0 if successes == trials else stats.beta.ppf(
        0.975, successes + 1, trials - successes
    )
    return float(lower), float(upper)


def _paired_counts(frame: pd.DataFrame, value: str) -> tuple[int, int, float]:
    paired = frame.pivot(index="iteration", columns="selection_event", values=value)
    same = paired["same_target"].astype(bool)
    exact = paired["exact_set"].astype(bool)
    same_only = int((same & ~exact).sum())
    exact_only = int((~same & exact).sum())
    discordant = same_only + exact_only
    p_value = (
        1.0
        if discordant == 0
        else float(stats.binomtest(same_only, discordant, 0.5).pvalue)
    )
    return same_only, exact_only, p_value


def summarize(null_path: Path, power_path: Path) -> tuple[pd.DataFrame, dict]:
    null = pd.read_csv(null_path)
    targets = pd.read_csv(power_path / "target_results.csv")
    features = pd.read_csv(power_path / "feature_results.csv")

    rows = []
    for event in EVENTS:
        null_event = null[null["selection_event"] == event]
        null_rejections = int((null_event["p_value"] < 0.05).sum())
        null_lower, null_upper = _clopper_pearson(
            null_rejections, len(null_event)
        )

        target_event = targets[targets["selection_event"] == event]
        signal = target_event[target_event["target_is_signal"].astype(bool)]
        detections = int(signal["rejected"].astype(bool).sum())
        power_lower, power_upper = _clopper_pearson(detections, len(signal))

        diagnostics = features[features["selection_event"] == event]
        rows.append(
            {
                "selection_event": event,
                "null_rejections": null_rejections,
                "null_iterations": len(null_event),
                "null_rejection_rate": null_rejections / len(null_event),
                "null_ci_95_lower": null_lower,
                "null_ci_95_upper": null_upper,
                "signal_detections": detections,
                "signal_targets": len(signal),
                "conditional_power": detections / len(signal),
                "conditional_power_ci_95_lower": power_lower,
                "conditional_power_ci_95_upper": power_upper,
                "overall_detection_rate": float(
                    target_event["successful_detection"].astype(bool).mean()
                ),
                "mean_selection_probability": float(
                    diagnostics["selection_probability_estimate"].mean()
                ),
                "mean_denominator_ess": float(
                    diagnostics["denominator_ess"].mean()
                ),
                "mean_tail_ess": float(diagnostics["tail_ess"].mean()),
                "mean_mc_se": float(diagnostics["mc_se"].mean()),
            }
        )

    null_for_pairing = null.assign(rejected=null["p_value"] < 0.05)
    null_same_only, null_exact_only, null_p = _paired_counts(
        null_for_pairing, "rejected"
    )
    power_same_only, power_exact_only, power_p = _paired_counts(
        targets, "successful_detection"
    )
    paired = {
        "null_same_target_only_rejections": null_same_only,
        "null_exact_set_only_rejections": null_exact_only,
        "null_paired_exact_p_value": null_p,
        "power_same_target_only_detections": power_same_only,
        "power_exact_set_only_detections": power_exact_only,
        "power_paired_exact_p_value": power_p,
    }
    return pd.DataFrame(rows), paired


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    x = np.arange(len(summary))
    labels = [LABELS[event] for event in summary["selection_event"]]
    colors = [COLORS[event] for event in summary["selection_event"]]

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
            "Power given signal was targeted",
        ),
    )
    for axis, (value_col, lower_col, upper_col, title) in zip(axes, panels):
        values = summary[value_col].to_numpy(float)
        lower = summary[lower_col].to_numpy(float)
        upper = summary[upper_col].to_numpy(float)
        errors = np.vstack((values - lower, upper - values))
        bars = axis.bar(x, values, color=colors, alpha=0.9, width=0.65)
        axis.errorbar(x, values, yerr=errors, fmt="none", color="black", capsize=5)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        axis.set_xticks(x, labels)
        axis.set_ylim(0, max(upper) * 1.25)
        axis.set_title(title)
        axis.set_ylabel("Rate (95% exact CI)")
        axis.grid(axis="y", linestyle=":", alpha=0.5)

    axes[0].axhline(0.05, color="#d62728", linestyle="--", linewidth=1.3)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-dir", type=Path, required=True)
    parser.add_argument("--power-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, paired = summarize(
        args.null_dir / "p_value_results.csv", args.power_dir
    )
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    pd.DataFrame([paired]).to_csv(
        args.output_dir / "paired_comparison_summary.csv", index=False
    )
    plot_summary(summary, args.output_dir / "calibration_and_power.png")
    print(summary.to_string(index=False))
    print(pd.Series(paired).to_string())


if __name__ == "__main__":
    main()
