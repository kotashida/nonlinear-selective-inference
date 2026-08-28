"""Plot k=2 and k=5 variable-size counterexamples without weighted inclusion."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K2_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "variable_size_selection_event_counterexample"
    / "calibration_summary.csv"
)
DEFAULT_K5_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "variable_size_selection_event_counterexample_k5"
    / "calibration_summary.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "variable_size_selection_event_counterexample_k_comparison"
    / "rejection_rates_k2_k5_without_weighted.png"
)
METHODS = ("feature_inclusion", "same_target", "exact_set")
METHOD_LABELS = ("feature inclusion", "same target", "exact set")


def _load_rates(summary_path: Path, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    summary = pd.read_csv(summary_path).set_index("method")
    missing = [method for method in METHODS if method not in summary.index]
    if missing:
        raise ValueError(f"Missing methods in {summary_path}: {missing}")

    label = f"{alpha:g}"
    observed_column = f"rejection_rate_{label}"
    expected_column = f"expected_rejection_rate_{label}"
    for column in (observed_column, expected_column):
        if column not in summary.columns:
            raise ValueError(f"Missing column {column!r} in {summary_path}")

    observed = summary.loc[list(METHODS), observed_column].to_numpy(dtype=float)
    expected = summary.loc[list(METHODS), expected_column].to_numpy(dtype=float)
    return observed, expected


def create_plot(
    *,
    k2_summary: Path = DEFAULT_K2_SUMMARY,
    k5_summary: Path = DEFAULT_K5_SUMMARY,
    output_path: Path = DEFAULT_OUTPUT,
    alpha: float = 0.05,
) -> Path:
    """Create a two-panel null-rejection-rate comparison."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")

    rates = {
        2: _load_rates(Path(k2_summary), alpha),
        5: _load_rates(Path(k5_summary), alpha),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    positions = np.arange(len(METHODS), dtype=float)

    for axis, (large_set_size, (observed, expected)) in zip(axes, rates.items()):
        axis.scatter(
            positions,
            observed,
            s=58,
            color="#1F77B4",
            label="Observed",
            zorder=3,
        )
        axis.scatter(
            positions,
            expected,
            s=62,
            color="#FF7F0E",
            marker="x",
            linewidths=2,
            label="Expected",
            zorder=4,
        )
        axis.axhline(
            alpha,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Nominal 5%",
        )
        axis.set_xticks(positions, METHOD_LABELS, rotation=15, ha="right")
        axis.set_title(f"Large selected-set size k = {large_set_size}")
        axis.grid(axis="y", alpha=0.2)

    maximum = max(np.max(observed) for observed, _ in rates.values())
    axes[0].set_ylabel("Null rejection rate")
    axes[0].set_ylim(0.0, max(0.10, maximum * 1.18))
    axes[1].legend(loc="upper right", fontsize=8)
    figure.suptitle(f"Variable-size selection counterexample (alpha = {alpha:g})")
    figure.tight_layout()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k2-summary", type=Path, default=DEFAULT_K2_SUMMARY)
    parser.add_argument("--k5-summary", type=Path, default=DEFAULT_K5_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = create_plot(
        k2_summary=args.k2_summary,
        k5_summary=args.k5_summary,
        output_path=args.output,
        alpha=args.alpha,
    )
    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
