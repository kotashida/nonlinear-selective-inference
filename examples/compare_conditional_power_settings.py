"""Compare conditional power across benchmark methods and MI tuning settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_analysis_power(root: Path) -> pd.DataFrame:
    path = root / "analysis" / "unified_power_decisions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _comparison_data(
    confirmation_dir: Path, pilot_dir: Path, benchmark_dir: Path
) -> pd.DataFrame:
    benchmark = pd.read_csv(benchmark_dir / "power_decisions.csv")
    benchmark = benchmark.loc[
        benchmark["selection_method"].isin(
            ["shap", "spline_screening", "mutual_information"]
        )
    ].copy()
    benchmark["source"] = "full benchmark"
    benchmark["series"] = benchmark["selection_method"].map(
        {
            "shap": "SHAP (steps=8, r=100)",
            "spline_screening": "Spline screening (exact)",
            "mutual_information": "MI k=3 (steps=8, r=100)",
        }
    )
    benchmark["mi_neighbors"] = benchmark["selection_method"].map(
        {"mutual_information": 3}
    )
    benchmark["mcmc_steps"] = benchmark["selection_method"].map(
        {"shap": 8, "mutual_information": 8}
    )
    benchmark["mcmc_replicates"] = benchmark["selection_method"].map(
        {"shap": 100, "mutual_information": 100}
    )

    confirmation = _load_analysis_power(confirmation_dir)
    confirmation["source"] = "confirmation"
    confirmation["selection_method"] = "mutual_information"
    confirmation["series"] = confirmation.apply(
        lambda row: (
            f"MI k={int(row.mi_neighbors)} "
            f"(steps={int(row.mcmc_steps)}, r={int(row.mcmc_replicates)})"
        ),
        axis=1,
    )

    pilot = _load_analysis_power(pilot_dir)
    pilot["source"] = "pilot"
    pilot["selection_method"] = "mutual_information"
    pilot["series"] = pilot.apply(
        lambda row: (
            f"MI k={int(row.mi_neighbors)}, steps={int(row.mcmc_steps)} "
            f"(pilot r={int(row.mcmc_replicates)})"
        ),
        axis=1,
    )

    columns = [
        "source",
        "series",
        "selection_method",
        "mi_neighbors",
        "mcmc_steps",
        "mcmc_replicates",
        "signal_strength",
        "n_iterations",
        "n_signal_targets",
        "target_signal_rate",
        "conditional_power_given_signal_target",
        "conditional_power_ci_95_lower",
        "conditional_power_ci_95_upper",
        "marginal_detection_power",
    ]
    return pd.concat(
        [benchmark[columns], confirmation[columns], pilot[columns]],
        ignore_index=True,
    )


def _plot_method_comparison(data: pd.DataFrame, output_dir: Path) -> None:
    series = [
        "Spline screening (exact)",
        "SHAP (steps=8, r=100)",
        "MI k=3 (steps=8, r=100)",
        "MI k=12 (steps=12, r=800)",
        "MI k=16 (steps=12, r=800)",
    ]
    subset = data.loc[data["series"].isin(series)].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    markers = ["s", "^", "o", "D", "P"]
    for label, marker in zip(series, markers):
        group = subset.loc[subset["series"].eq(label)].sort_values("signal_strength")
        ax.plot(
            group["signal_strength"],
            group["conditional_power_given_signal_target"],
            marker=marker,
            linewidth=2,
            label=label,
        )
    ax.set(
        xlabel="Signal strength",
        ylabel="Conditional power given signal selected",
        ylim=(0.2, 1.02),
        xticks=sorted(subset["signal_strength"].unique()),
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Conditional power: SHAP, spline screening, and MI")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"conditional-power-method-comparison.{suffix}",
            dpi=180 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_mi_settings(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data.loc[data["selection_method"].eq("mutual_information")].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for (source, label), group in subset.groupby(["source", "series"], sort=False):
        group = group.sort_values("signal_strength")
        ax.plot(
            group["signal_strength"],
            group["conditional_power_given_signal_target"],
            marker="o" if source != "pilot" else ".",
            linestyle="-" if source != "pilot" else "--",
            linewidth=2 if source != "pilot" else 1.1,
            alpha=1.0 if source != "pilot" else 0.65,
            label=label,
        )
    ax.set(
        xlabel="Signal strength",
        ylabel="Conditional power given signal selected",
        ylim=(0.2, 1.02),
        xticks=sorted(subset["signal_strength"].unique()),
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Conditional power across mutual-information settings")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"conditional-power-mi-settings.{suffix}",
            dpi=180 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _comparison_data(
        args.confirmation_dir.resolve(),
        args.pilot_dir.resolve(),
        args.benchmark_dir.resolve(),
    )
    data.to_csv(output_dir / "conditional_power_comparison.csv", index=False)
    summary = (
        data.groupby(["source", "series"], as_index=False)
        .agg(
            mean_conditional_power=(
                "conditional_power_given_signal_target",
                "mean",
            ),
            minimum_signal_targets=("n_signal_targets", "min"),
            iterations_per_cell=("n_iterations", "min"),
        )
        .sort_values("mean_conditional_power", ascending=False)
    )
    summary.to_csv(output_dir / "conditional_power_summary.csv", index=False)
    _plot_method_comparison(data, output_dir)
    _plot_mi_settings(data, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
