"""Unify and analyze a mutual-information tuning sweep.

The input directory contains a sweep manifest plus one completed validation
directory per (MI neighbors, MCMC steps, replicates) configuration.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_PATTERN = re.compile(r"_k(?P<neighbors>\d+)_s(?P<steps>\d+)_r(?P<replicates>\d+)$")
POWER_PATTERN = re.compile(r"power_feature_\d+_beta_(?P<strength>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Directory containing SHAP/spline pooled power_decisions.csv.",
    )
    return parser.parse_args()


def read_config(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        result[key] = value
    return result


def run_metadata(run: Path) -> dict[str, object]:
    config = read_config(run / "run_config.txt")
    match = RUN_PATTERN.search(run.name)
    if match is None:
        raise ValueError(f"Cannot parse configuration from {run.name}")
    parsed = {key: int(value) for key, value in match.groupdict().items()}
    metadata: dict[str, object] = {
        "configuration": run.name,
        "mi_neighbors": int(config.get("MI_NEIGHBORS", parsed["neighbors"])),
        "mcmc_steps": int(config.get("MCMC_STEPS", parsed["steps"])),
        "mcmc_replicates": int(config.get("MC_PROPOSALS", parsed["replicates"])),
    }
    return metadata


def prepend_metadata(frame: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    result = frame.copy()
    for name, value in metadata.items():
        result[name] = value
    leading = list(metadata)
    return result[leading + [column for column in result.columns if column not in leading]]


def shard_number(path: Path) -> int:
    shard = next(part for part in path.parts if part.startswith("shard_"))
    return int(shard.removeprefix("shard_"))


def signal_strength(path: Path) -> float:
    match = POWER_PATTERN.match(path.parent.name)
    if match is None:
        raise ValueError(f"Cannot parse signal strength from {path}")
    return float(match.group("strength").replace("p", "."))


def wilson_interval(successes: pd.Series, totals: pd.Series) -> tuple[pd.Series, pd.Series]:
    z = 1.959963984540054
    n = totals.astype(float)
    p = successes.astype(float) / n
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return (center - half).clip(lower=0.0), (center + half).clip(upper=1.0)


def load_sweep(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = sorted(path for path in input_dir.iterdir() if path.is_dir() and RUN_PATTERN.search(path.name))
    if not runs:
        raise FileNotFoundError(f"No tuning runs found below {input_dir}")

    pooled_calibration: list[pd.DataFrame] = []
    pooled_power: list[pd.DataFrame] = []
    records: list[pd.DataFrame] = []

    for run in runs:
        metadata = run_metadata(run)
        pooled = run / "pooled_summary"
        pooled_calibration.append(
            prepend_metadata(pd.read_csv(pooled / "calibration_decisions.csv"), metadata)
        )
        pooled_power.append(prepend_metadata(pd.read_csv(pooled / "power_decisions.csv"), metadata))

        for path in sorted(run.glob("shard_*/mutual_information/baseline/null_fresh/p_value_results.csv")):
            frame = pd.read_csv(path)
            frame.insert(0, "analysis_type", "calibration")
            frame.insert(1, "signal_strength", np.nan)
            frame.insert(2, "shard", shard_number(path))
            records.append(prepend_metadata(frame, metadata))

        for path in sorted(
            run.glob("shard_*/mutual_information/baseline/power_feature_*/target_results.csv")
        ):
            frame = pd.read_csv(path)
            frame.insert(0, "analysis_type", "power")
            frame.insert(1, "signal_strength", signal_strength(path))
            frame.insert(2, "shard", shard_number(path))
            records.append(prepend_metadata(frame, metadata))

    calibration = pd.concat(pooled_calibration, ignore_index=True)
    power = pd.concat(pooled_power, ignore_index=True)
    unified = pd.concat(records, ignore_index=True, sort=False)
    unified.insert(
        7,
        "configuration_iteration",
        unified["shard"].astype(int).astype(str) + ":" + unified["iteration"].astype(int).astype(str),
    )
    return calibration, power, unified


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    columns = list(frame.columns)

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if pd.isna(value) else f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def plot_calibration(calibration: pd.DataFrame, output_dir: Path) -> None:
    data = calibration.loc[np.isclose(calibration["alpha"], 0.05)].copy()
    data["ci_lower"], data["ci_upper"] = wilson_interval(data["n_rejected"], data["n_iterations"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for steps, group in data.groupby("mcmc_steps"):
        group = group.sort_values("mi_neighbors")
        yerr = np.vstack(
            [group["rejection_rate"] - group["ci_lower"], group["ci_upper"] - group["rejection_rate"]]
        )
        ax.errorbar(
            group["mi_neighbors"], group["rejection_rate"], yerr=yerr,
            marker="o", capsize=4, linewidth=1.8, label=f"{steps} MCMC steps",
        )
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="Nominal 0.05")
    ax.set(xlabel="MI neighbors", ylabel="Null rejection rate", ylim=(0, None))
    ax.set_xticks(sorted(data["mi_neighbors"].unique()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.suptitle("Mutual-information tuning: null calibration")
    save_figure(fig, output_dir, "null-calibration")


def plot_power(power: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("target_signal_rate", "Signal selection rate"),
        ("conditional_power_given_signal_target", "Conditional power"),
        ("marginal_detection_power", "Marginal detection power"),
    ]
    steps_values = sorted(power["mcmc_steps"].unique())
    fig, axes = plt.subplots(
        len(metrics),
        len(steps_values),
        figsize=(11.2, 10.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column, steps in enumerate(steps_values):
        subset = power.loc[power["mcmc_steps"].eq(steps)]
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, column]
            for neighbors, group in subset.groupby("mi_neighbors"):
                group = group.sort_values("signal_strength")
                ax.plot(group["signal_strength"], group[metric], marker="o", label=f"k={neighbors}")
            if row == 0:
                ax.set_title(f"{steps} MCMC steps")
            if column == 0:
                ax.set_ylabel(label)
            if row == len(metrics) - 1:
                ax.set_xlabel("Signal strength")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Selection and inference power across MI tuning parameters", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "power-curves")


def plot_null_diagnostics(unified: pd.DataFrame, output_dir: Path) -> None:
    data = unified.loc[unified["analysis_type"].eq("calibration")].copy()
    data["setting"] = data["mi_neighbors"].astype(str) + " / " + data["mcmc_steps"].astype(str)
    order = [f"{k} / {s}" for s in sorted(data["mcmc_steps"].unique()) for k in sorted(data["mi_neighbors"].unique())]
    metrics = [
        ("mcmc_replicate_acceptance_rate", "Replicate acceptance rate"),
        ("denominator_ess", "Denominator ESS"),
        ("tail_ess", "Tail ESS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    for ax, (metric, label) in zip(axes, metrics):
        values = [data.loc[data["setting"].eq(setting), metric].dropna().to_numpy() for setting in order]
        ax.boxplot(values, tick_labels=order, showfliers=False)
        ax.set(xlabel="MI neighbors / MCMC steps", ylabel=label)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Null-run MCMC diagnostics")
    fig.tight_layout()
    save_figure(fig, output_dir, "mcmc-diagnostics")


def plot_null_ecdf(unified: pd.DataFrame, output_dir: Path) -> None:
    data = unified.loc[
        unified["analysis_type"].eq("calibration") & unified["p_value"].notna()
    ].copy()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for (neighbors, steps), group in data.groupby(["mi_neighbors", "mcmc_steps"]):
        values = np.sort(group["p_value"].to_numpy())
        ecdf = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, ecdf, where="post", linewidth=1.4, label=f"k={neighbors}, steps={steps}")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.1, label="Uniform reference")
    ax.set(xlabel="Selective p-value", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Pooled null p-value calibration")
    save_figure(fig, output_dir, "null-pvalue-ecdf")


def load_benchmark(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    source = path / "power_decisions.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark power decisions not found: {source}")
    return pd.read_csv(source)


def plot_benchmark(power: pd.DataFrame, benchmark: pd.DataFrame, output_dir: Path) -> None:
    best_key = (
        power.groupby(["mi_neighbors", "mcmc_steps"])["marginal_detection_power"]
        .mean()
        .idxmax()
    )
    best = power.loc[
        power["mi_neighbors"].eq(best_key[0]) & power["mcmc_steps"].eq(best_key[1])
    ].copy()
    best["series"] = f"Tuned MI (k={best_key[0]}, steps={best_key[1]})"

    reference = benchmark.loc[
        benchmark["selection_method"].isin(["mutual_information", "shap", "spline_screening"])
    ].copy()
    reference["series"] = reference["selection_method"].map(
        {
            "mutual_information": "Original MI (k=3, steps=8)",
            "shap": "SHAP (steps=8)",
            "spline_screening": "Spline screening (exact inference)",
        }
    )
    comparison = pd.concat([reference, best], ignore_index=True, sort=False)
    comparison.to_csv(output_dir / "benchmark_power_comparison.csv", index=False)

    metrics = [
        ("target_signal_rate", "Signal selection rate"),
        ("conditional_power_given_signal_target", "Conditional power"),
        ("marginal_detection_power", "Marginal detection power"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        for series, group in comparison.groupby("series", sort=False):
            group = group.sort_values("signal_strength")
            ax.plot(group["signal_strength"], group[metric], marker="o", linewidth=1.8, label=series)
        ax.set(xlabel="Signal strength", ylabel=label, ylim=(-0.03, 1.03))
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Best evaluated MI setting versus full-run method benchmarks", y=1.12)
    fig.tight_layout()
    save_figure(fig, output_dir, "method-benchmark-power")


def write_report(
    calibration: pd.DataFrame,
    power: pd.DataFrame,
    unified: pd.DataFrame,
    output_dir: Path,
    benchmark: pd.DataFrame | None,
) -> None:
    cal = calibration.loc[np.isclose(calibration["alpha"], 0.05)].copy()
    cal["absolute_calibration_error"] = (cal["rejection_rate"] - 0.05).abs()
    summary = (
        power.groupby(["mi_neighbors", "mcmc_steps"], as_index=False)
        .agg(
            mean_signal_selection_rate=("target_signal_rate", "mean"),
            mean_conditional_power=("conditional_power_given_signal_target", "mean"),
            mean_marginal_power=("marginal_detection_power", "mean"),
            minimum_signal_targets=("n_signal_targets", "min"),
        )
        .merge(
            cal[["mi_neighbors", "mcmc_steps", "rejection_rate", "calibration_pass", "absolute_calibration_error"]],
            on=["mi_neighbors", "mcmc_steps"], how="left",
        )
        .sort_values(["mean_marginal_power", "absolute_calibration_error"], ascending=[False, True])
    )
    summary.to_csv(output_dir / "configuration_summary.csv", index=False)

    best = summary.iloc[0]
    benchmark_lines: list[str] = []
    if benchmark is not None:
        benchmark_summary = (
            benchmark.groupby("selection_method", as_index=False)
            .agg(
                mean_signal_selection_rate=("target_signal_rate", "mean"),
                mean_conditional_power=("conditional_power_given_signal_target", "mean"),
                mean_marginal_power=("marginal_detection_power", "mean"),
            )
            .sort_values("mean_marginal_power", ascending=False)
        )
        benchmark_summary.to_csv(output_dir / "benchmark_method_summary.csv", index=False)
        benchmark_lines = [
            "",
            "## Comparison with full-run SHAP and spline benchmarks",
            "",
            frame_to_markdown(benchmark_summary),
            "",
            f"The best evaluated MI setting has mean marginal power {best.mean_marginal_power:.3f}. "
            "The benchmark comparison uses the existing 1,500-iteration exact-set run; "
            "its spline results use exact spline inference, so conditional-power differences "
            "are not solely screening differences.",
        ]
    minimum_cell_iterations = int(power["n_iterations"].min())
    maximum_cell_iterations = int(power["n_iterations"].max())
    if minimum_cell_iterations == maximum_cell_iterations:
        iteration_description = f"{minimum_cell_iterations:,} iterations"
    else:
        iteration_description = (
            f"{minimum_cell_iterations:,}--{maximum_cell_iterations:,} iterations"
        )

    lines = [
        "# Mutual-information tuning analysis",
        "",
        f"The unified record-level dataset contains {len(unified):,} rows: "
        f"{int(unified['analysis_type'].eq('calibration').sum()):,} null targets and "
        f"{int(unified['analysis_type'].eq('power').sum()):,} power targets.",
        "",
        "## Configuration comparison",
        "",
        frame_to_markdown(summary),
        "",
        "## Interpretation",
        "",
        f"- The highest mean marginal power among the {len(summary)} evaluated settings was "
        f"k={int(best.mi_neighbors)}, "
        f"steps={int(best.mcmc_steps)} ({best.mean_marginal_power:.3f}).",
        f"- Its null rejection rate at alpha=0.05 was {best.rejection_rate:.3f}.",
        f"- Each signal-strength cell has {iteration_description}; conditional power can still "
        "be based on fewer selected-signal targets.",
        "- MCMC steps tune conditional-inference mixing; MI neighbors primarily tune the screening rule. "
        "Cross-method conclusions should use matched seeds, inference settings, and Monte Carlo budgets.",
        *benchmark_lines,
        "",
        "## Figures",
        "",
        "- `null-calibration.png`: rejection rate with Wilson 95% intervals.",
        "- `power-curves.png`: signal selection, conditional power, and marginal power.",
        "- `mcmc-diagnostics.png`: acceptance and effective-sample-size distributions.",
        "- `null-pvalue-ecdf.png`: full pooled null p-value ECDFs.",
        "- `method-benchmark-power.png`: best tuned MI against the existing full-run method benchmarks.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = args.benchmark_dir
    if benchmark_dir is None:
        candidate = (
            input_dir.parent
            / "selection_method_validation"
            / "exact_set"
            / "mcmc_rank"
            / "shap_mutual_information"
            / "pooled_summary"
        )
        benchmark_dir = candidate if (candidate / "power_decisions.csv").is_file() else None
    benchmark = load_benchmark(benchmark_dir.resolve() if benchmark_dir is not None else None)

    calibration, power, unified = load_sweep(input_dir)
    calibration.to_csv(output_dir / "unified_calibration_decisions.csv", index=False)
    power.to_csv(output_dir / "unified_power_decisions.csv", index=False)
    unified.to_csv(output_dir / "unified_target_records.csv", index=False)
    plot_calibration(calibration, output_dir)
    plot_power(power, output_dir)
    plot_null_diagnostics(unified, output_dir)
    plot_null_ecdf(unified, output_dir)
    if benchmark is not None:
        plot_benchmark(power, benchmark, output_dir)
    write_report(calibration, power, unified, output_dir, benchmark)

    metadata = {
        "input_directory": str(input_dir),
        "configurations": int(calibration[["mi_neighbors", "mcmc_steps", "mcmc_replicates"]].drop_duplicates().shape[0]),
        "calibration_rows": int(len(calibration)),
        "power_rows": int(len(power)),
        "record_rows": int(len(unified)),
        "benchmark_directory": str(benchmark_dir.resolve()) if benchmark_dir is not None else None,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
