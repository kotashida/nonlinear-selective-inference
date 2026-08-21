"""Pool null-calibration shards and create tabular and graphical summaries."""

from __future__ import annotations

import argparse
import json
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

from si_shap.null_calibration import (
    _calibration_summary,
    _paired_rejection_comparisons,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "null_calibration_seed_123_shards"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing shard_* subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: INPUT_DIR/pooled_summary).",
    )
    parser.add_argument(
        "--alpha-levels",
        type=float,
        nargs="+",
        help="Override alpha levels found in the shard settings.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def _ordered_shard_dirs(input_dir: Path) -> list[Path]:
    def sort_key(path: Path):
        suffix = path.name.removeprefix("shard_")
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    directories = [
        path
        for path in input_dir.glob("shard_*")
        if path.is_dir() and (path / "p_value_results.csv").is_file()
    ]
    if not directories:
        raise FileNotFoundError(
            f"No shard_*/p_value_results.csv files found under {input_dir}."
        )
    return sorted(directories, key=sort_key)


def _load_settings(shard_dir: Path) -> dict:
    path = shard_dir / "settings.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing shard settings: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_shards(shard_dirs: list[Path], settings: list[dict]) -> None:
    if len(shard_dirs) != len(settings):
        raise ValueError("Every shard directory must have one settings record.")
    compatible_keys = (
        "n_samples",
        "n_features",
        "k_select",
        "sigma",
        "selection_events",
        "target_rule",
        "auxiliary_randomization_mode",
        "fixed_auxiliary_u",
        "multiplicity",
        "inference_method",
        "alpha_levels",
        "design_seed",
        "fixed_design",
        "selection_decimals",
        "pilot_iters",
        "pilot_samples",
        "final_batch_size",
        "max_final_samples",
        "min_denominator_ess",
        "min_tail_ess",
        "stop_when_ess_met",
        "variance_method",
        "rf_params",
        "selector_settings",
    )
    reference = settings[0]
    for shard_dir, candidate in zip(shard_dirs[1:], settings[1:]):
        mismatches = [
            key for key in compatible_keys if candidate.get(key) != reference.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"{shard_dir.name} has incompatible settings: "
                + ", ".join(mismatches)
            )

    seeds = [candidate.get("seed") for candidate in settings]
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("Every shard must record a nonnegative integer seed.")
    duplicate_seeds = sorted(
        seed for seed in set(seeds) if seeds.count(seed) > 1
    )
    if duplicate_seeds:
        raise ValueError(
            "Shard seeds overlap; duplicate seed(s): "
            + ", ".join(str(seed) for seed in duplicate_seeds)
        )

    runtime_metadata = [candidate.get("runtime_metadata") for candidate in settings]
    if any(value is not None for value in runtime_metadata):
        if any(value is None for value in runtime_metadata):
            raise ValueError("Runtime metadata is missing from one or more shards.")
        reproducibility_keys = ("python_version", "package_versions", "git_commit")
        reference_runtime = runtime_metadata[0]
        for shard_dir, candidate_runtime in zip(
            shard_dirs[1:], runtime_metadata[1:]
        ):
            mismatches = [
                key
                for key in reproducibility_keys
                if candidate_runtime.get(key) != reference_runtime.get(key)
            ]
            if mismatches:
                raise ValueError(
                    f"{shard_dir.name} has incompatible runtime metadata: "
                    + ", ".join(mismatches)
                )

    design_path = shard_dirs[0] / "fixed_design.npy"
    if not design_path.is_file():
        raise FileNotFoundError(f"Missing fixed design: {design_path}")
    reference_design = np.load(design_path)
    for shard_dir in shard_dirs[1:]:
        candidate_path = shard_dir / "fixed_design.npy"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Missing fixed design: {candidate_path}")
        if not np.array_equal(np.load(candidate_path), reference_design):
            raise ValueError(f"{shard_dir.name} does not use the shared fixed design.")


def _load_pooled_results(
    shard_dirs: list[Path], settings: list[dict]
) -> pd.DataFrame:
    frames = []
    offset = 0
    for shard_dir, shard_settings in zip(shard_dirs, settings):
        frame = pd.read_csv(shard_dir / "p_value_results.csv")
        required = {
            "iteration",
            "selection_event",
            "p_value",
            "denominator_ess",
            "tail_ess",
            "mc_se",
            "finite_sample_valid",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{shard_dir.name} is missing columns: {', '.join(sorted(missing))}"
            )
        if frame.empty:
            raise ValueError(f"{shard_dir.name} contains no p-value rows.")
        if frame["iteration"].isna().any():
            raise ValueError(f"{shard_dir.name} contains missing iteration IDs.")
        events = tuple(shard_settings["selection_events"])
        observed_events = set(frame["selection_event"])
        if observed_events != set(events):
            raise ValueError(
                f"{shard_dir.name} event rows do not match selection_events."
            )
        counts = frame.groupby(["iteration", "selection_event"], dropna=False).size()
        if not bool((counts == 1).all()):
            raise ValueError(
                f"{shard_dir.name} must contain exactly one row per iteration/event."
            )
        local_iterations = pd.unique(frame["iteration"])
        expected_rows = len(local_iterations) * len(events)
        if len(frame) != expected_rows:
            raise ValueError(
                f"{shard_dir.name} has missing iteration/event combinations."
            )
        if len(local_iterations) != shard_settings.get("n_iters"):
            raise ValueError(
                f"{shard_dir.name} iteration count does not match settings.json."
            )
        p_values = frame["p_value"].to_numpy(dtype=float)
        finite_p_values = p_values[np.isfinite(p_values)]
        if np.any(np.isinf(p_values)) or np.any(
            (finite_p_values < 0.0) | (finite_p_values > 1.0)
        ):
            raise ValueError(
                f"{shard_dir.name} contains p-values outside [0, 1] or infinity."
            )
        frame.insert(0, "shard", shard_dir.name)
        frame.insert(1, "iteration_in_shard", frame["iteration"])
        iteration_map = {
            iteration: offset + index
            for index, iteration in enumerate(local_iterations, start=1)
        }
        frame["iteration"] = frame["iteration"].map(iteration_map)
        offset += len(local_iterations)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _pooled_summary(
    frame: pd.DataFrame, events: list[str], alpha_levels: list[float]
) -> pd.DataFrame:
    n_iterations = frame["iteration"].nunique()
    summaries = [
        _calibration_summary(
            frame.loc[frame["selection_event"] == event],
            n_iters=n_iterations,
            alpha_levels=alpha_levels,
        )
        for event in events
    ]
    summary = pd.DataFrame(summaries)
    for column, threshold in (
        ("denominator_ess", "min_denominator_ess"),
        ("tail_ess", "min_tail_ess"),
    ):
        counts = (
            frame.assign(_below=frame[column] < frame.attrs[threshold])
            .groupby("selection_event", sort=False)["_below"]
            .agg(["sum", "mean"])
        )
        summary = summary.merge(
            counts.rename(
                columns={
                    "sum": f"n_below_{column}_threshold",
                    "mean": f"rate_below_{column}_threshold",
                }
            ),
            left_on="selection_event",
            right_index=True,
            how="left",
        )
    return summary


def _shard_summary(shard_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for shard_dir in shard_dirs:
        path = shard_dir / "calibration_summary.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame.insert(0, "shard", shard_dir.name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _event_values(frame: pd.DataFrame, event: str) -> np.ndarray:
    values = frame.loc[frame["selection_event"] == event, "p_value"].to_numpy(
        dtype=float
    )
    return values[np.isfinite(values)]


def _plot_p_value_histograms(frame, events, output_path, dpi):
    figure, axes = plt.subplots(
        1, len(events), figsize=(5 * len(events), 4), squeeze=False
    )
    for axis, event in zip(axes[0], events):
        axis.hist(
            _event_values(frame, event),
            bins=np.linspace(0.0, 1.0, 21),
            density=True,
            alpha=0.75,
        )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set(title=event, xlabel="Selective p-value", ylabel="Density", xlim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_uniform_qq(frame, events, output_path, dpi):
    figure, axes = plt.subplots(
        1, len(events), figsize=(5 * len(events), 4), squeeze=False
    )
    for axis, event in zip(axes[0], events):
        observed = np.sort(_event_values(frame, event))
        expected = (
            (np.arange(observed.size) + 0.5) / observed.size
            if observed.size
            else np.array([])
        )
        axis.scatter(expected, observed, s=8, alpha=0.7)
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.set(
            title=event,
            xlabel="Uniform quantile",
            ylabel="Observed quantile",
            xlim=(0, 1),
            ylim=(0, 1),
            aspect="equal",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_ecdf_difference(frame, events, output_path, dpi):
    grid = np.linspace(0.0, 1.0, 501)
    figure, axes = plt.subplots(
        1, len(events), figsize=(5 * len(events), 4), squeeze=False
    )
    for axis, event in zip(axes[0], events):
        values = np.sort(_event_values(frame, event))
        if values.size:
            difference = (
                np.searchsorted(values, grid, side="right") / values.size - grid
            )
            dkw = np.sqrt(np.log(1.0 / 0.05) / (2.0 * values.size))
            axis.plot(grid, difference, label="ECDF(p) - p")
            axis.axhline(
                dkw, color="gray", linestyle=":", label="95% upper band"
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(title=event, xlabel="p", ylabel="ECDF(p) - p", xlim=(0, 1))
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_rejection_rates(summary, events, alpha_levels, output_path, dpi):
    figure, axes = plt.subplots(
        1, len(alpha_levels), figsize=(5 * len(alpha_levels), 4), squeeze=False
    )
    positions = np.arange(len(events))
    indexed = summary.set_index("selection_event")
    for axis, alpha in zip(axes[0], alpha_levels):
        label = format(float(alpha), ".12g")
        rates = indexed.loc[events, f"rejection_rate_{label}"].to_numpy(dtype=float)
        lower = indexed.loc[events, f"rejection_ci_95_lower_{label}"].to_numpy(
            dtype=float
        )
        upper = indexed.loc[events, f"rejection_ci_95_upper_{label}"].to_numpy(
            dtype=float
        )
        axis.errorbar(
            positions,
            rates,
            yerr=np.vstack((rates - lower, upper - rates)),
            fmt="o",
            capsize=4,
        )
        axis.axhline(alpha, color="black", linestyle="--", label="Nominal level")
        axis.set(
            title=f"alpha = {alpha:g}",
            ylabel="Null rejection rate",
            xticks=positions,
            xticklabels=events,
        )
        axis.tick_params(axis="x", rotation=25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_ess(frame, events, settings, output_path, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for axis, column, threshold, title in (
        (
            axes[0, 0],
            "denominator_ess",
            settings["min_denominator_ess"],
            "Denominator ESS",
        ),
        (axes[0, 1], "tail_ess", settings["min_tail_ess"], "Tail ESS"),
    ):
        data = [
            frame.loc[frame["selection_event"] == event, column]
            .dropna()
            .to_numpy(dtype=float)
            for event in events
        ]
        axis.boxplot(data, tick_labels=events, showfliers=False)
        axis.axhline(threshold, color="red", linestyle="--", label="Diagnostic threshold")
        axis.set(title=title, ylabel="Effective sample size")
        axis.tick_params(axis="x", rotation=25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def summarize_shards(input_dir: Path, output_dir: Path, alpha_levels=None, dpi=180):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    shard_dirs = _ordered_shard_dirs(input_dir)
    settings_by_shard = [_load_settings(path) for path in shard_dirs]
    _validate_shards(shard_dirs, settings_by_shard)
    settings = settings_by_shard[0]
    events = list(settings["selection_events"])
    levels = sorted(
        float(value)
        for value in (
            settings["alpha_levels"] if alpha_levels is None else alpha_levels
        )
    )
    if not levels or any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError("Alpha levels must lie strictly between zero and one.")

    pooled = _load_pooled_results(shard_dirs, settings_by_shard)
    pooled.attrs["min_denominator_ess"] = float(settings["min_denominator_ess"])
    pooled.attrs["min_tail_ess"] = float(settings["min_tail_ess"])
    summary = _pooled_summary(pooled, events, levels)
    comparisons = _paired_rejection_comparisons(pooled, events, levels)
    per_shard = _shard_summary(shard_dirs)

    output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_dir / "pooled_p_value_results.csv", index=False)
    summary.to_csv(output_dir / "pooled_calibration_summary.csv", index=False)
    comparisons.to_csv(
        output_dir / "pooled_paired_rejection_comparisons.csv", index=False
    )
    if not per_shard.empty:
        per_shard.to_csv(output_dir / "per_shard_calibration_summary.csv", index=False)

    metadata = {
        "input_dir": str(input_dir),
        "n_shards": len(shard_dirs),
        "shards": [path.name for path in shard_dirs],
        "seeds": [value.get("seed") for value in settings_by_shard],
        "design_seed": settings.get("design_seed"),
        "n_iterations": int(pooled["iteration"].nunique()),
        "selection_events": events,
        "alpha_levels": levels,
        "shared_fixed_design": True,
        "auxiliary_randomization_mode": settings.get(
            "auxiliary_randomization_mode", "redrawn_each_iteration"
        ),
        "fixed_auxiliary_u": settings.get("fixed_auxiliary_u"),
    }
    with (output_dir / "pooled_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    _plot_p_value_histograms(
        pooled, events, output_dir / "pooled_p_value_histograms.png", dpi
    )
    _plot_uniform_qq(pooled, events, output_dir / "pooled_uniform_qq_plots.png", dpi)
    _plot_ecdf_difference(
        pooled, events, output_dir / "pooled_ecdf_difference.png", dpi
    )
    _plot_rejection_rates(
        summary, events, levels, output_dir / "pooled_rejection_rates.png", dpi
    )
    _plot_ess(pooled, events, settings, output_dir / "pooled_ess_diagnostics.png", dpi)
    return {
        "calibration_summary": summary,
        "p_value_results": pooled,
        "paired_rejection_comparisons": comparisons,
        "per_shard_calibration_summary": per_shard,
        "metadata": metadata,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.input_dir / "pooled_summary"
    )
    result = summarize_shards(
        args.input_dir,
        output_dir,
        alpha_levels=args.alpha_levels,
        dpi=args.dpi,
    )
    print(result["calibration_summary"].to_string(index=False))
    print(f"\nSaved pooled summaries to {output_dir.resolve()}")
    return result


if __name__ == "__main__":
    main()
