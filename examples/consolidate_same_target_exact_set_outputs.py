"""Consolidate completed same-target versus exact-set experiment outputs.

This migration combines the completed 1,000-iteration SHAP, spline-screening,
and marginal-screening results into one normalized directory. It deliberately
does not delete legacy inputs; removal is a separate, explicit filesystem step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples import plot_same_target_exact_set_method_shards as plots


METHODS = ("shap", "spline_screening", "marginal_screening")
EVENTS = ("same_target", "exact_set")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_result(frame: pd.DataFrame, path: Path, expected_iterations: int = 1000):
    required = {"iteration", "selection_event"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if set(frame["selection_event"]) != set(EVENTS):
        raise ValueError(f"{path} does not contain exactly {EVENTS}.")
    counts = frame.groupby("selection_event")["iteration"].nunique()
    if not (counts == expected_iterations).all():
        raise ValueError(f"{path} does not contain {expected_iterations} iterations per event.")
    if "failed" in frame and frame["failed"].astype(bool).any():
        raise ValueError(f"{path} contains failed results.")


def _add_shards(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "shard" not in result:
        shard = ((result["iteration"].astype(int) - 1) // 100).map(lambda value: f"shard_{value}")
        result.insert(0, "shard", shard)
    return result


def _copy_sharded_method(source_root: Path, output_root: Path, method: str):
    source = source_root / method / "pooled_summary"
    destination = output_root / method / "pooled_summary"
    null = pd.read_csv(source / "pooled_null" / "p_value_results.csv")
    targets = pd.read_csv(source / "pooled_power" / "target_results.csv")
    features = pd.read_csv(source / "pooled_power" / "feature_results.csv")
    for frame, path in (
        (null, source / "pooled_null" / "p_value_results.csv"),
        (targets, source / "pooled_power" / "target_results.csv"),
        (features, source / "pooled_power" / "feature_results.csv"),
    ):
        _validate_result(frame, path)
    destination.joinpath("pooled_null").mkdir(parents=True, exist_ok=True)
    destination.joinpath("pooled_power").mkdir(parents=True, exist_ok=True)
    null.to_csv(destination / "pooled_null" / "p_value_results.csv", index=False)
    targets.to_csv(destination / "pooled_power" / "target_results.csv", index=False)
    features.to_csv(destination / "pooled_power" / "feature_results.csv", index=False)
    shutil.copy2(source / "comparison_summary.csv", destination / "comparison_summary.csv")
    shutil.copy2(
        source / "paired_comparison_summary.csv",
        destination / "paired_comparison_summary.csv",
    )
    shutil.copy2(source / "pooled_metadata.json", destination / "pooled_metadata.json")
    fixed_design = source_root / method / "shard_0" / "null" / "fixed_design.npy"
    if not fixed_design.is_file():
        fixed_design = source / "pooled_null" / "fixed_design.npy"
    if fixed_design.is_file():
        shutil.copy2(fixed_design, destination / "pooled_null" / "fixed_design.npy")


def _copy_marginal_method(outputs_dir: Path, output_root: Path):
    method = "marginal_screening"
    null_source = outputs_dir / "same_target_vs_exact_set_marginal_null"
    power_source = outputs_dir / "same_target_vs_exact_set_marginal_power"
    summary_source = outputs_dir / "same_target_vs_exact_set_marginal_summary"
    destination = output_root / method / "pooled_summary"
    null = _add_shards(pd.read_csv(null_source / "p_value_results.csv"))
    targets = _add_shards(pd.read_csv(power_source / "target_results.csv"))
    features = _add_shards(pd.read_csv(power_source / "feature_results.csv"))
    for frame, path in (
        (null, null_source / "p_value_results.csv"),
        (targets, power_source / "target_results.csv"),
        (features, power_source / "feature_results.csv"),
    ):
        _validate_result(frame, path)
    destination.joinpath("pooled_null").mkdir(parents=True, exist_ok=True)
    destination.joinpath("pooled_power").mkdir(parents=True, exist_ok=True)
    null.to_csv(destination / "pooled_null" / "p_value_results.csv", index=False)
    targets.to_csv(destination / "pooled_power" / "target_results.csv", index=False)
    features.to_csv(destination / "pooled_power" / "feature_results.csv", index=False)
    shutil.copy2(
        summary_source / "comparison_summary.csv", destination / "comparison_summary.csv"
    )
    shutil.copy2(
        summary_source / "paired_comparison_summary.csv",
        destination / "paired_comparison_summary.csv",
    )
    shutil.copy2(null_source / "fixed_design.npy", destination / "pooled_null" / "fixed_design.npy")
    metadata = {
        "n_iterations_per_experiment": 1000,
        "n_shards": 10,
        "selection_method": method,
        "selection_events": list(EVENTS),
        "null_settings": _read_json(null_source / "settings.json"),
        "power_settings": _read_json(power_source / "settings.json"),
        "note": "Shard labels were derived deterministically in blocks of 100 iterations.",
    }
    with (destination / "pooled_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def _unified_tables(output_root: Path):
    summaries = []
    paired = []
    for method in METHODS:
        method_root = output_root / method / "pooled_summary"
        summary = pd.read_csv(method_root / "comparison_summary.csv")
        summary.insert(0, "selection_method", method)
        summaries.append(summary)
        method_paired = pd.read_csv(method_root / "paired_comparison_summary.csv")
        method_paired.insert(0, "selection_method", method)
        paired.append(method_paired)
    summary = pd.concat(summaries, ignore_index=True)
    paired_summary = pd.concat(paired, ignore_index=True)
    pooled = output_root / "pooled_summary"
    pooled.mkdir(parents=True, exist_ok=True)
    summary.to_csv(pooled / "unified_dataset.csv", index=False)
    paired_summary.to_csv(pooled / "unified_paired_comparisons.csv", index=False)
    return summary, paired_summary


def main(argv=None):
    args = parse_args(argv)
    outputs_dir = args.outputs_dir.resolve()
    output_root = (args.output_dir or outputs_dir / "same_target_vs_exact_set").resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")

    existing_source = outputs_dir / "same_target_vs_exact_set"
    _copy_sharded_method(existing_source, output_root, "shap")
    spline_source = outputs_dir / "same_target_vs_exact_set_spline_screening_shards"
    _copy_sharded_method(spline_source, output_root, "spline_screening")
    _copy_sharded_method(existing_source, output_root, "marginal_screening")
    summary, paired = _unified_tables(output_root)

    analysis_dir = output_root / "pooled_summary" / "analysis_plots"
    plots.main(
        [
            str(output_root),
            "--methods",
            *METHODS,
            "--output-dir",
            str(analysis_dir),
            "--dpi",
            str(args.dpi),
        ]
    )
    shutil.copy2(
        analysis_dir / "01_headline_rates.png",
        output_root / "pooled_summary" / "unified_plot.png",
    )
    manifest = {
        "selection_methods": list(METHODS),
        "selection_events": list(EVENTS),
        "n_iterations_per_experiment": 1000,
        "retained_legacy_sources": [
            "same_target_vs_exact_set (pooled SHAP and marginal-screening results)",
            "same_target_vs_exact_set_spline_screening_shards (pooled spline-screening result)",
        ],
        "excluded_obsolete_sources": {
            "same_target_vs_exact_set_mutual_information_benchmark": "one-iteration benchmark",
            "same_target_vs_exact_set_mutual_information_shards": "interrupted log-only run with no result CSVs",
            "same_target_vs_exact_set_pilot": "30-iteration pilot with signal_strength=0.35 and seed=123",
        },
        "unified_dataset_rows": len(summary),
        "unified_paired_comparison_rows": len(paired),
    }
    with (output_root / "consolidation_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    print(summary.to_string(index=False))
    print(f"\nConsolidated outputs saved to {output_root}")
    return summary, paired


if __name__ == "__main__":
    main()
