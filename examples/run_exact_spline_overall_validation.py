"""Re-run the slide's spline validation with an analytic exact-set event.

The source shard settings are reused verbatim for data, design, target, and
iteration seeds.  Only the conditioning event and inference engine change:
``same_target``/conditional MC becomes ``exact_set``/exact spline intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from examples.compare_selection_event_null_calibration import (
    _runtime_metadata,
    _write_results as _write_null_results,
)
from examples.compare_selection_event_power import (
    _write_results_to_directory as _write_power_results,
)
from si_shap import (
    compare_selection_event_null_calibration,
    compare_selection_event_power,
)


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "outputs"
    / "selection_method_validation_same_target_spline_screening_shards"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "selection_method_validation_exact_spline_shards"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _discover_tasks(source_root: Path, output_root: Path, limit_shards):
    shard_dirs = sorted(
        (path for path in source_root.glob("shard_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if limit_shards is not None:
        shard_dirs = shard_dirs[:limit_shards]
    tasks = []
    for shard_dir in shard_dirs:
        experiment_root = shard_dir / "spline_screening" / "baseline"
        for settings_path in sorted(experiment_root.glob("*/settings.json")):
            relative = settings_path.parent.relative_to(source_root)
            kind = "null" if settings_path.parent.name == "null_fresh" else "power"
            tasks.append(
                {
                    "kind": kind,
                    "source_dir": str(settings_path.parent),
                    "output_dir": str(output_root / relative),
                    "settings_path": str(settings_path),
                }
            )
    if not tasks:
        raise FileNotFoundError(f"No spline validation shards found under {source_root}")
    return tasks


def _run_task(task, overwrite=False):
    source_dir = Path(task["source_dir"])
    output_dir = Path(task["output_dir"])
    settings = json.loads(Path(task["settings_path"]).read_text(encoding="utf-8"))
    result_name = "p_value_results.csv" if task["kind"] == "null" else "target_results.csv"
    if (output_dir / result_name).is_file() and not overwrite:
        return str(output_dir)

    common = {
        "n_iters": settings["n_iters"],
        "n_samples": settings["n_samples"],
        "n_features": settings["n_features"],
        "k_select": settings["k_select"],
        "feature_correlation": settings["feature_correlation"],
        "selection_events": ("exact_set",),
        "selection_method": "spline_screening",
        "seed": settings["seed"],
        "iteration_start": settings.get("iteration_start", 0),
        "selection_decimals": settings.get("selection_decimals", 10),
        "inference_method": "exact_spline",
        "final_batch_size": settings.get("final_batch_size", 80),
        "max_final_samples": settings.get("max_final_samples", 800),
    }
    if task["kind"] == "null":
        result = compare_selection_event_null_calibration(
            **common,
            sigma=settings["sigma"],
            alpha_levels=tuple(settings["alpha_levels"]),
            fixed_auxiliary_u=settings.get("fixed_auxiliary_u"),
            design_seed=settings.get("design_seed"),
        )
        result["settings"]["runtime_metadata"] = _runtime_metadata()
        _write_null_results(result, output_dir)
    else:
        result = compare_selection_event_power(
            **common,
            signal_features=tuple(settings["signal_features"]),
            signal_strength=settings["signal_strength"],
            alpha=settings["alpha"],
            multiplicity=settings["multiplicity"],
        )
        result["settings"]["runtime_metadata"] = _runtime_metadata()
        _write_power_results(result, output_dir)
    return str(output_dir)


def _load_with_shard(paths, filename):
    frames = []
    for path in paths:
        frame = pd.read_csv(path / filename)
        settings_path = path / "settings.json"
        if settings_path.is_file():
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if "signal_strength" in settings and "signal_strength" not in frame:
                frame["signal_strength"] = settings["signal_strength"]
        frame.insert(0, "shard", path.parents[2].name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _summarize(source_root: Path, output_root: Path):
    analysis_dir = output_root / "pooled_summary"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    exact_null_paths = sorted(output_root.glob("shard_*/spline_screening/baseline/null_fresh"))
    exact_power_paths = sorted(output_root.glob("shard_*/spline_screening/baseline/power_*"))
    exact_null = _load_with_shard(exact_null_paths, "p_value_results.csv")
    exact_power = _load_with_shard(exact_power_paths, "target_results.csv")
    exact_null.to_csv(analysis_dir / "null_results_combined.csv", index=False)
    exact_power.to_csv(analysis_dir / "target_results_combined.csv", index=False)

    null_rows = []
    for alpha in (0.01, 0.05, 0.10):
        rejected = exact_null["p_value"].lt(alpha)
        null_rows.append(
            {
                "selection_event": "exact_set",
                "alpha": alpha,
                "n_iterations": len(exact_null),
                "n_rejected": int(rejected.sum()),
                "rejection_rate": float(rejected.mean()),
            }
        )
    null_summary = pd.DataFrame(null_rows)

    power_rows = []
    for strength, frame in exact_power.groupby("signal_strength", sort=True):
        signal = frame["target_is_signal"].astype(bool)
        rejected = frame["rejected"].astype(bool)
        power_rows.append(
            {
                "selection_event": "exact_set",
                "signal_strength": strength,
                "n_iterations": len(frame),
                "n_failed": int(frame["failed"].sum()),
                "n_signal_targets": int(signal.sum()),
                "target_signal_rate": float(signal.mean()),
                "signal_inclusion_rate": float(min(1.0, 2.0 * signal.mean())),
                "marginal_detection_power": float((signal & rejected).mean()),
                "conditional_power": float(rejected[signal].mean()),
            }
        )
    power_summary = pd.DataFrame(power_rows)
    null_summary.to_csv(analysis_dir / "null_calibration_summary.csv", index=False)
    power_summary.to_csv(analysis_dir / "power_summary.csv", index=False)

    paired_rows = []
    source_null_paths = [source_root / path.relative_to(output_root) for path in exact_null_paths]
    same_null = _load_with_shard(source_null_paths, "p_value_results.csv")
    null_pair = same_null.merge(
        exact_null,
        on=["shard", "iteration", "target_feature"],
        suffixes=("_same", "_exact"),
        validate="one_to_one",
    )
    for alpha in (0.01, 0.05, 0.10):
        same = null_pair["p_value_same"].lt(alpha)
        exact = null_pair["p_value_exact"].lt(alpha)
        discordant = int((same ^ exact).sum())
        paired_rows.append(
            {
                "context": "null",
                "level": alpha,
                "same_target_rate": float(same.mean()),
                "exact_set_rate": float(exact.mean()),
                "difference_same_minus_exact": float(same.mean() - exact.mean()),
                "mcnemar_exact_p_value": (
                    float(stats.binomtest(int((same & ~exact).sum()), discordant).pvalue)
                    if discordant
                    else 1.0
                ),
            }
        )
    for strength, exact_frame in exact_power.groupby("signal_strength", sort=True):
        exact_paths = [
            path for path in exact_power_paths
            if json.loads((path / "settings.json").read_text(encoding="utf-8"))[
                "signal_strength"
            ] == strength
        ]
        source_paths = [source_root / path.relative_to(output_root) for path in exact_paths]
        same_frame = _load_with_shard(source_paths, "target_results.csv")
        pair = same_frame.merge(
            exact_frame,
            on=["shard", "iteration", "target_feature", "target_is_signal"],
            suffixes=("_same", "_exact"),
            validate="one_to_one",
        )
        pair = pair[pair["target_is_signal"].astype(bool)]
        same = pair["rejected_same"].astype(bool)
        exact = pair["rejected_exact"].astype(bool)
        discordant = int((same ^ exact).sum())
        paired_rows.append(
            {
                "context": "conditional_power",
                "level": strength,
                "same_target_rate": float(same.mean()),
                "exact_set_rate": float(exact.mean()),
                "difference_same_minus_exact": float(same.mean() - exact.mean()),
                "mcnemar_exact_p_value": (
                    float(stats.binomtest(int((same & ~exact).sum()), discordant).pvalue)
                    if discordant
                    else 1.0
                ),
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        analysis_dir / "paired_comparison_summary.csv", index=False
    )
    print("\nExact-set null calibration:\n", null_summary.to_string(index=False))
    print("\nExact-set power:\n", power_summary.to_string(index=False))
    print("\nPaired comparison:\n", pd.DataFrame(paired_rows).to_string(index=False))


def main():
    args = parse_args()
    tasks = _discover_tasks(args.source_root, args.output_root, args.limit_shards)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(tasks)} exact-spline shard experiments with {args.max_workers} workers")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_run_task, task, args.overwrite) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            print(f"[{index}/{len(futures)}] {future.result()}")
    _summarize(args.source_root, args.output_root)


if __name__ == "__main__":
    main()
