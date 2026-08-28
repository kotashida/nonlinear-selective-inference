"""Validate and pool sharded same-target versus exact-set experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.summarize_same_target_exact_set_experiment import (
    plot_summary,
    summarize,
)


EVENTS = ("same_target", "exact_set")
RESULT_FILES = {
    "null": ("p_value_results.csv",),
    "power": ("target_results.csv", "feature_results.csv"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--expected-shards", type=int, default=10)
    parser.add_argument("--expected-iterations", type=int, default=1000)
    parser.add_argument(
        "--expected-method",
        choices=("shap", "mutual_information", "marginal_screening"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing settings file: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _shard_dirs(input_dir: Path, expected_shards: int) -> list[Path]:
    directories = [input_dir / f"shard_{index}" for index in range(expected_shards)]
    missing = [path for path in directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing shard directories: " + ", ".join(path.name for path in missing)
        )
    return directories


def _validate_settings(records: list[tuple[Path, dict]], expected_iterations: int):
    ignored = {"n_iters", "iteration_start", "runtime_metadata", "run_preset"}
    reference_path, reference = records[0]
    reference_core = {key: value for key, value in reference.items() if key not in ignored}
    ranges = []
    for path, settings in records:
        candidate_core = {
            key: value for key, value in settings.items() if key not in ignored
        }
        if candidate_core != reference_core:
            mismatches = sorted(
                key
                for key in set(reference_core) | set(candidate_core)
                if reference_core.get(key) != candidate_core.get(key)
            )
            raise ValueError(
                f"{path} is incompatible with {reference_path}: "
                + ", ".join(mismatches)
            )
        start = settings.get("iteration_start")
        count = settings.get("n_iters")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or start < 0
            or count < 1
        ):
            raise ValueError(f"Invalid iteration range in {path}.")
        ranges.append((start, start + count))
    ranges.sort()
    cursor = 0
    for start, end in ranges:
        if start != cursor:
            raise ValueError(
                f"Iteration ranges overlap or have a gap at zero-based index {cursor}."
            )
        cursor = end
    if cursor != expected_iterations:
        raise ValueError(
            f"Pooled iteration count is {cursor}, expected {expected_iterations}."
        )
    return reference


def _pool_csvs(
    records: list[tuple[Path, dict]], filename: str, expected_iterations: int
) -> pd.DataFrame:
    frames = []
    for directory, settings in records:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard result: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Shard result is empty: {path}")
        expected_ids = set(
            range(
                int(settings["iteration_start"]) + 1,
                int(settings["iteration_start"]) + int(settings["n_iters"]) + 1,
            )
        )
        observed_ids = set(frame["iteration"].astype(int))
        if observed_ids != expected_ids:
            raise ValueError(f"Iteration IDs do not match settings in {path}.")
        frame.insert(0, "shard", directory.parent.name)
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    if set(pooled["iteration"].astype(int)) != set(range(1, expected_iterations + 1)):
        raise ValueError(f"{filename} does not cover every requested iteration.")
    return pooled.sort_values(["iteration", "selection_event"]).reset_index(drop=True)


def summarize_shards(
    input_dir: Path,
    output_dir: Path,
    *,
    expected_shards: int,
    expected_iterations: int,
    expected_method: str,
):
    shards = _shard_dirs(input_dir.resolve(), expected_shards)
    by_experiment = {}
    pooled = {}
    references = {}
    for experiment, filenames in RESULT_FILES.items():
        records = []
        for shard in shards:
            directory = shard / experiment
            settings = _load_json(directory / "settings.json")
            records.append((directory, settings))
        references[experiment] = _validate_settings(records, expected_iterations)
        by_experiment[experiment] = records
        for filename in filenames:
            pooled[(experiment, filename)] = _pool_csvs(
                records, filename, expected_iterations
            )

    if references["null"].get("selection_method") != expected_method or references[
        "power"
    ].get("selection_method") != expected_method:
        raise ValueError(
            f"Every shard must use {expected_method!r} feature selection."
        )
    if tuple(references["null"].get("selection_events", ())) != EVENTS or tuple(
        references["power"].get("selection_events", ())
    ) != EVENTS:
        raise ValueError("Every shard must compare same_target and exact_set.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_null = output_dir / "pooled_null"
    pooled_power = output_dir / "pooled_power"
    pooled_null.mkdir(exist_ok=True)
    pooled_power.mkdir(exist_ok=True)
    null_results = pooled[("null", "p_value_results.csv")]
    target_results = pooled[("power", "target_results.csv")]
    feature_results = pooled[("power", "feature_results.csv")]
    null_results.to_csv(pooled_null / "p_value_results.csv", index=False)
    target_results.to_csv(pooled_power / "target_results.csv", index=False)
    feature_results.to_csv(pooled_power / "feature_results.csv", index=False)

    summary, paired = summarize(
        pooled_null / "p_value_results.csv", pooled_power
    )
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    pd.DataFrame([paired]).to_csv(
        output_dir / "paired_comparison_summary.csv", index=False
    )
    plot_summary(summary, output_dir / "calibration_and_power.png")
    metadata = {
        "n_shards": expected_shards,
        "n_iterations_per_experiment": expected_iterations,
        "selection_method": expected_method,
        "selection_events": list(EVENTS),
        "null_settings": references["null"],
        "power_settings": references["power"],
    }
    with (output_dir / "pooled_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    return summary, paired


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir or args.input_dir / "pooled_summary"
    summary, paired = summarize_shards(
        args.input_dir,
        output_dir,
        expected_shards=args.expected_shards,
        expected_iterations=args.expected_iterations,
        expected_method=args.expected_method,
    )
    print(summary.to_string(index=False))
    print(pd.Series(paired).to_string())
    print(f"\nSaved pooled results to {output_dir.resolve()}")
    return summary, paired


if __name__ == "__main__":
    main()
