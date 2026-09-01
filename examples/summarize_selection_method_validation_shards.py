"""Pool independent validation shards and make global calibration/power decisions."""

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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.validate_selection_methods import (
    DESIGNS,
    PRIMARY_EVENT,
    _apply_calibration_decisions,
    _auxiliary_label,
    _calibration_rows,
    _attach_resolution_diagnostics,
    _power_row,
    _signal_feature,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-shards", type=int)
    return parser.parse_args(argv)


def _ordered_shards(input_dir: Path) -> list[Path]:
    def key(path: Path):
        suffix = path.name.removeprefix("shard_")
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    shards = [
        path
        for path in input_dir.glob("shard_*")
        if path.is_dir() and (path / "validation_settings.json").is_file()
    ]
    if not shards:
        raise FileNotFoundError(
            f"No completed shard_*/validation_settings.json files under {input_dir}."
        )
    return sorted(shards, key=key)


def _read_settings(shard: Path) -> dict:
    with (shard / "validation_settings.json").open(encoding="utf-8") as file:
        return json.load(file)


def _validate_settings(shards: list[Path], settings: list[dict]) -> None:
    compatible = (
        "preset",
        "methods",
        "designs",
        "max_final_samples",
        "inference_method",
        "spline_inference_method",
        "method_inference_methods",
        "mcmc_steps",
        "fixed_auxiliary_values",
        "signal_strengths",
        "signal_positions",
        "minimum_conditional_power",
        "minimum_calibration_iterations",
        "minimum_signal_targets",
        "primary_selection_event",
        "comparison_events",
        "alpha_levels",
        "design_seed",
        "rf_jobs",
        "cross_method_randomness",
    )
    reference = settings[0]
    for shard, candidate in zip(shards[1:], settings[1:]):
        mismatches = [
            name for name in compatible if candidate.get(name) != reference.get(name)
        ]
        if mismatches:
            raise ValueError(
                f"{shard.name} has incompatible settings: {', '.join(mismatches)}"
            )

    seeds = [item.get("seed") for item in settings]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("Every shard must have an integer root seed.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Shard root seeds must be unique.")

    reference_runtime = reference.get("runtime_metadata")
    runtime_keys = ("python_version", "package_versions", "git_commit", "git_dirty")
    for shard, candidate in zip(shards[1:], settings[1:]):
        candidate_runtime = candidate.get("runtime_metadata")
        mismatches = [
            name
            for name in runtime_keys
            if (candidate_runtime or {}).get(name)
            != (reference_runtime or {}).get(name)
        ]
        if mismatches:
            raise ValueError(
                f"{shard.name} has incompatible runtime metadata: "
                + ", ".join(mismatches)
            )


def _read_configuration_frames(
    shards: list[Path],
    relative_path: Path,
    *,
    expected_per_shard: list[int],
    primary_event: str = PRIMARY_EVENT,
) -> pd.DataFrame:
    frames = []
    for shard, expected in zip(shards, expected_per_shard):
        path = shard / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard result: {path}")
        frame = pd.read_csv(path)
        primary_count = int((frame["selection_event"] == primary_event).sum())
        if primary_count != expected:
            raise ValueError(
                f"{path} has {primary_count} {primary_event} rows; expected {expected}."
            )
        frame.insert(0, "validation_shard", shard.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _validate_shared_design(shards: list[Path], relative_directory: Path) -> None:
    paths = [shard / relative_directory / "fixed_design.npy" for shard in shards]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fixed null design: {missing[0]}")
    reference = np.load(paths[0])
    for path in paths[1:]:
        if not np.array_equal(np.load(path), reference):
            raise ValueError(f"{path} does not match the shared fixed null design.")


def summarize_shards(input_dir: Path, output_dir: Path, expected_shards=None):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    shards = _ordered_shards(input_dir)
    if expected_shards is not None and len(shards) != expected_shards:
        raise ValueError(
            f"Found {len(shards)} completed shards; expected {expected_shards}."
        )
    settings = [_read_settings(shard) for shard in shards]
    _validate_settings(shards, settings)
    reference = settings[0]
    primary_event = reference["primary_selection_event"]
    null_counts = [int(item["n_null_iters"]) for item in settings]
    power_counts = [int(item["n_power_iters"]) for item in settings]

    calibration_rows = []
    power_rows = []
    for method in reference["methods"]:
        for design_name in reference["designs"]:
            design = DESIGNS[design_name]
            for fixed_u in reference["fixed_auxiliary_values"]:
                regime = _auxiliary_label(fixed_u)
                relative_directory = Path(method) / design_name / f"null_{regime}"
                _validate_shared_design(shards, relative_directory)
                relative = relative_directory / "p_value_results.csv"
                frame = _read_configuration_frames(
                    shards,
                    relative,
                    expected_per_shard=null_counts,
                    primary_event=primary_event,
                )
                calibration_rows.extend(
                    _calibration_rows(
                        frame,
                        method=method,
                        design=design_name,
                        auxiliary_regime=regime,
                        primary_event=primary_event,
                    )
                )

            for position in reference["signal_positions"]:
                feature = _signal_feature(position, design["n_features"])
                for strength in reference["signal_strengths"]:
                    label = str(strength).replace(".", "p")
                    relative_directory = (
                        Path(method)
                        / design_name
                        / f"power_feature_{feature}_beta_{label}"
                    )
                    relative = relative_directory / "target_results.csv"
                    frame = _read_configuration_frames(
                        shards,
                        relative,
                        expected_per_shard=power_counts,
                        primary_event=primary_event,
                    )
                    feature_relative = relative_directory / "feature_results.csv"
                    if all((shard / feature_relative).is_file() for shard in shards):
                        feature_frame = _read_configuration_frames(
                            shards,
                            feature_relative,
                            expected_per_shard=power_counts,
                            primary_event=primary_event,
                        )
                        frame = _attach_resolution_diagnostics(
                            frame, feature_frame, alpha=0.05
                        )
                    power_rows.append(
                        _power_row(
                            frame,
                            method=method,
                            design=design_name,
                            signal_feature=feature,
                            signal_strength=strength,
                            minimum_conditional_power=reference[
                                "minimum_conditional_power"
                            ],
                            minimum_signal_targets=reference[
                                "minimum_signal_targets"
                            ],
                            primary_event=primary_event,
                        )
                    )

    calibration = _apply_calibration_decisions(
        pd.DataFrame.from_records(calibration_rows),
        minimum_iterations=reference["minimum_calibration_iterations"],
    )
    power = pd.DataFrame.from_records(power_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_dir / "calibration_decisions.csv", index=False)
    power.to_csv(output_dir / "power_decisions.csv", index=False)
    pooled_settings = {
        **reference,
        "seed": None,
        "shard_root_seeds": [item["seed"] for item in settings],
        "n_shards": len(shards),
        "n_null_iters": sum(null_counts),
        "n_power_iters": sum(power_counts),
        "n_null_iters_per_shard": null_counts,
        "n_power_iters_per_shard": power_counts,
    }
    with (output_dir / "validation_settings.json").open("w", encoding="utf-8") as file:
        json.dump(pooled_settings, file, indent=2, sort_keys=True)

    print(f"\nPooled calibration decisions ({primary_event}):")
    print(calibration.to_string(index=False))
    print(f"\nPooled power decisions ({primary_event}):")
    print(power.to_string(index=False))
    print(f"\nSaved pooled validation bundle to {output_dir}")
    return {"calibration": calibration, "power": power, "settings": pooled_settings}


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir or args.input_dir / "pooled_summary"
    return summarize_shards(args.input_dir, output_dir, args.expected_shards)


if __name__ == "__main__":
    main()
