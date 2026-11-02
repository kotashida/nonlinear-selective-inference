"""Benchmark the audited selected-five nonlinear selective-inference pipeline.

The defaults reproduce the statistical configuration recorded in
``selected_five_kde2p5_300_20260923``.  The only change is the data source:
the synthetic fixed design/response is replaced by paired row subsamples from
``benchmarks/table_dataset.pkl``, which was generated from ``dataset/`` by
``process.ipynb``.  Run ``check`` before a long run and use ``--repetitions``
only to create an explicitly labelled pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from si_shap import selective_inference
from si_shap.selection import LimeSelector, MutualInformationSelector


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "benchmarks" / "table_dataset.pkl"
RAW_DATA_FILES = (
    "abalone.csv",
    "Concrete_Data.xls",
    "ENB2012_data.xlsx",
    "gt_2015.csv",
    "Real estate valuation data set.xlsx",
    "winequality-red.csv",
    "winequality-white.csv",
)
REFERENCE = (
    ROOT
    / "outputs"
    / "selection_method_validation"
    / "exact_set"
    / "selected_five_kde2p5_300_20260923"
)
REFERENCE_SNAPSHOT = ROOT / "benchmarks" / "exact_pipeline_reference.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "benchmarks" / "selected_five_kde2p5_1000_exact_pipeline"
)

# The seven source files yield eight regression tasks because Energy Efficiency
# supplies both heating-load and cooling-load responses.
DATASET_ORDER = (
    "heating_load",
    "cooling_load",
    "gas_turbine",
    "red_wine",
    "white_wine",
    "abalone",
    "concrete",
    "housing",
)
METHOD_ORDER = (
    "shap_ais",
    "lime_40_ais",
    "spline_screening_ais",
    "spline_screening_exact",
    "mutual_information_kde_2p5_ais",
)
METHOD_LABELS = {
    "shap_ais": "SHAP (AIS)",
    "lime_40_ais": "LIME (40, AIS)",
    "spline_screening_ais": "Spline screening (AIS)",
    "spline_screening_exact": "Spline screening (exact)",
    "mutual_information_kde_2p5_ais": "Mutual information (KDE 2.5, AIS)",
}

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_REPETITIONS = 1_000
MASTER_SEED = 20260908
ALPHA = 0.05

COMMON = {
    "k_select": 2,
    "sigma": 1.0,
    "test_statistic": "spline",
    "selection_event": "exact_set",
    "target_rule": "uniform_from_selected",
    "multiplicity": "none",
    "selection_decimals": 10,
    "pilot_iters": 10,
    "pilot_samples": 320,
    "final_batch_size": 800,
    "max_final_samples": 51_200,
    "min_denominator_ess": 80.0,
    "min_tail_ess": 15.0,
    "mcmc_steps": 20,
}
RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": 5,
    "random_state": 42,
    "n_jobs": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_datasets(path: Path = DATA_FILE) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    missing = [name for name in DATASET_ORDER if name not in raw]
    if missing:
        raise RuntimeError(f"Benchmark input is missing datasets: {missing}")
    datasets = {}
    for name in DATASET_ORDER:
        X_raw, y_raw = raw[name]
        X = np.asarray(X_raw, dtype=float)
        y = np.asarray(y_raw, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
            raise ValueError(f"Invalid shapes for {name}: X={X.shape}, y={y.shape}")
        if X.shape[0] < DEFAULT_SAMPLE_SIZE or not (
            np.all(np.isfinite(X)) and np.all(np.isfinite(y))
        ):
            raise ValueError(f"Dataset {name} is too small or contains non-finite values.")
        datasets[name] = (X, y)
    return datasets


def method_arguments(method: str) -> dict:
    """Construct the selector and inference engine exactly as in the archive."""
    if method == "shap_ais":
        return {
            "selection_method": "shap",
            "estimator": RandomForestRegressor(**RF_PARAMS),
            "inference_method": "ais",
            "ais_strategy": "a_robust_draw_floor",
            "stop_when_ess_met": True,
        }
    if method == "lime_40_ais":
        return {
            "selector": LimeSelector(
                RandomForestRegressor(**RF_PARAMS),
                selection_decimals=10,
                random_state=42,
                num_samples=64,
                num_explanations=40,
                kernel_width=None,
                ridge_alpha=1.0,
            ),
            "inference_method": "ais",
            "ais_strategy": "a_robust_draw_floor",
            "stop_when_ess_met": True,
        }
    if method == "spline_screening_ais":
        return {
            "selection_method": "spline_screening",
            "inference_method": "ais",
            "ais_strategy": "a_robust_draw_floor",
            "stop_when_ess_met": True,
        }
    if method == "spline_screening_exact":
        return {
            "selection_method": "spline_screening",
            "inference_method": "exact_spline",
            # The archived exact arm records these necessarily inactive values.
            "ais_strategy": "baseline",
            "stop_when_ess_met": False,
        }
    if method == "mutual_information_kde_2p5_ais":
        return {
            "selector": MutualInformationSelector(
                random_state=42,
                n_neighbors=30,  # Recorded but inactive for the KDE estimator.
                estimator="kde",
                n_bins=5,  # Recorded but inactive for the KDE estimator.
                bandwidth=2.5,
            ),
            "inference_method": "ais",
            "ais_strategy": "a_robust_draw_floor",
            "stop_when_ess_met": True,
        }
    raise ValueError(f"Unknown benchmark method: {method}")


def seed_bundle(dataset_index: int, sample_size: int, repetition: int) -> dict:
    """Derive method-shared subset, target, and Monte Carlo seeds."""
    sequence = np.random.SeedSequence(
        [MASTER_SEED, int(dataset_index), int(sample_size), int(repetition)]
    )
    subset, target, inference = sequence.spawn(3)
    return {
        "subset_seed": int(subset.generate_state(1, dtype=np.uint32)[0]),
        "target_seed": int(target.generate_state(1, dtype=np.uint32)[0]),
        "ais_seed": int(inference.generate_state(1, dtype=np.uint32)[0]),
    }


def configuration(
    *, sample_size: int, repetitions: int, repetition_start: int = 0, datasets, methods
) -> dict:
    if REFERENCE.is_dir():
        archived = json.loads((REFERENCE / "selected_configuration.json").read_text())
        source_hashes = json.loads((REFERENCE / "source_hashes.json").read_text())
    else:
        snapshot = json.loads(REFERENCE_SNAPSHOT.read_text(encoding="utf-8"))
        archived = snapshot["selected_configuration"]
        source_hashes = snapshot["source_hashes"]
    relevant_sources = {
        name: source_hashes[name]
        for name in (
            "src/si_shap/api.py",
            "src/si_shap/inference.py",
            "src/si_shap/selection.py",
        )
    }
    return {
        "benchmark": "selected_five_kde2p5_1000_exact_pipeline",
        "reference_output": str(REFERENCE.relative_to(ROOT)).replace("\\", "/"),
        "reference_methods": archived["methods"],
        "datasets": list(datasets),
        "methods": list(methods),
        "sample_size": int(sample_size),
        "repetitions": int(repetitions),
        "repetition_start": int(repetition_start),
        "master_seed": MASTER_SEED,
        "alpha": ALPHA,
        "common_arguments": COMMON,
        "rf_params": RF_PARAMS,
        "method_specific": {
            "ais": {
                "ais_strategy": "a_robust_draw_floor",
                "stop_when_ess_met": True,
            },
            "exact_spline": {
                "ais_strategy": "baseline",
                "stop_when_ess_met": False,
            },
            "lime": {
                "random_state": 42,
                "num_samples": 64,
                "num_explanations": 40,
                "kernel_width": None,
                "ridge_alpha": 1.0,
            },
            "mutual_information": {
                "random_state": 42,
                "n_neighbors": 30,
                "estimator": "kde",
                "n_bins": 5,
                "bandwidth": 2.5,
            },
        },
        "input": {
            "path": str(DATA_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(DATA_FILE),
            "preprocessing": "benchmarks/process.ipynb; full-dataset standardization and documented jitter",
            "raw_source_sha256": {
                name: _sha256(ROOT / "benchmarks" / "dataset" / name)
                for name in RAW_DATA_FILES
            },
        },
        "archived_core_source_hashes": relevant_sources,
        "compatibility_changes": [
            {
                "change": "Replace the synthetic n=100, p=20 design and simulated response with a row subsample from each preprocessed benchmark dataset.",
                "reason": "Strictly necessary to evaluate the existing pipeline on the requested real datasets.",
            },
            {
                "change": "Use the benchmark dataset's native number of features.",
                "reason": "Strictly necessary because the supplied datasets have 7 to 11 features rather than 20.",
            },
        ],
        "non_changes": [
            "n=100",
            "k_select=2",
            "one uniformly randomized selected target",
            "exact-set conditioning",
            "spline test statistic and sigma=1 working scale",
            "all selector hyperparameters",
            "all active AIS proposal, ESS, stopping, and strategy settings",
        ],
    }


def audit_reference_sources(config: dict) -> None:
    drift = []
    for relative, expected in config["archived_core_source_hashes"].items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else "missing"
        if actual != expected:
            drift.append(f"{relative}: expected {expected}, found {actual}")
    if drift:
        raise RuntimeError(
            "Core pipeline source differs from the archived run:\n" + "\n".join(drift)
        )


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _result_row(result, *, dataset, sample_size, repetition, method, seeds, rows, elapsed):
    feature = result["feature_results"].iloc[0].to_dict()
    row = {
        "dataset": dataset,
        "sample_size": sample_size,
        "repetition": repetition,
        "method": method,
        "method_label": METHOD_LABELS[method],
        **seeds,
        "row_indices": json.dumps([int(value) for value in rows]),
        "observed_selected_features": json.dumps(
            [int(value) for value in result["observed_selected_features"]]
        ),
        "observed_target_feature": int(result["observed_target_feature"]),
        "runtime_seconds": elapsed,
        "error_type": "",
        "error_message": "",
    }
    row.update(feature)
    return row


def _error_row(error, *, dataset, sample_size, repetition, method, seeds, rows, elapsed):
    return {
        "dataset": dataset,
        "sample_size": sample_size,
        "repetition": repetition,
        "method": method,
        "method_label": METHOD_LABELS[method],
        **seeds,
        "row_indices": json.dumps([int(value) for value in rows]),
        "runtime_seconds": elapsed,
        "status": "exception",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "p_value": np.nan,
        "raw_selective_p_value": np.nan,
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "sample_size", "method", "method_label"]
    for values, group in results.groupby(keys, sort=False, dropna=False):
        p_values = pd.to_numeric(group["p_value"], errors="coerce")
        finite = np.isfinite(p_values)
        ais = group["method"].ne("spline_screening_exact")
        denominator_ess = pd.to_numeric(
            group.get("denominator_ess", pd.Series(np.nan, index=group.index)),
            errors="coerce",
        )
        tail_ess = pd.to_numeric(
            group.get("tail_ess", pd.Series(np.nan, index=group.index)), errors="coerce"
        )
        rows.append(
            {
                **dict(zip(keys, values)),
                "completed_repetitions": int(group["repetition"].nunique()),
                "n_results": len(group),
                "n_finite": int(finite.sum()),
                "n_failed": int((~finite).sum()),
                "selective_rejection_rate": float(np.mean(p_values[finite] < ALPHA))
                if finite.any()
                else np.nan,
                "failure_as_nonrejection_rate": float(
                    np.mean(np.where(finite, p_values < ALPHA, False))
                ),
                "median_runtime_seconds": float(group["runtime_seconds"].median()),
                "median_proposals": float(
                    pd.to_numeric(
                        group.get("proposals", pd.Series(np.nan, index=group.index)),
                        errors="coerce",
                    ).median()
                ),
                "ais_denominator_ess_target_met_rate": float(
                    np.mean(denominator_ess[ais] >= COMMON["min_denominator_ess"])
                )
                if ais.any()
                else np.nan,
                "ais_tail_ess_target_met_rate": float(
                    np.mean(tail_ess[ais] >= COMMON["min_tail_ess"])
                )
                if ais.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def audit_spline_pairing(results: pd.DataFrame) -> pd.DataFrame:
    subset = results[results["method"].isin(
        ("spline_screening_ais", "spline_screening_exact")
    )]
    keys = ["dataset", "sample_size", "repetition"]
    records = []
    for values, group in subset.groupby(keys, sort=False):
        by_method = group.set_index("method")
        complete = set(by_method.index) == {
            "spline_screening_ais", "spline_screening_exact"
        }
        matches = False
        if complete:
            left = by_method.loc["spline_screening_ais"]
            right = by_method.loc["spline_screening_exact"]
            matches = (
                left.get("observed_selected_features") == right.get("observed_selected_features")
                and left.get("observed_target_feature") == right.get("observed_target_feature")
                and np.isclose(float(left.get("t_obs", np.nan)), float(right.get("t_obs", np.nan)), rtol=0, atol=1e-12)
            )
        records.append({**dict(zip(keys, values)), "pair_complete": complete, "selection_target_statistic_match": matches})
    return pd.DataFrame(records)


def assigned_repetitions(repetition_start: int, repetitions: int, n_shards: int, shard_index: int):
    if n_shards < 1 or not 0 <= shard_index < n_shards:
        raise ValueError("Require n_shards >= 1 and 0 <= shard_index < n_shards.")
    return tuple(
        repetition
        for repetition in range(repetition_start, repetition_start + repetitions)
        if (repetition - repetition_start) % n_shards == shard_index
    )


def verify_results(results: pd.DataFrame, config: dict) -> dict:
    keys = ["dataset", "sample_size", "repetition", "method"]
    if results.duplicated(keys).any():
        raise RuntimeError("Duplicate benchmark result keys were found.")
    expected = {
        (dataset, int(config["sample_size"]), repetition, method)
        for dataset in config["datasets"]
        for repetition in range(
            int(config["repetition_start"]),
            int(config["repetition_start"]) + int(config["repetitions"]),
        )
        for method in config["methods"]
    }
    observed = {
        (str(row.dataset), int(row.sample_size), int(row.repetition), str(row.method))
        for row in results[keys].itertuples(index=False)
    }
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise RuntimeError(
            f"Benchmark key mismatch: {len(missing)} missing, {len(unexpected)} unexpected."
        )

    trial_keys = ["dataset", "sample_size", "repetition"]
    trial_rows = []
    for values, group in results.groupby(trial_keys, sort=False):
        row_sets = group["row_indices"].nunique(dropna=False)
        target_seeds = group["target_seed"].nunique(dropna=False)
        ais_seeds = group["ais_seed"].nunique(dropna=False)
        subset_seeds = group["subset_seed"].nunique(dropna=False)
        matched = row_sets == target_seeds == ais_seeds == subset_seeds == 1
        trial_rows.append({
            **dict(zip(trial_keys, values)),
            "method_count": len(group),
            "rows_and_seeds_match": bool(matched),
        })
    trial_audit = pd.DataFrame(trial_rows)
    if not trial_audit["rows_and_seeds_match"].all():
        raise RuntimeError("Methods were not matched on rows and random seeds.")

    spline_audit = audit_spline_pairing(results)
    spline_methods = {
        "spline_screening_ais", "spline_screening_exact"
    }.issubset(config["methods"])
    if spline_methods and not (
        spline_audit["pair_complete"].all()
        and spline_audit["selection_target_statistic_match"].all()
    ):
        raise RuntimeError("Spline AIS/exact pairing audit failed.")
    return {
        "expected_rows": len(expected),
        "observed_rows": len(results),
        "trial_audit": trial_audit,
        "spline_audit": spline_audit,
    }


def merge(args) -> None:
    shard_directories = [args.input_root / f"shard_{index}" for index in range(args.n_shards)]
    missing = [str(path) for path in shard_directories if not (path / "results.csv").is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} completed shard result files: {missing[:5]}")
    configs = [json.loads((path / "configuration.json").read_text(encoding="utf-8")) for path in shard_directories]
    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError("Shard configurations differ.")
    frames = [pd.read_csv(path / "results.csv") for path in shard_directories]
    results = pd.concat(frames, ignore_index=True)
    results = results.sort_values(
        ["dataset", "sample_size", "repetition", "method"], kind="stable"
    ).reset_index(drop=True)
    audit = verify_results(results, configs[0])
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "configuration.json", configs[0])
    _atomic_csv(args.output / "results.csv", results)
    _atomic_csv(args.output / "summary.csv", summarize(results))
    _atomic_csv(args.output / "trial_audit.csv", audit["trial_audit"])
    _atomic_csv(args.output / "spline_pairing_audit.csv", audit["spline_audit"])
    (args.output / "COMPLETED").write_text(
        f"Verified {audit['observed_rows']} matched benchmark tests across {args.n_shards} shards.\n",
        encoding="utf-8",
    )
    print((args.output / "COMPLETED").read_text(encoding="utf-8").strip())


def verify_output(args) -> None:
    config = json.loads((args.output / "configuration.json").read_text(encoding="utf-8"))
    audit_reference_sources(config)
    results = pd.read_csv(args.output / "results.csv")
    audit = verify_results(results, config)
    expected_summary = summarize(results)
    stored_summary = pd.read_csv(args.output / "summary.csv")
    if list(expected_summary.columns) != list(stored_summary.columns) or len(expected_summary) != len(stored_summary):
        raise RuntimeError("Stored summary schema or row count does not match recomputation.")
    print(f"Verified {audit['observed_rows']} benchmark tests and all matching audits.")


def run(args) -> None:
    datasets = load_datasets()
    selected_datasets = tuple(args.datasets or DATASET_ORDER)
    selected_methods = tuple(args.methods or METHOD_ORDER)
    unknown_datasets = sorted(set(selected_datasets) - set(DATASET_ORDER))
    unknown_methods = sorted(set(selected_methods) - set(METHOD_ORDER))
    if unknown_datasets or unknown_methods:
        raise ValueError(f"Unknown datasets={unknown_datasets}; methods={unknown_methods}")
    if args.sample_size < 5:
        raise ValueError("sample-size must exceed 4 for the three-dimensional spline basis.")
    for name in selected_datasets:
        if args.sample_size > datasets[name][0].shape[0]:
            raise ValueError(f"sample-size exceeds the row count for {name}.")

    config = configuration(
        sample_size=args.sample_size,
        repetitions=args.repetitions,
        repetition_start=args.repetition_start,
        datasets=selected_datasets,
        methods=selected_methods,
    )
    audit_reference_sources(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != _jsonable(config):
            raise RuntimeError("Output directory contains a different benchmark configuration.")
    else:
        _atomic_json(config_path, config)

    result_path = output / "results.csv"
    if result_path.exists():
        result_rows = pd.read_csv(result_path).to_dict("records")
    else:
        result_rows = []
    completed = {
        (str(row["dataset"]), int(row["sample_size"]), int(row["repetition"]), str(row["method"]))
        for row in result_rows
    }

    repetitions = assigned_repetitions(
        args.repetition_start, args.repetitions, args.n_shards, args.shard_index
    )
    _atomic_json(
        output / "execution.json",
        {
            "n_shards": args.n_shards,
            "shard_index": args.shard_index,
            "assigned_repetitions": repetitions,
        },
    )
    for dataset_index, name in enumerate(DATASET_ORDER):
        if name not in selected_datasets:
            continue
        X_full, y_full = datasets[name]
        for repetition in repetitions:
            seeds = seed_bundle(dataset_index, args.sample_size, repetition)
            row_rng = np.random.default_rng(seeds["subset_seed"])
            row_indices = np.sort(
                row_rng.choice(len(y_full), size=args.sample_size, replace=False)
            )
            X = X_full[row_indices]
            y = y_full[row_indices]
            for method in selected_methods:
                key = (name, args.sample_size, repetition, method)
                if key in completed:
                    continue
                started = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="Exact-set conditioning can make the event rare"
                        )
                        result = selective_inference(
                            X,
                            y,
                            **COMMON,
                            target_seed=seeds["target_seed"],
                            ais_seed=seeds["ais_seed"],
                            **method_arguments(method),
                        )
                    row = _result_row(
                        result,
                        dataset=name,
                        sample_size=args.sample_size,
                        repetition=repetition,
                        method=method,
                        seeds=seeds,
                        rows=row_indices,
                        elapsed=time.perf_counter() - started,
                    )
                except Exception as error:  # Preserve failures in the benchmark denominator.
                    row = _error_row(
                        error,
                        dataset=name,
                        sample_size=args.sample_size,
                        repetition=repetition,
                        method=method,
                        seeds=seeds,
                        rows=row_indices,
                        elapsed=time.perf_counter() - started,
                    )
                result_rows.append(row)
                completed.add(key)
                frame = pd.DataFrame(result_rows)
                _atomic_csv(result_path, frame)
                _atomic_csv(output / "summary.csv", summarize(frame))
                _atomic_csv(output / "spline_pairing_audit.csv", audit_spline_pairing(frame))
                print(
                    f"{name} n={args.sample_size} repetition={repetition} {method}: "
                    f"{row.get('status', 'ok')}",
                    flush=True,
                )


def check(args) -> None:
    datasets = load_datasets()
    config = configuration(
        sample_size=args.sample_size,
        repetitions=args.repetitions,
        repetition_start=0,
        datasets=DATASET_ORDER,
        methods=METHOD_ORDER,
    )
    audit_reference_sources(config)
    report = {
        "configuration": config,
        "dataset_shapes": {
            name: {"X": list(X.shape), "y": list(y.shape)}
            for name, (X, y) in datasets.items()
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "status": "ready",
    }
    print(json.dumps(_jsonable(report), indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    check_parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    run_parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    run_parser.add_argument("--repetition-start", type=int, default=0)
    run_parser.add_argument("--n-shards", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER)
    run_parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input-root", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--n-shards", type=int, default=32)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "repetitions") and args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if getattr(args, "repetition_start", 0) < 0:
        parser.error("--repetition-start must be nonnegative")
    if args.action == "run" and (
        args.n_shards < 1 or not 0 <= args.shard_index < args.n_shards
    ):
        parser.error("require --n-shards >= 1 and 0 <= --shard-index < --n-shards")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.action == "check":
        check(args)
    elif args.action == "run":
        run(args)
    elif args.action == "merge":
        merge(args)
    else:
        verify_output(args)


if __name__ == "__main__":
    main()
