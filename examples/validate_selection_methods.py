"""Validate calibration and power across all built-in selection methods.

The comprehensive preset is intentionally expensive. Runs are resumable: an
existing result directory is reused unless ``--overwrite`` is supplied. Use
``--methods`` and ``--designs`` with distinct output directories to distribute
independent jobs across workers.
"""

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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "selection_method_validation"
METHODS = (
    "shap",
    "mutual_information",
    "marginal_screening",
    "spline_screening",
)
DESIGNS = {
    "baseline": {
        "n_samples": 100,
        "n_features": 20,
        "k_select": 2,
        "feature_correlation": 0.0,
    },
    "correlated": {
        "n_samples": 100,
        "n_features": 20,
        "k_select": 2,
        "feature_correlation": 0.5,
    },
    "larger_sample": {
        "n_samples": 200,
        "n_features": 20,
        "k_select": 2,
        "feature_correlation": 0.0,
    },
    "high_dimensional": {
        "n_samples": 100,
        "n_features": 50,
        "k_select": 5,
        "feature_correlation": 0.0,
    },
}
PRESETS = {
    "smoke": {
        "n_null_iters": 3,
        "n_power_iters": 3,
        "max_final_samples": 40,
        "fixed_auxiliary_values": (None,),
        "signal_strengths": (0.30,),
        "designs": ("baseline",),
        "signal_positions": ("first",),
    },
    "pilot": {
        "n_null_iters": 200,
        "n_power_iters": 200,
        "max_final_samples": 800,
        "fixed_auxiliary_values": (None, 0.25, 0.75),
        "signal_strengths": (0.30, 0.50, 0.75),
        "designs": ("baseline",),
        "signal_positions": ("first", "middle"),
    },
    "comprehensive": {
        "n_null_iters": 1000,
        "n_power_iters": 1000,
        "max_final_samples": 2000,
        "fixed_auxiliary_values": (None, 0.25, 0.75),
        "signal_strengths": (0.15, 0.30, 0.50, 0.75, 1.00),
        "designs": (
            "baseline",
            "correlated",
            "larger_sample",
            "high_dimensional",
        ),
        "signal_positions": ("first", "middle"),
    },
}
PRIMARY_EVENT = "same_target"
COMPARISON_EVENTS = ("feature_inclusion", PRIMARY_EVENT)
SELECTION_EVENT_CHOICES = (
    "feature_inclusion",
    "exact_set",
    "same_target",
    "exact_ranking",
)
ALPHA_LEVELS = (0.01, 0.05, 0.10)
DEFAULT_MIN_CALIBRATION_ITERATIONS = 1000
DEFAULT_MIN_SIGNAL_TARGETS = 100


def _auxiliary_argument(value: str) -> float | None:
    if value.lower() in {"fresh", "random", "none"}:
        return None
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError(
            "auxiliary values must be 'fresh' or numbers in [0, 1)"
        )
    return parsed


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        return np.nan, np.nan
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(0.025, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(0.975, successes + 1, trials - successes))
    )
    return lower, upper


def _max_ecdf_excess(values) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    return float(np.max(np.arange(1, values.size + 1) / values.size - values))


def _auxiliary_label(value) -> str:
    return "fresh" if value is None else f"fixed_{value:.2f}".replace(".", "p")


def _signal_feature(position: str, n_features: int) -> int:
    return 0 if position == "first" else n_features // 2


def _seed(*parts: int) -> int:
    return int(
        np.random.SeedSequence(parts).generate_state(1, dtype=np.uint32)[0]
    )


def _load_or_run_null(
    output_dir: Path,
    *,
    overwrite: bool,
    method: str,
    design: dict,
    n_iters: int,
    fixed_auxiliary_u: float | None,
    max_final_samples: int,
    seed: int,
    design_seed: int | None = None,
    rf_jobs: int = 32,
    selection_events: tuple[str, ...] = COMPARISON_EVENTS,
):
    result_path = output_dir / "p_value_results.csv"
    if result_path.is_file() and not overwrite:
        return pd.read_csv(result_path)
    result = compare_selection_event_null_calibration(
        n_iters=n_iters,
        **design,
        selection_method=method,
        selection_events=selection_events,
        alpha_levels=ALPHA_LEVELS,
        fixed_auxiliary_u=fixed_auxiliary_u,
        seed=seed,
        design_seed=design_seed,
        rf_params={"n_jobs": rf_jobs} if method == "shap" else None,
        inference_method="conditional_mc",
        final_batch_size=min(80, max_final_samples),
        max_final_samples=max_final_samples,
    )
    result["settings"]["runtime_metadata"] = _runtime_metadata()
    _write_null_results(result, output_dir)
    return result["p_value_results"]


def _load_or_run_power(
    output_dir: Path,
    *,
    overwrite: bool,
    method: str,
    design: dict,
    n_iters: int,
    signal_feature: int,
    signal_strength: float,
    max_final_samples: int,
    seed: int,
    rf_jobs: int = 32,
    selection_events: tuple[str, ...] = COMPARISON_EVENTS,
):
    result_path = output_dir / "target_results.csv"
    if result_path.is_file() and not overwrite:
        return pd.read_csv(result_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    result = compare_selection_event_power(
        n_iters=n_iters,
        **design,
        signal_features=(signal_feature,),
        signal_strength=signal_strength,
        selection_method=method,
        rf_params={"n_jobs": rf_jobs} if method == "shap" else None,
        selection_events=selection_events,
        seed=seed,
        inference_method="conditional_mc",
        final_batch_size=min(80, max_final_samples),
        max_final_samples=max_final_samples,
    )
    result["settings"]["runtime_metadata"] = _runtime_metadata()
    _write_power_results(result, output_dir)
    return result["target_results"]


def _calibration_rows(
    frame, *, method, design, auxiliary_regime, primary_event=PRIMARY_EVENT
):
    primary = frame[frame["selection_event"] == primary_event]
    p_values = primary["p_value"].to_numpy(dtype=float)
    finite = p_values[np.isfinite(p_values)]
    rows = []
    for alpha in ALPHA_LEVELS:
        rejections = int(np.sum(finite < alpha))
        rows.append(
            {
                "selection_method": method,
                "design": design,
                "auxiliary_regime": auxiliary_regime,
                "selection_event": primary_event,
                "alpha": alpha,
                "n_iterations": len(primary),
                "n_failed": int(np.sum(~np.isfinite(p_values))),
                "n_rejected": rejections,
                "rejection_rate": (
                    rejections / finite.size if finite.size else np.nan
                ),
                "anti_conservative_binomial_p_value": (
                    float(
                        stats.binomtest(
                            rejections,
                            finite.size,
                            alpha,
                            alternative="greater",
                        ).pvalue
                    )
                    if finite.size
                    else np.nan
                ),
                "max_ecdf_excess": _max_ecdf_excess(finite),
            }
        )
    return rows


def _power_row(
    frame,
    *,
    method,
    design,
    signal_feature,
    signal_strength,
    minimum_conditional_power,
    minimum_signal_targets=DEFAULT_MIN_SIGNAL_TARGETS,
    primary_event=PRIMARY_EVENT,
):
    primary = frame[frame["selection_event"] == primary_event].copy()
    signal_targets = primary[primary["target_is_signal"].astype(bool)]
    finite_signal = signal_targets[np.isfinite(signal_targets["p_value"])]
    signal_successes = int(finite_signal["rejected"].astype(bool).sum())
    conditional_trials = len(finite_signal)
    conditional_power = (
        signal_successes / conditional_trials if conditional_trials else np.nan
    )
    conditional_lower, conditional_upper = _clopper_pearson(
        signal_successes, conditional_trials
    )
    marginal_successes = int(primary["successful_detection"].astype(bool).sum())
    marginal_lower, marginal_upper = _clopper_pearson(
        marginal_successes, len(primary)
    )
    return {
        "selection_method": method,
        "design": design,
        "selection_event": primary_event,
        "signal_feature": signal_feature,
        "signal_strength": signal_strength,
        "n_iterations": len(primary),
        "n_signal_targets": len(signal_targets),
        "n_failed_signal_targets": len(signal_targets) - conditional_trials,
        "target_signal_rate": (
            len(signal_targets) / len(primary) if len(primary) else np.nan
        ),
        "marginal_detection_power": (
            marginal_successes / len(primary) if len(primary) else np.nan
        ),
        "marginal_power_ci_95_lower": marginal_lower,
        "marginal_power_ci_95_upper": marginal_upper,
        "conditional_power_given_signal_target": conditional_power,
        "conditional_power_ci_95_lower": conditional_lower,
        "conditional_power_ci_95_upper": conditional_upper,
        "minimum_conditional_power": minimum_conditional_power,
        "minimum_signal_targets": minimum_signal_targets,
        "power_evidence_sufficient": bool(
            conditional_trials >= minimum_signal_targets
            and conditional_trials == len(signal_targets)
        ),
        "enough_power": bool(
            conditional_trials >= minimum_signal_targets
            and conditional_lower >= minimum_conditional_power
            and conditional_trials == len(signal_targets)
        ),
    }


def _apply_calibration_decisions(
    frame: pd.DataFrame,
    minimum_iterations: int = DEFAULT_MIN_CALIBRATION_ITERATIONS,
) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        return frame
    familywise_threshold = 0.05 / len(frame)
    n_runs = frame[
        ["selection_method", "design", "auxiliary_regime"]
    ].drop_duplicates().shape[0]
    dkw_delta = 0.05 / n_runs
    frame["familywise_binomial_threshold"] = familywise_threshold
    frame["simultaneous_dkw_bound"] = np.sqrt(
        np.log(1.0 / dkw_delta) / (2.0 * frame["n_iterations"])
    )
    frame["minimum_calibration_iterations"] = minimum_iterations
    frame["calibration_evidence_sufficient"] = (
        frame["n_iterations"] >= minimum_iterations
    )
    frame["calibration_pass"] = (
        frame["calibration_evidence_sufficient"]
        & (frame["n_failed"] == 0)
        & (frame["anti_conservative_binomial_p_value"] >= familywise_threshold)
        & (frame["max_ecdf_excess"] <= frame["simultaneous_dkw_bound"])
    )
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="pilot")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--designs", nargs="+", choices=DESIGNS)
    parser.add_argument("--n-null-iters", type=int)
    parser.add_argument("--n-power-iters", type=int)
    parser.add_argument("--max-final-samples", type=int)
    parser.add_argument(
        "--auxiliary-values",
        nargs="+",
        type=_auxiliary_argument,
        help="Override null regimes, using 'fresh' and/or fixed values in [0, 1).",
    )
    parser.add_argument("--signal-strengths", nargs="+", type=float)
    parser.add_argument(
        "--signal-positions", nargs="+", choices=("first", "middle")
    )
    parser.add_argument("--minimum-conditional-power", type=float, default=0.80)
    parser.add_argument(
        "--minimum-calibration-iterations",
        type=int,
        default=DEFAULT_MIN_CALIBRATION_ITERATIONS,
    )
    parser.add_argument(
        "--minimum-signal-targets", type=int, default=DEFAULT_MIN_SIGNAL_TARGETS
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--design-seed",
        type=int,
        help=(
            "Optional root seed for fixed null designs. Supplying the same value "
            "to every shard guarantees identical designs across shards."
        ),
    )
    parser.add_argument(
        "--rf-jobs",
        type=int,
        choices=range(1, 33),
        default=32,
        metavar="1..32",
        help="Random-forest workers per SHAP process (default: 32).",
    )
    parser.add_argument(
        "--selection-events",
        nargs="+",
        choices=SELECTION_EVENT_CHOICES,
        default=list(COMPARISON_EVENTS),
        help=(
            "conditioning events to compare; include same_target because it is "
            "the primary validation event"
        ),
    )
    parser.add_argument(
        "--primary-selection-event",
        choices=SELECTION_EVENT_CHOICES,
        default=PRIMARY_EVENT,
        help="event used for calibration and power threshold decisions",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selection_events = tuple(args.selection_events)
    if not selection_events or len(set(selection_events)) != len(selection_events):
        raise ValueError("--selection-events must contain at least one unique event.")
    primary_event = args.primary_selection_event
    if primary_event not in selection_events:
        raise ValueError(
            f"--selection-events must include the primary event {primary_event!r}."
        )
    preset = dict(PRESETS[args.preset])
    designs = tuple(args.designs or preset["designs"])
    n_null_iters = args.n_null_iters or preset["n_null_iters"]
    n_power_iters = args.n_power_iters or preset["n_power_iters"]
    max_final_samples = args.max_final_samples or preset["max_final_samples"]
    fixed_auxiliary_values = tuple(
        preset["fixed_auxiliary_values"]
        if args.auxiliary_values is None
        else args.auxiliary_values
    )
    signal_strengths = tuple(
        preset["signal_strengths"]
        if args.signal_strengths is None
        else args.signal_strengths
    )
    signal_positions = tuple(
        preset["signal_positions"]
        if args.signal_positions is None
        else args.signal_positions
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_rows = []
    power_rows = []
    for method_index, method in enumerate(args.methods):
        for design_index, design_name in enumerate(designs):
            design = DESIGNS[design_name]
            shared_design_seed = (
                None
                if args.design_seed is None
                else _seed(args.design_seed, tuple(DESIGNS).index(design_name))
            )
            for auxiliary_index, fixed_u in enumerate(
                fixed_auxiliary_values
            ):
                regime = _auxiliary_label(fixed_u)
                run_dir = output_dir / method / design_name / f"null_{regime}"
                frame = _load_or_run_null(
                    run_dir,
                    overwrite=args.overwrite,
                    method=method,
                    design=design,
                    n_iters=n_null_iters,
                    fixed_auxiliary_u=fixed_u,
                    max_final_samples=max_final_samples,
                    seed=_seed(args.seed, method_index, design_index, auxiliary_index),
                    design_seed=shared_design_seed,
                    rf_jobs=args.rf_jobs,
                    selection_events=selection_events,
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

            for position_index, position in enumerate(signal_positions):
                feature = _signal_feature(position, design["n_features"])
                for strength_index, strength in enumerate(
                    signal_strengths
                ):
                    strength_label = str(strength).replace(".", "p")
                    run_dir = (
                        output_dir
                        / method
                        / design_name
                        / f"power_feature_{feature}_beta_{strength_label}"
                    )
                    frame = _load_or_run_power(
                        run_dir,
                        overwrite=args.overwrite,
                        method=method,
                        design=design,
                        n_iters=n_power_iters,
                        signal_feature=feature,
                        signal_strength=strength,
                        max_final_samples=max_final_samples,
                        seed=_seed(
                            args.seed,
                            10_000,
                            method_index,
                            design_index,
                            position_index,
                            strength_index,
                        ),
                        rf_jobs=args.rf_jobs,
                        selection_events=selection_events,
                    )
                    power_rows.append(
                        _power_row(
                            frame,
                            method=method,
                            design=design_name,
                            signal_feature=feature,
                            signal_strength=strength,
                            minimum_conditional_power=args.minimum_conditional_power,
                            minimum_signal_targets=args.minimum_signal_targets,
                            primary_event=primary_event,
                        )
                    )

    calibration = _apply_calibration_decisions(
        pd.DataFrame.from_records(calibration_rows),
        minimum_iterations=args.minimum_calibration_iterations,
    )
    power = pd.DataFrame.from_records(power_rows)
    calibration.to_csv(output_dir / "calibration_decisions.csv", index=False)
    power.to_csv(output_dir / "power_decisions.csv", index=False)
    settings = {
        "preset": args.preset,
        "methods": args.methods,
        "designs": designs,
        "n_null_iters": n_null_iters,
        "n_power_iters": n_power_iters,
        "max_final_samples": max_final_samples,
        "fixed_auxiliary_values": fixed_auxiliary_values,
        "signal_strengths": signal_strengths,
        "signal_positions": signal_positions,
        "minimum_conditional_power": args.minimum_conditional_power,
        "minimum_calibration_iterations": args.minimum_calibration_iterations,
        "minimum_signal_targets": args.minimum_signal_targets,
        "primary_selection_event": primary_event,
        "comparison_events": selection_events,
        "alpha_levels": ALPHA_LEVELS,
        "seed": args.seed,
        "design_seed": args.design_seed,
        "rf_jobs": args.rf_jobs,
        "runtime_metadata": _runtime_metadata(),
    }
    with (output_dir / "validation_settings.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(settings, file, indent=2, sort_keys=True)

    print(f"\nCalibration decisions ({primary_event}):")
    print(calibration.to_string(index=False))
    print(f"\nPower decisions ({primary_event}):")
    print(power.to_string(index=False))
    print(f"\nSaved validation bundle to {output_dir}")
    return {"calibration": calibration, "power": power, "settings": settings}


if __name__ == "__main__":
    main()
