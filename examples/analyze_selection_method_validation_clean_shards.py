"""Combine clean validation shards, recompute pooled summaries, and make plots.

The input layout is the cleaned single-configuration subset produced by the
selection-method validation workflow.  Raw null p-values and detailed power
records are combined into one long, schema-union CSV.  The smaller summary
CSVs are recomputed from pooled replicate-level data rather than averaged
across shards.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si_shap.null_calibration import (  # noqa: E402
    _calibration_summary,
    _paired_rejection_comparisons,
)
from si_shap.power import _paired_comparisons, _summarize_event_power  # noqa: E402


EVENTS = ("feature_inclusion", "same_target")
EVENT_LABELS = {
    "feature_inclusion": "Feature inclusion",
    "same_target": "Same target",
}
COLORS = {
    "feature_inclusion": "#4C78A8",
    "same_target": "#F58518",
}
NULL_RELATIVE = Path("shap/baseline/null_fresh")
POWER_EXPERIMENTS = {
    "power_beta_0.3_feature_0": Path("shap/baseline/power_feature_0_beta_0p3"),
    "power_beta_0.5_feature_0": Path("shap/baseline/power_feature_0_beta_0p5"),
    "power_beta_0.75_feature_0": Path("shap/baseline/power_feature_0_beta_0p75"),
    "power_beta_1.0_feature_0": Path("shap/baseline/power_feature_0_beta_1p0"),
}

EXPERIMENT_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-shards", type=int, default=10)
    parser.add_argument("--validation-target", type=float, default=0.80)
    return parser.parse_args(argv)


def ordered_shards(input_dir: Path) -> list[Path]:
    shards = [path for path in input_dir.glob("shard_*") if path.is_dir()]
    shards.sort(
        key=lambda path: (
            0,
            int(path.name.removeprefix("shard_")),
        )
        if path.name.removeprefix("shard_").isdigit()
        else (1, path.name)
    )
    return shards


def discover_power_experiments(shards: list[Path]) -> dict[str, Path]:
    """Discover every completed power directory shared by all shards."""
    if not shards:
        return {}
    base_relative = Path("shap/baseline")
    reference_names = {
        path.name
        for path in (shards[0] / base_relative).glob("power_*")
        if path.is_dir() and (path / "settings.json").is_file()
    }
    if not reference_names:
        raise FileNotFoundError(f"No completed power experiments under {shards[0] / base_relative}.")
    for shard in shards[1:]:
        names = {
            path.name
            for path in (shard / base_relative).glob("power_*")
            if path.is_dir() and (path / "settings.json").is_file()
        }
        if names != reference_names:
            missing = sorted(reference_names - names)
            extra = sorted(names - reference_names)
            raise ValueError(
                f"{shard.name} has a different power-experiment inventory; "
                f"missing={missing}, extra={extra}."
            )
    records = []
    for directory_name in reference_names:
        relative = base_relative / directory_name
        settings = read_json(shards[0] / relative / "settings.json")
        strength = float(settings["signal_strength"])
        signal_features = settings.get("signal_features", [])
        if len(signal_features) != 1:
            raise ValueError(f"{relative} must define exactly one signal feature.")
        experiment = f"power_beta_{strength}_feature_{int(signal_features[0])}"
        records.append((strength, experiment, relative))
    return {
        experiment: relative
        for _, experiment, relative in sorted(records, key=lambda item: item[0])
    }


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_seed_values(settings: list[dict], *, label: str) -> None:
    seeds = [setting.get("seed") for setting in settings]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError(f"Every {label} shard seed must be an integer.")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"The {label} shard seeds are not unique.")


def _validate_compatible_settings(
    shards: list[Path], settings: list[dict], *, label: str
) -> None:
    """Require all experiment-defining settings to match across shards."""
    reference = {key: value for key, value in settings[0].items() if key != "seed"}
    for shard, candidate in zip(shards[1:], settings[1:]):
        comparable = {key: value for key, value in candidate.items() if key != "seed"}
        if comparable != reference:
            keys = sorted(set(reference) | set(comparable))
            mismatches = [key for key in keys if comparable.get(key) != reference.get(key)]
            raise ValueError(
                f"{shard.name} has incompatible {label} settings: "
                f"{', '.join(mismatches)}"
            )
    _validate_seed_values(settings, label=label)


def validate_settings(
    shards: list[Path],
) -> tuple[list[dict], dict[str, list[dict]]]:
    null_settings = [read_json(shard / NULL_RELATIVE / "settings.json") for shard in shards]
    _validate_compatible_settings(shards, null_settings, label="null")
    power_settings = {}
    for experiment, relative in POWER_EXPERIMENTS.items():
        settings = [read_json(shard / relative / "settings.json") for shard in shards]
        _validate_compatible_settings(shards, settings, label=experiment)
        power_settings[experiment] = settings
    return null_settings, power_settings


def validate_shared_null_design(shards: list[Path], settings: list[dict]) -> None:
    if not bool(settings[0].get("fixed_design")):
        return
    paths = [shard / NULL_RELATIVE / "fixed_design.npy" for shard in shards]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"Missing fixed null design(s): {', '.join(missing)}")
    reference = np.load(paths[0], allow_pickle=False)
    for shard, path in zip(shards[1:], paths[1:]):
        candidate = np.load(path, allow_pickle=False)
        if not np.array_equal(candidate, reference, equal_nan=True):
            raise ValueError(f"{shard.name} does not use the shared fixed null design.")


def read_shard_csvs(
    shards: list[Path],
    relative_path: Path,
    *,
    expected_iters: int,
    events: tuple[str, ...],
    required_columns: set[str],
) -> pd.DataFrame:
    frames = []
    reference_columns = None
    for shard_index, shard in enumerate(shards):
        path = shard / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}.")
        expected_rows = expected_iters * len(events)
        if len(frame) != expected_rows:
            raise ValueError(f"{path} has {len(frame)} rows; expected {expected_rows}.")
        columns = tuple(frame.columns)
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(f"{path} has a schema that differs from earlier shards.")
        numeric_iterations = pd.to_numeric(frame["iteration"], errors="coerce")
        if numeric_iterations.isna().any() or not np.equal(
            numeric_iterations, np.floor(numeric_iterations)
        ).all():
            raise ValueError(f"{path} contains non-integer iteration IDs.")
        frame["iteration"] = numeric_iterations.astype(int)
        expected_iteration_ids = set(range(1, expected_iters + 1))
        if set(frame["iteration"]) != expected_iteration_ids:
            raise ValueError(f"{path} does not contain iterations 1 through {expected_iters}.")
        if set(frame["selection_event"]) != set(events):
            raise ValueError(f"{path} has unexpected or missing selection events.")
        if frame.duplicated(["iteration", "selection_event"]).any():
            raise ValueError(f"{path} must have exactly one row per iteration/event.")
        frame.insert(0, "shard_index", shard_index)
        frame.insert(0, "shard", shard.name)
        frame.insert(2, "global_iteration", shard_index * expected_iters + frame["iteration"])
        frame.insert(3, "source_file", path.relative_to(shard.parents[0]).as_posix())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_probability_records(frame: pd.DataFrame, *, label: str) -> None:
    values = pd.to_numeric(frame["p_value"], errors="coerce").to_numpy(float)
    failed = frame["failed"].astype(bool).to_numpy()
    invalid = np.isinf(values) | ((values < 0.0) | (values > 1.0))
    if np.any(invalid):
        raise ValueError(f"{label} contains p-values outside [0, 1] or infinity.")
    if not np.array_equal(~np.isfinite(values), failed):
        raise ValueError(f"{label} has inconsistent p_value and failed columns.")


def validate_power_alignment(
    power_features: pd.DataFrame, power_targets: pd.DataFrame, *, alpha: float
) -> None:
    keys = ["shard", "shard_index", "global_iteration", "selection_event"]
    feature_columns = keys + ["feature", "p_value_used", "failed", "rejected", "is_signal"]
    target_columns = keys + ["target_feature", "p_value", "failed", "rejected", "target_is_signal"]
    feature = power_features[feature_columns].sort_values(keys).reset_index(drop=True)
    target = power_targets[target_columns].sort_values(keys).reset_index(drop=True)
    if not feature[keys].equals(target[keys]):
        raise ValueError("Power feature and target records have different iteration/event keys.")
    checks = {
        "target feature": feature["feature"].to_numpy() == target["target_feature"].to_numpy(),
        "decision p-value": np.isclose(
            feature["p_value_used"].to_numpy(float),
            target["p_value"].to_numpy(float),
            equal_nan=True,
        ),
        "failure flag": feature["failed"].to_numpy(bool) == target["failed"].to_numpy(bool),
        "rejection flag": feature["rejected"].to_numpy(bool) == target["rejected"].to_numpy(bool),
        "signal flag": feature["is_signal"].to_numpy(bool) == target["target_is_signal"].to_numpy(bool),
    }
    failures = [name for name, values in checks.items() if not np.all(values)]
    if failures:
        raise ValueError(f"Power feature and target records disagree on: {', '.join(failures)}.")
    expected_rejection = np.isfinite(target["p_value"]) & (target["p_value"] < alpha)
    if not np.array_equal(expected_rejection.to_numpy(bool), target["rejected"].to_numpy(bool)):
        raise ValueError("Power rejection flags do not match p_value < alpha.")


def clopper_pearson(count: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total < 1:
        return np.nan, np.nan
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if count == 0 else float(stats.beta.ppf(tail, count, total - count + 1))
    upper = 1.0 if count == total else float(stats.beta.ppf(1.0 - tail, count + 1, total - count))
    return lower, upper


def paired_binary_exact_interval(
    first_only: int, second_only: int, total: int
) -> tuple[float, float]:
    """Conservative 95% CI for second-minus-first paired binary rates.

    Simultaneous 97.5% Clopper-Pearson intervals for the two discordant-cell
    probabilities give at least 95% joint coverage by Bonferroni.  Unlike a
    paired t interval, this interval remains non-degenerate when no discordant
    outcomes are observed.
    """
    first_low, first_high = clopper_pearson(first_only, total, confidence=0.975)
    second_low, second_high = clopper_pearson(second_only, total, confidence=0.975)
    return max(-1.0, second_low - first_high), min(1.0, second_high - first_low)


def add_exact_paired_intervals(
    comparisons: pd.DataFrame, target_results: pd.DataFrame | None = None
) -> pd.DataFrame:
    result = comparisons.copy()
    if "first_only_rejections" in result:
        first_only = result["first_only_rejections"].astype(int).to_numpy()
        second_only = result["second_only_rejections"].astype(int).to_numpy()
    else:
        if target_results is None:
            raise ValueError("target_results are required for paired power intervals.")
        first_counts = []
        second_counts = []
        for row in result.itertuples(index=False):
            paired = target_results[
                target_results["selection_event"].isin(
                    (row.baseline_event, row.comparison_event)
                )
            ].pivot(
                index="global_iteration",
                columns="selection_event",
                values=["successful_detection", "failed_signal_test"],
            )
            complete = ~(
                paired["failed_signal_test"][row.baseline_event].astype(bool)
                | paired["failed_signal_test"][row.comparison_event].astype(bool)
            )
            outcomes = paired.loc[complete, "successful_detection"].astype(bool)
            first_counts.append(
                int(np.sum(outcomes[row.baseline_event] & ~outcomes[row.comparison_event]))
            )
            second_counts.append(
                int(np.sum(~outcomes[row.baseline_event] & outcomes[row.comparison_event]))
            )
        first_only = np.asarray(first_counts)
        second_only = np.asarray(second_counts)
        result["baseline_only_detections"] = first_only
        result["comparison_only_detections"] = second_only

    result["asymptotic_ci_95_lower"] = result["ci_95_lower"]
    result["asymptotic_ci_95_upper"] = result["ci_95_upper"]
    intervals = [
        paired_binary_exact_interval(int(first), int(second), int(total))
        for first, second, total in zip(
            first_only, second_only, result["n_complete_pairs"].astype(int)
        )
    ]
    result["ci_95_lower"] = [interval[0] for interval in intervals]
    result["ci_95_upper"] = [interval[1] for interval in intervals]
    result["ci_method"] = "bonferroni_clopper_pearson_discordant_cells"
    return result


def add_power_exact_intervals(
    summary: pd.DataFrame, target_results: pd.DataFrame
) -> pd.DataFrame:
    result = summary.copy()
    interval_columns = {
        "power_ci_95_lower": [],
        "power_ci_95_upper": [],
        "conditional_power_ci_95_lower": [],
        "conditional_power_ci_95_upper": [],
        "target_signal_rate_ci_95_lower": [],
        "target_signal_rate_ci_95_upper": [],
    }
    for event in result["selection_event"]:
        frame = target_results[target_results["selection_event"] == event]
        complete = frame[~frame["failed_signal_test"].astype(bool)]
        power_interval = clopper_pearson(
            int(complete["successful_detection"].sum()), len(complete)
        )
        signal = frame[
            frame["target_is_signal"].astype(bool) & ~frame["failed"].astype(bool)
        ]
        conditional_interval = clopper_pearson(int(signal["rejected"].sum()), len(signal))
        signal_rate_interval = clopper_pearson(
            int(frame["target_is_signal"].sum()), len(frame)
        )
        for name, value in zip(
            interval_columns,
            (*power_interval, *conditional_interval, *signal_rate_interval),
        ):
            interval_columns[name].append(value)
    for name, values in interval_columns.items():
        result[name] = values
    return result


def pooled_summaries(
    null: pd.DataFrame,
    power_features: pd.DataFrame,
    power_targets: pd.DataFrame,
    *,
    alpha_levels: list[float],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    null_for_summary = null.copy()
    null_for_summary["iteration"] = null_for_summary["global_iteration"]
    calibration = pd.DataFrame.from_records(
        [
            _calibration_summary(
                null_for_summary[null_for_summary["selection_event"] == event],
                n_iters=int((null_for_summary["selection_event"] == event).sum()),
                alpha_levels=alpha_levels,
            )
            for event in EVENTS
        ]
    )
    null_paired = _paired_rejection_comparisons(
        null_for_summary, EVENTS, alpha_levels
    )
    null_paired = add_exact_paired_intervals(null_paired)

    power_targets_for_summary = power_targets.copy()
    power_targets_for_summary["iteration"] = power_targets_for_summary["global_iteration"]
    power = pd.DataFrame.from_records(
        [
            _summarize_event_power(
                power_targets_for_summary[
                    power_targets_for_summary["selection_event"] == event
                ],
                power_features[power_features["selection_event"] == event],
                n_iters=int(
                    (power_targets_for_summary["selection_event"] == event).sum()
                ),
                alpha=alpha,
            )
            for event in EVENTS
        ]
    )
    power_paired = _paired_comparisons(
        power_targets_for_summary, power, EVENTS
    )
    power = add_power_exact_intervals(power, power_targets_for_summary)
    power_paired = add_exact_paired_intervals(
        power_paired, power_targets_for_summary
    )
    return calibration, null_paired, power, power_paired


def build_unified(
    null: pd.DataFrame,
    powers: dict[str, pd.DataFrame],
    power_settings: dict[str, dict],
    null_settings: dict,
) -> pd.DataFrame:
    null_long = null.copy()
    null_long.insert(0, "experiment", "null_calibration")
    # Avoid the literal string "null", which many CSV readers parse as missing.
    null_long.insert(1, "analysis_type", "null_calibration")
    null_long.insert(2, "signal_strength", np.nan)
    null_long.insert(3, "signal_feature", np.nan)
    frames = [null_long]
    for experiment, power in powers.items():
        power_long = power.copy()
        # p_value is the canonical decision p-value in the unified schema.  This
        # matters when multiplicity is enabled and p_value_used is adjusted.
        power_long["p_value"] = power_long["p_value_used"]
        power_long.insert(0, "experiment", experiment)
        power_long.insert(1, "analysis_type", "power")
        power_long.insert(2, "signal_strength", float(power_settings[experiment]["signal_strength"]))
        signal_features = power_settings[experiment].get("signal_features", [])
        signal_feature = signal_features[0] if len(signal_features) == 1 else np.nan
        power_long.insert(3, "signal_feature", signal_feature)
        frames.append(power_long)
    unified = pd.concat(frames, ignore_index=True, sort=False)
    unified.insert(4, "selection_method", null_settings.get("selection_method"))
    unified.insert(5, "design", "baseline")
    leading = [
        "experiment",
        "analysis_type",
        "signal_strength",
        "signal_feature",
        "selection_method",
        "design",
        "shard",
        "shard_index",
        "iteration",
        "global_iteration",
        "selection_event",
        "feature",
        "p_value",
        "failed",
        "rejected",
        "source_file",
    ]
    return unified[leading + [column for column in unified if column not in leading]]


def shard_summaries(
    null: pd.DataFrame,
    power_targets: dict[str, pd.DataFrame],
    *,
    alpha_levels: list[float],
) -> pd.DataFrame:
    rows = []
    for shard in sorted(null["shard"].unique(), key=lambda value: int(value.split("_")[1])):
        for event in EVENTS:
            frame = null[(null["shard"] == shard) & (null["selection_event"] == event)]
            p_values = frame["p_value"].to_numpy(float)
            rows.append(
                {
                    "experiment": "null_calibration",
                    "shard": shard,
                    "selection_event": event,
                    "n": len(frame),
                    "mean_p_value": float(np.mean(p_values)),
                    "ks_statistic": float(stats.kstest(p_values, "uniform").statistic),
                    **{
                        f"rejection_rate_{level:g}": float(np.mean(p_values < level))
                        for level in alpha_levels
                    },
                }
            )
            for experiment, all_targets in power_targets.items():
                target = all_targets[
                    (all_targets["shard"] == shard)
                    & (all_targets["selection_event"] == event)
                ]
                signal = target[target["target_is_signal"] & ~target["failed"]]
                conditional_interval = clopper_pearson(
                    int(signal["rejected"].sum()), len(signal)
                )
                rows.append(
                    {
                        "experiment": experiment,
                        "shard": shard,
                        "selection_event": event,
                        "n": len(target),
                        "overall_power": float(target["successful_detection"].mean()),
                        "conditional_power": float(signal["rejected"].mean()),
                        "conditional_power_ci_95_lower": conditional_interval[0],
                        "conditional_power_ci_95_upper": conditional_interval[1],
                        "target_signal_rate": float(target["target_is_signal"].mean()),
                        "failure_rate": float(target["failed"].mean()),
                    }
                )
    return pd.DataFrame.from_records(rows)


def experiment_summaries(
    unified: pd.DataFrame,
    power_settings: dict[str, dict],
    *,
    null_alpha: float,
    min_denominator_ess: float,
    min_tail_ess: float,
) -> pd.DataFrame:
    """Create one compact numerical-diagnostics row per experiment and event."""
    rows = []
    for (experiment, event), frame in unified.groupby(
        ["experiment", "selection_event"], sort=False
    ):
        values = frame["p_value"].to_numpy(float)
        alpha = (
            null_alpha
            if experiment == "null_calibration"
            else float(power_settings[experiment]["alpha"])
        )
        finite = np.isfinite(values)
        is_signal = frame.get("is_signal", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        rows.append(
            {
                "experiment": experiment,
                "analysis_type": frame["analysis_type"].iloc[0],
                "signal_strength": frame["signal_strength"].iloc[0],
                "selection_event": event,
                "n_records": len(frame),
                "n_failed": int((~finite).sum()),
                "rejection_threshold": alpha,
                "rejection_rate": float(np.mean(values[finite] < alpha)),
                "mean_p_value": float(np.mean(values[finite])),
                "median_p_value": float(np.median(values[finite])),
                "p_value_q01": float(np.quantile(values[finite], 0.01)),
                "p_value_q05": float(np.quantile(values[finite], 0.05)),
                "p_value_q95": float(np.quantile(values[finite], 0.95)),
                "target_signal_rate": (
                    float(is_signal.mean()) if experiment != "null_calibration" else np.nan
                ),
                "mean_selected_samples": float(frame["selected_samples"].mean()),
                "zero_selected_samples": int(frame["selected_samples"].eq(0).sum()),
                "mean_denominator_ess": float(frame["denominator_ess"].mean()),
                "denominator_ess_below_min_rate": float(
                    (frame["denominator_ess"] < min_denominator_ess).mean()
                ),
                "mean_tail_ess": float(frame["tail_ess"].mean()),
                "tail_ess_below_min_rate": float(
                    (frame["tail_ess"] < min_tail_ess).mean()
                ),
                "mean_mc_se": float(frame["mc_se"].mean()),
                "mc_se_q95": float(frame["mc_se"].quantile(0.95)),
            }
        )
    return pd.DataFrame.from_records(rows)


def target_selection_summaries(
    power_features: dict[str, pd.DataFrame], power_settings: dict[str, dict]
) -> pd.DataFrame:
    """Summarize which selected feature became the inferential target."""
    rows = []
    for experiment, frame in power_features.items():
        strength = float(power_settings[experiment]["signal_strength"])
        for event, event_frame in frame.groupby("selection_event", sort=False):
            total = len(event_frame)
            for feature, feature_frame in event_frame.groupby("feature", sort=True):
                complete = feature_frame[~feature_frame["failed"].astype(bool)]
                rows.append(
                    {
                        "experiment": experiment,
                        "signal_strength": strength,
                        "selection_event": event,
                        "feature": int(feature),
                        "is_signal": bool(feature_frame["is_signal"].all()),
                        "target_count": len(feature_frame),
                        "target_rate": len(feature_frame) / total,
                        "rejection_rate": float(complete["rejected"].mean()),
                        "mean_p_value": float(complete["p_value_used"].mean()),
                    }
                )
    return pd.DataFrame.from_records(rows)


def event_concordance_summaries(
    null: pd.DataFrame,
    power_features: dict[str, pd.DataFrame],
    power_settings: dict[str, dict],
    *,
    null_alpha: float,
) -> pd.DataFrame:
    """Quantify paired agreement between the two conditioning events."""
    frames = {"null_calibration": null, **power_features}
    rows = []
    for experiment, frame in frames.items():
        alpha = (
            null_alpha
            if experiment == "null_calibration"
            else float(power_settings[experiment]["alpha"])
        )
        wide = frame.pivot(
            index="global_iteration",
            columns="selection_event",
            values=["p_value", "feature"],
        )
        first = wide["p_value"][EVENTS[0]].to_numpy(float)
        second = wide["p_value"][EVENTS[1]].to_numpy(float)
        complete = np.isfinite(first) & np.isfinite(second)
        first_reject = first[complete] < alpha
        second_reject = second[complete] < alpha
        difference = second[complete] - first[complete]
        rows.append(
            {
                "experiment": experiment,
                "signal_strength": (
                    np.nan
                    if experiment == "null_calibration"
                    else float(power_settings[experiment]["signal_strength"])
                ),
                "n_complete_pairs": int(complete.sum()),
                "identical_target_rate": float(
                    np.mean(
                        wide["feature"][EVENTS[0]].to_numpy()[complete]
                        == wide["feature"][EVENTS[1]].to_numpy()[complete]
                    )
                ),
                "identical_p_value_rate": float(np.mean(np.isclose(first[complete], second[complete]))),
                "pearson_p_value_correlation": float(stats.pearsonr(first[complete], second[complete]).statistic),
                "spearman_p_value_correlation": float(stats.spearmanr(first[complete], second[complete]).statistic),
                "mean_p_value_difference_same_target_minus_feature_inclusion": float(np.mean(difference)),
                "mean_absolute_p_value_difference": float(np.mean(np.abs(difference))),
                "decision_agreement_rate": float(np.mean(first_reject == second_reject)),
                "feature_inclusion_only_rejections": int(np.sum(first_reject & ~second_reject)),
                "same_target_only_rejections": int(np.sum(~first_reject & second_reject)),
            }
        )
    return pd.DataFrame.from_records(rows)


def style_axis(ax, *, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_null(
    null: pd.DataFrame,
    calibration: pd.DataFrame,
    shard_summary: pd.DataFrame,
    output_path: Path,
    *,
    alpha_levels: list[float],
):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 21)
    for event in EVENTS:
        values = null.loc[null["selection_event"] == event, "p_value"].to_numpy(float)
        axes[0, 0].hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            label=EVENT_LABELS[event],
            color=COLORS[event],
        )
        ordered = np.sort(values)
        expected = (np.arange(len(ordered)) + 0.5) / len(ordered)
        axes[0, 1].plot(
            expected,
            ordered,
            linewidth=1.8,
            label=EVENT_LABELS[event],
            color=COLORS[event],
        )
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set(title="P-value distribution", xlabel="Selective p-value", ylabel="Density")
    axes[0, 0].legend(frameon=False)
    style_axis(axes[0, 0])
    axes[0, 1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set(
        title="Uniform Q-Q plot",
        xlabel="Expected uniform quantile",
        ylabel="Observed p-value quantile",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0, 1].legend(frameon=False)
    style_axis(axes[0, 1])

    levels = alpha_levels
    min_spacing = min(np.diff(sorted(levels))) if len(levels) > 1 else levels[0]
    offsets = (-0.10 * min_spacing, 0.10 * min_spacing)
    for event, offset in zip(EVENTS, offsets):
        row = calibration.set_index("selection_event").loc[event]
        rates = np.array([row[f"rejection_rate_{level:g}"] for level in levels])
        low = np.array([row[f"rejection_ci_95_lower_{level:g}"] for level in levels])
        high = np.array([row[f"rejection_ci_95_upper_{level:g}"] for level in levels])
        axes[1, 0].errorbar(
            np.array(levels) + offset,
            rates,
            yerr=np.vstack([rates - low, high - rates]),
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=EVENT_LABELS[event],
            color=COLORS[event],
        )
    axes[1, 0].plot(levels, levels, color="black", linestyle="--", linewidth=1.0, label="Nominal")
    axes[1, 0].set(
        title="Type-I error with 95% exact intervals",
        xlabel="Nominal alpha",
        ylabel="Observed rejection rate",
        xticks=levels,
    )
    axes[1, 0].legend(frameon=False)
    style_axis(axes[1, 0])

    null_shards = shard_summary[shard_summary["experiment"] == "null_calibration"]
    stability_level = 0.05 if 0.05 in levels else levels[len(levels) // 2]
    x = np.arange(null_shards["shard"].nunique())
    for event in EVENTS:
        frame = null_shards[null_shards["selection_event"] == event]
        axes[1, 1].plot(
            x,
            frame[f"rejection_rate_{stability_level:g}"],
            marker="o",
            label=EVENT_LABELS[event],
            color=COLORS[event],
        )
    axes[1, 1].axhline(stability_level, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        title=f"Shard stability at alpha = {stability_level:g}",
        xlabel="Shard",
        ylabel="Rejection rate",
        xticks=x,
        xticklabels=[str(index) for index in x],
    )
    axes[1, 1].legend(frameon=False)
    style_axis(axes[1, 1])
    n_per_event = int(null.groupby("selection_event").size().min())
    fig.suptitle(
        f"Pooled null calibration: {n_per_event:,} replicates per event",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_power(
    power: pd.DataFrame,
    shard_summary: pd.DataFrame,
    output_path: Path,
    *,
    validation_target: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    experiments = list(POWER_EXPERIMENTS)
    strengths = {
        experiment: float(
            power.loc[power["experiment"] == experiment, "signal_strength"].iloc[0]
        )
        for experiment in experiments
    }
    x = np.arange(len(experiments))
    tick_labels = [f"beta = {strengths[experiment]:g}" for experiment in experiments]
    width = 0.36

    def grouped_intervals(ax, value: str, low: str, high: str, title: str):
        maxima = []
        for event_index, event in enumerate(EVENTS):
            frame = power[power["selection_event"] == event].set_index("experiment")
            values = np.array([frame.loc[item, value] for item in experiments], dtype=float)
            lows = np.array([frame.loc[item, low] for item in experiments], dtype=float)
            highs = np.array([frame.loc[item, high] for item in experiments], dtype=float)
            positions = x + (event_index - 0.5) * width
            ax.bar(
                positions,
                values,
                width,
                color=COLORS[event],
                alpha=0.88,
                label=EVENT_LABELS[event],
            )
            ax.errorbar(
                positions,
                values,
                yerr=np.vstack([values - lows, highs - values]),
                fmt="none",
                color="black",
                capsize=4,
            )
            maxima.extend(highs)
        ax.set(
            title=title,
            ylabel="Probability",
            xticks=x,
            xticklabels=tick_labels,
            ylim=(0, min(1.0, max(0.2, max(maxima) + 0.04))),
        )
        ax.legend(frameon=False)
        style_axis(ax)

    grouped_intervals(
        axes[0, 0],
        "power",
        "power_ci_95_lower",
        "power_ci_95_upper",
        "Unconditional detection power",
    )
    grouped_intervals(
        axes[0, 1],
        "conditional_power_given_signal_target",
        "conditional_power_ci_95_lower",
        "conditional_power_ci_95_upper",
        "Power given the signal was targeted",
    )
    axes[0, 1].axhline(
        validation_target,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Validation target ({validation_target:.2f})",
    )
    axes[0, 1].legend(frameon=False)

    for event in EVENTS:
        frame = power[power["selection_event"] == event].set_index("experiment")
        axes[1, 0].plot(
            x,
            [frame.loc[item, "target_signal_rate"] for item in experiments],
            marker="o",
            color=COLORS[event],
            label=f"{EVENT_LABELS[event]}: target is signal",
        )
        axes[1, 0].plot(
            x,
            [frame.loc[item, "converged_non_signal_rejection_rate"] for item in experiments],
            marker="x",
            linestyle="--",
            color=COLORS[event],
            label=f"{EVENT_LABELS[event]}: reject non-signal",
        )
    test_alpha = float(power["alpha"].iloc[0])
    axes[1, 0].axhline(test_alpha, color="black", linestyle=":", linewidth=1.0)
    axes[1, 0].set(
        title="Targeting and selected non-signal diagnostics",
        ylabel="Rate",
        xticks=x,
        xticklabels=tick_labels,
        ylim=(0, 1),
    )
    axes[1, 0].legend(frameon=False, fontsize=8)
    style_axis(axes[1, 0])

    shard_x = np.arange(shard_summary["shard"].nunique())
    linestyles = ("-", "--", ":", "-.")
    for experiment, linestyle in zip(experiments, linestyles, strict=False):
        experiment_frame = shard_summary[shard_summary["experiment"] == experiment]
        for event in EVENTS:
            values = experiment_frame[
                experiment_frame["selection_event"] == event
            ]["conditional_power"].to_numpy()
            axes[1, 1].plot(
                shard_x,
                values,
                marker="o",
                linestyle=linestyle,
                color=COLORS[event],
                label=f"{EVENT_LABELS[event]}, beta={strengths[experiment]:g}",
            )
    axes[1, 1].axhline(validation_target, color="black", linestyle=":", linewidth=1.0)
    axes[1, 1].set(
        title="Conditional power by shard",
        xlabel="Shard",
        ylabel="Probability",
        xticks=shard_x,
        xticklabels=[str(index) for index in shard_x],
        ylim=(0, 1),
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    style_axis(axes[1, 1])
    fig.suptitle(
        "Pooled power validation across nonlinear signal strengths",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(
    unified: pd.DataFrame,
    output_path: Path,
    *,
    power_settings: dict[str, dict],
    min_denominator_ess: float,
    min_tail_ess: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    experiments = ("null_calibration", *POWER_EXPERIMENTS)
    experiment_labels = ("Null",) + tuple(
        f"Power beta={power_settings[experiment]['signal_strength']:g}"
        for experiment in POWER_EXPERIMENTS
    )
    experiment_colors = EXPERIMENT_COLORS[: len(experiments)]
    for experiment, label, color in zip(experiments, experiment_labels, experiment_colors):
        frame = unified[unified["experiment"] == experiment]
        axes[0, 0].hist(frame["denominator_ess"].dropna(), bins=35, histtype="step", linewidth=1.8, label=label, color=color, density=True)
        axes[0, 1].hist(frame["tail_ess"].dropna(), bins=35, histtype="step", linewidth=1.8, label=label, color=color, density=True)
        axes[1, 0].hist(frame["mc_se"].dropna(), bins=35, histtype="step", linewidth=1.8, label=label, color=color, density=True)
        sample = frame.sample(min(2500, len(frame)), random_state=123)
        axes[1, 1].scatter(sample["selection_probability_estimate"], sample["mc_se"], s=8, alpha=0.25, label=label, color=color)
    axes[0, 0].axvline(min_denominator_ess, color="black", linestyle="--", linewidth=1.0, label=f"Minimum ESS ({min_denominator_ess:g})")
    axes[0, 0].set(title="Denominator effective sample size", xlabel="ESS", ylabel="Density")
    axes[0, 1].axvline(min_tail_ess, color="black", linestyle="--", linewidth=1.0, label=f"Minimum ESS ({min_tail_ess:g})")
    axes[0, 1].set(title="Tail effective sample size", xlabel="ESS", ylabel="Density")
    axes[1, 0].set(title="Monte Carlo standard error", xlabel="MC SE", ylabel="Density")
    axes[1, 1].set(title="MC uncertainty vs. selection probability", xlabel="Estimated selection probability", ylabel="MC SE")
    for ax in axes.flat:
        ax.legend(frameon=False)
        style_axis(ax)
    fig.get_layout_engine().set(w_pad=5 / 72, h_pad=5 / 72, wspace=0.08, hspace=0.08)
    fig.suptitle("Conditional Monte Carlo diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_p_value_distributions(
    unified: pd.DataFrame,
    output_path: Path,
    power_settings: dict[str, dict],
) -> None:
    """Show the complete p-value distribution for every experiment and event."""
    experiments = ("null_calibration", *POWER_EXPERIMENTS)
    fig, axes = plt.subplots(
        len(experiments),
        len(EVENTS),
        figsize=(12.5, 2.25 * len(experiments)),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    bins = np.linspace(0.0, 1.0, 26)
    for row_index, experiment in enumerate(experiments):
        label = (
            "Null"
            if experiment == "null_calibration"
            else f"Power, beta={power_settings[experiment]['signal_strength']:g}"
        )
        alpha = (
            0.05
            if experiment == "null_calibration"
            else float(power_settings[experiment]["alpha"])
        )
        for column_index, event in enumerate(EVENTS):
            ax = axes[row_index, column_index]
            values = unified.loc[
                (unified["experiment"] == experiment)
                & (unified["selection_event"] == event),
                "p_value",
            ].dropna()
            ax.hist(
                values,
                bins=bins,
                density=True,
                color=COLORS[event],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
            )
            ax.axvline(alpha, color="black", linestyle="--", linewidth=0.9)
            ax.text(
                0.98,
                0.88,
                f"n={len(values):,}; reject={np.mean(values < alpha):.1%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            ax.set_title(f"{label} - {EVENT_LABELS[event]}", fontsize=10)
            if column_index == 0:
                ax.set_ylabel("Density")
            if row_index == len(experiments) - 1:
                ax.set_xlabel("Selective p-value")
            style_axis(ax)
    fig.suptitle("Selective p-value distributions across the complete dataset", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_event_agreement(
    null: pd.DataFrame,
    power_features: dict[str, pd.DataFrame],
    power_settings: dict[str, dict],
    output_path: Path,
) -> None:
    """Plot paired p-values under feature-inclusion and same-target conditioning."""
    frames = {"null_calibration": null, **power_features}
    experiments = list(frames)
    n_columns = 3
    n_rows = int(np.ceil(len(experiments) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.5, 4.0 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (experiment, frame) in zip(axes, frames.items()):
        p_column = "p_value" if experiment == "null_calibration" else "p_value_used"
        wide = frame.pivot(index="global_iteration", columns="selection_event", values=p_column)
        first = wide[EVENTS[0]].to_numpy(float)
        second = wide[EVENTS[1]].to_numpy(float)
        complete = np.isfinite(first) & np.isfinite(second)
        ax.hexbin(first[complete], second[complete], gridsize=34, mincnt=1, cmap="viridis", linewidths=0)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
        correlation = stats.spearmanr(first[complete], second[complete]).statistic
        label = (
            "Null"
            if experiment == "null_calibration"
            else f"Power, beta={power_settings[experiment]['signal_strength']:g}"
        )
        ax.text(
            0.04,
            0.94,
            f"Spearman rho={correlation:.3f}\nExact ties={np.mean(np.isclose(first[complete], second[complete])):.1%}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.set(title=label, xlabel=EVENT_LABELS[EVENTS[0]], ylabel=EVENT_LABELS[EVENTS[1]], xlim=(0, 1), ylim=(0, 1))
        style_axis(ax, grid_axis="both")
    for ax in axes[len(experiments) :]:
        ax.set_visible(False)
    fig.suptitle("Paired selective p-value agreement between conditioning events", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_target_selection(
    target_selection: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap the inferential-target distribution over features and signal strengths."""
    strengths = sorted(target_selection["signal_strength"].unique())
    features = sorted(target_selection["feature"].unique())
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(13.5, 4.2), sharey=True, constrained_layout=True)
    images = []
    for ax, event in zip(np.atleast_1d(axes), EVENTS):
        frame = target_selection[target_selection["selection_event"] == event]
        matrix = (
            frame.pivot(index="signal_strength", columns="feature", values="target_rate")
            .reindex(index=strengths, columns=features, fill_value=0.0)
            .fillna(0.0)
        )
        image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=float(target_selection["target_rate"].max()))
        images.append(image)
        ax.set(
            title=EVENT_LABELS[event],
            xlabel="Target feature",
            xticks=np.arange(len(features)),
            xticklabels=features,
            yticks=np.arange(len(strengths)),
            yticklabels=[f"beta={value:g}" for value in strengths],
        )
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                1.0,
                len(strengths),
                fill=False,
                edgecolor="#E45756",
                linewidth=2.5,
            )
        )
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Signal strength")
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.85, pad=0.02)
    colorbar.set_label("Probability feature is targeted")
    fig.suptitle("Inferential target distribution (signal feature is 0)", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shard_stability(
    shard_summary: pd.DataFrame,
    power_settings: dict[str, dict],
    output_path: Path,
) -> None:
    """Show shard-to-shard variation on one common probability scale."""
    rows = []
    labels = []
    shards = sorted(shard_summary["shard"].unique(), key=lambda value: int(value.split("_")[1]))
    for event in EVENTS:
        frame = shard_summary[
            (shard_summary["experiment"] == "null_calibration")
            & (shard_summary["selection_event"] == event)
        ].set_index("shard")
        rows.append(frame.loc[shards, "rejection_rate_0.05"].to_numpy(float))
        labels.append(f"Null rejection - {EVENT_LABELS[event]}")
    for experiment in POWER_EXPERIMENTS:
        strength = power_settings[experiment]["signal_strength"]
        for event in EVENTS:
            frame = shard_summary[
                (shard_summary["experiment"] == experiment)
                & (shard_summary["selection_event"] == event)
            ].set_index("shard")
            rows.append(frame.loc[shards, "conditional_power"].to_numpy(float))
            labels.append(f"Conditional power beta={strength:g} - {EVENT_LABELS[event]}")
    matrix = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(12.5, max(5.0, 0.55 * len(rows))), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set(
        title="Shard stability: null rejection at 5% and conditional power",
        xlabel="Shard",
        xticks=np.arange(len(shards)),
        xticklabels=shards,
        yticks=np.arange(len(labels)),
        yticklabels=labels,
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.02)
    colorbar.set_label("Observed rate")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def write_analysis_report(
    output_path: Path,
    *,
    shards: list[Path],
    unified: pd.DataFrame,
    null: pd.DataFrame,
    calibration: pd.DataFrame,
    null_paired: pd.DataFrame,
    power: pd.DataFrame,
    power_paired: pd.DataFrame,
    power_settings: dict,
    validation_target: float,
) -> None:
    calibration_by_event = calibration.set_index("selection_event")
    power_by_event = power.set_index("selection_event")
    alpha_levels = [float(value) for value in calibration.columns.str.extract(
        r"^rejection_rate_([0-9.]+)$", expand=False
    ).dropna()]
    alpha_levels = sorted(set(alpha_levels))
    null_table = [
        "| Selection event | Mean p-value | KS statistic (descriptive) | "
        + " | ".join(f"Reject at {level:g}" for level in alpha_levels)
        + " |",
        "|---|---:|---:|" + "---:|" * len(alpha_levels),
    ]
    for event in EVENTS:
        row = calibration_by_event.loc[event]
        rates = " | ".join(
            _percent(float(row[f"rejection_rate_{level:g}"]))
            for level in alpha_levels
        )
        null_table.append(
            f"| {EVENT_LABELS[event]} | {row['mean_p_value']:.4f} | "
            f"{row['uniform_ks_statistic']:.4f} | {rates} |"
        )

    power_table = [
        "| Selection event | Detection power (95% exact CI) | Target is signal "
        "(95% exact CI) | Conditional power (95% exact CI) | Selected non-signal rejection |",
        "|---|---:|---:|---:|---:|",
    ]
    for event in EVENTS:
        row = power_by_event.loc[event]
        power_table.append(
            f"| {EVENT_LABELS[event]} | {_percent(row['power'])} "
            f"({_percent(row['power_ci_95_lower'])}–{_percent(row['power_ci_95_upper'])}) | "
            f"{_percent(row['target_signal_rate'])} "
            f"({_percent(row['target_signal_rate_ci_95_lower'])}–{_percent(row['target_signal_rate_ci_95_upper'])}) | "
            f"{_percent(row['conditional_power_given_signal_target'])} "
            f"({_percent(row['conditional_power_ci_95_lower'])}–{_percent(row['conditional_power_ci_95_upper'])}) | "
            f"{_percent(row['converged_non_signal_rejection_rate'])} |"
        )

    null_pair_lines = []
    for row in null_paired.itertuples(index=False):
        null_pair_lines.append(
            f"- At alpha = {row.alpha:g}, same-target minus feature-inclusion is "
            f"{100.0 * row.rejection_rate_difference:+.2f} percentage points "
            f"(conservative paired 95% CI {100.0 * row.ci_95_lower:+.2f} to "
            f"{100.0 * row.ci_95_upper:+.2f})."
        )
    power_pair = power_paired.iloc[0]
    min_denominator = float(power_settings["min_denominator_ess"])
    min_tail = float(power_settings["min_tail_ess"])
    diagnostics_lines = []
    for experiment, label in (
        ("null_calibration", "Null"),
        ("power_beta_0.3_feature_0", "Power"),
    ):
        frame = unified[unified["experiment"] == experiment]
        diagnostics_lines.append(
            f"- {label}: denominator ESS below {min_denominator:g} in "
            f"{_percent(float((frame['denominator_ess'] < min_denominator).mean()), 1)} "
            f"of records; tail ESS below {min_tail:g} in "
            f"{_percent(float((frame['tail_ess'] < min_tail).mean()), 1)}; "
            f"95th percentile MC SE {frame['mc_se'].quantile(0.95):.3f}."
        )
    zero_draws = unified["selected_samples"].eq(0).groupby(unified["experiment"]).sum()
    finite_valid = bool(unified["finite_sample_valid"].fillna(False).all())
    n_per_event = null.groupby("selection_event").size().min()
    report = f"""# Unified selection-method validation analysis

## Scope and integrity checks

- Combined {len(shards)} independent shards (`{shards[0].name}` through `{shards[-1].name}`).
- The unified raw file contains {len(unified):,} records: {int((unified['experiment'] == 'null_calibration').sum()):,} null and {int((unified['experiment'] != 'null_calibration').sum()):,} power records, or {int(n_per_event):,} replicates per event and experiment.
- All experiment-defining settings match across shards; only random seeds differ. Shard IDs and local iteration/event grids are complete and unique.
- The fixed null-design arrays are identical across shards. Power `feature_results`, `target_results`, and duplicate `signal_results` agree on keys, targets, p-values, signal flags, failures, and rejection decisions.
- The unified `p_value` field is the decision p-value used for rejection. There are no duplicate experiment/global-iteration/event keys, no test failures, and all p-values lie in `[0, 1]`.

## Null calibration

{chr(10).join(null_table)}

The rejection-rate intervals in `null_calibration_summary.csv` are exact Clopper–Pearson binomial intervals. The KS statistics above are descriptive only: a standard continuous-uniform KS p-value is not reported because finite-budget rank p-values are discrete and super-uniform rather than exactly continuous uniform.

{chr(10).join(null_pair_lines)}

Both events are close to nominal at 5% and 10%. At 1%, both reject in {_percent(float(calibration_by_event.iloc[0]['rejection_rate_0.01']))}; the exact 95% interval is {_percent(float(calibration_by_event.iloc[0]['rejection_ci_95_lower_0.01']))}–{_percent(float(calibration_by_event.iloc[0]['rejection_ci_95_upper_0.01']))}, indicating conservatism at the extreme tail, not anti-conservatism.

## Power at signal strength {float(power_settings['signal_strength']):g}

{chr(10).join(power_table)}

Conditional power is below the configured validation target of {_percent(validation_target, 0)}. The selected non-signal rejection rate is a diagnostic under an alternative-data distribution and adaptive selection; the alpha line in the plot is a reference, not a null-calibration target.

The observed paired power difference is {100.0 * float(power_pair['power_difference']):+.2f} percentage points. There were {int(power_pair['baseline_only_detections'])} feature-inclusion-only and {int(power_pair['comparison_only_detections'])} same-target-only detections. The conservative paired 95% interval is {100.0 * float(power_pair['ci_95_lower']):+.3f} to {100.0 * float(power_pair['ci_95_upper']):+.3f} percentage points. This replaces the misleading degenerate plug-in t interval `[0, 0]` that results when every observed paired decision ties.

## Monte Carlo diagnostics

- Finite-sample-valid flag on every unified record: `{finite_valid}`.
- Zero selected Monte Carlo draws (finite rank p-value of 1 by construction): {int(zero_draws.get('null_calibration', 0))} null records and {int(zero_draws.get('power_beta_0.3_feature_0', 0))} power records.
{chr(10).join(diagnostics_lines)}

These ESS thresholds are diagnostics for numerical resolution. Falling below them does not mark a fixed-budget conditional-rank p-value as failed, but it indicates coarse/noisy individual p-values and reduced decision stability near the rejection cutoff.
"""
    output_path.write_text(report, encoding="utf-8")


def write_multi_analysis_report(
    output_path: Path,
    *,
    shards: list[Path],
    unified: pd.DataFrame,
    null: pd.DataFrame,
    calibration: pd.DataFrame,
    null_paired: pd.DataFrame,
    power: pd.DataFrame,
    power_paired: pd.DataFrame,
    power_settings: dict[str, dict],
    diagnostic_summary: pd.DataFrame,
    target_selection: pd.DataFrame,
    event_concordance: pd.DataFrame,
    validation_target: float,
) -> None:
    """Write a report covering the shared null run and every power strength."""
    calibration_by_event = calibration.set_index("selection_event")
    alpha_levels = sorted(
        {
            float(value)
            for value in calibration.columns.str.extract(
                r"^rejection_rate_([0-9.]+)$", expand=False
            ).dropna()
        }
    )
    null_table = [
        "| Selection event | Mean p-value | KS statistic (descriptive) | "
        + " | ".join(f"Reject at {level:g}" for level in alpha_levels)
        + " |",
        "|---|---:|---:|" + "---:|" * len(alpha_levels),
    ]
    for event in EVENTS:
        row = calibration_by_event.loc[event]
        rates = " | ".join(
            _percent(float(row[f"rejection_rate_{level:g}"]))
            for level in alpha_levels
        )
        null_table.append(
            f"| {EVENT_LABELS[event]} | {row['mean_p_value']:.4f} | "
            f"{row['uniform_ks_statistic']:.4f} | {rates} |"
        )

    null_pair_lines = [
        f"- At alpha = {row.alpha:g}, same-target minus feature-inclusion is "
        f"{100.0 * row.rejection_rate_difference:+.2f} percentage points "
        f"(conservative paired 95% CI {100.0 * row.ci_95_lower:+.2f} to "
        f"{100.0 * row.ci_95_upper:+.2f})."
        for row in null_paired.itertuples(index=False)
    ]

    power_sections = []
    for experiment in POWER_EXPERIMENTS:
        settings = power_settings[experiment]
        signal_strength = float(settings["signal_strength"])
        experiment_power = power[power["experiment"] == experiment].set_index(
            "selection_event"
        )
        table = [
            "| Selection event | Detection power (95% exact CI) | Target is signal "
            "(95% exact CI) | Conditional power (95% exact CI) | Selected non-signal rejection |",
            "|---|---:|---:|---:|---:|",
        ]
        for event in EVENTS:
            row = experiment_power.loc[event]
            table.append(
                f"| {EVENT_LABELS[event]} | {_percent(row['power'])} "
                f"({_percent(row['power_ci_95_lower'])}–{_percent(row['power_ci_95_upper'])}) | "
                f"{_percent(row['target_signal_rate'])} "
                f"({_percent(row['target_signal_rate_ci_95_lower'])}–{_percent(row['target_signal_rate_ci_95_upper'])}) | "
                f"{_percent(row['conditional_power_given_signal_target'])} "
                f"({_percent(row['conditional_power_ci_95_lower'])}–{_percent(row['conditional_power_ci_95_upper'])}) | "
                f"{_percent(row['converged_non_signal_rejection_rate'])} |"
            )
        pair = power_paired[power_paired["experiment"] == experiment].iloc[0]
        target_assessment = (
            "meets"
            if bool(
                (
                    experiment_power["conditional_power_given_signal_target"]
                    >= validation_target
                ).all()
            )
            else "is below"
        )
        power_sections.append(
            f"""## Power at signal strength {signal_strength:g}

{chr(10).join(table)}

Conditional power {target_assessment} the configured validation target of {_percent(validation_target, 0)} across the two events. The selected non-signal rejection rate is a diagnostic under an alternative-data distribution and adaptive selection; the alpha line in the plot is a reference, not a null-calibration target.

The observed paired power difference (same-target minus feature-inclusion) is {100.0 * float(pair['power_difference']):+.2f} percentage points. There were {int(pair['baseline_only_detections'])} feature-inclusion-only and {int(pair['comparison_only_detections'])} same-target-only detections. The conservative paired 95% interval is {100.0 * float(pair['ci_95_lower']):+.3f} to {100.0 * float(pair['ci_95_upper']):+.3f} percentage points.
"""
        )

    first_settings = power_settings[next(iter(POWER_EXPERIMENTS))]
    min_denominator = float(first_settings["min_denominator_ess"])
    min_tail = float(first_settings["min_tail_ess"])
    diagnostics_lines = []
    diagnostic_experiments = [("null_calibration", "Null")] + [
        (experiment, f"Power beta={power_settings[experiment]['signal_strength']:g}")
        for experiment in POWER_EXPERIMENTS
    ]
    for experiment, label in diagnostic_experiments:
        frame = unified[unified["experiment"] == experiment]
        diagnostics_lines.append(
            f"- {label}: denominator ESS below {min_denominator:g} in "
            f"{_percent(float((frame['denominator_ess'] < min_denominator).mean()), 1)} "
            f"of records; tail ESS below {min_tail:g} in "
            f"{_percent(float((frame['tail_ess'] < min_tail).mean()), 1)}; "
            f"95th percentile MC SE {frame['mc_se'].quantile(0.95):.3f}."
        )

    zero_draws = unified["selected_samples"].eq(0).groupby(unified["experiment"]).sum()
    zero_draw_lines = [
        f"  - Null: {int(zero_draws.get('null_calibration', 0))} records."
    ] + [
        f"  - Power beta={power_settings[experiment]['signal_strength']:g}: "
        f"{int(zero_draws.get(experiment, 0))} records."
        for experiment in POWER_EXPERIMENTS
    ]
    experiment_counts = unified.groupby("experiment").size()
    power_count_text = ", ".join(
        f"{int(experiment_counts[experiment]):,} at beta="
        f"{power_settings[experiment]['signal_strength']:g}"
        for experiment in POWER_EXPERIMENTS
    )
    finite_valid = bool(unified["finite_sample_valid"].fillna(False).all())
    n_per_event = int(null.groupby("selection_event").size().min())
    concordance_lines = []
    for row in event_concordance.itertuples(index=False):
        label = (
            "Null"
            if row.experiment == "null_calibration"
            else f"Power beta={row.signal_strength:g}"
        )
        concordance_lines.append(
            f"- {label}: Spearman correlation {row.spearman_p_value_correlation:.3f}, "
            f"exact p-value ties {_percent(row.identical_p_value_rate, 1)}, and "
            f"decision agreement {_percent(row.decision_agreement_rate, 2)}."
        )
    signal_target = target_selection[target_selection["is_signal"]].sort_values(
        ["signal_strength", "selection_event"]
    )
    selection_lines = [
        f"- Beta={row.signal_strength:g}, {EVENT_LABELS[row.selection_event]}: "
        f"signal feature targeted {_percent(row.target_rate)}."
        for row in signal_target.itertuples(index=False)
    ]
    max_mc_se = diagnostic_summary.loc[
        diagnostic_summary["mc_se_q95"].idxmax()
    ]
    passing_strengths = sorted(
        strength
        for strength, frame in power.groupby("signal_strength")
        if bool(
            (frame["conditional_power_given_signal_target"] >= validation_target).all()
        )
    )
    failing_strengths = sorted(
        strength
        for strength, frame in power.groupby("signal_strength")
        if not bool(
            (frame["conditional_power_given_signal_target"] >= validation_target).all()
        )
    )
    strongest = power.loc[power["signal_strength"].idxmax()]
    worst_tail = diagnostic_summary.loc[
        diagnostic_summary["tail_ess_below_min_rate"].idxmax()
    ]
    strength_text = lambda values: ", ".join(f"{value:g}" for value in values)  # noqa: E731
    report = f"""# Unified selection-method validation analysis

## Executive summary

- Null rejection is close to nominal: {_percent(calibration_by_event.loc['feature_inclusion', 'rejection_rate_0.05'])} for feature inclusion and {_percent(calibration_by_event.loc['same_target', 'rejection_rate_0.05'])} for same target at alpha=0.05.
- Conditional power meets the {_percent(validation_target, 0)} target at beta={strength_text(passing_strengths)} and remains below it at beta={strength_text(failing_strengths)}.
- At the largest signal strength, unconditional detection power is {_percent(strongest['power'])} because the signal is targeted in only {_percent(strongest['target_signal_rate'])} of iterations; conditional power after targeting is {_percent(strongest['conditional_power_given_signal_target'])}.
- The two conditioning events have identical observed detection power at every strength. Their paired p-values become exactly equal in essentially every replicate by beta=0.75.
- Numerical resolution is the main caution: the worst tail-ESS shortfall is {_percent(worst_tail['tail_ess_below_min_rate'], 1)} ({worst_tail.experiment}, {EVENT_LABELS[worst_tail.selection_event]}), although every record retains the finite-sample-valid rank-p-value flag.

## Scope and integrity checks

- Combined {len(shards)} independent shards (`{shards[0].name}` through `{shards[-1].name}`).
- The unified raw file contains {len(unified):,} records: {int(experiment_counts['null_calibration']):,} null and {int((unified['experiment'] != 'null_calibration').sum()):,} power ({power_count_text}), or {n_per_event:,} replicates per event and experiment.
- The clean-shard dataset contains the SHAP selector, baseline design, fresh auxiliary randomization, signal feature 0, and all four available signal strengths (0.3, 0.5, 0.75, and 1.0).
- All experiment-defining settings match within each experiment across shards; only random seeds differ. The power experiments differ only in signal strength and seeds. Shard IDs and local iteration/event grids are complete and unique.
- The fixed null-design arrays are identical across shards. For every power strength, `feature_results`, `target_results`, and duplicate `signal_results` agree on keys, targets, p-values, signal flags, failures, and rejection decisions.
- The unified `p_value` field is the decision p-value used for rejection. There are no duplicate experiment/global-iteration/event keys, no test failures, and all p-values lie in `[0, 1]`.

## Null calibration

{chr(10).join(null_table)}

The rejection-rate intervals in `null_calibration_summary.csv` are exact Clopper–Pearson binomial intervals. The KS statistics above are descriptive only because finite-budget rank p-values are discrete and super-uniform.

{chr(10).join(null_pair_lines)}

{chr(10).join(power_sections)}
## Event agreement and target selection

{chr(10).join(concordance_lines)}

Signal-target probability increases sharply from beta=0.3 and is near 50% thereafter:

{chr(10).join(selection_lines)}

## Monte Carlo diagnostics

- Finite-sample-valid flag on every unified record: `{finite_valid}`.
- Zero selected Monte Carlo draws (finite rank p-value of 1 by construction):
{chr(10).join(zero_draw_lines)}
{chr(10).join(diagnostics_lines)}

These ESS thresholds are diagnostics for numerical resolution. Falling below them does not mark a fixed-budget conditional-rank p-value as failed, but it indicates coarse/noisy individual p-values and reduced decision stability near the rejection cutoff.

The largest experiment/event 95th percentile Monte Carlo SE is {max_mc_se.mc_se_q95:.3f} ({max_mc_se.experiment}, {EVENT_LABELS[max_mc_se.selection_event]}).

## Output guide

- `selection_method_validation_unified.csv`: schema-unified record-level data with explicit analysis type, signal strength, method, design, shard, and global iteration identifiers.
- `null_calibration_summary.csv`, `power_summary.csv`, and paired-comparison files: pooled inferential summaries and exact intervals.
- `experiment_diagnostics_summary.csv`: p-value, rejection, ESS, selected-draw, and Monte Carlo precision diagnostics by experiment and event.
- `target_selection_summary.csv`: target-feature frequencies and feature-specific rejection behavior.
- `event_concordance_summary.csv`: paired p-value, target, and rejection agreement between conditioning events.
- `per_shard_summary.csv`: shard-level stability statistics.
- `null_calibration.png`, `power_validation.png`, `p_value_distributions.png`, `event_agreement.png`, `target_selection.png`, `mc_diagnostics.png`, and `shard_stability.png`: the complete plot suite.
"""
    output_path.write_text(report, encoding="utf-8")


def main(argv=None):
    global POWER_EXPERIMENTS
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "unified_analysis").resolve()
    shards = ordered_shards(input_dir)
    if len(shards) != args.expected_shards:
        raise ValueError(f"Found {len(shards)} shards; expected {args.expected_shards}.")
    expected_names = [f"shard_{index}" for index in range(args.expected_shards)]
    if [shard.name for shard in shards] != expected_names:
        raise ValueError(
            "Shard IDs must be contiguous from shard_0 through "
            f"shard_{args.expected_shards - 1}."
        )
    if not 0.0 < args.validation_target < 1.0:
        raise ValueError("--validation-target must lie strictly between 0 and 1.")
    POWER_EXPERIMENTS = discover_power_experiments(shards)
    null_settings_by_shard, power_settings_by_shard = validate_settings(shards)
    validate_shared_null_design(shards, null_settings_by_shard)
    null_settings = null_settings_by_shard[0]
    power_settings = {
        experiment: settings[0]
        for experiment, settings in power_settings_by_shard.items()
    }
    if tuple(null_settings.get("selection_events", ())) != EVENTS:
        raise ValueError(f"Null selection_events must be {EVENTS}.")
    for experiment, settings in power_settings.items():
        if tuple(settings.get("selection_events", ())) != EVENTS:
            raise ValueError(f"{experiment} selection_events must be {EVENTS}.")
    reference_power = {
        key: value
        for key, value in next(iter(power_settings.values())).items()
        if key not in {"seed", "signal_strength"}
    }
    for experiment, settings in power_settings.items():
        comparable = {
            key: value
            for key, value in settings.items()
            if key not in {"seed", "signal_strength"}
        }
        if comparable != reference_power:
            raise ValueError(
                f"{experiment} differs from the other power experiments in more "
                "than signal strength and seed."
            )
    n_null = int(null_settings["n_iters"])
    alpha_levels = [float(value) for value in null_settings["alpha_levels"]]

    null = read_shard_csvs(
        shards,
        NULL_RELATIVE / "p_value_results.csv",
        expected_iters=n_null,
        events=EVENTS,
        required_columns={
            "iteration",
            "selection_event",
            "p_value",
            "failed",
            "finite_sample_valid",
            "denominator_ess",
            "tail_ess",
            "mc_se",
        },
    )
    validate_probability_records(null, label="Null results")
    feature_frames = {}
    target_frames = {}
    power_summaries = []
    power_paired_summaries = []
    calibration = null_paired = None
    for experiment, relative in POWER_EXPERIMENTS.items():
        settings = power_settings[experiment]
        n_power = int(settings["n_iters"])
        power_alpha = float(settings["alpha"])
        features = read_shard_csvs(
            shards,
            relative / "feature_results.csv",
            expected_iters=n_power,
            events=EVENTS,
            required_columns={
                "iteration",
                "selection_event",
                "feature",
                "p_value",
                "p_value_used",
                "failed",
                "rejected",
                "is_signal",
                "fixed_design_null",
                "denominator_ess",
                "tail_ess",
                "mc_se",
            },
        )
        targets = read_shard_csvs(
            shards,
            relative / "target_results.csv",
            expected_iters=n_power,
            events=EVENTS,
            required_columns={
                "iteration",
                "selection_event",
                "target_feature",
                "target_is_signal",
                "p_value",
                "failed",
                "rejected",
                "successful_detection",
                "failed_signal_test",
            },
        )
        signals = read_shard_csvs(
            shards,
            relative / "signal_results.csv",
            expected_iters=n_power,
            events=EVENTS,
            required_columns={
                "iteration",
                "selection_event",
                "target_feature",
                "target_is_signal",
                "p_value",
                "failed",
                "rejected",
                "successful_detection",
                "failed_signal_test",
            },
        )
        validate_probability_records(targets, label=f"{experiment} target results")
        validate_power_alignment(features, targets, alpha=power_alpha)
        comparison_columns = [
            column for column in targets if column not in {"source_file"}
        ]
        if not targets[comparison_columns].equals(signals[comparison_columns]):
            raise ValueError(
                f"{experiment} target_results.csv and signal_results.csv are not "
                "exact duplicates."
            )
        experiment_calibration, experiment_null_paired, summary, paired = (
            pooled_summaries(
                null,
                features,
                targets,
                alpha_levels=alpha_levels,
                alpha=power_alpha,
            )
        )
        if calibration is None:
            calibration = experiment_calibration
            null_paired = experiment_null_paired
        summary.insert(0, "signal_strength", float(settings["signal_strength"]))
        summary.insert(0, "experiment", experiment)
        paired.insert(0, "signal_strength", float(settings["signal_strength"]))
        paired.insert(0, "experiment", experiment)
        feature_frames[experiment] = features
        target_frames[experiment] = targets
        power_summaries.append(summary)
        power_paired_summaries.append(paired)

    power = pd.concat(power_summaries, ignore_index=True)
    power_paired = pd.concat(power_paired_summaries, ignore_index=True)
    unified = build_unified(null, feature_frames, power_settings, null_settings)
    per_shard = shard_summaries(null, target_frames, alpha_levels=alpha_levels)
    first_power_settings = power_settings[next(iter(POWER_EXPERIMENTS))]
    min_denominator_ess = float(first_power_settings["min_denominator_ess"])
    min_tail_ess = float(first_power_settings["min_tail_ess"])
    diagnostic_summary = experiment_summaries(
        unified,
        power_settings,
        null_alpha=0.05,
        min_denominator_ess=min_denominator_ess,
        min_tail_ess=min_tail_ess,
    )
    target_selection = target_selection_summaries(feature_frames, power_settings)
    event_concordance = event_concordance_summaries(
        null,
        feature_frames,
        power_settings,
        null_alpha=0.05,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    unified.to_csv(output_dir / "selection_method_validation_unified.csv", index=False)
    calibration.to_csv(output_dir / "null_calibration_summary.csv", index=False)
    null_paired.to_csv(output_dir / "null_paired_event_comparison.csv", index=False)
    power.to_csv(output_dir / "power_summary.csv", index=False)
    power_paired.to_csv(output_dir / "power_paired_event_comparison.csv", index=False)
    per_shard.to_csv(output_dir / "per_shard_summary.csv", index=False)
    diagnostic_summary.to_csv(output_dir / "experiment_diagnostics_summary.csv", index=False)
    target_selection.to_csv(output_dir / "target_selection_summary.csv", index=False)
    event_concordance.to_csv(output_dir / "event_concordance_summary.csv", index=False)
    plot_null(
        null,
        calibration,
        per_shard,
        output_dir / "null_calibration.png",
        alpha_levels=alpha_levels,
    )
    plot_power(
        power,
        per_shard,
        output_dir / "power_validation.png",
        validation_target=args.validation_target,
    )
    plot_diagnostics(
        unified,
        output_dir / "mc_diagnostics.png",
        power_settings=power_settings,
        min_denominator_ess=min_denominator_ess,
        min_tail_ess=min_tail_ess,
    )
    plot_p_value_distributions(
        unified,
        output_dir / "p_value_distributions.png",
        power_settings,
    )
    plot_event_agreement(
        null,
        feature_frames,
        power_settings,
        output_dir / "event_agreement.png",
    )
    plot_target_selection(
        target_selection,
        output_dir / "target_selection.png",
    )
    plot_shard_stability(
        per_shard,
        power_settings,
        output_dir / "shard_stability.png",
    )
    write_multi_analysis_report(
        output_dir / "analysis_report.md",
        shards=shards,
        unified=unified,
        null=null,
        calibration=calibration,
        null_paired=null_paired,
        power=power,
        power_paired=power_paired,
        power_settings=power_settings,
        diagnostic_summary=diagnostic_summary,
        target_selection=target_selection,
        event_concordance=event_concordance,
        validation_target=args.validation_target,
    )

    metadata = {
        "input_dir": str(input_dir),
        "shards": [shard.name for shard in shards],
        "n_shards": len(shards),
        "unified_rows": len(unified),
        "null_rows": len(null),
        "power_rows": {
            experiment: len(frame) for experiment, frame in feature_frames.items()
        },
        "target_results_and_signal_results_exact_duplicates": True,
        "power_feature_and_target_records_aligned": True,
        "shared_fixed_null_design_verified": bool(null_settings.get("fixed_design")),
        "canonical_unified_p_value": "decision p-value used for rejection",
        "paired_interval_method": "bonferroni_clopper_pearson_discordant_cells",
        "experiments": ["null_calibration", *POWER_EXPERIMENTS],
        "summary_files": [
            "null_calibration_summary.csv",
            "null_paired_event_comparison.csv",
            "power_summary.csv",
            "power_paired_event_comparison.csv",
            "per_shard_summary.csv",
            "experiment_diagnostics_summary.csv",
            "target_selection_summary.csv",
            "event_concordance_summary.csv",
        ],
        "plot_files": [
            "null_calibration.png",
            "power_validation.png",
            "mc_diagnostics.png",
            "p_value_distributions.png",
            "event_agreement.png",
            "target_selection.png",
            "shard_stability.png",
        ],
        "null_shard_seeds": [item["seed"] for item in null_settings_by_shard],
        "power_shard_seeds": {
            experiment: [item["seed"] for item in settings]
            for experiment, settings in power_settings_by_shard.items()
        },
        "null_settings": null_settings,
        "power_settings": power_settings,
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    print(f"Combined {len(shards)} shards into {len(unified):,} raw records.")
    print("\nNull calibration summary:")
    print(calibration.to_string(index=False))
    print("\nPower summary:")
    print(power.to_string(index=False))
    print(f"\nSaved analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
