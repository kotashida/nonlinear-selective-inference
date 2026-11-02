"""Pool and audit the matched 200-iteration AIS follow-up experiment."""

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

import numpy as np
import pandas as pd
from scipy import stats


CONDITIONS = (
    "baseline_control",
    "multimodal_qadapt",
    "reallocated_pilot_exploration",
)
STRATEGIES = {
    "baseline_control": "baseline",
    "multimodal_qadapt": "multimodal_qadapt",
    "reallocated_pilot_exploration": "reallocated_pilot_exploration",
}
EXPECTED_SHARDS = 48
EXPECTED_ITERATIONS = 200


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    return parser.parse_args()


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().eq("true")


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(0.975)
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return float(center - radius), float(center + radius)


def load_condition(root: Path, condition: str) -> pd.DataFrame:
    rows = []
    settings_paths = sorted((root / condition).glob("shard_*/validation_settings.json"))
    if len(settings_paths) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"{condition}: expected {EXPECTED_SHARDS} completed shards, "
            f"found {len(settings_paths)}"
        )

    null_iterations = 0
    power_iterations = 0
    for settings_path in settings_paths:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["methods"] == ["lime"]
        assert settings["designs"] == ["baseline"]
        assert settings["comparison_events"] == ["exact_set"]
        assert settings["signal_strengths"] == [1.0]
        assert settings["pilot_iters"] == 10 and settings["pilot_samples"] == 320
        assert settings["max_final_samples"] == 51200
        assert settings["final_batch_size"] == 800
        assert settings["min_denominator_ess"] == 80.0
        assert settings["min_tail_ess"] == 15.0
        assert settings["stop_when_ess_met"] is True
        assert settings["lime_params"] == {
            "kernel_width": None,
            "num_explanations": 40,
            "ridge_alpha": 1.0,
        }
        assert settings["ais_strategy"] == STRATEGIES[condition]
        null_iterations += int(settings["n_null_iters"])
        power_iterations += int(settings["n_power_iters"])

        shard_dir = settings_path.parent
        files = (
            ("null", shard_dir / "lime" / "baseline" / "null_fresh" / "p_value_results.csv"),
            (
                "power",
                shard_dir
                / "lime"
                / "baseline"
                / "power_feature_0_beta_1p0"
                / "feature_results.csv",
            ),
        )
        for scope, path in files:
            frame = pd.read_csv(path)
            frame = frame.loc[frame["selection_event"].eq("exact_set")].copy()
            frame["scope"] = scope
            frame["signal_strength"] = np.nan if scope == "null" else 1.0
            frame["shard"] = shard_dir.name
            rows.append(frame)

    if null_iterations != EXPECTED_ITERATIONS or power_iterations != EXPECTED_ITERATIONS:
        raise RuntimeError(
            f"{condition}: expected {EXPECTED_ITERATIONS} null and power iterations, "
            f"found {null_iterations} and {power_iterations}"
        )
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 2 * EXPECTED_ITERATIONS:
        raise RuntimeError(
            f"{condition}: expected {2 * EXPECTED_ITERATIONS} target tests, "
            f"found {len(result)}"
        )
    result.insert(0, "condition", condition)
    result["failed"] = _as_bool(result["failed"])
    return result


def failure_summary(condition: str, scope: str, group: pd.DataFrame) -> dict:
    failures = int(group["failed"].sum())
    trials = len(group)
    lower, upper = _wilson_interval(failures, trials)
    denominator = pd.to_numeric(group["denominator_ess"], errors="coerce").fillna(0.0)
    tail = pd.to_numeric(group["tail_ess"], errors="coerce").fillna(0.0)
    proposals = pd.to_numeric(group["proposals"], errors="raise")
    evaluations = pd.to_numeric(group["aj_evaluations"], errors="raise")
    return {
        "condition": condition,
        "scope": scope,
        "trials": trials,
        "failures": failures,
        "failure_rate": failures / trials,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "below_denominator_ess_80": int((denominator < 80.0).sum()),
        "below_tail_ess_15": int((tail < 15.0).sum()),
        "cap_hits": int((proposals == 51200).sum()),
        "mean_aj_evaluations": float(evaluations.mean()),
        "median_denominator_ess": float(denominator.median()),
        "median_tail_ess": float(tail.median()),
    }


def main():
    root = parse_args().input_dir.resolve()
    frames = [load_condition(root, condition) for condition in CONDITIONS]
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(root / "all_target_diagnostics.csv", index=False)

    summaries = []
    for condition, group in detail.groupby("condition", sort=False):
        summaries.append(failure_summary(condition, "all", group))
        for scope, scoped in group.groupby("scope", sort=False):
            summaries.append(failure_summary(condition, scope, scoped))
        signal = group.loc[
            group["scope"].eq("power") & _as_bool(group["is_signal"])
        ]
        summaries.append(failure_summary(condition, "power_signal_targets", signal))
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "failure_summary.csv", index=False)

    # Confirm that each arm used the same observed problem before pairing outcomes.
    keys = ["scope", "shard", "iteration"]
    matched_fields = ["target_feature", "observed_selected_features", "t_obs"]
    for field in matched_fields:
        counts = detail.groupby(keys, dropna=False)[field].nunique(dropna=False)
        if not counts.le(1).all():
            raise RuntimeError(f"The three arms are not matched on {field}.")

    failure_wide = detail.pivot(index=keys, columns="condition", values="failed")
    paired_rows = []
    for scope in ("all", "null", "power"):
        scoped = failure_wide if scope == "all" else failure_wide.xs(scope, level="scope")
        for candidate in CONDITIONS[1:]:
            fixed = int((scoped["baseline_control"] & ~scoped[candidate]).sum())
            regressed = int((~scoped["baseline_control"] & scoped[candidate]).sum())
            discordant = fixed + regressed
            p_value = (
                1.0
                if discordant == 0
                else float(
                    stats.binomtest(
                        min(fixed, regressed), discordant, 0.5, alternative="two-sided"
                    ).pvalue
                )
            )
            paired_rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "baseline_failed_candidate_succeeded": fixed,
                    "baseline_succeeded_candidate_failed": regressed,
                    "net_failures_prevented": fixed - regressed,
                    "discordant_pairs": discordant,
                    "mcnemar_exact_two_sided_p": p_value,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "paired_failure_comparisons.csv", index=False)

    (root / "COMPLETED").touch()
    print(summary.loc[summary["scope"].eq("power")].to_string(index=False))
    print("\nPaired power comparisons against baseline:")
    print(paired.loc[paired["scope"].eq("power")].to_string(index=False))


if __name__ == "__main__":
    main()
