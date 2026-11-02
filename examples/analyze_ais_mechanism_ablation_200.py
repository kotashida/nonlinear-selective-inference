"""Pool the paired 200-trial AIS A/B/C mechanism ablation."""

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


REFERENCE_CONDITIONS = (
    "baseline_control",
    "reallocated_pilot_exploration",
)
NEW_CONDITIONS = {
    "a_multimodal_retention": "ablation_a",
    "b_denominator_targeting": "ablation_b",
    "c_stratified_exploration": "ablation_c",
    "ab_multimodal_denominator": "ablation_ab",
    "ac_multimodal_stratified": "ablation_ac",
    "bc_denominator_stratified": "ablation_bc",
    "abc_all_mechanisms": "ablation_abc",
}
EXPECTED_SHARDS = 48
EXPECTED_ITERATIONS = 200
REFERENCE = "reallocated_pilot_exploration"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
        help="all_target_diagnostics.csv from ais_fix_followup_200",
    )
    return parser.parse_args()


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().eq("true")


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(0.975)
    rate = successes / trials
    scale = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / scale
    radius = z * np.sqrt(
        rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2)
    ) / scale
    return float(center - radius), float(center + radius)


def load_references(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["condition"].isin(REFERENCE_CONDITIONS)].copy()
    expected = 2 * len(REFERENCE_CONDITIONS) * EXPECTED_ITERATIONS
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} carried reference rows, found {len(frame)}.")
    frame["failed"] = _as_bool(frame["failed"])
    frame["is_signal"] = _as_bool(frame["is_signal"])
    return frame


def load_new_condition(root: Path, condition: str, strategy: str) -> pd.DataFrame:
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
        assert settings["seed"] == 2026091501 + int(settings_path.parent.name[6:])
        assert settings["design_seed"] == 314159
        assert settings["inference_method"] == "ais"
        assert settings["primary_selection_event"] == "exact_set"
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
        assert settings["ais_strategy"] == strategy
        null_iterations += int(settings["n_null_iters"])
        power_iterations += int(settings["n_power_iters"])
        shard = settings_path.parent.name
        paths = (
            ("null", settings_path.parent / "lime/baseline/null_fresh/p_value_results.csv"),
            (
                "power",
                settings_path.parent
                / "lime/baseline/power_feature_0_beta_1p0/feature_results.csv",
            ),
        )
        for scope, path in paths:
            part = pd.read_csv(path)
            part = part.loc[part["selection_event"].eq("exact_set")].copy()
            part["scope"] = scope
            part["signal_strength"] = np.nan if scope == "null" else 1.0
            part["shard"] = shard
            rows.append(part)
    if null_iterations != EXPECTED_ITERATIONS or power_iterations != EXPECTED_ITERATIONS:
        raise RuntimeError(
            f"{condition}: expected 200 null and power iterations; found "
            f"{null_iterations} and {power_iterations}."
        )
    frame = pd.concat(rows, ignore_index=True)
    if len(frame) != 2 * EXPECTED_ITERATIONS:
        raise RuntimeError(f"{condition}: expected 400 target rows, found {len(frame)}.")
    frame.insert(0, "condition", condition)
    frame["failed"] = _as_bool(frame["failed"])
    frame["is_signal"] = _as_bool(frame["is_signal"])
    return frame


def summarize(condition: str, scope: str, group: pd.DataFrame) -> dict:
    failures = int(group["failed"].sum())
    trials = len(group)
    lower, upper = _wilson(failures, trials)
    denominator = pd.to_numeric(group["denominator_ess"], errors="coerce").fillna(0.0)
    tail = pd.to_numeric(group["tail_ess"], errors="coerce").fillna(0.0)
    proposals = pd.to_numeric(group["proposals"], errors="raise")
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
        "median_denominator_ess": float(denominator.median()),
        "median_tail_ess": float(tail.median()),
    }


def paired_comparisons(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["scope", "shard", "iteration"]
    wide = detail.pivot(index=keys, columns="condition", values="failed")
    rows = []
    candidates = (*NEW_CONDITIONS.keys(), "baseline_control")
    for scope in ("power", "power_signal_targets"):
        if scope == "power":
            scoped_detail = detail.loc[detail["scope"].eq("power")]
        else:
            scoped_detail = detail.loc[
                detail["scope"].eq("power") & detail["is_signal"]
            ]
        scoped = scoped_detail.pivot(index=keys, columns="condition", values="failed")
        for candidate in candidates:
            fixed = int((scoped[REFERENCE] & ~scoped[candidate]).sum())
            regressed = int((~scoped[REFERENCE] & scoped[candidate]).sum())
            discordant = fixed + regressed
            p_value = (
                1.0
                if discordant == 0
                else float(
                    stats.binomtest(
                        min(fixed, regressed),
                        discordant,
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                )
            )
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "reference": REFERENCE,
                    "reference_failed_candidate_succeeded": fixed,
                    "reference_succeeded_candidate_failed": regressed,
                    "net_failures_prevented": fixed - regressed,
                    "discordant_pairs": discordant,
                    "mcnemar_exact_two_sided_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def validate_matched_scenarios(detail: pd.DataFrame) -> None:
    """Confirm exact scenario identity, allowing only roundoff in t_obs."""
    keys = ["scope", "shard", "iteration"]
    for field in ("feature", "is_signal", "auxiliary_u"):
        counts = detail.groupby(keys, dropna=False)[field].nunique(dropna=False)
        if not counts.le(1).all():
            mismatched = counts.loc[counts.gt(1)]
            raise RuntimeError(
                f"The nine arms are not matched on {field}; first mismatched "
                f"scenario: {mismatched.index[0]}."
            )

    worst_key = None
    worst_spread = 0.0
    worst_scale = 1.0
    for key, group in detail.groupby(keys, dropna=False, sort=False):
        values = pd.to_numeric(group["t_obs"], errors="coerce").to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size != values.size or not finite.size:
            raise RuntimeError(f"Nonfinite t_obs in matched scenario {key}.")
        spread = float(np.max(finite) - np.min(finite))
        scale = max(1.0, float(np.max(np.abs(finite))))
        if spread > worst_spread:
            worst_key = key
            worst_spread = spread
            worst_scale = scale
    if worst_spread > 1e-12 + 1e-10 * worst_scale:
        raise RuntimeError(
            "The nine arms have a substantive t_obs mismatch: "
            f"scenario={worst_key}, spread={worst_spread:.17g}, "
            f"scale={worst_scale:.17g}."
        )
    print(
        "Matched-scenario audit passed; maximum cross-arm t_obs spread="
        f"{worst_spread:.3g} at {worst_key}."
    )


def write_report(root: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    power = summary.loc[summary["scope"].eq("power")].set_index("condition")
    signal = summary.loc[
        summary["scope"].eq("power_signal_targets")
    ].set_index("condition")
    power_comparisons = paired.loc[
        paired["scope"].eq("power")
    ].set_index("candidate")
    signal_comparisons = paired.loc[
        paired["scope"].eq("power_signal_targets")
    ].set_index("candidate")
    order = [*REFERENCE_CONDITIONS, *NEW_CONDITIONS.keys()]
    lines = [
        "# AIS exact-set A/B/C ablation (200 paired trials)",
        "",
        "All arms use the exact same scenario seed schedule, exact-set conditioning, "
        "3,200 pilot evaluations, and a maximum of 51,200 final draws.",
        "",
        "| Arm | Power failures | Signal-target failures | Net power failures | Power McNemar p | Net signal failures | Signal McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in order:
        p = power.loc[condition]
        s = signal.loc[condition]
        if condition == REFERENCE:
            power_net = signal_net = "reference"
            power_p = signal_p = "—"
        else:
            power_comparison = power_comparisons.loc[condition]
            signal_comparison = signal_comparisons.loc[condition]
            power_net = f"{int(power_comparison.net_failures_prevented):+d}"
            signal_net = f"{int(signal_comparison.net_failures_prevented):+d}"
            power_p = f"{power_comparison.mcnemar_exact_two_sided_p:.4g}"
            signal_p = f"{signal_comparison.mcnemar_exact_two_sided_p:.4g}"
        lines.append(
            f"| {condition} | {int(p.failures)}/{int(p.trials)} | "
            f"{int(s.failures)}/{int(s.trials)} | {power_net} | {power_p} | "
            f"{signal_net} | {signal_p} |"
        )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = args.input_dir.resolve()
    frames = [load_references(args.reference_csv.resolve())]
    frames.extend(
        load_new_condition(root, condition, strategy)
        for condition, strategy in NEW_CONDITIONS.items()
    )
    detail = pd.concat(frames, ignore_index=True, sort=False)

    validate_matched_scenarios(detail)

    summaries = []
    for condition, group in detail.groupby("condition", sort=False):
        power = group.loc[group["scope"].eq("power")]
        signal = power.loc[power["is_signal"]]
        summaries.append(summarize(condition, "power", power))
        summaries.append(summarize(condition, "power_signal_targets", signal))
    summary = pd.DataFrame(summaries)
    paired = paired_comparisons(detail)
    detail.to_csv(root / "all_target_diagnostics.csv", index=False)
    summary.to_csv(root / "failure_summary.csv", index=False)
    paired.to_csv(root / "paired_failure_comparisons.csv", index=False)
    write_report(root, summary, paired)
    (root / "COMPLETED").touch()
    print(summary.to_string(index=False))
    print("\nPaired comparisons against current best:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
