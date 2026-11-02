"""Pool the six paired 200-trial robust-A refinement arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats


NEW_CONDITIONS = {
    "a_robust_consolidation": ("a_robust_consolidation", 10),
    "a_robust_draw_floor": ("a_robust_draw_floor", 10),
    "a_robust_draw_floor_pilot15": ("a_robust_draw_floor_pilot15", 15),
    "a_cdf_interval": ("a_cdf_interval", 10),
    "a_robust_floor_multimodal_denominator": (
        "a_robust_floor_multimodal_denominator", 10
    ),
    "a_full_stack_optimized": ("a_full_stack_optimized", 15),
}
REFERENCES = ("baseline_control", "a_multimodal_retention")
EXPECTED_SHARDS = 48
EXPECTED_ITERATIONS = 200


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--mechanism-csv", type=Path, required=True)
    return parser.parse_args()


def _as_bool(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().eq("true")


def load_reference(path, condition):
    frame = pd.read_csv(path)
    frame = frame.loc[frame["condition"].eq(condition)].copy()
    if len(frame) != 2 * EXPECTED_ITERATIONS:
        raise RuntimeError(
            f"{condition}: expected 400 reference rows, found {len(frame)}."
        )
    frame["failed"] = _as_bool(frame["failed"])
    frame["is_signal"] = _as_bool(frame["is_signal"])
    frame["scope"] = frame["scope"].fillna("").replace("", "null")
    return frame


def load_new_condition(root, condition, strategy, pilot_iters):
    rows = []
    paths = sorted((root / condition).glob("shard_*/validation_settings.json"))
    if len(paths) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"{condition}: expected {EXPECTED_SHARDS} completed shards, found {len(paths)}."
        )
    totals = {"null": 0, "power": 0}
    for settings_path in paths:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        shard = settings_path.parent.name
        shard_index = int(shard[6:])
        assert settings["methods"] == ["lime"]
        assert settings["designs"] == ["baseline"]
        assert settings["comparison_events"] == ["exact_set"]
        assert settings["primary_selection_event"] == "exact_set"
        assert settings["signal_strengths"] == [1.0]
        assert settings["seed"] == 2026091501 + shard_index
        assert settings["design_seed"] == 314159
        assert settings["inference_method"] == "ais"
        assert settings["pilot_iters"] == pilot_iters
        assert settings["pilot_samples"] == 320
        assert settings["max_final_samples"] == 51200
        assert settings["final_batch_size"] == 800
        assert settings["min_denominator_ess"] == 80.0
        assert settings["min_tail_ess"] == 15.0
        assert settings["stop_when_ess_met"] is True
        assert settings["ais_strategy"] == strategy
        assert settings["lime_params"] == {
            "kernel_width": None, "num_explanations": 40, "ridge_alpha": 1.0
        }
        totals["null"] += int(settings["n_null_iters"])
        totals["power"] += int(settings["n_power_iters"])
        result_paths = {
            "null": settings_path.parent / "lime/baseline/null_fresh/p_value_results.csv",
            "power": settings_path.parent
            / "lime/baseline/power_feature_0_beta_1p0/feature_results.csv",
        }
        for scope, result_path in result_paths.items():
            part = pd.read_csv(result_path)
            part = part.loc[part["selection_event"].eq("exact_set")].copy()
            part["scope"] = scope
            part["signal_strength"] = np.nan if scope == "null" else 1.0
            part["shard"] = shard
            rows.append(part)
    if totals != {"null": 200, "power": 200}:
        raise RuntimeError(f"{condition}: unexpected iteration totals {totals}.")
    frame = pd.concat(rows, ignore_index=True)
    if len(frame) != 400:
        raise RuntimeError(f"{condition}: expected 400 rows, found {len(frame)}.")
    frame.insert(0, "condition", condition)
    frame["failed"] = _as_bool(frame["failed"])
    frame["is_signal"] = _as_bool(frame["is_signal"])
    required = {
        "pre_merge_cluster_count", "post_merge_cluster_count",
        "component_diagnostics_json", "max_normalized_weight", "weight_gini",
        "psis_khat", "tail_psis_khat",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{condition}: missing forward diagnostics {sorted(missing)}.")
    return frame


def validate_matched_scenarios(detail):
    keys = ["scope", "shard", "iteration"]
    counts = detail.groupby(keys, dropna=False)["condition"].nunique()
    if not counts.eq(len(REFERENCES) + len(NEW_CONDITIONS)).all():
        raise RuntimeError("Arms do not contain identical scenario keys.")
    for field in ("feature", "is_signal", "auxiliary_u"):
        unique = detail.groupby(keys, dropna=False)[field].nunique(dropna=False)
        if not unique.le(1).all():
            raise RuntimeError(f"Arms are not matched on {field}.")
    spread = detail.groupby(keys, dropna=False)["t_obs"].agg(
        lambda values: pd.to_numeric(values).max() - pd.to_numeric(values).min()
    )
    if float(spread.max()) > 1e-10:
        raise RuntimeError(f"Substantive matched-scenario t_obs spread: {spread.max()}.")


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, np.nan, np.nan
    radius = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return mean, mean - radius, mean + radius


def _median_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, values.size), replace=True)
    medians = np.median(draws, axis=1)
    return (
        float(np.median(values)), float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def summarize(detail):
    rows = []
    for index, (condition, group) in enumerate(detail.groupby("condition", sort=False)):
        power = group.loc[group["scope"].eq("power")]
        signal = power.loc[power["is_signal"]]
        mean, mean_low, mean_high = _mean_ci(power["aj_evaluations"])
        median, median_low, median_high = _median_ci(
            power["aj_evaluations"], 9017 + index
        )
        rows.append({
            "condition": condition,
            "power_trials": len(power),
            "power_failures": int(power["failed"].sum()),
            "signal_targets": len(signal),
            "signal_failures": int(signal["failed"].sum()),
            "mean_cost": mean, "mean_cost_ci_95_lower": mean_low,
            "mean_cost_ci_95_upper": mean_high,
            "median_cost": median, "median_cost_ci_95_lower": median_low,
            "median_cost_ci_95_upper": median_high,
        })
    return pd.DataFrame(rows)


def paired_failure_comparisons(detail):
    keys = ["scope", "shard", "iteration"]
    rows = []
    for scope in ("power", "power_signal_targets"):
        scoped = detail.loc[detail["scope"].eq("power")]
        if scope.endswith("signal_targets"):
            scoped = scoped.loc[scoped["is_signal"]]
        wide = scoped.pivot(index=keys, columns="condition", values="failed")
        for candidate in NEW_CONDITIONS:
            for reference in REFERENCES:
                fixed = int((wide[reference] & ~wide[candidate]).sum())
                regressed = int((~wide[reference] & wide[candidate]).sum())
                discordant = fixed + regressed
                p_value = 1.0 if not discordant else float(
                    stats.binomtest(min(fixed, regressed), discordant, 0.5).pvalue
                )
                rows.append({
                    "scope": scope, "candidate": candidate, "reference": reference,
                    "reference_failed_candidate_succeeded": fixed,
                    "reference_succeeded_candidate_failed": regressed,
                    "net_failures_prevented": fixed - regressed,
                    "discordant_pairs": discordant,
                    "mcnemar_exact_two_sided_p": p_value,
                })
    return pd.DataFrame(rows)


def paired_cost_comparisons(detail):
    keys = ["shard", "iteration"]
    power = detail.loc[detail["scope"].eq("power")]
    wide = power.pivot(index=keys, columns="condition", values="aj_evaluations")
    rows = []
    for index, candidate in enumerate(NEW_CONDITIONS):
        for reference in REFERENCES:
            differences = wide[candidate] - wide[reference]
            mean, low, high = _mean_ci(differences)
            median, median_low, median_high = _median_ci(
                differences, 17011 + index
            )
            rows.append({
                "candidate": candidate, "reference": reference,
                "mean_paired_cost_difference": mean,
                "mean_difference_ci_95_lower": low,
                "mean_difference_ci_95_upper": high,
                "median_paired_cost_difference": median,
                "median_difference_ci_95_lower": median_low,
                "median_difference_ci_95_upper": median_high,
            })
    return pd.DataFrame(rows)


def write_report(root, summary, failures, costs):
    lines = [
        "# Robust-A refinement study (200 paired trials)", "",
        "All arms use exact-set conditioning, the historical 200-scenario seed schedule, "
        "and a 51,200 final-draw cap. Arms 3 and 6 use 4,800 pilot evaluations; "
        "the other new arms use 3,200.", "",
        "| Arm | Power failures | Signal failures | Mean cost (95% CI) | Median cost (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    order = [*REFERENCES, *NEW_CONDITIONS]
    indexed = summary.set_index("condition")
    for condition in order:
        row = indexed.loc[condition]
        lines.append(
            f"| {condition} | {int(row.power_failures)}/{int(row.power_trials)} | "
            f"{int(row.signal_failures)}/{int(row.signal_targets)} | "
            f"{row.mean_cost:.1f} ({row.mean_cost_ci_95_lower:.1f}, "
            f"{row.mean_cost_ci_95_upper:.1f}) | {row.median_cost:.1f} "
            f"({row.median_cost_ci_95_lower:.1f}, {row.median_cost_ci_95_upper:.1f}) |"
        )
    lines.extend([
        "", "Paired exact McNemar comparisons are in `paired_failure_comparisons.csv`; "
        "paired mean and median cost-difference intervals are in "
        "`paired_cost_comparisons.csv`. Negative cost differences favor the new arm.",
        "Forward-only component and weight diagnostics remain in "
        "`all_target_diagnostics.csv`.",
    ])
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = args.input_dir.resolve()
    frames = [
        load_reference(args.baseline_csv.resolve(), "baseline_control"),
        load_reference(args.mechanism_csv.resolve(), "a_multimodal_retention"),
    ]
    frames.extend(
        load_new_condition(root, condition, strategy, pilot_iters)
        for condition, (strategy, pilot_iters) in NEW_CONDITIONS.items()
    )
    detail = pd.concat(frames, ignore_index=True, sort=False)
    validate_matched_scenarios(detail)
    summary = summarize(detail)
    failures = paired_failure_comparisons(detail)
    costs = paired_cost_comparisons(detail)
    detail.to_csv(root / "all_target_diagnostics.csv", index=False)
    summary.to_csv(root / "arm_summary.csv", index=False)
    failures.to_csv(root / "paired_failure_comparisons.csv", index=False)
    costs.to_csv(root / "paired_cost_comparisons.csv", index=False)
    write_report(root, summary, failures, costs)
    (root / "COMPLETED").touch()
    print(summary.to_string(index=False))
    print("\nPaired failure comparisons:\n", failures.to_string(index=False))
    print("\nPaired cost comparisons:\n", costs.to_string(index=False))


if __name__ == "__main__":
    main()
