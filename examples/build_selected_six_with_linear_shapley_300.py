"""Combine the selected-five KDE-2.5 study with matched Linear Shapley results."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "examples")]

import numpy as np
import pandas as pd

import analyze_ais_method_comparison as raw_analysis
import analyze_matched_ais_methods as summary_analysis
import plot_combined_ais_analysis as plotting

BASE = ROOT / "outputs/selection_method_validation/exact_set"
REFERENCE = BASE / "selected_five_kde2p5_300_20260923"
LINEAR = BASE / "linear_shapley_selected_five_matched_300_20260928"
LINEAR_RUN = LINEAR / "runs/linear_shapley"
OUT = BASE / "selected_six_with_linear_shapley_300_20260928"
BETAS = (0.3, 0.5, 0.75, 1.0)
METHODS = {
    "shap": "Tree SHAP (AIS)",
    "lime": "LIME (40, AIS)",
    "spline_screening": "Spline screening (AIS)",
    "spline_screening_exact": "Spline screening (exact)",
    "mutual_information": "Mutual information (KDE 2.5, AIS)",
    "linear_shapley": "Linear RSS Shapley (AIS)",
}
COLORS = {
    "shap": "#2674a6",
    "lime": "#d58122",
    "spline_screening": "#258568",
    "spline_screening_exact": "#c44e52",
    "mutual_information": "#9256a0",
    "linear_shapley": "#6b6ecf",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_linear_rows():
    settings = read_json(LINEAR_RUN / "pooled_summary/validation_settings.json")
    raw_analysis.METHOD_LABELS["linear_shapley"] = METHODS["linear_shapley"]
    null, target, feature = raw_analysis._read_raw(LINEAR_RUN, "ais", settings, 32)
    keys = [
        "selection_method", "validation_shard", "iteration",
        "selection_event", "signal_strength",
    ]
    diagnostics = [
        "feature", "selection_position", "importance_score", "selection_rank",
        "shap_importance", "shap_rank", "test_rank", "t_obs", "radial_t_obs",
        "test_statistic", "unadjusted_p_value", "unadjusted_p_value_conditioning",
        "raw_selective_p_value", "target_rule", "selection_event_definition",
        "status", "proposals", "selected_samples", "tail_samples",
        "denominator_ess", "tail_ess", "mc_se", "mc_ci_95_lower",
        "mc_ci_95_upper", "sampling_mode", "p_value_method",
        "finite_sample_valid", "ais_strategy", "pilot_evaluations",
        "aj_evaluations", "pilot_sampling_scheme", "pilot_selected_samples",
        "pilot_low_selected_samples", "pilot_validation_samples",
        "pre_merge_cluster_count", "post_merge_cluster_count",
        "adaptive_component_count", "minimum_adapted_component_weight",
        "absolute_component_draw_floor", "denominator_component_count",
        "final_base_weight", "final_observed_weight", "final_adapted_weight",
        "final_denominator_weight", "final_component_count",
        "final_allocation_method", "mixture_optimization_succeeded",
        "max_normalized_weight", "weight_gini", "psis_khat", "tail_psis_khat",
        "component_diagnostics_json",
    ]
    available = [column for column in diagnostics if column in feature.columns]
    target = target.merge(feature[keys + available], on=keys, validate="one_to_one")
    relative_run = LINEAR_RUN.relative_to(BASE).as_posix()
    for frame, scope in ((null, "null_calibration"), (target, "power")):
        frame["scope"] = scope
        frame["ais_setting"] = "combined"
        frame["source_run"] = relative_run
        frame["max_final_samples"] = 51200
        frame["pilot_iters"] = 10
        frame["pilot_samples"] = 320
        frame["total_proposals"] = frame["proposals"] + 3200
        frame["option"] = METHODS["linear_shapley"]
        frame["mi_neighbors"] = np.nan
        frame["mi_estimator"] = None
        frame["mi_bandwidth"] = np.nan
    return pd.concat([null, target], ignore_index=True, sort=False), settings


def audit_pairing(reference, linear):
    rows = []
    comparator = reference[reference["selection_method"].eq("shap")]
    for scope, expected in (("null_calibration", 300), ("power", 1200)):
        left = comparator[comparator["scope"].eq(scope)]
        right = linear[linear["scope"].eq(scope)]
        keys = ["validation_shard", "iteration", "selection_event"]
        if scope == "power":
            keys.append("signal_strength")
        pair = left.merge(right, on=keys, suffixes=("_reference", "_linear"),
                          validate="one_to_one")
        require(len(pair) == expected, f"Pairing count drift: {scope}")
        np.testing.assert_array_equal(pair["target_seed_reference"], pair["target_seed_linear"])
        np.testing.assert_allclose(pair["auxiliary_u_reference"], pair["auxiliary_u_linear"],
                                   rtol=0, atol=0)
        if scope == "null_calibration":
            np.testing.assert_array_equal(pair["ais_seed_reference"], pair["ais_seed_linear"])
        rows.append({
            "scope": scope,
            "matched_trials": len(pair),
            "shared_condition_keys": True,
            "target_seeds_match": True,
            "auxiliary_draws_match": True,
            "comparison_reference": "shap",
            "note": "Targets and observed statistics may differ because the selectors differ.",
        })
    return pd.DataFrame(rows)


def main():
    require((REFERENCE / "COMPLETED").is_file(), "Reference study is incomplete")
    require((LINEAR / "COMPLETED").is_file(), "Linear Shapley study is incomplete")
    reference = pd.read_csv(REFERENCE / "unified_tests.csv")
    require(len(reference) == 7500, "Expected 7,500 selected-five rows")
    linear, linear_settings = read_linear_rows()
    require(len(linear) == 1500, "Expected 1,500 Linear Shapley rows")
    require(not linear["failed"].isna().any(), "Missing Linear Shapley failure indicators")
    pairing = audit_pairing(reference, linear)

    extra_columns = [column for column in linear if column not in reference]
    data = pd.concat(
        [reference, linear.reindex(columns=[*reference.columns, *extra_columns])],
        ignore_index=True,
        sort=False,
    )
    keys = [
        "selection_method", "scope", "validation_shard", "iteration",
        "selection_event", "signal_strength",
    ]
    require(len(data) == 9000, f"Expected 9,000 combined rows, got {len(data)}")
    require(set(data["selection_method"]) == set(METHODS), "Method set drift")
    require(not data.duplicated(keys).any(), "Duplicate method/trial rows")
    counts = data.groupby(["selection_method", "scope"]).size().unstack()
    require((counts["null_calibration"] == 300).all(), "Null count drift")
    require((counts["power"] == 1200).all(), "Power count drift")

    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "unified_tests.csv", index=False)
    pairing.to_csv(OUT / "trial_pairing_audit.csv", index=False)
    selected = {
        "methods": list(METHODS),
        "reference": REFERENCE.relative_to(ROOT).as_posix(),
        "linear_shapley_source": LINEAR.relative_to(ROOT).as_posix(),
        "trials_per_condition": 300,
        "total_tests": 9000,
        "intentional_difference": "Feature selector and matching test statistic",
        "linear_shapley_selector": linear_settings,
    }
    (OUT / "selected_configuration.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    manifest = read_json(REFERENCE / "source_manifest.json")
    manifest[LINEAR_RUN.relative_to(BASE).as_posix()] = linear_settings
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary_analysis.OUT = OUT
    power, calibration = summary_analysis.summarize(data)
    require(len(power) == len(METHODS) * len(BETAS), "Power summary row drift")
    require(len(calibration) == len(METHODS) * 3, "Calibration summary row drift")

    plotting.SOURCE = plotting.OUT = OUT
    plotting.DATA_FILENAME = "unified_tests.csv"
    plotting.TRIALS_PER_CONDITION = 300
    plotting.REPORT_LABEL = "Selected six-method matched comparison"
    plotting.REPORT_INTRO = (
        "Matched comparison with 300 null trials and 300 power trials per beta per method. "
        "All methods share the baseline design, shard allocation, simulation seeds, exact-set "
        "conditioning, target randomization, and applicable AIS controls. Linear RSS Shapley "
        "uses its matching quadratic test; spline exact uses analytic inference."
    )
    plotting.METHODS = METHODS.copy()
    plotting.COLORS = COLORS.copy()
    plotting.EXPECTED_PILOT_ITERS = (10,)
    plotting.ALLOW_INCOMPLETE = False
    plotting.FIGURES.clear()
    plotting.main()
    (OUT / "COMPLETED").write_text(
        "Verified 9,000 matched records and generated the six-method analysis.\n",
        encoding="utf-8",
    )
    print(f"Built {OUT} from {len(data):,} verified records.")


if __name__ == "__main__":
    main()
