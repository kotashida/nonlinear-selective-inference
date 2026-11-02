"""Add the matched exact-spline reference to the selected four-method dataset."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_matched_ais_methods as analysis
from analyze_ais_method_comparison import _read_raw
import plot_combined_ais_analysis as plotting


BASE = analysis.BASE
AIS_DIR = BASE / "selected_four_methods_combined_ais_mi30"
EXACT_DIR = BASE / "exact_spline_matched_ais_mi30"
OUT = BASE / "selected_five_methods_combined_ais_exact_mi30"
EXACT_METHOD = "spline_screening_exact"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ais = pd.read_csv(AIS_DIR / "unified_tests.csv")
    exact_settings = read_json(EXACT_DIR / "pooled_summary/validation_settings.json")
    ais_manifest = read_json(AIS_DIR / "source_manifest.json")
    ais_spline_rel = ais.loc[ais.selection_method.eq("spline_screening"), "source_run"].iat[0]
    ais_spline_dir = BASE / ais_spline_rel
    ais_settings = read_json(ais_spline_dir / "pooled_summary/validation_settings.json")

    # Pooled experimental settings must match. Method membership and the two
    # inference-specific controls are deliberately different for the exact arm.
    expected = {
        "design_seed": 314159, "designs": ["baseline"],
        "fixed_auxiliary_values": [None], "max_final_samples": 51200,
        "final_batch_size": 800, "mcmc_steps": 20,
        "min_denominator_ess": 80.0, "min_tail_ess": 15.0,
        "minimum_calibration_iterations": 200,
        "minimum_conditional_power": 0.8, "minimum_signal_targets": 30,
        "n_null_iters": 200, "n_power_iters": 200, "n_shards": 32,
        "pilot_iters": 10, "pilot_samples": 320, "preset": "comprehensive",
        "primary_selection_event": "exact_set", "rf_jobs": 1,
        "shard_root_seeds": list(range(20260908, 20260940)),
        "signal_positions": ["first"], "signal_strengths": [.3, .5, .75, 1.],
    }
    for key, value in expected.items():
        assert ais_settings[key] == value, ("AIS", key, ais_settings[key], value)
        assert exact_settings[key] == value, ("exact", key, exact_settings[key], value)
    selected = read_json(AIS_DIR / "selected_configuration.json")
    assert selected["methods"]["mutual_information"]["n_neighbors"] == 30
    assert selected["methods"]["lime"]["num_explanations"] == 40
    assert exact_settings["mi_neighbors"] == 30
    assert exact_settings["lime_params"]["num_explanations"] == 40
    assert ais_settings["method_inference_methods"]["spline_screening"] == "ais"
    assert exact_settings["method_inference_methods"] == {"spline_screening": "exact_spline"}
    assert ais_settings["stop_when_ess_met"] is True
    assert exact_settings["stop_when_ess_met"] is False
    assert ais_settings["runtime_metadata"]["package_versions"] == exact_settings["runtime_metadata"]["package_versions"]

    null, target, feature = _read_raw(EXACT_DIR, "exact", exact_settings, 32)
    merge_keys = ["selection_method", "validation_shard", "iteration", "selection_event", "signal_strength"]
    target = target.merge(
        feature[merge_keys + ["feature", "t_obs", "proposals", "denominator_ess", "tail_ess"]],
        on=merge_keys, validate="one_to_one"
    )
    exact_frames = []
    for frame, scope in ((null, "null_calibration"), (target, "power")):
        frame = frame.copy()
        frame["scope"] = scope
        frame["ais_setting"] = "exact"
        frame["source_run"] = str(EXACT_DIR.relative_to(BASE)).replace("\\", "/")
        frame["max_final_samples"] = 51200
        frame["pilot_iters"] = 10
        frame["pilot_samples"] = 320
        frame["total_proposals"] = 0
        frame["selection_method"] = EXACT_METHOD
        frame["option"] = "Spline screening (exact)"
        frame["mi_neighbors"] = np.nan
        exact_frames.append(frame)
    exact = pd.concat(exact_frames, ignore_index=True)
    assert len(exact) == 1000 and not exact.failed.any()

    # Compare all low-level condition settings, ignoring only inference-specific
    # and execution-provenance fields.
    audit_rows = []
    ignored = {"inference_method", "stop_when_ess_met", "ais_strategy", "runtime_metadata"}
    for shard in range(32):
        folders = ["null_fresh", *[f"power_feature_0_beta_{str(b).replace('.', 'p')}" for b in [.3, .5, .75, 1.]]]
        for folder in folders:
            suffix = Path(f"shard_{shard}/spline_screening/baseline/{folder}/settings.json")
            a = read_json(ais_spline_dir / suffix)
            e = read_json(EXACT_DIR / suffix)
            compared = sorted((a.keys() | e.keys()) - ignored)
            for key in compared:
                assert a.get(key) == e.get(key), (shard, folder, key, a.get(key), e.get(key))
            audit_rows.append({"shard": shard, "condition": folder,
                               "experimental_settings_match": True,
                               "excluded_expected_differences": ";".join(sorted(ignored))})

    # The generated data, chosen target, and observed statistic must pair exactly
    # (allowing only floating-point roundoff in the statistic/auxiliary draw).
    trial_rows = []
    ais_spline = ais[ais.selection_method.eq("spline_screening")]
    for scope in ["null_calibration", "power"]:
        left = ais_spline[ais_spline.scope.eq(scope)]
        right = exact[exact.scope.eq(scope)]
        keys = ["validation_shard", "iteration", "selection_event"]
        if scope == "power": keys.append("signal_strength")
        paired = left.merge(right, on=keys, suffixes=("_ais", "_exact"), validate="one_to_one")
        assert len(paired) == (200 if scope == "null_calibration" else 800)
        for key in ["target_seed", "target_feature"]:
            np.testing.assert_array_equal(paired[f"{key}_ais"], paired[f"{key}_exact"])
        for key in ["auxiliary_u", "t_obs"]:
            np.testing.assert_allclose(paired[f"{key}_ais"], paired[f"{key}_exact"], rtol=0, atol=1e-12)
        if scope == "null_calibration":
            np.testing.assert_array_equal(paired.observed_selected_features_ais,
                                          paired.observed_selected_features_exact)
        else:
            np.testing.assert_array_equal(paired.target_is_signal_ais,
                                          paired.target_is_signal_exact)
        trial_rows.append({"scope": scope, "matched_trials": len(paired),
                           "seeds_targets_statistics_match": True})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(OUT / "configuration_audit.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(OUT / "trial_audit.csv", index=False)
    data = pd.concat([ais, exact], ignore_index=True, sort=False)
    assert len(data) == 5000
    data.to_csv(OUT / "unified_tests.csv", index=False)
    manifest = dict(ais_manifest)
    manifest[str(EXACT_DIR.relative_to(BASE)).replace("\\", "/")] = exact_settings
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "selected_configuration.json").write_text(json.dumps({
        "methods": ["shap", "lime_40", "mutual_information_30",
                    "spline_screening_ais", "spline_screening_exact"],
        "matched_experimental_settings": expected,
        "expected_inference_differences": {
            "spline_ais": {"inference_method": "ais", "stop_when_ess_met": True},
            "spline_exact": {"inference_method": "exact_spline", "stop_when_ess_met": False},
        },
        "runtime_provenance_note": "Package versions match; AIS ran on Linux/Python 3.11 and exact ran on Windows/Python 3.12.",
    }, indent=2), encoding="utf-8")

    previous = analysis.OUT
    analysis.OUT = OUT
    analysis.summarize(data)
    analysis.OUT = previous
    plotting.SOURCE = OUT
    plotting.OUT = OUT
    plotting.DATA_FILENAME = "unified_tests.csv"
    plotting.REPORT_LABEL = "Combined AIS plus exact spline"
    plotting.METHODS.clear()
    plotting.METHODS.update({
        "shap": "SHAP", "lime": "LIME (40 explanations)",
        "mutual_information": "Mutual information (30 neighbors)",
        "spline_screening": "Spline screening (AIS)",
        EXACT_METHOD: "Spline screening (exact)",
    })
    plotting.COLORS.clear()
    plotting.COLORS.update(dict(zip(plotting.METHODS,
                                    ["#2674a6", "#d58122", "#9256a0", "#258568", "#c44e52"])))
    plotting.FIGURES.clear()
    plotting.main()
    print(f"Verified 160 matched condition configs and 1,000 paired spline trials; saved to {OUT}")


if __name__ == "__main__":
    main()
