"""Run and verify the matched five-method KDE-2.5 confirmation experiment.

Actions:
  check   validate and print the immutable experiment manifest
  run     run both arms, verify them, and build the combined report
  verify  verify existing raw results and rebuild the combined report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
MANIFEST = ROOT / "examples/selected_five_kde2p5_300.json"
BETAS = (0.3, 0.5, 0.75, 1.0)
AIS_METHODS = ("shap", "lime", "spline_screening", "mutual_information")
EXACT_METHOD = "spline_screening_exact"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_manifest():
    m = read_json(MANIFEST)
    e = m["environment"]
    from examples.validate_selection_methods import DESIGNS
    require(m["design"] == DESIGNS["baseline"], "Baseline design drift")
    require(e["TOTAL_NULL_ITERS"] == e["TOTAL_POWER_ITERS"] == "300", "Expected 300 trials")
    require(1 <= int(e["PARALLEL_SHARDS"]) <= int(e["N_SHARDS"]) <= 32, "Shard CPU cap exceeded")
    require(int(e["PARALLEL_SHARDS"]) * int(e["RF_JOBS"]) <= int(e["CPU_BUDGET"]) <= 32,
            "CPU budget exceeded")
    require(e["MI_ESTIMATOR"] == "kde" and e["MI_BANDWIDTH"] == "2.5", "MI setting drift")
    require(e["LIME_NUM_EXPLANATIONS"] == "40", "LIME setting drift")
    require(e["AIS_STRATEGY"] == "a_robust_draw_floor", "AIS strategy drift")
    require(m["arms"] == [
        {"name": "ais", "methods": "shap lime spline_screening mutual_information",
         "spline_inference_method": "inherit"},
        {"name": "exact_spline", "methods": "spline_screening",
         "spline_inference_method": "exact_spline"}], "Arm definition drift")
    return m


def source_hashes():
    files = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "examples").glob("*.py")),
             ROOT / "examples/run_selection_method_validation_shards.sh", MANIFEST,
             ROOT / "pyproject.toml"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}


def lock(m):
    out = ROOT / m["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    for filename, value in (("requested_manifest.json", m), ("source_hashes.json", source_hashes())):
        path = out / filename
        if path.exists():
            require(read_json(path) == value, f"{filename} changed; use a new output directory")
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return out


def arm_environment(m, arm):
    out = ROOT / m["output_root"] / "runs" / arm["name"]
    env = {**os.environ, **m["environment"], "PYTHON_BIN": sys.executable,
           "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
           "METHODS": arm["methods"], "SPLINE_INFERENCE_METHOD": arm["spline_inference_method"],
           "OUTPUT_ROOT": str(out)}
    # Exact inference does not sample and rejects this AIS-only stopping flag.
    # This is an expected inference-method difference, not a design change.
    if arm["name"] == "exact_spline":
        env["AIS_STOP_WHEN_ESS_MET"] = "false"
        env["AIS_STRATEGY"] = "baseline"
    return env


def verify_arm(m, arm):
    import pandas as pd
    root = ROOT / m["output_root"] / "runs" / arm["name"]
    pooled = read_json(root / "pooled_summary/validation_settings.json")
    e = m["environment"]
    expected_methods = arm["methods"].split()
    checks = {
        "methods": expected_methods, "designs": ["baseline"], "comparison_events": ["exact_set"],
        "primary_selection_event": "exact_set", "n_shards": 32, "n_null_iters": 300,
        "n_power_iters": 300, "shard_root_seeds": list(range(20260908, 20260940)),
        "design_seed": 314159, "max_final_samples": 51200, "pilot_iters": 10,
        "pilot_samples": 320, "final_batch_size": 800, "min_denominator_ess": 80.0,
        "min_tail_ess": 15.0,
        "ais_strategy": "baseline" if arm["name"] == "exact_spline" else "a_robust_draw_floor",
        "rf_jobs": 1,
        "signal_strengths": list(BETAS), "signal_positions": ["first"],
        "mi_neighbors": 30, "mi_params": {"estimator": "kde", "n_bins": 5, "bandwidth": 2.5},
        "lime_params": {"ridge_alpha": 1.0, "kernel_width": None, "num_explanations": 40},
        "spline_inference_method": arm["spline_inference_method"],
        "stop_when_ess_met": arm["name"] != "exact_spline",
    }
    for key, value in checks.items():
        require(pooled.get(key) == value, f"{arm['name']} pooled setting drift: {key}")
    require(pooled["method_inference_methods"] == {
        method: ("exact_spline" if arm["name"] == "exact_spline" else "ais")
        for method in expected_methods}, f"{arm['name']} inference mapping drift")
    for shard in range(32):
        expected_n = 10 if shard < 12 else 9
        settings = read_json(root / f"shard_{shard}/validation_settings.json")
        require(settings["n_null_iters"] == settings["n_power_iters"] == expected_n,
                f"Wrong allocation in {arm['name']}/shard_{shard}")
        require(settings["seed"] == 20260908 + shard, f"Wrong seed in {arm['name']}/shard_{shard}")
        for method in expected_methods:
            folders = ["null_fresh", *[f"power_feature_0_beta_{str(b).replace('.', 'p')}" for b in BETAS]]
            for index, folder in enumerate(folders):
                base = root / f"shard_{shard}/{method}/baseline/{folder}"
                result = base / ("p_value_results.csv" if index == 0 else "target_results.csv")
                frame = pd.read_csv(result)
                require(len(frame) == expected_n and frame.iteration.nunique() == expected_n,
                        f"Wrong row count: {result}")
                require(frame.selection_event.eq("exact_set").all(), f"Wrong event: {result}")
    return pooled


def comparable_settings(value):
    ignored = {"inference_method", "stop_when_ess_met", "runtime_metadata", "ais_strategy"}
    return {k: v for k, v in value.items() if k not in ignored}


def finalize(m, settings):
    import numpy as np
    import pandas as pd
    import analyze_matched_ais_methods as analysis
    from analyze_ais_method_comparison import _read_raw
    import plot_combined_ais_analysis as plotting

    out = ROOT / m["output_root"]
    base = analysis.BASE.resolve()
    frames = []
    manifest = {}
    labels = {"shap": "SHAP", "lime": "LIME (40 explanations)",
              "spline_screening": "Spline screening (AIS)",
              "mutual_information": "Mutual information (KDE bandwidth 2.5)"}
    for arm_name, setting in settings.items():
        run = out / "runs" / arm_name
        null, target, feature = _read_raw(run, arm_name, setting, 32)
        keys = ["selection_method", "validation_shard", "iteration", "selection_event", "signal_strength"]
        target = target.merge(feature[keys + ["feature", "t_obs", "proposals",
                                               "denominator_ess", "tail_ess"]],
                              on=keys, validate="one_to_one")
        for frame, scope in ((null, "null_calibration"), (target, "power")):
            frame = frame.copy()
            frame["scope"] = scope
            frame["ais_setting"] = "exact" if arm_name == "exact_spline" else "combined"
            frame["source_run"] = run.resolve().relative_to(base).as_posix()
            frame["max_final_samples"] = 51200
            frame["pilot_iters"] = 10
            frame["pilot_samples"] = 320
            frame["total_proposals"] = 0 if arm_name == "exact_spline" else frame.proposals + 3200
            if arm_name == "exact_spline":
                frame["selection_method"] = EXACT_METHOD
                frame["option"] = "Spline screening (exact)"
            else:
                frame["option"] = frame.selection_method.map(labels)
            # Neighbor count is not an active KDE tuning parameter. Keep the
            # legacy report column empty rather than mislabeling KDE as k-NN.
            frame["mi_neighbors"] = np.nan
            frame["mi_estimator"] = np.where(frame.selection_method.eq("mutual_information"), "kde", None)
            frame["mi_bandwidth"] = np.where(frame.selection_method.eq("mutual_information"), 2.5, np.nan)
            frames.append(frame)
        manifest[run.resolve().relative_to(base).as_posix()] = setting

    data = pd.concat(frames, ignore_index=True, sort=False)
    require(len(data) == 7500, f"Expected 7,500 unified tests, got {len(data)}")
    require(set(data.selection_method) == {*AIS_METHODS, EXACT_METHOD}, "Combined method drift")

    # Exact and AIS spline must have identical generated trials, targets, and observed statistics.
    trial_audit = []
    ais_spline = data[data.selection_method.eq("spline_screening")]
    exact = data[data.selection_method.eq(EXACT_METHOD)]
    for scope, expected in (("null_calibration", 300), ("power", 1200)):
        left = ais_spline[ais_spline.scope.eq(scope)]
        right = exact[exact.scope.eq(scope)]
        keys = ["validation_shard", "iteration", "selection_event"]
        if scope == "power": keys.append("signal_strength")
        paired = left.merge(right, on=keys, suffixes=("_ais", "_exact"), validate="one_to_one")
        require(len(paired) == expected, f"Spline pairing count drift: {scope}")
        for key in ("target_seed", "target_feature"):
            np.testing.assert_array_equal(paired[f"{key}_ais"], paired[f"{key}_exact"])
        for key in ("auxiliary_u", "t_obs"):
            np.testing.assert_allclose(paired[f"{key}_ais"], paired[f"{key}_exact"], rtol=0, atol=1e-12)
        trial_audit.append({"scope": scope, "matched_trials": len(paired),
                            "seeds_targets_statistics_match": True})

    # Low-level spline settings differ only in the expected inference controls.
    config_audit = []
    ais_root = out / "runs/ais"; exact_root = out / "runs/exact_spline"
    for shard in range(32):
        folders = ["null_fresh", *[f"power_feature_0_beta_{str(b).replace('.', 'p')}" for b in BETAS]]
        for folder in folders:
            suffix = Path(f"shard_{shard}/spline_screening/baseline/{folder}/settings.json")
            a = comparable_settings(read_json(ais_root / suffix))
            e = comparable_settings(read_json(exact_root / suffix))
            require(a == e, f"AIS/exact experimental setting drift: {suffix}")
            config_audit.append({"shard": shard, "condition": folder,
                                 "experimental_settings_match": True,
                                 "expected_difference": "inference method"})

    data.to_csv(out / "unified_tests.csv", index=False)
    pd.DataFrame(trial_audit).to_csv(out / "trial_audit.csv", index=False)
    pd.DataFrame(config_audit).to_csv(out / "configuration_audit.csv", index=False)
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    selected = {"methods": ["shap_ais", "lime_40_ais", "spline_screening_ais",
                             "spline_screening_exact", "mutual_information_kde_2p5_ais"],
                "matched_experimental_settings": m["environment"],
                "design": m["design"],
                "expected_inference_difference": {
                    "spline_ais": {"method": "ais", "stop_when_ess_met": True,
                                   "ais_strategy": "a_robust_draw_floor"},
                    "spline_exact": {"method": "exact_spline", "stop_when_ess_met": False,
                                     "ais_strategy": "baseline"}}}
    (out / "selected_configuration.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    analysis.OUT = out
    analysis.summarize(data)
    plotting.SOURCE = plotting.OUT = out
    plotting.DATA_FILENAME = "unified_tests.csv"
    plotting.REPORT_LABEL = "Five-method KDE-2.5 confirmation (300 trials)"
    plotting.REPORT_INTRO = ("Matched confirmation experiment with 300 null trials and 300 power trials "
                             "per beta per method. MI uses KDE bandwidth 2.5; LIME uses 40 explanations.")
    plotting.METHODS = {"shap": "SHAP (AIS)", "lime": "LIME (40, AIS)",
                        "spline_screening": "Spline screening (AIS)",
                        EXACT_METHOD: "Spline screening (exact)",
                        "mutual_information": "Mutual information (KDE 2.5, AIS)"}
    plotting.COLORS = dict(zip(plotting.METHODS, ["#2674a6", "#d58122", "#258568", "#c44e52", "#9256a0"]))
    plotting.TRIALS_PER_CONDITION = 300
    plotting.EXPECTED_PILOT_ITERS = (10,)
    plotting.ALLOW_INCOMPLETE = False
    plotting.FIGURES.clear()
    plotting.main()
    with (out / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\n\nAll five methods use the same 300 trials per condition, shard allocation, root seeds, generated data, target randomization, and design. Exact spline uses analytic inference; its AIS-only strategy and stopping controls are necessarily inactive. See `configuration_audit.csv` and `trial_audit.csv`.\n")
    (out / "COMPLETED").write_text("Verified 7,500 matched tests and built the five-method report.\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "verify"))
    args = parser.parse_args(argv)
    m = read_manifest()
    if args.action == "check":
        print(json.dumps(m, indent=2)); return
    lock(m)
    if args.action == "run":
        for arm in m["arms"]:
            print(f"Running {arm['name']} arm", flush=True)
            subprocess.run(["bash", "examples/run_selection_method_validation_shards.sh"],
                           cwd=ROOT, env=arm_environment(m, arm), check=True)
    settings = {arm["name"]: verify_arm(m, arm) for arm in m["arms"]}
    finalize(m, settings)
    print("Verified 7,500 tests; combined report complete.", flush=True)


if __name__ == "__main__":
    main()
