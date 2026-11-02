"""Replay the selected five-method experiment with only the AIS strategy changed.

Default action audits inputs and prints a plan. --run executes new AIS trials;
--finalize audits completed trials and produces the historical report format.
The exact-spline rows are reused verbatim, never simulated again.
"""
from __future__ import annotations

import argparse
import json
import inspect
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import validate_selection_methods as validation
import analyze_matched_ais_methods as analysis
from analyze_ais_method_comparison import _read_raw
import plot_combined_ais_analysis as plotting
from si_shap.selection import make_selector, _resolve_rf_params
from si_shap.power import SIMULATION_SIGMA

METHODS = ("shap", "lime", "mutual_information", "spline_screening")
STRATEGIES = ("a_robust_draw_floor", "a_robust_floor_multimodal_denominator")
BETAS = (.3, .5, .75, 1.)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_settings(old, new, path=""):
    """Check historical keys recursively; tolerate absent historical strategy.

    Newly recorded metadata keys do not change historical parameters. Every
    value of every historical key is checked, including nested selector data.
    """
    for key, value in old.items():
        if not path and key in {"runtime_metadata", "ais_strategy"}:
            continue
        name = f"{path}.{key}" if path else key
        require(key in new, f"Missing setting: {name}")
        if isinstance(value, dict):
            compare_settings(value, new[key], name)
        else:
            require(value == new[key], f"Setting drift: {name}: {value!r} != {new[key]!r}")


def audit_current_parameters(low, settings, method, condition_index):
    """Check current API defaults and selector metadata without running AIS."""
    api = (validation.compare_selection_event_null_calibration if condition_index == 0
           else validation.compare_selection_event_power)
    defaults = {k: p.default for k, p in inspect.signature(api).parameters.items()
                if p.default is not inspect.Parameter.empty}
    passed = {**validation.DESIGNS["baseline"],
              **{key: settings[key] for key in (
                  "max_final_samples", "inference_method", "pilot_iters", "pilot_samples",
                  "final_batch_size", "min_denominator_ess", "min_tail_ess",
                  "stop_when_ess_met", "mcmc_steps")},
              "selection_events": settings["comparison_events"], "n_iters": low["n_iters"],
              "seed": low["seed"], "rf_params": _resolve_rf_params({"n_jobs": settings["rf_jobs"]})
              if method == "shap" else None}
    if condition_index == 0:
        passed.update(alpha_levels=list(validation.ALPHA_LEVELS), fixed_auxiliary_u=None,
                      design_seed=validation._seed(settings["design_seed"], 0))
    else:
        passed.update(signal_features=[0], signal_strength=BETAS[condition_index - 1])
        defaults["sigma"] = SIMULATION_SIGMA
    predicted = {**defaults, **passed}
    for key in low.keys() & predicted.keys():
        if key in {"selection_method", "selector", "estimator", "ais_strategy"}:
            continue
        value = predicted[key]
        if isinstance(value, tuple):
            value = list(value)
        require(low[key] == value, f"Current API parameter drift: {method}/{key}")
    selector_args = validation._selector_arguments(method, settings["mi_neighbors"],
                                                    settings["lime_params"], settings["rf_jobs"],
                                                    {"estimator": "knn", "n_bins": 5, "bandwidth": 1.0})
    selector = make_selector(**selector_args,
                             selection_decimals=defaults["selection_decimals"],
                             rf_params=passed["rf_params"])
    compare_settings(low["selector_settings"], dict(selector.get_settings()), "selector_settings")


def environment(settings, method, strategy, output, workers, python):
    env = {
        "PYTHON_BIN": python, "OUTPUT_ROOT": str(output),
        "PRESET": settings["preset"], "METHODS": method,
        "DESIGNS": " ".join(settings["designs"]), "AUXILIARY_VALUES": "fresh",
        "SIGNAL_STRENGTHS": " ".join(map(str, settings["signal_strengths"])),
        "SIGNAL_POSITIONS": " ".join(settings["signal_positions"]),
        "SELECTION_EVENTS": " ".join(settings["comparison_events"]),
        "PRIMARY_SELECTION_EVENT": settings["primary_selection_event"],
        "TOTAL_NULL_ITERS": settings["n_null_iters"],
        "TOTAL_POWER_ITERS": settings["n_power_iters"],
        "N_SHARDS": settings["n_shards"], "PARALLEL_SHARDS": workers,
        "CPU_BUDGET": workers, "RF_JOBS": settings["rf_jobs"],
        "BASE_SEED": settings["shard_root_seeds"][0], "DESIGN_SEED": settings["design_seed"],
        "INFERENCE_METHOD": settings["inference_method"],
        "SPLINE_INFERENCE_METHOD": settings["spline_inference_method"],
        "MC_PROPOSALS": settings["max_final_samples"], "MCMC_STEPS": settings["mcmc_steps"],
        "MI_NEIGHBORS": settings["mi_neighbors"],
        "MI_ESTIMATOR": "knn", "MI_BINS": 5, "MI_BANDWIDTH": 1.0,
        "LIME_RIDGE_ALPHA": settings["lime_params"]["ridge_alpha"],
        "LIME_KERNEL_WIDTH": "", "LIME_NUM_EXPLANATIONS": settings["lime_params"]["num_explanations"],
        "AIS_PILOT_ITERS": settings["pilot_iters"], "AIS_PILOT_SAMPLES": settings["pilot_samples"],
        "AIS_FINAL_BATCH_SIZE": settings["final_batch_size"],
        "AIS_MIN_DENOMINATOR_ESS": settings["min_denominator_ess"],
        "AIS_MIN_TAIL_ESS": settings["min_tail_ess"],
        "AIS_STOP_WHEN_ESS_MET": str(settings["stop_when_ess_met"]).lower(),
        "AIS_STRATEGY": strategy,
        "MIN_CALIBRATION_ITERS": settings["minimum_calibration_iterations"],
        "MIN_SIGNAL_TARGETS": settings["minimum_signal_targets"],
        "MIN_CONDITIONAL_POWER": settings["minimum_conditional_power"],
    }
    return {key: str(value) for key, value in env.items()}


def prepare(config, workers=32, python=sys.executable):
    require(config["ais_strategy"] in STRATEGIES, "Unsupported study strategy")
    reference = (ROOT / config["reference"]).resolve()
    output = (ROOT / config["output"]).resolve()
    require(output.is_relative_to(analysis.BASE.resolve()), "Output must be under exact_set")
    require(output != reference, "Output must preserve the reference")
    original = pd.read_csv(reference / "unified_tests.csv")
    manifest = read(reference / "source_manifest.json")
    require(len(original) == 5000, "Expected 5,000 historical rows")
    require(set(original.selection_method) == {*METHODS, "spline_screening_exact"}, "Method drift")
    sources = original.groupby("selection_method").source_run.first().to_dict()
    jobs, audit = [], []
    for method in METHODS:
        source = analysis.BASE / sources[method]
        settings = manifest[sources[method]]
        expected = read(reference / "selected_configuration.json")["matched_experimental_settings"]
        for key, value in expected.items():
            require(settings[key] == value, f"Historical {method}: {key} drift")
        require(settings["shard_root_seeds"] == list(range(20260908, 20260940)), "Seed drift")
        require(settings["stop_when_ess_met"] is True, "Stopping drift")
        require(settings.get("ais_strategy", "baseline") == "baseline", "Unexpected historical strategy")
        require(settings["inference_method"] == "ais", "Inference drift")
        require(settings["lime_params"]["kernel_width"] is None, "Unexpected kernel width")
        if method == "lime":
            require(settings["lime_params"]["num_explanations"] == 40, "LIME drift")
        if method == "mutual_information":
            require(settings["mi_neighbors"] == 30, "MI drift")
        for shard in range(32):
            for index, folder in enumerate(["null_fresh", *[f"power_feature_0_beta_{str(b).replace('.', 'p')}" for b in BETAS]]):
                suffix = Path(f"shard_{shard}/{method}/baseline/{folder}/settings.json")
                low = read(source / suffix)
                require(low["n_iters"] == (7 if shard < 8 else 6), f"Replication drift: {suffix}")
                compare_settings(validation.DESIGNS["baseline"], low)
                seed = (validation._seed(20260908 + shard, 0, 0) if index == 0 else
                        validation._seed(20260908 + shard, 10000, 0, 0, index - 1))
                require(low["seed"] == seed, f"Condition seed drift: {suffix}")
                require(low["selection_events"] == ["exact_set"], f"Event drift: {suffix}")
                audit_current_parameters(low, settings, method, index)
                audit.append({"method": method, "shard": shard, "condition": folder,
                              "source_settings": str(source / suffix), "seed": seed,
                              "n_iters": low["n_iters"]})
        jobs.append({"method": method, "source": source, "settings": settings,
                     "env": environment(settings, method, config["ais_strategy"],
                                        output / "runs" / method, workers, python)})
    return reference, output, original, manifest, jobs, audit


def pair_trials(old, new, method):
    rows = []
    for scope in ("null_calibration", "power"):
        a = old[old.scope.eq(scope) & old.selection_method.eq(method)]
        b = new[new.scope.eq(scope)]
        keys = ["validation_shard", "iteration", "selection_event"]
        if scope == "power":
            keys.append("signal_strength")
        pair = a.merge(b, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
        require(len(pair) == len(a) == len(b) == (200 if scope == "null_calibration" else 800), "Trial count drift")
        for key in ("target_seed", "ais_seed", "target_feature"):
            np.testing.assert_array_equal(pair[key + "_old"], pair[key + "_new"])
        for key in ("auxiliary_u", "t_obs"):
            np.testing.assert_allclose(pair[key + "_old"], pair[key + "_new"], rtol=0, atol=1e-12)
        key = "observed_selected_features" if scope == "null_calibration" else "target_is_signal"
        np.testing.assert_array_equal(pair[key + "_old"], pair[key + "_new"])
        rows.append({"method": method, "scope": scope, "matched_trials": len(pair),
                     "seeds_targets_statistics_match": True})
    return rows


def finalize(config, prepared):
    reference, output, original, manifest, jobs, audit = prepared
    frames, trials, used = [], [], {}
    for job in jobs:
        method = job["method"]
        run = Path(job["env"]["OUTPUT_ROOT"])
        settings = read(run / "pooled_summary/validation_settings.json")
        expected = {**job["settings"], "methods": [method],
                    "method_inference_methods": {method: "ais"}}
        compare_settings(expected, settings)
        require(settings["ais_strategy"] == config["ais_strategy"], "Wrong AIS strategy")
        for row in (r for r in audit if r["method"] == method):
            suffix = Path(f"shard_{row['shard']}/{method}/baseline/{row['condition']}/settings.json")
            low = read(run / suffix)
            compare_settings(read(job["source"] / suffix), low)
            require(low["ais_strategy"] == config["ais_strategy"], "Wrong condition strategy")
            row["experimental_settings_match"] = True
        n, t, f = _read_raw(run, "ais", settings, 32)
        keys = ["selection_method", "validation_shard", "iteration", "selection_event", "signal_strength"]
        t = t.merge(f[keys + ["feature", "t_obs", "proposals", "denominator_ess", "tail_ess"]],
                    on=keys, validate="one_to_one")
        rel = run.relative_to(analysis.BASE).as_posix()
        used[rel] = settings
        for frame, scope in ((n, "null_calibration"), (t, "power")):
            frame["scope"] = scope
            # Preserve the historical report category; strategy identity is
            # explicit in source_manifest and selected_configuration.
            frame["ais_setting"] = "combined"
            frame["source_run"] = rel
            frame["max_final_samples"] = 51200
            frame["pilot_iters"] = 10
            frame["pilot_samples"] = 320
            frame["total_proposals"] = frame.proposals + 3200
            template = original[original.selection_method.eq(method)].iloc[0]
            frame["option"] = template.option
            frame["mi_neighbors"] = template.mi_neighbors
        new = pd.concat([n, t], ignore_index=True)
        trials.extend(pair_trials(original, new, method))
        frames.append(new)
    exact = original[original.selection_method.eq("spline_screening_exact")].copy()
    require(len(exact) == 1000, "Exact reference count drift")
    used[exact.source_run.iat[0]] = manifest[exact.source_run.iat[0]]
    data = pd.concat([*frames, exact], ignore_index=True, sort=False)
    require(len(data) == 5000, "Combined count drift")
    # Keep historical columns first, retaining new diagnostic columns afterward.
    data = data.reindex(columns=[*original.columns, *[c for c in data if c not in original]])
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "unified_tests.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "configuration_audit.csv", index=False)
    pd.DataFrame(trials).to_csv(output / "trial_audit.csv", index=False)
    (output / "source_manifest.json").write_text(json.dumps(used, indent=2), encoding="utf-8")
    selected = read(reference / "selected_configuration.json")
    selected.update(ais_strategy=config["ais_strategy"], reference=str(reference),
                    exact_reference_reused=True)
    (output / "selected_configuration.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    analysis.OUT = output
    analysis.summarize(data)
    plotting.SOURCE = plotting.OUT = output
    plotting.DATA_FILENAME = "unified_tests.csv"
    plotting.REPORT_LABEL = config["ais_strategy"] + " plus exact spline"
    plotting.METHODS = {"shap": "SHAP", "lime": "LIME (40 explanations)",
                        "mutual_information": "Mutual information (30 neighbors)",
                        "spline_screening": "Spline screening (AIS)",
                        "spline_screening_exact": "Spline screening (exact)"}
    plotting.COLORS = dict(zip(plotting.METHODS, ["#2674a6", "#d58122", "#9256a0", "#258568", "#c44e52"]))
    plotting.FIGURES.clear()
    # Failures are outcomes under test. Counts and pairings were checked above.
    plotting.ALLOW_INCOMPLETE = True
    plotting.main()
    with (output / "README.md").open("a", encoding="utf-8") as file:
        file.write(f"\n\nAIS strategy: `{config['ais_strategy']}`. Exact-spline rows reused verbatim from `{reference.name}`. The `combined` report category denotes the matched AIS budget; the strategy is recorded in the manifests. All historical condition settings and trial seeds/targets/statistics were audited.\n")
    (output / "COMPLETED").write_text("Matched settings and trials verified.\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--workers", type=int, default=32, help="Execution concurrency only")
    args = parser.parse_args(argv)
    require(1 <= args.workers <= 32, "Workers must be between 1 and 32")
    config = read(args.config)
    prepared = prepare(config, args.workers)
    reference, output, original, manifest, jobs, audit = prepared
    print(f"Audited {len(audit)} historical AIS condition configs; reference: {reference}")
    print(f"Strategy: {config['ais_strategy']}; output: {output}")
    print("New trials: 800 null + 3,200 power; reuse 1,000 exact-spline rows.")
    if not (args.run or args.finalize):
        print(json.dumps({j["method"]: j["env"] for j in jobs}, indent=2))
        return
    if args.run:
        versions = validation._runtime_metadata()["package_versions"]
        for job in jobs:
            require(versions == job["settings"]["runtime_metadata"]["package_versions"],
                    "Package versions must match the historical manifest")
        output.mkdir(parents=True, exist_ok=True)
        lock = output / "experiment_config.json"
        if lock.exists():
            require(read(lock) == config, "Existing experiment config differs")
        else:
            lock.write_text(json.dumps(config, indent=2), encoding="utf-8")
        for job in jobs:
            subprocess.run(["bash", "examples/run_selection_method_validation_shards.sh"],
                           cwd=ROOT, env={**os.environ, **job["env"]}, check=True)
    finalize(config, prepared)


if __name__ == "__main__":
    main()
