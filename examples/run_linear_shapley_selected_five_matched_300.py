"""Run and verify the 300-trial Linear Shapley matched follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS",
):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
MANIFEST = ROOT / "examples/linear_shapley_selected_five_matched_300_20260928.json"
BETAS = (0.3, 0.5, 0.75, 1.0)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_manifest():
    manifest = read_json(MANIFEST)
    env = manifest["environment"]
    from examples.validate_selection_methods import DESIGNS

    require(manifest["design"] == DESIGNS["baseline"], "Baseline design drift")
    require(env["METHODS"] == "linear_shapley", "Method drift")
    require(env["TOTAL_NULL_ITERS"] == env["TOTAL_POWER_ITERS"] == "300",
            "Expected 300 trials per condition")
    require(env["SELECTION_EVENTS"] == env["PRIMARY_SELECTION_EVENT"] == "exact_set",
            "Expected exact-set conditioning")
    require(1 <= int(env["PARALLEL_SHARDS"]) <= int(env["N_SHARDS"]) <= 32,
            "Shard CPU cap exceeded")
    require(int(env["PARALLEL_SHARDS"]) * int(env["RF_JOBS"])
            <= int(env["CPU_BUDGET"]) <= 32, "CPU budget exceeded")
    require(env["AIS_STRATEGY"] == "a_robust_draw_floor", "AIS strategy drift")
    require(env["AIS_STOP_WHEN_ESS_MET"] == "true", "AIS stopping drift")
    require(manifest["linear_shapley"] == {
        "mode": "auto", "exact_max_features": 10, "n_permutations": 1000,
        "random_state": 42, "fit_intercept": True, "resolved_mode": "monte_carlo",
    }, "Linear Shapley setting drift")
    return manifest


def source_hashes():
    files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "examples").glob("*.py")),
        ROOT / "examples/run_selection_method_validation_shards.sh",
        MANIFEST,
        ROOT / "pyproject.toml",
    ]
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in files
    }


def lock(manifest):
    output = ROOT / manifest["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("requested_manifest.json", manifest),
        ("source_hashes.json", source_hashes()),
    ):
        path = output / filename
        if path.exists():
            require(read_json(path) == value,
                    f"{filename} changed; use a new output directory")
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return output


def run_environment(manifest):
    output = ROOT / manifest["output_root"] / "runs" / "linear_shapley"
    return {
        **os.environ,
        **manifest["environment"],
        "PYTHON_BIN": sys.executable,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
        "OUTPUT_ROOT": str(output),
    }


def verify(manifest):
    import numpy as np
    import pandas as pd

    output = ROOT / manifest["output_root"]
    run = output / "runs" / "linear_shapley"
    pooled = read_json(run / "pooled_summary/validation_settings.json")
    expected = {
        "methods": ["linear_shapley"],
        "designs": ["baseline"],
        "comparison_events": ["exact_set"],
        "primary_selection_event": "exact_set",
        "n_shards": 32,
        "n_null_iters": 300,
        "n_power_iters": 300,
        "shard_root_seeds": list(range(20260908, 20260940)),
        "design_seed": 314159,
        "max_final_samples": 51200,
        "inference_method": "ais",
        "method_inference_methods": {"linear_shapley": "ais"},
        "pilot_iters": 10,
        "pilot_samples": 320,
        "final_batch_size": 800,
        "min_denominator_ess": 80.0,
        "min_tail_ess": 15.0,
        "stop_when_ess_met": True,
        "ais_strategy": "a_robust_draw_floor",
        "rf_jobs": 1,
        "signal_strengths": list(BETAS),
        "signal_positions": ["first"],
    }
    for key, value in expected.items():
        require(pooled.get(key) == value, f"Pooled setting drift: {key}")

    audit = []
    selector_expected = {
        "selector": "LinearShapleySelector",
        "selection_method": "linear_shapley",
        "importance": "linear_shapley_rss_contribution",
        "mode": "auto",
        "exact_max_features": 10,
        "n_permutations": 1000,
        "random_state": 42,
        "fit_intercept": True,
        "resolved_mode": "monte_carlo",
    }
    folders = [
        ("null_fresh", "p_value_results.csv"),
        *[(f"power_feature_0_beta_{str(beta).replace('.', 'p')}", "target_results.csv")
          for beta in BETAS],
    ]
    total = 0
    for shard in range(32):
        expected_n = 10 if shard < 12 else 9
        shard_settings = read_json(run / f"shard_{shard}/validation_settings.json")
        require(shard_settings["seed"] == 20260908 + shard, f"Seed drift: shard {shard}")
        require(shard_settings["n_null_iters"] == shard_settings["n_power_iters"] == expected_n,
                f"Allocation drift: shard {shard}")
        for folder, filename in folders:
            base = run / f"shard_{shard}/linear_shapley/baseline/{folder}"
            settings = read_json(base / "settings.json")
            for key, value in selector_expected.items():
                require(settings["selector_settings"].get(key) == value,
                        f"Selector setting drift: shard {shard}/{folder}/{key}")
            frame = pd.read_csv(base / filename)
            require(len(frame) == expected_n and frame["iteration"].nunique() == expected_n,
                    f"Result count drift: shard {shard}/{folder}")
            require(frame["selection_event"].eq("exact_set").all(),
                    f"Selection-event drift: shard {shard}/{folder}")
            diagnostics = (frame if folder == "null_fresh"
                           else pd.read_csv(base / "feature_results.csv"))
            require(diagnostics["test_statistic"].eq("linear_shapley").all(),
                    f"Test-statistic drift: shard {shard}/{folder}")
            np.testing.assert_allclose(
                diagnostics["importance_score"], diagnostics["t_obs"], rtol=0, atol=1e-10,
                err_msg=f"Selection/test score mismatch: shard {shard}/{folder}",
            )
            total += len(frame)
            audit.append({"shard": shard, "condition": folder, "rows": len(frame),
                          "settings_verified": True})
    require(total == 1500, f"Expected 1,500 tests, got {total}")
    pd.DataFrame(audit).to_csv(output / "configuration_audit.csv", index=False)
    verification = {
        "verified_tests": total,
        "null_tests": 300,
        "power_tests": 1200,
        "conditions": 5,
        "shards": 32,
        "maximum_cpu_cores": 32,
        "reference": manifest["reference"],
        "intentional_differences": manifest["intentional_differences"],
    }
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    (output / "COMPLETED").write_text(
        "Verified 1,500 Linear Shapley tests matched to the selected-five 300-trial design.\n",
        encoding="utf-8",
    )
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "verify"))
    args = parser.parse_args(argv)
    manifest = read_manifest()
    if args.action == "check":
        print(json.dumps(manifest, indent=2))
        return
    lock(manifest)
    if args.action == "run":
        subprocess.run(
            ["bash", "examples/run_selection_method_validation_shards.sh"],
            cwd=ROOT,
            env=run_environment(manifest),
            check=True,
        )
    result = verify(manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
