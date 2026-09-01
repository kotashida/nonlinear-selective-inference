"""Small paired audit of SHAP, spline screening, and mutual information.

The three methods receive identical data, target-randomization seeds, and
Monte-Carlo seeds.  The script is deliberately a diagnostic, not a
replacement for the server-scale validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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
from itertools import combinations

from si_shap import (
    compare_selection_event_null_calibration,
    compare_selection_event_power,
)


METHODS = ("shap", "spline_screening", "mutual_information")


def _run_method(config: dict) -> dict:
    method = config["method"]
    rf_params = (
        {
            "n_estimators": config["rf_trees"],
            "max_depth": 4,
            "random_state": 42,
            "n_jobs": 1,
        }
        if method == "shap"
        else None
    )
    common = {
        "n_samples": config["n_samples"],
        "n_features": config["n_features"],
        "k_select": config["k_select"],
        "selection_method": method,
        "rf_params": rf_params,
        "selection_events": ("same_target",),
        "inference_method": config["inference_method"],
        "final_batch_size": min(40, config["mc_proposals"]),
        "max_final_samples": config["mc_proposals"],
        "mcmc_steps": config["mcmc_steps"],
    }
    null = compare_selection_event_null_calibration(
        n_iters=config["n_null_iters"],
        alpha_levels=(0.05,),
        seed=config["seed"],
        design_seed=config["design_seed"],
        **common,
    )
    power = compare_selection_event_power(
        n_iters=config["n_power_iters"],
        signal_features=(0,),
        signal_strength=config["signal_strength"],
        alpha=0.05,
        seed=config["seed"],
        **common,
    )
    return {
        "method": method,
        "k_select": config["k_select"],
        "null": null["p_value_results"],
        "power_features": power["feature_results"],
        "power_targets": power["target_results"],
        "power_summary": power["summary"],
    }


def _interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return np.nan, np.nan
    lower = 0.0 if successes == 0 else stats.beta.ppf(
        0.025, successes, trials - successes + 1
    )
    upper = 1.0 if successes == trials else stats.beta.ppf(
        0.975, successes + 1, trials - successes
    )
    return float(lower), float(upper)


def _summarize(result: dict, alpha: float = 0.05) -> dict:
    null = result["null"]
    targets = result["power_targets"]
    features = result["power_features"]
    signal_targets = targets[targets["target_is_signal"].astype(bool)]
    signal_features = features[features["is_signal"].astype(bool)]
    null_rejections = int((null["p_value"].astype(float) < alpha).sum())
    signal_rejections = int(signal_targets["rejected"].astype(bool).sum())
    null_lo, null_hi = _interval(null_rejections, len(null))
    power_lo, power_hi = _interval(signal_rejections, len(signal_targets))
    summary = result["power_summary"].iloc[0]
    return {
        "selection_method": result["method"],
        "null_iterations": len(null),
        "null_rejection_rate": null_rejections / len(null),
        "null_ci_95_lower": null_lo,
        "null_ci_95_upper": null_hi,
        "null_mean_p_value": float(null["p_value"].astype(float).mean()),
        "null_resolution_limited_rate": float(
            (null["minimum_attainable_p_value"].astype(float) >= alpha).mean()
        ),
        "signal_target_rate": float(targets["target_is_signal"].astype(bool).mean()),
        "signal_inclusion_rate_estimate": float(
            min(
                1.0,
                result["k_select"]
                * targets["target_is_signal"].astype(bool).mean(),
            )
        ),
        "signal_targets": len(signal_targets),
        "mean_signal_t_obs": float(signal_features["t_obs"].astype(float).mean()),
        "unadjusted_power_given_signal_target": float(
            (signal_features["unadjusted_p_value"].astype(float) < alpha).mean()
        ),
        "selective_power_given_signal_target": (
            signal_rejections / len(signal_targets)
            if len(signal_targets)
            else np.nan
        ),
        "selective_power_ci_95_lower": power_lo,
        "selective_power_ci_95_upper": power_hi,
        "signal_resolution_limited_rate": float(
            signal_targets["resolution_limited"].astype(bool).mean()
        ),
        "mean_signal_selected_samples": float(
            signal_features["selected_samples"].astype(float).mean()
        ),
        "mean_signal_event_probability": float(
            signal_features["selection_probability_estimate"].astype(float).mean()
        ),
        "power_resolution_lower_bound": float(
            summary["conditional_power_resolution_lower_bound"]
        ),
        "power_resolution_upper_bound": float(
            summary["conditional_power_resolution_upper_bound"]
        ),
    }


def _paired_binary_comparisons(results: list[dict], alpha: float = 0.05):
    """Compare paired binary outcomes without relying on large-sample tests."""
    records = []
    frames = {}
    for result in results:
        null = result["null"][["iteration", "p_value", "minimum_attainable_p_value"]].copy()
        null["null_rejected"] = null["p_value"].astype(float) < alpha
        null["null_resolution_limited"] = (
            null["minimum_attainable_p_value"].astype(float) >= alpha
        )
        targets = result["power_targets"][
            ["iteration", "target_is_signal", "rejected", "resolution_limited"]
        ].copy()
        targets["signal_detected"] = (
            targets["target_is_signal"].astype(bool)
            & targets["rejected"].astype(bool)
        )
        frames[result["method"]] = {
            "null": null,
            "power": targets,
        }

    metrics = {
        "null_rejected": "null",
        "null_resolution_limited": "null",
        "target_is_signal": "power",
        "signal_detected": "power",
        "resolution_limited": "power",
    }
    for left, right in combinations(METHODS, 2):
        for metric, frame_name in metrics.items():
            paired = frames[left][frame_name][["iteration", metric]].merge(
                frames[right][frame_name][["iteration", metric]],
                on="iteration",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            left_values = paired[f"{metric}_left"].astype(bool)
            right_values = paired[f"{metric}_right"].astype(bool)
            left_only = int((left_values & ~right_values).sum())
            right_only = int((~left_values & right_values).sum())
            discordant = left_only + right_only
            p_value = (
                1.0
                if discordant == 0
                else float(
                    stats.binomtest(
                        min(left_only, right_only), discordant, 0.5
                    ).pvalue
                )
            )
            records.append(
                {
                    "method_left": left,
                    "method_right": right,
                    "metric": metric,
                    "paired_iterations": len(paired),
                    "left_rate": float(left_values.mean()),
                    "right_rate": float(right_values.mean()),
                    "left_only": left_only,
                    "right_only": right_only,
                    "exact_mcnemar_p_value": p_value,
                }
            )
    return pd.DataFrame.from_records(records)


def _shared_test_statistic_check(results: list[dict]):
    frames = []
    for result in results:
        frame = result["power_features"][
            ["iteration", "feature", "t_obs", "unadjusted_p_value"]
        ].copy()
        frame["selection_method"] = result["method"]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    shared = combined.groupby(["iteration", "feature"], as_index=False).filter(
        lambda group: group["selection_method"].nunique() == len(METHODS)
    )
    if shared.empty:
        return {
            "shared_iteration_feature_tests": 0,
            "maximum_t_obs_spread": np.nan,
            "maximum_unadjusted_p_value_spread": np.nan,
        }
    spreads = shared.groupby(["iteration", "feature"])[
        ["t_obs", "unadjusted_p_value"]
    ].agg(lambda values: float(values.max() - values.min()))
    return {
        "shared_iteration_feature_tests": int(len(spreads)),
        "maximum_t_obs_spread": float(spreads["t_obs"].max()),
        "maximum_unadjusted_p_value_spread": float(
            spreads["unadjusted_p_value"].max()
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-null-iters", type=int, default=12)
    parser.add_argument("--n-power-iters", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--k-select", type=int, default=2)
    parser.add_argument("--signal-strength", type=float, default=0.75)
    parser.add_argument("--mc-proposals", type=int, default=39)
    parser.add_argument(
        "--inference-method",
        choices=("conditional_mc", "mcmc_rank"),
        default="mcmc_rank",
    )
    parser.add_argument("--mcmc-steps", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--design-seed", type=int, default=314159)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "selector_pipeline_small",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = vars(args).copy()
    output_dir = Path(config.pop("output_dir")).resolve()
    workers = int(config.pop("workers"))
    jobs = [{**config, "method": method} for method in METHODS]
    if workers == 1:
        results = [_run_method(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_method, jobs))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame.from_records([_summarize(result) for result in results])
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired = _paired_binary_comparisons(results)
    paired.to_csv(output_dir / "paired_exact_comparisons.csv", index=False)
    statistic_check = _shared_test_statistic_check(results)
    for result in results:
        method = result["method"]
        result["null"].to_csv(output_dir / f"{method}_null.csv", index=False)
        result["power_features"].to_csv(
            output_dir / f"{method}_power_features.csv", index=False
        )
        result["power_targets"].to_csv(
            output_dir / f"{method}_power_targets.csv", index=False
        )
    with (output_dir / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(
            {**config, "methods": METHODS, "shared_statistic_check": statistic_check},
            file,
            indent=2,
            sort_keys=True,
        )
    print(summary.to_string(index=False))
    print("\nPaired exact comparisons:")
    print(paired.to_string(index=False))
    print(f"\nShared-statistic check: {statistic_check}")
    print(f"Saved diagnostic bundle to {output_dir}")
    return summary


if __name__ == "__main__":
    main()
