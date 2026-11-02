"""Pool and audit the five matched LIME AIS-fix experiments."""

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


CONDITIONS = (
    "baseline_reduced",
    "multimodal_qadapt",
    "raised_sd_floor",
    "adaptive_mixture_weights",
    "reallocated_pilot_exploration",
)
STRATEGIES = {
    "baseline_reduced": "baseline",
    "multimodal_qadapt": "multimodal_qadapt",
    "raised_sd_floor": "raised_sd_floor",
    "adaptive_mixture_weights": "adaptive_mixture_weights",
    "reallocated_pilot_exploration": "reallocated_pilot_exploration",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    return parser.parse_args()


def load_condition(root: Path, condition: str) -> pd.DataFrame:
    rows = []
    condition_dir = root / condition
    settings_paths = sorted(condition_dir.glob("shard_*/validation_settings.json"))
    if len(settings_paths) != 32:
        raise RuntimeError(f"{condition}: expected 32 completed shards, found {len(settings_paths)}")
    for settings_path in settings_paths:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["methods"] == ["lime"]
        assert settings["designs"] == ["baseline"]
        assert settings["comparison_events"] == ["exact_set"]
        assert settings["signal_strengths"] == [0.5, 0.75, 1.0]
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
        shard_dir = settings_path.parent
        files = [
            shard_dir / "lime" / "baseline" / "null_fresh" / "p_value_results.csv",
            *(shard_dir / "lime" / "baseline" / f"power_feature_0_beta_{beta}" / "feature_results.csv"
              for beta in ("0p5", "0p75", "1p0")),
        ]
        for path in files:
            frame = pd.read_csv(path)
            frame = frame.loc[frame["selection_event"].eq("exact_set")].copy()
            frame["scope"] = "null" if "null_fresh" in str(path) else "power"
            frame["signal_strength"] = (
                np.nan if frame["scope"].iat[0] == "null"
                else float(path.parent.name.rsplit("_", 1)[-1].replace("p", "."))
            )
            frame["shard"] = shard_dir.name
            rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 400:
        raise RuntimeError(f"{condition}: expected 400 target tests, found {len(result)}")
    result.insert(0, "condition", condition)
    return result


def summarize(group: pd.DataFrame) -> dict:
    denominator = pd.to_numeric(group["denominator_ess"], errors="coerce").fillna(0.0)
    tail = pd.to_numeric(group["tail_ess"], errors="coerce").fillna(0.0)
    evaluations = pd.to_numeric(group["aj_evaluations"], errors="raise")
    row = {
        "tests": len(group),
        "below_denominator_ess_80": int((denominator < 80.0).sum()),
        "below_tail_ess_15": int((tail < 15.0).sum()),
        "below_either_threshold": int(((denominator < 80.0) | (tail < 15.0)).sum()),
        "total_aj_evaluations": int(evaluations.sum()),
        "mean_aj_evaluations": float(evaluations.mean()),
    }
    for label, values in (("denominator_ess", denominator), ("tail_ess", tail)):
        row[f"{label}_min"] = float(values.min())
        for quantile in (0.10, 0.25, 0.50, 0.75, 0.90):
            row[f"{label}_q{int(100 * quantile):02d}"] = float(values.quantile(quantile))
        row[f"{label}_max"] = float(values.max())
        row[f"{label}_mean"] = float(values.mean())
    return row


def main():
    root = parse_args().input_dir.resolve()
    frames = [load_condition(root, condition) for condition in CONDITIONS]
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(root / "all_target_diagnostics.csv", index=False)
    summary_rows = []
    for condition, group in detail.groupby("condition", sort=False):
        summary_rows.append({"condition": condition, "scope": "all", **summarize(group)})
        for scope, scoped in group.groupby("scope", sort=False):
            summary_rows.append({"condition": condition, "scope": scope, **summarize(scoped)})
        for beta, scoped in group.loc[group.scope.eq("power")].groupby("signal_strength"):
            summary_rows.append({"condition": condition, "scope": f"power_beta_{beta:g}", **summarize(scoped)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "ais_fix_summary.csv", index=False)
    (root / "COMPLETED").touch()
    print(summary.loc[summary.scope.eq("all"), [
        "condition", "tests", "below_denominator_ess_80", "below_tail_ess_15",
        "below_either_threshold", "total_aj_evaluations",
        "denominator_ess_q10", "denominator_ess_q50", "tail_ess_q10", "tail_ess_q50",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
