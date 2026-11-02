"""Create the focused four-pipeline analysis from the verified six-method data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "examples")]

import pandas as pd

import analyze_matched_ais_methods as summary_analysis
import plot_combined_ais_analysis as plotting

BASE = ROOT / "outputs/selection_method_validation/exact_set"
SOURCE = BASE / "selected_six_with_linear_shapley_300_20260928"
OUT = BASE / "selected_four_with_linear_shapley_300_20260928"
METHODS = {
    "spline_screening_exact": "Spline screening (exact)",
    "spline_screening": "Spline screening (AIS)",
    "shap": "Tree SHAP (AIS)",
    "linear_shapley": "Linear RSS Shapley (AIS)",
}
COLORS = {
    "spline_screening_exact": "#c44e52",
    "spline_screening": "#258568",
    "shap": "#2674a6",
    "linear_shapley": "#6b6ecf",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require((SOURCE / "COMPLETED").is_file(), "Six-method source analysis is incomplete")
    source = pd.read_csv(SOURCE / "unified_tests.csv", low_memory=False)
    data = source[source["selection_method"].isin(METHODS)].copy()
    data["selection_method"] = pd.Categorical(
        data["selection_method"], categories=list(METHODS), ordered=True
    )
    data = data.sort_values(
        ["selection_method", "scope", "validation_shard", "iteration", "signal_strength"],
        na_position="first",
    )
    data["selection_method"] = data["selection_method"].astype(str)
    require(len(data) == 6000, f"Expected 6,000 rows, got {len(data)}")
    counts = data.groupby(["selection_method", "scope"]).size().unstack()
    require((counts["null_calibration"] == 300).all(), "Null count drift")
    require((counts["power"] == 1200).all(), "Power count drift")
    require(set(data["selection_method"]) == set(METHODS), "Method set drift")

    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "unified_tests.csv", index=False)
    source_manifest = json.loads((SOURCE / "source_manifest.json").read_text(encoding="utf-8"))
    selected_runs = set(data["source_run"].dropna().unique())
    require(selected_runs <= set(source_manifest), "Missing source-run manifest")
    (OUT / "source_manifest.json").write_text(
        json.dumps({run: source_manifest[run] for run in sorted(selected_runs)}, indent=2) + "\n",
        encoding="utf-8",
    )
    configuration = {
        "methods": list(METHODS),
        "labels": METHODS,
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "trials_per_condition": 300,
        "total_tests": 6000,
        "comparison_scope": "end-to-end selective-inference pipelines",
        "interpretation": {
            "spline_exact_vs_spline_ais": "isolates the inference engine for the spline selector",
            "spline_ais_vs_shap_ais": "compares selectors with the spline test and AIS held fixed",
            "linear_shapley": "uses a different rank-20 quadratic association null",
        },
    }
    (OUT / "selected_configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    summary_analysis.OUT = OUT
    power, calibration = summary_analysis.summarize(data)
    require(len(power) == 16, "Power summary row drift")
    require(len(calibration) == 12, "Calibration summary row drift")

    plotting.SOURCE = plotting.OUT = OUT
    plotting.DATA_FILENAME = "unified_tests.csv"
    plotting.TRIALS_PER_CONDITION = 300
    plotting.REPORT_LABEL = "Focused four-pipeline matched comparison"
    plotting.REPORT_INTRO = (
        "Focused comparison with 300 null trials and 300 power trials per beta per pipeline. "
        "Spline exact versus spline AIS isolates analytic versus AIS inference; spline AIS "
        "versus Tree SHAP AIS compares selectors while retaining the spline test; Linear RSS "
        "Shapley AIS is a distinct pipeline with a rank-20 quadratic association null."
    )
    plotting.METHODS = METHODS.copy()
    plotting.COLORS = COLORS.copy()
    plotting.EXPECTED_PILOT_ITERS = (10,)
    plotting.ALLOW_INCOMPLETE = False
    plotting.FIGURES.clear()
    plotting.main()
    (OUT / "COMPLETED").write_text(
        "Verified 6,000 records and generated the focused four-pipeline analysis.\n",
        encoding="utf-8",
    )
    print(f"Built {OUT} from {len(data):,} verified records.")


if __name__ == "__main__":
    main()
