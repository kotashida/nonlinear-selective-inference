"""Build the seven reference-style plots from the Linear RSS Shapley raw run."""
from __future__ import annotations

import json

import build_selected_six_with_linear_shapley_300 as source
import analyze_matched_ais_methods as summary_analysis
import plot_combined_ais_analysis as plotting


def main():
    out = source.LINEAR
    source.require((out / "COMPLETED").is_file(), "Linear Shapley run is incomplete")
    data, settings = source.read_linear_rows()
    source.require(len(data) == 1500, "Expected 1,500 Linear Shapley records")
    source.require(set(data.selection_method) == {"linear_shapley"}, "Unexpected method")
    source.require(not data.failed.isna().any(), "Missing failure indicators")
    keys = ["scope", "validation_shard", "iteration", "selection_event", "signal_strength"]
    source.require(not data.duplicated(keys).any(), "Duplicate trial records")
    null = data[data.scope.eq("null_calibration")]
    power = data[data.scope.eq("power")]
    source.require(len(null) == 300, "Expected 300 null trials")
    counts = power.groupby("signal_strength").size().to_dict()
    source.require(counts == {beta: 300 for beta in source.BETAS}, "Power trial count drift")

    data.to_csv(out / "unified_tests.csv", index=False)
    run = source.LINEAR_RUN.relative_to(source.BASE).as_posix()
    (out / "source_manifest.json").write_text(
        json.dumps({run: settings}, indent=2) + "\n", encoding="utf-8"
    )
    (out / "selected_configuration.json").write_text(
        json.dumps({
            "methods": ["linear_shapley"],
            "labels": {"linear_shapley": source.METHODS["linear_shapley"]},
            "source": source.LINEAR_RUN.relative_to(source.ROOT).as_posix(),
            "plot_reference": source.OUT.relative_to(source.ROOT).as_posix(),
            "trials_per_condition": 300,
            "total_tests": len(data),
        }, indent=2) + "\n", encoding="utf-8"
    )
    summary_analysis.OUT = out
    p, c = summary_analysis.summarize(data)
    source.require(len(p) == 4 and len(c) == 3, "Summary condition count drift")

    plotting.SOURCE = plotting.OUT = out
    plotting.DATA_FILENAME = "unified_tests.csv"
    plotting.TRIALS_PER_CONDITION = 300
    plotting.REPORT_LABEL = "Linear RSS Shapley (AIS)"
    plotting.REPORT_INTRO = (
        "Linear RSS Shapley selection with its matching quadratic test statistic and AIS "
        "inference: 300 null trials and 300 power trials at each beta (0.30, 0.50, 0.75, 1.00). "
        "Baseline design, exact-set conditioning, final sampling cap 51,200, pilot 10 x 320, "
        "batch size 800, ESS thresholds 80/15, and early stopping enabled. "
        "Plots use the same definitions and styling as the six-method matched comparison."
    )
    plotting.METHODS = {"linear_shapley": source.METHODS["linear_shapley"]}
    plotting.COLORS = {"linear_shapley": source.COLORS["linear_shapley"]}
    plotting.EXPECTED_PILOT_ITERS = (10,)
    plotting.ALLOW_INCOMPLETE = False
    plotting.FIGURES.clear()
    plotting.main()
    gallery = out / "index.html"
    gallery.write_text(
        gallery.read_text(encoding="utf-8").replace("Combined AIS analysis", "Linear RSS Shapley analysis"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
