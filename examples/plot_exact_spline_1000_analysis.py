"""Create the reference seven-plot gallery for the 1,000-trial exact spline run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _read_raw
import analyze_matched_ais_methods as analysis
import plot_combined_ais_analysis as plotting


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/selection_method_validation/exact_set/exact_spline_matched_ais_mi30_1000"
OUT = RUN / "pooled_summary"
METHOD = "spline_screening_exact"


def main():
    settings = json.loads((OUT / "validation_settings.json").read_text(encoding="utf-8"))
    null, targets, features = _read_raw(RUN, "exact", settings, settings["n_shards"])
    keys = ["selection_method", "validation_shard", "iteration", "selection_event", "signal_strength"]
    targets = targets.merge(
        features[keys + ["feature", "t_obs", "proposals", "denominator_ess", "tail_ess"]],
        on=keys,
        validate="one_to_one",
    )
    frames = []
    for frame, scope in ((null, "null_calibration"), (targets, "power")):
        frame = frame.copy()
        frame["scope"] = scope
        frame["ais_setting"] = "exact"
        frame["source_run"] = str(RUN.relative_to(ROOT)).replace("\\", "/")
        frame["max_final_samples"] = settings["max_final_samples"]
        frame["pilot_iters"] = settings["pilot_iters"]
        frame["pilot_samples"] = settings["pilot_samples"]
        frame["total_proposals"] = 0
        frame["selection_method"] = METHOD
        frame["option"] = "Spline screening (exact)"
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    assert len(null) == settings["n_null_iters"]
    assert len(targets) == 4 * settings["n_power_iters"]
    assert not data.failed.any()

    calibration = pd.read_csv(OUT / "calibration_decisions.csv")
    power = pd.read_csv(OUT / "power_decisions.csv")
    for row in calibration.itertuples():
        assert len(null) == row.n_iterations
        assert int(null.failed.sum()) == row.n_failed
        assert int(null.p_value.lt(row.alpha).sum()) == row.n_rejected
    for row in power.itertuples():
        group = targets[targets.signal_strength.eq(row.signal_strength)]
        signal = group[group.target_is_signal.astype(bool)]
        assert len(group) == row.n_iterations
        assert len(signal) == row.n_signal_targets
        assert int((signal.p_value.lt(.05) & ~signal.failed).sum()) == int(round(row.marginal_detection_power * row.n_iterations))

    data.to_csv(OUT / "unified_tests.csv", index=False)
    previous = analysis.OUT
    analysis.OUT = OUT
    analysis.summarize(data)
    analysis.OUT = previous
    (OUT / "source_manifest.json").write_text(
        json.dumps({str(RUN.relative_to(ROOT)).replace("\\", "/"): settings}, indent=2),
        encoding="utf-8",
    )

    plotting.SOURCE = OUT
    plotting.OUT = OUT
    plotting.DATA_FILENAME = "unified_tests.csv"
    plotting.TRIALS_PER_CONDITION = settings["n_null_iters"]
    plotting.REPORT_LABEL = "Exact spline screening"
    plotting.METHODS.clear()
    plotting.METHODS.update({METHOD: "Spline screening (exact)"})
    plotting.COLORS.clear()
    plotting.COLORS.update({METHOD: "#c44e52"})
    plotting.FIGURES.clear()
    plotting.main()
    print(f"Verified {len(data):,} records and saved the seven-plot gallery to {OUT}")


if __name__ == "__main__":
    main()
