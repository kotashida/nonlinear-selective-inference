import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import json
import numpy as np
import pandas as pd

from examples import validate_selection_methods as validation
from examples import summarize_selection_method_validation_shards as shard_summary


def test_calibration_decision_detects_obvious_anti_conservatism():
    rows = []
    for method, values in (
        ("calibrated", np.linspace(0.001, 0.999, 1000)),
        ("anti", np.linspace(0.0001, 0.4999, 1000)),
    ):
        frame = pd.DataFrame(
            {
                "selection_event": ["same_target"] * len(values),
                "p_value": values,
            }
        )
        rows.extend(
            validation._calibration_rows(
                frame,
                method=method,
                design="baseline",
                auxiliary_regime="fresh",
            )
        )

    decisions = validation._apply_calibration_decisions(
        pd.DataFrame.from_records(rows)
    )

    assert decisions.loc[
        decisions["selection_method"] == "calibrated", "calibration_pass"
    ].all()
    assert not decisions.loc[
        decisions["selection_method"] == "anti", "calibration_pass"
    ].all()


def test_power_decision_uses_conditional_lower_confidence_bound():
    frame = pd.DataFrame(
        {
            "selection_event": ["same_target"] * 100,
            "target_is_signal": [True] * 100,
            "p_value": [0.01] * 90 + [0.5] * 10,
            "rejected": [True] * 90 + [False] * 10,
            "successful_detection": [True] * 90 + [False] * 10,
        }
    )

    row = validation._power_row(
        frame,
        method="marginal_screening",
        design="baseline",
        signal_feature=0,
        signal_strength=1.0,
        minimum_conditional_power=0.8,
    )

    assert row["conditional_power_given_signal_target"] == 0.9
    assert row["conditional_power_ci_95_lower"] > 0.8
    assert row["enough_power"]


def test_comprehensive_preset_covers_multiple_designs_and_signal_strengths():
    preset = validation.PRESETS["comprehensive"]

    assert preset["n_null_iters"] >= 1000
    assert preset["n_power_iters"] >= 1000
    assert len(preset["designs"]) >= 3
    assert len(preset["signal_strengths"]) >= 5
    assert preset["fixed_auxiliary_values"] == (None, 0.25, 0.75)


def test_validation_shards_are_pooled_before_decisions(tmp_path):
    root = tmp_path / "validation_shards"
    common = {
        "preset": "smoke",
        "methods": ["marginal_screening"],
        "designs": ["baseline"],
        "n_null_iters": 2,
        "n_power_iters": 2,
        "max_final_samples": 40,
        "fixed_auxiliary_values": [None],
        "signal_strengths": [0.3],
        "signal_positions": ["first"],
        "minimum_conditional_power": 0.8,
        "minimum_calibration_iterations": 4,
        "minimum_signal_targets": 3,
        "primary_selection_event": "same_target",
        "comparison_events": ["feature_inclusion", "same_target"],
        "alpha_levels": [0.01, 0.05, 0.1],
        "design_seed": 99,
        "rf_jobs": 1,
        "runtime_metadata": {
            "python_version": "test",
            "package_versions": {},
            "git_commit": "abc",
        },
    }
    for index, p_values in enumerate(((0.2, 0.4), (0.6, 0.8))):
        shard = root / f"shard_{index}"
        null_dir = shard / "marginal_screening" / "baseline" / "null_fresh"
        power_dir = (
            shard
            / "marginal_screening"
            / "baseline"
            / "power_feature_0_beta_0p3"
        )
        null_dir.mkdir(parents=True)
        power_dir.mkdir(parents=True)
        settings = {**common, "seed": 100 + index}
        (shard / "validation_settings.json").write_text(
            json.dumps(settings), encoding="utf-8"
        )
        pd.DataFrame(
            {"selection_event": ["same_target"] * 2, "p_value": p_values}
        ).to_csv(null_dir / "p_value_results.csv", index=False)
        np.save(null_dir / "fixed_design.npy", np.arange(6).reshape(3, 2))
        pd.DataFrame(
            {
                "selection_event": ["same_target"] * 2,
                "target_is_signal": [True, True],
                "p_value": [0.01, 0.02],
                "rejected": [True, True],
                "successful_detection": [True, True],
            }
        ).to_csv(power_dir / "target_results.csv", index=False)

    result = shard_summary.summarize_shards(
        root, root / "pooled_summary", expected_shards=2
    )

    assert set(result["calibration"]["n_iterations"]) == {4}
    assert set(result["power"]["n_iterations"]) == {4}
    assert result["settings"]["n_null_iters"] == 4
    assert result["settings"]["n_power_iters"] == 4
    assert result["settings"]["shard_root_seeds"] == [100, 101]


def test_validation_cli_accepts_shared_design_and_rf_worker_limits():
    args = validation.parse_args(
        [
            "--design-seed",
            "99",
            "--rf-jobs",
            "3",
            "--auxiliary-values",
            "fresh",
            "0.75",
            "--signal-strengths",
            "0.5",
            "--signal-positions",
            "first",
        ]
    )

    assert args.design_seed == 99
    assert args.rf_jobs == 3
    assert args.auxiliary_values == [None, 0.75]
    assert args.signal_strengths == [0.5]
    assert args.signal_positions == ["first"]
