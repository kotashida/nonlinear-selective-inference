import numpy as np
import pandas as pd

from examples import validate_selection_methods as validation


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
