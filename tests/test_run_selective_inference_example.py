import argparse

import numpy as np
import pandas as pd

from examples import run_selective_inference as example


def test_parse_rf_parameter_preserves_json_types():
    assert example._parse_rf_parameter("n_estimators=100") == (
        "n_estimators",
        100,
    )
    assert example._parse_rf_parameter("max_depth=null") == ("max_depth", None)


def test_parse_rf_parameter_requires_name_value_syntax():
    try:
        example._parse_rf_parameter("n_estimators")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("Expected argparse.ArgumentTypeError")


def test_main_saves_feature_level_selective_p_values(monkeypatch, tmp_path):
    diagnostics = pd.DataFrame(
        [
            {
                "iteration": 1,
                "feature": 2,
                "rank": 3,
                "t_obs": 2.1,
                "p_value": 0.14,
                "status": "ok",
                "proposals": 160,
                "selected_samples": 90,
                "tail_samples": 30,
                "denominator_ess": 82.0,
                "tail_ess": 18.0,
                "mc_se": 0.02,
            }
        ]
    )
    fake_result = {
        "summary": pd.DataFrame([{"method": "Selective SHAP (AIS)"}]),
        "p_values": {"Selective SHAP (AIS)": np.array([0.14])},
        "ais_diagnostics": diagnostics,
        "settings": {"seed": 123, "rf_params": {}},
    }
    monkeypatch.setattr(example, "run_simulation", lambda **kwargs: fake_result)

    example.main(["--output-dir", str(tmp_path)])

    saved = pd.read_csv(tmp_path / "selective_inference.csv")
    assert saved.loc[0, "p_value"] == 0.14
    assert saved.loc[0, "status"] == "ok"
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "converged_p_values.csv").exists()
    assert (tmp_path / "settings.json").exists()
