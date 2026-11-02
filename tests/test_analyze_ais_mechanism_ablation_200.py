import pandas as pd
import pytest

from examples.analyze_ais_mechanism_ablation_200 import validate_matched_scenarios


def _matched_frame(t_values):
    return pd.DataFrame(
        {
            "scope": ["power"] * len(t_values),
            "shard": ["shard_0"] * len(t_values),
            "iteration": [1] * len(t_values),
            "feature": [0] * len(t_values),
            "is_signal": [True] * len(t_values),
            "auxiliary_u": [0.25] * len(t_values),
            "t_obs": t_values,
        }
    )


def test_matched_scenario_audit_allows_serialization_roundoff():
    validate_matched_scenarios(
        _matched_frame([10.0, 10.0 + 2e-13, 10.0 - 3e-13])
    )


def test_matched_scenario_audit_rejects_substantive_t_obs_difference():
    with pytest.raises(RuntimeError, match="substantive t_obs mismatch"):
        validate_matched_scenarios(_matched_frame([10.0, 10.01]))


def test_matched_scenario_audit_still_requires_exact_target_identity():
    frame = _matched_frame([10.0, 10.0])
    frame.loc[1, "feature"] = 1
    with pytest.raises(RuntimeError, match="not matched on feature"):
        validate_matched_scenarios(frame)
