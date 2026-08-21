import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pandas as pd
import pytest

from si_shap import simulation
from si_shap.selection import SelectionResult
from si_shap.simulation import (
    _generate_gaussian_design,
    _generate_null_dataset,
    _method_summary,
    _validate_inputs,
)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 100, 10, 1, 0.05),
        (1, 4, 10, 1, 0.05),
        (1, 100, 0, 1, 0.05),
        (1, 100, 10, 0, 0.05),
        (1, 100, 10, 1, 1.0),
    ],
)
def test_validate_inputs_rejects_invalid_settings(arguments):
    with pytest.raises(ValueError):
        _validate_inputs(*arguments)


def test_validate_inputs_rejects_noninteger_counts():
    with pytest.raises(TypeError, match="n_iters"):
        _validate_inputs(1.5, 100, 10, 1, 0.05)
    with pytest.raises(TypeError, match="k_select"):
        _validate_inputs(1, 100, 10, True, 0.05)


def test_method_summary_reports_failures_without_silently_estimating_fpr():
    summary, converged = _method_summary(
        "Selective SHAP (approximate AIS)",
        [np.array([0.01]), np.array([np.nan])],
        alpha=0.05,
    )

    assert np.isnan(summary["fpr"])
    assert summary["failure_rate"] == 0.5
    np.testing.assert_array_equal(converged, [0.01])


def test_generate_null_dataset_is_reproducible():
    first = _generate_null_dataset(np.random.default_rng(123), 10, 4)
    second = _generate_null_dataset(np.random.default_rng(123), 10, 4)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_gaussian_design_supports_ar1_feature_correlation():
    X = _generate_gaussian_design(np.random.default_rng(123), 5000, 4, 0.6)

    correlations = np.corrcoef(X, rowvar=False)
    assert correlations[0, 1] == pytest.approx(0.6, abs=0.04)
    assert correlations[0, 2] == pytest.approx(0.6**2, abs=0.04)


@pytest.mark.parametrize("value", [-0.1, 1.0, np.nan])
def test_gaussian_design_rejects_invalid_feature_correlation(value):
    with pytest.raises(ValueError, match="feature_correlation"):
        _generate_gaussian_design(np.random.default_rng(123), 20, 4, value)


def test_method_summary_reports_global_null_family_metrics():
    summary, _ = _method_summary(
        "method",
        [np.array([0.01, 0.9]), np.array([0.2, 0.3])],
        alpha=0.05,
    )

    assert summary["fpr"] == 0.25
    assert summary["familywise_error_rate"] == 0.5
    assert summary["false_discovery_rate"] == 0.5
    assert summary["mean_rejections"] == 0.5
    assert np.isfinite(summary["fpr_ci_95_lower"])
    assert 0.0 <= summary["familywise_error_ci_95_lower"] <= 0.5
    assert 0.5 <= summary["familywise_error_ci_95_upper"] <= 1.0
    assert np.isfinite(summary["uniform_ks_statistic"])


class FixedSelector:
    def select(self, X, response, k_select):
        ranking = np.arange(X.shape[1])
        return SelectionResult(
            ranking[:k_select],
            np.arange(X.shape[1], 0, -1, dtype=float),
            ranking,
        )

    def get_settings(self):
        return {"selector": "fixed"}


def test_run_simulation_uses_requested_adjusted_p_values(monkeypatch):
    def fake_selective_inference(X, y, **kwargs):
        return {
            "observed_selected_features": np.array([0, 1]),
            "feature_results": pd.DataFrame(
                {
                    "feature": [0, 1],
                    "test_rank": [3, 3],
                    "unadjusted_p_value": [0.01, 0.4],
                    "raw_selective_p_value": [0.01, 0.4],
                    "adjusted_selective_p_value": [0.02, 0.8],
                    "p_value": [0.01, 0.4],
                    "status": ["ok", "ok"],
                }
            ),
        }

    monkeypatch.setattr(simulation, "selective_inference", fake_selective_inference)
    result = simulation.run_simulation(
        1,
        20,
        3,
        2,
        alpha=0.015,
        selector=FixedSelector(),
        multiplicity="bonferroni",
    )

    np.testing.assert_allclose(
        result["p_values"]["Selective SHAP (conditional MC)"], [0.02, 0.8]
    )
    # The unadjusted and random baselines receive the same family-size
    # correction rather than being compared on a different p-value scale.
    assert set(result["summary"]["p_value_scale"]) == {
        "multiplicity_adjusted"
    }
    assert result["settings"]["inference_method"] == "conditional_mc"
    assert result["settings"]["pilot_iters"] == 3
    assert result["settings"]["max_final_samples"] == 800
