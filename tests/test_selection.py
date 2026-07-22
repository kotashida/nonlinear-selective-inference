import numpy as np
import pytest

from si_shap.selection import RF_PARAMS, _resolve_rf_params, _top_k


def test_top_k_breaks_ties_by_feature_index():
    selected = _top_k([1.0, 1.0, 0.5], 2)

    np.testing.assert_array_equal(selected, [0, 1])


@pytest.mark.parametrize("scores", [[1.0, np.nan], [[1.0, 2.0]]])
def test_top_k_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        _top_k(scores, 1)


@pytest.mark.parametrize("k", [0, 4, 1.5])
def test_top_k_rejects_invalid_selection_sizes(k):
    with pytest.raises(ValueError, match="k must satisfy"):
        _top_k([3.0, 2.0, 1.0], k)


def test_rf_params_override_defaults_without_mutating_inputs():
    overrides = {"n_estimators": 12, "min_samples_leaf": 3}

    resolved = _resolve_rf_params(overrides)

    assert resolved == {
        **RF_PARAMS,
        "n_estimators": 12,
        "min_samples_leaf": 3,
    }
    assert overrides == {"n_estimators": 12, "min_samples_leaf": 3}


def test_rf_params_must_be_a_mapping():
    with pytest.raises(TypeError, match="mapping or None"):
        _resolve_rf_params([("n_estimators", 12)])
