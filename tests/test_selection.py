import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from si_shap.selection import (
    RF_PARAMS,
    _resolve_rf_params,
    _top_k,
    ShapSelector,
    selection_event_holds,
)


def test_top_k_breaks_ties_by_feature_index():
    selected = _top_k([1.0, 1.0, 0.5], 2)

    np.testing.assert_array_equal(selected, [0, 1])


@pytest.mark.parametrize("scores", [[1.0, np.nan], [[1.0, 2.0]]])
def test_top_k_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        _top_k(scores, 1)


@pytest.mark.parametrize("k", [0, 4, 1.5, True])
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


def test_exact_set_rejects_replacement_while_inclusion_accepts_it():
    observed = np.array([2, 5])
    candidate = np.array([2, 7])

    assert not selection_event_holds(candidate, observed, 2, "exact_set")
    assert selection_event_holds(candidate, observed, 2, "feature_inclusion")


def test_exact_set_is_unordered_but_exact_ranking_is_ordered():
    observed = np.array([2, 5])
    reversed_candidate = np.array([5, 2])

    assert selection_event_holds(reversed_candidate, observed, 2, "exact_set")
    assert not selection_event_holds(
        reversed_candidate, observed, 2, "exact_ranking"
    )


def test_stochastic_estimator_requires_fixed_random_state():
    with pytest.raises(ValueError, match="random_state"):
        ShapSelector(RandomForestRegressor(random_state=None))


def test_shap_selector_returns_one_deterministic_importance_per_feature():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(24, 4))
    y = rng.normal(size=24)
    selector = ShapSelector(
        RandomForestRegressor(
            n_estimators=3, max_depth=2, random_state=42
        )
    )

    first = selector.select(X, y, 2)
    second = selector.select(X, y, 2)

    assert first.importance.shape == (X.shape[1],)
    assert first.ranking.shape == (X.shape[1],)
    np.testing.assert_array_equal(first.selected_features, first.ranking[:2])
    np.testing.assert_array_equal(first.ranking, second.ranking)
    np.testing.assert_array_equal(first.importance, second.importance)


def test_shap_selector_explicitly_rejects_multioutput_shap_values():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(20, 3))
    response = rng.normal(size=(20, 2))
    selector = ShapSelector(
        RandomForestRegressor(n_estimators=3, max_depth=2, random_state=42)
    )

    with pytest.raises(ValueError, match="Unexpected SHAP shape|single-output"):
        selector.select(X, response, 1)
