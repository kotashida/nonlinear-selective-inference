import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from si_shap.selection import (
    MAX_CPU_CORES,
    MarginalCorrelationSelector,
    MutualInformationSelector,
    RF_PARAMS,
    _resolve_rf_params,
    _top_k,
    ShapSelector,
    make_selector,
    selection_event_holds,
    target_from_selected_set,
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


def test_rf_params_are_explicitly_limited_to_32_jobs():
    assert MAX_CPU_CORES == 32
    assert _resolve_rf_params(None)["n_jobs"] == MAX_CPU_CORES
    assert _resolve_rf_params({"n_jobs": 1})["n_jobs"] == 1
    assert _resolve_rf_params({"n_jobs": None})["n_jobs"] is None
    with pytest.raises(ValueError, match="n_jobs must be an integer from 1 to 32"):
        _resolve_rf_params({"n_jobs": 33})
    with pytest.raises(ValueError, match="n_jobs must be an integer from 1 to 32"):
        _resolve_rf_params({"n_jobs": -1})


def test_custom_estimator_cannot_exceed_cpu_limit():
    with pytest.raises(ValueError, match="n_jobs must be an integer from 1 to 32"):
        ShapSelector(RandomForestRegressor(random_state=42, n_jobs=33))


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
        reversed_candidate,
        observed,
        2,
        "exact_ranking",
        ranking=[5, 2, 0, 1, 3, 4],
        observed_ranking=[2, 5, 0, 1, 3, 4],
    )


def test_exact_ranking_compares_features_below_the_top_k():
    with pytest.raises(ValueError, match="complete candidate"):
        selection_event_holds([0, 1], [0, 1], 0, "exact_ranking")
    assert not selection_event_holds(
        [0, 1],
        [0, 1],
        0,
        "exact_ranking",
        ranking=[0, 1, 3, 2],
        observed_ranking=[0, 1, 2, 3],
    )


def test_same_target_reuses_fixed_auxiliary_randomness():
    observed = [1, 3, 5, 6]
    auxiliary_u = 0.45

    assert target_from_selected_set(observed, auxiliary_u) == 3
    assert selection_event_holds([3], observed, 3, "same_target", auxiliary_u)
    assert selection_event_holds(
        [1, 2, 3, 4, 5], observed, 3, "same_target", auxiliary_u
    )
    assert not selection_event_holds(
        [1, 2, 4, 5], observed, 3, "same_target", auxiliary_u
    )


@pytest.mark.parametrize(("selected", "u"), [([], 0.5), ([1], -0.1), ([1], 1.0)])
def test_target_from_selected_set_rejects_invalid_inputs(selected, u):
    with pytest.raises(ValueError):
        target_from_selected_set(selected, u)


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

    with pytest.raises(ValueError, match="one-dimensional responses"):
        selector.select(X, response, 1)


@pytest.mark.parametrize(
    ("method", "expected_type"),
    [
        ("shap", ShapSelector),
        ("mutual_information", MutualInformationSelector),
        ("marginal_screening", MarginalCorrelationSelector),
    ],
)
def test_make_selector_resolves_three_builtin_methods(method, expected_type):
    assert isinstance(make_selector(selection_method=method), expected_type)


def test_make_selector_rejects_unknown_or_conflicting_methods():
    with pytest.raises(ValueError, match="selection_method must be"):
        make_selector(selection_method="unknown")
    with pytest.raises(ValueError, match="either selection_method or selector"):
        make_selector(selection_method="shap", selector=MarginalCorrelationSelector())
    with pytest.raises(ValueError, match="only for SHAP"):
        make_selector(
            selection_method="mutual_information",
            estimator=RandomForestRegressor(random_state=42),
        )


@pytest.mark.parametrize(
    "selector", [MutualInformationSelector(), MarginalCorrelationSelector()]
)
def test_non_shap_builtin_selectors_are_deterministic(selector):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(30, 5))
    y = X[:, 3] + 0.1 * rng.normal(size=30)

    first = selector.select(X, y, 2)
    second = selector.select(X, y, 2)

    assert first.scores.shape == (X.shape[1],)
    np.testing.assert_array_equal(first.selected_features, first.ranking[:2])
    np.testing.assert_array_equal(first.ranking, second.ranking)
    np.testing.assert_array_equal(first.scores, second.scores)


def test_marginal_selector_handles_constant_columns_deterministically():
    X = np.column_stack((np.ones(10), np.arange(10), -np.arange(10)))
    result = MarginalCorrelationSelector().select(X, np.arange(10), 2)

    np.testing.assert_array_equal(result.selected_features, [1, 2])
    assert result.scores[0] == 0.0
