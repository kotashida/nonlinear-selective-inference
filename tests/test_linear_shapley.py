from itertools import combinations
from math import comb

import numpy as np
import pytest

from si_shap import LinearShapleySelector, LinearShapleyStatistic, selective_inference
from si_shap.selection import make_selector, selection_event_holds


def direct_scores(X, y, intercept):
    """Independent least-squares RSS implementation of the definition."""
    p = X.shape[1]

    def rss(columns):
        design = X[:, columns]
        if intercept:
            design = np.column_stack((np.ones(len(y)), design))
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        return residual @ residual

    scores = np.zeros(p)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for size in range(p):
            for subset in combinations(others, size):
                scores[j] += (rss(list(subset)) - rss([*subset, j])) / (p * comb(p - 1, size))
    return scores


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("aliased", [True, False])
def test_quadratic_forms_match_direct_rss(intercept, aliased):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(25, 4)) + 2
    if aliased:
        X[:, 2] = X[:, 0]
        X[:, 3] = 1
    y = rng.normal(size=25) + 3
    statistic = LinearShapleyStatistic(X, fit_intercept=intercept)
    expected = direct_scores(X, y, intercept)
    np.testing.assert_allclose(statistic.scores(y), expected, atol=1e-10)
    np.testing.assert_allclose(np.einsum("n,jnm,m->j", y, statistic.matrices, y), expected, atol=1e-10)
    assert np.linalg.eigvalsh(statistic.matrices).min() > -1e-12
    # Efficiency: contributions partition the full model's explained RSS.
    design = np.column_stack((np.ones(25), X)) if intercept else X
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    baseline = y - y.mean() if intercept else y
    np.testing.assert_allclose(expected.sum(), baseline @ baseline - (y-fitted) @ (y-fitted), atol=1e-10)


def test_permutation_approximation_agrees_and_is_fixed():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 3))
    X[:, 1] += X[:, 0]
    y = X[:, 0] + rng.normal(size=20)
    exact = LinearShapleyStatistic(X)
    mc = LinearShapleyStatistic(X, mode="monte_carlo", n_permutations=12000)
    np.testing.assert_allclose(mc.matrices, exact.matrices, atol=0.015)
    np.testing.assert_allclose(mc.scores(y), exact.scores(y), rtol=0.04, atol=0.1)
    duplicate = LinearShapleyStatistic(X, mode="monte_carlo", n_permutations=12000)
    np.testing.assert_array_equal(mc.reduced_matrices, duplicate.reduced_matrices)
    assert LinearShapleyStatistic(X, exact_max_features=2, n_permutations=3).mode == "monte_carlo"


def test_selector_top_k_and_cache_mutation():
    X = np.column_stack((np.arange(10.), np.ones(10)))
    selector = make_selector(selection_method="linear_shapley")
    result = selector.select(X, X[:, 0], 1)
    assert result.selected_features.tolist() == [0]
    assert result.scores[1] == 0
    cached = selector.statistic(X)
    assert selector.statistic(X.copy()) is cached
    X[:, 0] = 1
    assert selector.statistic(X) is not cached
    assert selector.select(X, np.ones(10), 1).ranking.tolist() == [0, 1]


def test_wide_design_and_intercept_shift_invariance():
    rng = np.random.default_rng(11)
    X, y = rng.normal(size=(5, 6)), rng.normal(size=5)
    statistic = LinearShapleyStatistic(X)
    np.testing.assert_allclose(statistic.scores(y), direct_scores(X, y, True), atol=1e-10)
    np.testing.assert_allclose(statistic.scores(y + 7), statistic.scores(y), atol=1e-10)
    np.testing.assert_allclose(LinearShapleyStatistic(X + 10).scores(y), statistic.scores(y), atol=1e-10)


def test_single_feature_empty_coalition():
    X = np.arange(1., 11.)[:, None]
    y = X[:, 0] + 3
    for intercept in (True, False):
        statistic = LinearShapleyStatistic(X, fit_intercept=intercept)
        np.testing.assert_allclose(statistic.scores(y), direct_scores(X, y, intercept), atol=1e-10)


@pytest.mark.parametrize("method", ["conditional_mc", "mcmc_rank", "ais"])
def test_pipeline_same_score_and_candidate_event(monkeypatch, method):
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(30, 3)), rng.normal(size=30)
    selector = LinearShapleySelector()
    observed = selector.select(X, y, 2)
    statistic = selector.statistic(X)
    checked = []

    def sampler(radius, rank, is_selected, rng, **kwargs):
        feature = int(observed.selected_features[len(checked)])
        basis = statistic.support_basis(feature)
        projected = basis @ (basis.T @ y)
        direction = projected / np.linalg.norm(projected)
        assert is_selected(radius)
        for z in (0.1, 1.0, 3.0, 10.0):
            candidate = y - projected + direction * z
            expected = selection_event_holds(selector.select(X, candidate, 2).selected_features,
                                             observed.selected_features, feature)
            assert is_selected(z) == expected
            np.testing.assert_allclose(statistic.scores(candidate)[feature],
                                       statistic.scores(direction)[feature] * z**2, atol=1e-10)
        checked.append(feature)
        return 0.5, {"status": "ok"}

    monkeypatch.setattr(f"si_shap.api._run_{method}", sampler)
    with pytest.warns(UserWarning):
        result = selective_inference(X, y, k_select=2, sigma=1, selector=selector,
                                     selection_event="exact_set", inference_method=method)
    rows = result["feature_results"]
    np.testing.assert_allclose(rows.t_obs, rows.importance_score)
    assert result["settings"]["test_statistic"] == "linear_shapley"
    assert len(checked) == 2


def test_actual_sampler_and_zero_rank():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(20, 3))
    result = selective_inference(X, rng.normal(size=20), k_select=3, sigma=2,
                                 selection_method="linear_shapley", max_final_samples=40)
    assert result["feature_results"].p_value.between(0, 1).all()
    result = selective_inference(np.ones((20, 2)), np.ones(20), k_select=2,
                                 sigma=1, selection_method="linear_shapley", max_final_samples=4)
    assert (result["feature_results"].p_value == 1).all()
    assert (result["feature_results"].test_rank == 0).all()


def test_experiment_helpers_use_linear_basis(monkeypatch):
    from si_shap import power, selection_regions, simulation

    def wrong_basis(*args):
        raise AssertionError("Linear experiments must use the linear range basis")

    for module in (power, selection_regions, simulation):
        monkeypatch.setattr(module, "_spline_effect_basis", wrong_basis)
    with pytest.warns(UserWarning):
        result = simulation.run_simulation(1, 20, 3, 1,
                                           selection_method="linear_shapley", max_final_samples=5)
    assert "Unadjusted Linear Shapley" in result["p_values"]
    power.compare_selection_event_power(1, 20, 3, 1, selection_method="linear_shapley",
                                         selection_events=("same_target",), max_final_samples=5)
    regions = selection_regions.compute_selection_regions(
        dataset_seeds=[2], n_samples=20, n_features=3, k_select=1,
        selection_method="linear_shapley", grid_size=11, grid_refinements=0,
        proposal_pilot_iters=0,
    )
    assert len(regions) == 1
    assert regions[0].rank >= 1


@pytest.mark.parametrize("options", [{"random_state": None}, {"mode": "bad"},
                                     {"n_permutations": 0}, {"fit_intercept": 1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        LinearShapleyStatistic(np.ones((10, 2)), **options)
