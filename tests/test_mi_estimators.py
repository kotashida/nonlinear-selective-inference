import numpy as np
import pytest
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KernelDensity

from si_shap.selection import MutualInformationSelector


def test_histogram_matches_discrete_mi_with_ties():
    X = np.array([[0, 1], [0, 1], [1, 1], [1, 1], [2, 1], [2, 1]], dtype=float)
    y = X[:, 0].copy()
    result = MutualInformationSelector(estimator="histogram", n_bins=3).select(X, y, 1)
    np.testing.assert_allclose(result.scores, [mutual_info_score(X[:, 0], y), 0])


def test_kde_matches_independent_leave_one_out_density_reference():
    rng = np.random.default_rng(89)
    X = rng.normal(size=(35, 2))
    y = X[:, 0] + rng.normal(scale=0.15, size=35)
    z = np.column_stack((X, y))
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)
    h = len(y) ** (-1 / 6)
    reference = []
    for j in range(2):
        terms = []
        for i in range(len(y)):
            train = np.arange(len(y)) != i
            def density(columns):
                return KernelDensity(bandwidth=h).fit(z[train][:, columns]).score_samples(z[i:i+1, columns])[0]
            terms.append(density([j, 2]) - density([j]) - density([2]))
        reference.append(max(0, np.mean(terms)))
    result = MutualInformationSelector(estimator="kde").select(X, y, 1)
    np.testing.assert_allclose(result.scores, reference, atol=1e-12)


@pytest.mark.parametrize("estimator", ["histogram", "kde"])
def test_response_recomputed_cache_invalidated_and_nonlinearity_detected(estimator):
    rng = np.random.default_rng(35)
    X = rng.normal(size=(100, 4))
    y = X[:, 0] ** 2 + 0.05 * rng.normal(size=100)
    selector = MutualInformationSelector(estimator=estimator)
    assert selector.select(X, y, 1).selected_features[0] == 0
    y = X[:, 1] ** 2 + 0.05 * rng.normal(size=100)
    np.testing.assert_allclose(selector.select(X, y, 1).scores,
                               MutualInformationSelector(estimator=estimator).select(X, y, 1).scores)
    assert selector.select(X, y, 1).selected_features[0] == 1
    X[:, 2] = y
    np.testing.assert_allclose(selector.select(X, y, 1).scores,
                               MutualInformationSelector(estimator=estimator).select(X, y, 1).scores)
    np.testing.assert_allclose(selector.select(X, np.ones(100), 1).scores, 0, atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"estimator": "unknown"}, {"n_bins": 1}, {"bandwidth": 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MutualInformationSelector(**kwargs)


def test_pooling_rejects_mixed_mi_estimators():
    from pathlib import Path
    from examples.summarize_selection_method_validation_shards import _validate_settings
    with pytest.raises(ValueError, match="mi_params"):
        _validate_settings([Path("shard_0"), Path("shard_1")],
                           [{"mi_params": {"estimator": "histogram"}},
                            {"mi_params": {"estimator": "kde"}}])
