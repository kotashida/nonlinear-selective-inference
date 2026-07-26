import numpy as np
import pytest

from si_shap import selective_inference
from si_shap.api import adjust_p_values
from si_shap.selection import SelectionResult


class FixedSelector:
    def select(self, X, response, k_select):
        importance = np.arange(X.shape[1], 0, -1, dtype=float)
        ranking = np.arange(X.shape[1])
        return SelectionResult(ranking[:k_select], importance, ranking)

    def get_settings(self):
        return {"selector": "fixed-test-selector"}


def _data():
    rng = np.random.default_rng(7)
    return rng.normal(size=(30, 4)), rng.normal(size=30)


def test_public_api_returns_k_featurewise_results_and_reproduces_t_obs(monkeypatch):
    checked = []

    def fake_ais(t_obs, rank, is_selected, rng, **kwargs):
        checked.append(is_selected(t_obs))
        return 0.25, {
            "status": "ok",
            "proposals": 100,
            "selected_samples": 100,
            "tail_samples": 25,
            "denominator_ess": 100.0,
            "tail_ess": 25.0,
            "mc_se": 0.02,
        }

    monkeypatch.setattr("si_shap.api._run_ais", fake_ais)
    X, y = _data()
    with pytest.warns(UserWarning, match="Exact-set"):
        result = selective_inference(
            X,
            y,
            k_select=2,
            sigma=1.5,
            selector=FixedSelector(),
        )

    assert checked == [True, True]
    assert result["observed_selected_features"].tolist() == [0, 1]
    assert len(result["feature_results"]) == 2
    assert set(result["feature_results"]["selection_event"]) == {"exact_set"}
    assert result["settings"]["variance_method"] == "known_user_supplied"
    assert result["settings"]["sigma"] == 1.5


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        ([[1.0, 2.0]], [1.0], "more than 4"),
        (np.ones((6, 2)), np.ones(5), "same number"),
        (np.array([[1.0, np.nan]] * 6), np.ones(6), "missing"),
    ],
)
def test_public_api_validates_data(X, y, message):
    with pytest.raises(ValueError, match=message):
        selective_inference(
            X, y, k_select=1, sigma=1.0, selector=FixedSelector()
        )


def test_public_api_requires_known_sigma():
    X, y = _data()
    with pytest.raises(ValueError, match="Unknown-variance"):
        selective_inference(
            X, y, k_select=1, sigma=None, selector=FixedSelector()
        )


def test_multiplicity_adjustments_preserve_failures():
    values = np.array([0.01, 0.04, np.nan])

    np.testing.assert_allclose(
        adjust_p_values(values, "bonferroni")[:2], [0.03, 0.12]
    )
    assert np.all(np.isnan(adjust_p_values(values, "holm")))
