"""Deterministic SHAP-based feature selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor


RF_PARAMS = {"n_estimators": 50, "max_depth": 5, "random_state": 42}


def _resolve_rf_params(rf_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return forest parameters with user overrides applied to the defaults."""
    if rf_params is not None and not isinstance(rf_params, Mapping):
        raise TypeError("rf_params must be a mapping or None.")
    return {**RF_PARAMS, **({} if rf_params is None else rf_params)}


def _top_k(scores, k):
    """Select by decreasing score, breaking ties by increasing feature index."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError(
            "SHAP importance scores must be a finite one-dimensional array."
        )
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= scores.size:
        raise ValueError("k must satisfy 1 <= k <= the number of scores.")
    return np.lexsort((np.arange(scores.size), -scores))[:k]


def _tree_shap_importance(X, response, selection_decimals, rf_params=None):
    """Fit the configured forest and return mean absolute Tree SHAP importance."""
    stable_response = np.round(response, decimals=selection_decimals)
    model = RandomForestRegressor(**_resolve_rf_params(rf_params))
    model.fit(X, stable_response)
    values = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent"
    ).shap_values(X)
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected a single-output SHAP result.")
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != X.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {X.shape}.")
    return np.mean(np.abs(values), axis=0)


def _select_features(X, response, k_select, selection_decimals, rf_params=None):
    importance = _tree_shap_importance(
        X, response, selection_decimals, rf_params=rf_params
    )
    return _top_k(importance, k_select)
