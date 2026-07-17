"""Deterministic SHAP-based feature selection."""

from __future__ import annotations

import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor


RF_PARAMS = {"n_estimators": 50, "max_depth": 5, "random_state": 42}


def _top_k(scores, k):
    """Select by decreasing score, breaking ties by increasing feature index."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError(
            "SHAP importance scores must be a finite one-dimensional array."
        )
    return np.lexsort((np.arange(scores.size), -scores))[:k]


def _tree_shap_importance(X, response, selection_decimals):
    """Fit the fixed forest and return mean absolute Tree SHAP importance."""
    stable_response = np.round(response, decimals=selection_decimals)
    model = RandomForestRegressor(**RF_PARAMS)
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


def _select_features(X, response, k_select, selection_decimals):
    importance = _tree_shap_importance(X, response, selection_decimals)
    return _top_k(importance, k_select)
