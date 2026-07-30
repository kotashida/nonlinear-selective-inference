"""Deterministic, configurable SHAP-based feature selection."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor


MAX_CPU_CORES = 32
RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": 5,
    "random_state": 42,
    "n_jobs": MAX_CPU_CORES,
}
SelectionEvent = Literal[
    "exact_set", "feature_inclusion", "same_target", "exact_ranking"
]
SELECTION_EVENTS = (
    "exact_set", "feature_inclusion", "same_target", "exact_ranking"
)
TargetRule = Literal["all_selected", "uniform_from_selected"]
TARGET_RULES = ("all_selected", "uniform_from_selected")


@dataclass(frozen=True)
class SelectionResult:
    """One complete fit -> importance -> rank -> top-k selection result."""

    selected_features: np.ndarray
    importance: np.ndarray
    ranking: np.ndarray


@runtime_checkable
class Selector(Protocol):
    """Interface for a refittable, deterministic feature-selection pipeline."""

    def select(self, X: np.ndarray, response: np.ndarray, k_select: int) -> SelectionResult:
        """Refit on ``response`` and return importance, ranking, and top-k features."""

    def get_settings(self) -> Mapping[str, Any]:
        """Return reproducibility metadata for result tables and configurations."""


def _resolve_rf_params(rf_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return forest parameters with user overrides applied to the defaults."""
    if rf_params is not None and not isinstance(rf_params, Mapping):
        raise TypeError("rf_params must be a mapping or None.")
    resolved = {**RF_PARAMS, **({} if rf_params is None else rf_params)}
    _validate_n_jobs(resolved.get("n_jobs"))
    return resolved


def _validate_n_jobs(n_jobs) -> None:
    """Allow serial execution or at most the lab's allocated CPU count."""
    if n_jobs is None:
        return
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or not 1 <= n_jobs <= MAX_CPU_CORES
    ):
        raise ValueError(f"n_jobs must be an integer from 1 to {MAX_CPU_CORES}.")


def _top_k(scores, k):
    """Select by decreasing score, breaking ties by increasing feature index."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError(
            "SHAP importance scores must be a finite one-dimensional array."
        )
    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or not 1 <= k <= scores.size
    ):
        raise ValueError("k must satisfy 1 <= k <= the number of scores.")
    return np.lexsort((np.arange(scores.size), -scores))[:k]


def _validate_selection_event(selection_event: str) -> str:
    if selection_event not in SELECTION_EVENTS:
        choices = ", ".join(repr(value) for value in SELECTION_EVENTS)
        raise ValueError(f"selection_event must be one of {choices}.")
    return selection_event


def _validate_target_rule(target_rule: str) -> str:
    if target_rule not in TARGET_RULES:
        choices = ", ".join(repr(value) for value in TARGET_RULES)
        raise ValueError(f"target_rule must be one of {choices}.")
    return target_rule


def target_from_selected_set(selected_features, auxiliary_u: float) -> int:
    """Choose uniformly from a set using fixed auxiliary randomness.

    Sorting makes the map independent of the selector's ranking representation.
    Reusing the same ``auxiliary_u`` conditions on the realized randomization.
    """
    selected = tuple(sorted({int(feature) for feature in selected_features}))
    if not selected:
        raise ValueError("selected_features must be nonempty.")
    if not np.isscalar(auxiliary_u) or not np.isfinite(auxiliary_u):
        raise ValueError("auxiliary_u must be a finite scalar in [0, 1).")
    auxiliary_u = float(auxiliary_u)
    if not 0.0 <= auxiliary_u < 1.0:
        raise ValueError("auxiliary_u must lie in [0, 1).")
    return selected[int(np.floor(auxiliary_u * len(selected)))]


def selection_event_definition(selection_event: str, target_feature: int) -> str:
    """Return the mathematical event description recorded in outputs."""
    _validate_selection_event(selection_event)
    if selection_event == "exact_set":
        return "frozenset(selected(z)) == frozenset(observed_selected)"
    if selection_event == "feature_inclusion":
        return f"{int(target_feature)} in selected(z)"
    if selection_event == "same_target":
        return f"target(sorted(selected(z)), fixed_u) == {int(target_feature)}"
    return "tuple(ranking(z)) == tuple(observed_ranking)"


def selection_event_holds(
    selected,
    observed_selected,
    target_feature: int,
    selection_event: str = "exact_set",
    auxiliary_u: float | None = None,
    ranking=None,
    observed_ranking=None,
) -> bool:
    """Evaluate a selection event using one shared definition everywhere."""
    _validate_selection_event(selection_event)
    selected_tuple = tuple(int(feature) for feature in selected)
    observed_tuple = tuple(int(feature) for feature in observed_selected)
    if selection_event == "exact_set":
        return frozenset(selected_tuple) == frozenset(observed_tuple)
    if selection_event == "feature_inclusion":
        return int(target_feature) in selected_tuple
    if selection_event == "same_target":
        if auxiliary_u is None:
            raise ValueError("same_target requires auxiliary_u.")
        return target_from_selected_set(selected_tuple, auxiliary_u) == int(
            target_feature
        )
    if ranking is None or observed_ranking is None:
        raise ValueError(
            "exact_ranking requires the complete candidate and observed rankings."
        )
    ranking_tuple = tuple(int(feature) for feature in ranking)
    observed_ranking_tuple = tuple(int(feature) for feature in observed_ranking)
    return ranking_tuple == observed_ranking_tuple


def _check_estimator_reproducibility(estimator) -> None:
    """Reject exposed stochastic controls that have not been fixed."""
    try:
        cloned = clone(estimator)
    except Exception as error:  # sklearn provides estimator-specific details
        raise TypeError("estimator must be scikit-learn cloneable.") from error
    if not hasattr(cloned, "fit"):
        raise TypeError("estimator must provide a fit method.")
    params = cloned.get_params(deep=True)
    unset_states = [
        name for name, value in params.items()
        if name.endswith("random_state") and value is None
    ]
    if unset_states:
        joined = ", ".join(sorted(unset_states))
        raise ValueError(
            "Every exposed random_state must be fixed for selective inference; "
            f"unset parameter(s): {joined}."
        )


def _tree_shap_importance_for_estimator(
    X, response, selection_decimals, estimator
):
    """Clone, refit, and compute mean absolute Tree SHAP importance."""
    stable_response = np.round(response, decimals=selection_decimals)
    model = clone(estimator)
    model.fit(X, stable_response)
    try:
        values = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        ).shap_values(X)
    except Exception as error:
        raise TypeError(
            "The estimator is not supported by SHAP TreeExplainer. Officially "
            "supported: sklearn RandomForestRegressor."
        ) from error
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


class ShapSelector:
    """Single-output regression selector based on mean absolute Tree SHAP."""

    def __init__(self, estimator=None, *, selection_decimals: int = 10):
        if (
            isinstance(selection_decimals, (bool, np.bool_))
            or not isinstance(selection_decimals, (int, np.integer))
            or selection_decimals < 0
        ):
            raise ValueError("selection_decimals must be a nonnegative integer.")
        if estimator is None:
            estimator = RandomForestRegressor(**RF_PARAMS)
        _check_estimator_reproducibility(estimator)
        estimator_params = estimator.get_params(deep=False)
        if "n_jobs" in estimator_params:
            _validate_n_jobs(estimator_params["n_jobs"])
        self.estimator = clone(estimator)
        self.selection_decimals = selection_decimals

    def select(self, X, response, k_select) -> SelectionResult:
        response = np.asarray(response)
        if response.ndim != 1:
            raise ValueError("ShapSelector supports only one-dimensional responses.")
        if (
            isinstance(k_select, (bool, np.bool_))
            or not isinstance(k_select, (int, np.integer))
            or not 1 <= k_select <= np.asarray(X).shape[1]
        ):
            raise ValueError("k_select must satisfy 1 <= k_select <= X.shape[1].")
        importance = _tree_shap_importance_for_estimator(
            X, response, self.selection_decimals, self.estimator
        )
        ranking = _top_k(importance, importance.size)
        return SelectionResult(
            selected_features=ranking[:k_select].copy(),
            importance=importance,
            ranking=ranking,
        )

    def get_settings(self) -> Mapping[str, Any]:
        return {
            "selector": type(self).__name__,
            "importance": "mean_absolute_tree_shap",
            "tie_breaking": "decreasing_importance_then_increasing_feature_index",
            "selection_decimals": self.selection_decimals,
            "estimator_class": (
                f"{type(self.estimator).__module__}.{type(self.estimator).__qualname__}"
            ),
            "estimator_params": self.estimator.get_params(deep=False),
        }


class _MemoizedSelector:
    """Cache deterministic selector results for one fixed design matrix."""

    def __init__(self, selector):
        self.selector = selector
        self.cache = {}

    def select(self, X, response, k_select):
        response_array = np.ascontiguousarray(response, dtype=float)
        key = (int(k_select), response_array.tobytes())
        if key not in self.cache:
            self.cache[key] = self.selector.select(X, response_array, k_select)
        return self.cache[key]

    def get_settings(self):
        return {
            **dict(self.selector.get_settings()),
            "memoized_within_iteration": True,
        }


def _validate_selector(selector) -> Selector:
    if not isinstance(selector, Selector):
        raise TypeError("selector must provide select(...) and get_settings() methods.")
    return selector


def make_selector(
    *, estimator=None, selector=None, selection_decimals=10, rf_params=None
) -> Selector:
    """Resolve public selector options while preserving the legacy RF shortcut."""
    supplied = sum(value is not None for value in (estimator, selector, rf_params))
    if supplied > 1:
        raise ValueError("Pass only one of estimator, selector, or rf_params.")
    if selector is not None:
        return _validate_selector(selector)
    if rf_params is not None:
        estimator = RandomForestRegressor(**_resolve_rf_params(rf_params))
    return ShapSelector(estimator, selection_decimals=selection_decimals)


def _tree_shap_importance(X, response, selection_decimals, rf_params=None):
    """Backward-compatible Random Forest Tree SHAP importance helper."""
    estimator = RandomForestRegressor(**_resolve_rf_params(rf_params))
    return _tree_shap_importance_for_estimator(
        X, response, selection_decimals, estimator
    )


def _select_features(X, response, k_select, selection_decimals, rf_params=None):
    importance = _tree_shap_importance(
        X, response, selection_decimals, rf_params=rf_params
    )
    return _top_k(importance, k_select)
