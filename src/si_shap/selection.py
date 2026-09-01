"""Deterministic, interchangeable feature-selection methods."""

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
from sklearn.feature_selection import mutual_info_regression

from .inference import _spline_effect_basis


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
    """One complete fit -> score -> rank -> top-k selection result.

    ``importance`` is retained as the field name for backward compatibility;
    it contains the generic feature score produced by any selector.
    """

    selected_features: np.ndarray
    importance: np.ndarray
    ranking: np.ndarray

    @property
    def scores(self) -> np.ndarray:
        """Generic alias for the backward-compatible ``importance`` field."""
        return self.importance


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
            "Feature scores must be a finite one-dimensional array."
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
            "selection_method": "shap",
            "importance": "mean_absolute_tree_shap",
            "tie_breaking": "decreasing_importance_then_increasing_feature_index",
            "selection_decimals": self.selection_decimals,
            "estimator_class": (
                f"{type(self.estimator).__module__}.{type(self.estimator).__qualname__}"
            ),
            "estimator_params": self.estimator.get_params(deep=False),
        }


class LimeSelector:
    """Global feature selection from deterministic tabular LIME explanations.

    A regression model is refitted for every candidate response.  Continuous
    Gaussian neighborhoods are generated around deterministic, evenly spaced
    rows of the fixed design, and a distance-weighted ridge surrogate is fit in
    standardized coordinates at each row.  The feature score is the mean
    absolute local coefficient across those explanation rows.

    Neighborhoods and weighted-ridge operators depend only on ``X`` and are
    cached.  This is both faster and, crucially for selective inference, makes
    the LIME randomization identical for the observed and candidate responses.
    """

    def __init__(
        self,
        estimator=None,
        *,
        selection_decimals: int = 10,
        random_state: int = 42,
        num_samples: int = 64,
        num_explanations: int = 10,
        kernel_width: float | None = None,
        ridge_alpha: float = 1.0,
    ):
        if (
            isinstance(selection_decimals, (bool, np.bool_))
            or not isinstance(selection_decimals, (int, np.integer))
            or selection_decimals < 0
        ):
            raise ValueError("selection_decimals must be a nonnegative integer.")
        if (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer.")
        for name, value, minimum in (
            ("num_samples", num_samples, 2),
            ("num_explanations", num_explanations, 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if kernel_width is not None and (
            not np.isscalar(kernel_width)
            or not np.isfinite(kernel_width)
            or float(kernel_width) <= 0.0
        ):
            raise ValueError("kernel_width must be positive and finite or None.")
        if (
            not np.isscalar(ridge_alpha)
            or not np.isfinite(ridge_alpha)
            or float(ridge_alpha) < 0.0
        ):
            raise ValueError("ridge_alpha must be nonnegative and finite.")
        if estimator is None:
            estimator = RandomForestRegressor(**RF_PARAMS)
        _check_estimator_reproducibility(estimator)
        estimator_params = estimator.get_params(deep=False)
        if "n_jobs" in estimator_params:
            _validate_n_jobs(estimator_params["n_jobs"])
        self.estimator = clone(estimator)
        self.selection_decimals = int(selection_decimals)
        self.random_state = int(random_state)
        self.num_samples = int(num_samples)
        self.num_explanations = int(num_explanations)
        self.kernel_width = (
            None if kernel_width is None else float(kernel_width)
        )
        self.ridge_alpha = float(ridge_alpha)
        self._cached_X = None
        self._cached_neighborhoods = None
        self._cached_operators = None
        self._cached_anchor_indices = None

    def _prepare_neighborhoods(self, X: np.ndarray) -> None:
        if self._cached_X is X and self._cached_neighborhoods is not None:
            return
        means = X.mean(axis=0)
        raw_scales = X.std(axis=0)
        variable = raw_scales > np.finfo(float).eps
        scales = np.where(variable, raw_scales, 1.0)
        standardized = (X - means) / scales
        count = min(self.num_explanations, X.shape[0])
        anchor_indices = np.linspace(0, X.shape[0] - 1, count, dtype=int)
        rng = np.random.default_rng(self.random_state)
        width = (
            0.75 * np.sqrt(X.shape[1])
            if self.kernel_width is None
            else self.kernel_width
        )
        neighborhoods = []
        operators = []
        penalty = np.eye(X.shape[1] + 1) * self.ridge_alpha
        penalty[0, 0] = 0.0
        for anchor_index in anchor_indices:
            anchor = standardized[anchor_index]
            local = anchor + rng.normal(
                size=(self.num_samples, X.shape[1])
            )
            local[:, ~variable] = anchor[~variable]
            local[0] = anchor
            distances = np.linalg.norm(local - anchor, axis=1)
            weights = np.exp(-np.square(distances) / np.square(width))
            design = np.column_stack((np.ones(self.num_samples), local))
            gram = design.T @ (weights[:, None] * design) + penalty
            operator = np.linalg.pinv(gram) @ (design.T * weights)
            neighborhoods.append(means + local * scales)
            operators.append(operator[1:])
        self._cached_X = X
        self._cached_neighborhoods = np.stack(neighborhoods)
        self._cached_operators = np.stack(operators)
        self._cached_anchor_indices = anchor_indices

    def select(self, X, response, k_select) -> SelectionResult:
        X = np.asarray(X, dtype=float)
        response = np.asarray(response, dtype=float)
        if X.ndim != 2 or response.ndim != 1 or X.shape[0] != response.size:
            raise ValueError("X must be two-dimensional and match the response.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(response)):
            raise ValueError("X and response must contain only finite values.")
        if (
            isinstance(k_select, (bool, np.bool_))
            or not isinstance(k_select, (int, np.integer))
            or not 1 <= k_select <= X.shape[1]
        ):
            raise ValueError("k_select must satisfy 1 <= k_select <= X.shape[1].")
        self._prepare_neighborhoods(X)
        stable_response = np.round(response, decimals=self.selection_decimals)
        model = clone(self.estimator)
        model.fit(X, stable_response)
        flat = self._cached_neighborhoods.reshape(-1, X.shape[1])
        predictions = np.asarray(model.predict(flat), dtype=float).reshape(
            self._cached_neighborhoods.shape[:2]
        )
        coefficients = np.einsum(
            "epm,em->ep", self._cached_operators, predictions, optimize=True
        )
        scores = np.mean(np.abs(coefficients), axis=0)
        ranking = _top_k(scores, X.shape[1])
        return SelectionResult(ranking[:k_select].copy(), scores, ranking)

    def get_settings(self) -> Mapping[str, Any]:
        return {
            "selector": type(self).__name__,
            "selection_method": "lime",
            "importance": "mean_absolute_standardized_lime_coefficient",
            "aggregation": "mean_absolute_coefficient_across_explanation_rows",
            "neighborhood_distribution": "gaussian_centered_at_explanation_row",
            "anchor_selection": "evenly_spaced_design_rows",
            "tie_breaking": "decreasing_importance_then_increasing_feature_index",
            "selection_decimals": self.selection_decimals,
            "random_state": self.random_state,
            "num_samples": self.num_samples,
            "num_explanations": self.num_explanations,
            "kernel_width": self.kernel_width,
            "effective_kernel_width_rule": "0.75 * sqrt(n_features)",
            "ridge_alpha": self.ridge_alpha,
            "estimator_class": (
                f"{type(self.estimator).__module__}.{type(self.estimator).__qualname__}"
            ),
            "estimator_params": self.estimator.get_params(deep=False),
        }


class MutualInformationSelector:
    """Top-k screening by estimated feature-response mutual information."""

    def __init__(self, *, random_state: int = 42, n_neighbors: int = 3):
        if (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer.")
        if (
            isinstance(n_neighbors, (bool, np.bool_))
            or not isinstance(n_neighbors, (int, np.integer))
            or n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer.")
        self.random_state = int(random_state)
        self.n_neighbors = int(n_neighbors)

    def select(self, X, response, k_select) -> SelectionResult:
        X = np.asarray(X, dtype=float)
        response = np.asarray(response, dtype=float)
        if X.ndim != 2 or response.ndim != 1 or X.shape[0] != response.size:
            raise ValueError("X must be two-dimensional and match the response.")
        scores = mutual_info_regression(
            X,
            response,
            discrete_features=False,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        selected = _top_k(scores, k_select)
        ranking = _top_k(scores, X.shape[1])
        return SelectionResult(selected, scores, ranking)

    def get_settings(self) -> Mapping[str, Any]:
        return {
            "selector": type(self).__name__,
            "selection_method": "mutual_information",
            "importance": "mutual_information_regression",
            "tie_breaking": "decreasing_importance_then_increasing_feature_index",
            "random_state": self.random_state,
            "n_neighbors": self.n_neighbors,
        }


class MarginalCorrelationSelector:
    """Top-k marginal screening by absolute Pearson correlation."""

    def select(self, X, response, k_select) -> SelectionResult:
        X = np.asarray(X, dtype=float)
        response = np.asarray(response, dtype=float)
        if X.ndim != 2 or response.ndim != 1 or X.shape[0] != response.size:
            raise ValueError("X must be two-dimensional and match the response.")
        centered_X = X - X.mean(axis=0, keepdims=True)
        centered_response = response - response.mean()
        denominator = np.linalg.norm(centered_X, axis=0) * np.linalg.norm(
            centered_response
        )
        numerator = np.abs(centered_X.T @ centered_response)
        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros(X.shape[1], dtype=float),
            where=denominator > 0.0,
        )
        selected = _top_k(scores, k_select)
        ranking = _top_k(scores, X.shape[1])
        return SelectionResult(selected, scores, ranking)

    def get_settings(self) -> Mapping[str, Any]:
        return {
            "selector": type(self).__name__,
            "selection_method": "marginal_screening",
            "importance": "absolute_pearson_correlation",
            "tie_breaking": "decreasing_importance_then_increasing_feature_index",
        }


class SplineScreeningSelector:
    """Top-k nonlinear screening by cubic B-spline projection norm.

    The score for feature ``j`` is ``||Q_j.T @ response||_2``, where ``Q_j``
    is the orthonormal centered spline basis used by the downstream chi test.
    Bases are cached for the most recently seen design because conditional
    Monte Carlo repeatedly evaluates the selector with one fixed ``X``.
    """

    def __init__(self):
        self._cached_X = None
        self._cached_bases = None

    def _bases(self, X: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if self._cached_X is X and self._cached_bases is not None:
            return self._cached_bases
        bases = []
        for feature in range(X.shape[1]):
            column = X[:, feature]
            if np.ptp(column) <= np.finfo(float).eps:
                bases.append(None)
            else:
                bases.append(_spline_effect_basis(column))
        self._cached_X = X
        self._cached_bases = tuple(bases)
        return self._cached_bases

    def select(self, X, response, k_select) -> SelectionResult:
        X = np.asarray(X, dtype=float)
        response = np.asarray(response, dtype=float)
        if X.ndim != 2 or response.ndim != 1 or X.shape[0] != response.size:
            raise ValueError("X must be two-dimensional and match the response.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(response)):
            raise ValueError("X and response must contain only finite values.")
        scores = np.array(
            [
                0.0 if basis is None else np.linalg.norm(basis.T @ response)
                for basis in self._bases(X)
            ],
            dtype=float,
        )
        selected = _top_k(scores, k_select)
        ranking = _top_k(scores, X.shape[1])
        return SelectionResult(selected, scores, ranking)

    def get_settings(self) -> Mapping[str, Any]:
        return {
            "selector": type(self).__name__,
            "selection_method": "spline_screening",
            "importance": "centered_cubic_bspline_projection_norm",
            "spline_df": 3,
            "spline_degree": 3,
            "tie_breaking": "decreasing_importance_then_increasing_feature_index",
        }


_BUILTIN_SELECTOR_TYPES = (
    ShapSelector,
    LimeSelector,
    MutualInformationSelector,
    MarginalCorrelationSelector,
    SplineScreeningSelector,
)


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
    *,
    selection_method: str | None = None,
    estimator=None,
    selector=None,
    selection_decimals=10,
    rf_params=None,
) -> Selector:
    """Resolve a built-in or custom selector.

    The built-in methods are ``shap`` (the backward-compatible default),
    ``lime``, ``mutual_information``, ``marginal_screening``, and
    ``spline_screening``.
    """
    supplied = sum(value is not None for value in (estimator, selector, rf_params))
    if supplied > 1:
        raise ValueError("Pass only one of estimator, selector, or rf_params.")
    if selector is not None:
        if selection_method is not None:
            raise ValueError("Pass either selection_method or selector, not both.")
        return _validate_selector(selector)
    if selection_method is not None and not isinstance(selection_method, str):
        raise TypeError("selection_method must be a string or None.")
    method = "shap" if selection_method is None else selection_method
    if method not in {
        "shap",
        "lime",
        "mutual_information",
        "marginal_screening",
        "spline_screening",
    }:
        raise ValueError(
            "selection_method must be 'shap', 'lime', 'mutual_information', "
            "'marginal_screening', or 'spline_screening'."
        )
    if method not in {"shap", "lime"} and (
        estimator is not None or rf_params is not None
    ):
        raise ValueError(
            "estimator and rf_params are available only for SHAP or LIME selection."
        )
    if method == "mutual_information":
        return MutualInformationSelector()
    if method == "marginal_screening":
        return MarginalCorrelationSelector()
    if method == "spline_screening":
        return SplineScreeningSelector()
    if rf_params is not None:
        estimator = RandomForestRegressor(**_resolve_rf_params(rf_params))
    if method == "lime":
        return LimeSelector(estimator, selection_decimals=selection_decimals)
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
