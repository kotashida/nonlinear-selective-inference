"""Public real-data API for featurewise SHAP selective inference."""

from __future__ import annotations

import os
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats

from .inference import (
    _chi_statistic,
    _run_ais,
    _run_conditional_mc,
    _spline_effect_basis,
)
from .selection import (
    SelectionResult,
    ShapSelector,
    _validate_selection_event,
    _validate_target_rule,
    make_selector,
    selection_event_definition,
    selection_event_holds,
    target_from_selected_set,
)


HYPOTHESIS = (
    "H0: the fixed-design mean response has zero projection onto the centered "
    "cubic B-spline basis of the target feature (a marginal nonlinear-association "
    "null; not a model coefficient, SHAP value, or conditional/causal effect)."
)
ASSUMPTIONS = (
    "X is fixed; errors are independent N(0, sigma^2); sigma is known and supplied; "
    "all preprocessing, tuning, k selection, selector rules, tie-breaking, and "
    "estimator randomness are fixed or included in the conditioned selection map."
)


def _validate_data(X, y, k_select, sigma):
    try:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("X and y must contain only numeric values.") from error
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if X.shape[0] != y.size:
        raise ValueError("X and y must have the same number of rows.")
    if X.shape[0] <= 4:
        raise ValueError("X and y must contain more than 4 observations.")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must not contain missing or infinite values.")
    if (
        isinstance(k_select, (bool, np.bool_))
        or not isinstance(k_select, (int, np.integer))
        or not 1 <= k_select <= X.shape[1]
    ):
        raise ValueError("k_select must satisfy 1 <= k_select <= X.shape[1].")
    if sigma is None:
        raise ValueError(
            "sigma is required. Unknown-variance selective inference is not "
            "implemented; do not estimate sigma silently from the selected data."
        )
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive scalar.")
    return X, y, int(k_select), float(sigma)


def _validate_selection_result(result, n_features, k_select) -> SelectionResult:
    if not isinstance(result, SelectionResult):
        raise TypeError("selector.select(...) must return a SelectionResult.")
    selected = np.asarray(result.selected_features)
    importance = np.asarray(result.importance, dtype=float)
    ranking = np.asarray(result.ranking)
    if importance.shape != (n_features,) or not np.all(np.isfinite(importance)):
        raise ValueError("selector importance must contain one finite value per feature.")
    if ranking.shape != (n_features,) or set(ranking.tolist()) != set(range(n_features)):
        raise ValueError("selector ranking must be a permutation of feature indices.")
    if selected.shape != (k_select,) or not np.array_equal(selected, ranking[:k_select]):
        raise ValueError("selected_features must equal the first k_select ranked features.")
    return SelectionResult(
        selected_features=selected.astype(int, copy=True),
        importance=importance.copy(),
        ranking=ranking.astype(int, copy=True),
    )


def adjust_p_values(p_values, method="none"):
    """Adjust finite featurewise p-values with none, Bonferroni, or Holm."""
    if method not in {"none", "bonferroni", "holm"}:
        raise ValueError("multiplicity must be 'none', 'bonferroni', or 'holm'.")
    values = np.asarray(p_values, dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError("p-values must lie in [0, 1] or be NaN, not infinite.")
    flat_values = values.ravel()
    flat_adjusted = np.full(flat_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(flat_values))
    if finite_indices.size == 0:
        return flat_adjusted.reshape(values.shape)
    finite = flat_values[finite_indices]
    if np.any((finite < 0.0) | (finite > 1.0)):
        raise ValueError("p-values must lie in [0, 1] or be NaN.")
    if method == "none":
        flat_adjusted[finite_indices] = finite
    elif method == "bonferroni":
        # The family size includes failed hypotheses; omitting them would make
        # the correction anticonservative.
        flat_adjusted[finite_indices] = np.minimum(1.0, finite * flat_values.size)
    else:
        # Holm depends on the ordering of every raw p-value. A failed estimate
        # has unknown order, so no familywise Holm result is reported.
        if finite_indices.size != flat_values.size:
            return flat_adjusted.reshape(values.shape)
        order = np.argsort(finite, kind="stable")
        ordered = finite[order]
        scaled = (finite.size - np.arange(finite.size)) * ordered
        ordered_adjusted = np.minimum(1.0, np.maximum.accumulate(scaled))
        restored = np.empty_like(ordered_adjusted)
        restored[order] = ordered_adjusted
        flat_adjusted[finite_indices] = restored
    return flat_adjusted.reshape(values.shape)


def selective_inference(
    X,
    y,
    *,
    k_select: int,
    sigma,
    estimator=None,
    selector=None,
    selection_event: str = "exact_set",
    target_rule: str = "all_selected",
    target_seed: int = 123,
    auxiliary_u: float | None = None,
    target_feature: int | None = None,
    multiplicity: str = "none",
    inference_method: str = "conditional_mc",
    selection_decimals: int = 10,
    ais_seed: int = 123,
    pilot_iters: int = 3,
    pilot_samples: int = 40,
    final_batch_size: int = 80,
    max_final_samples: int = 800,
    min_denominator_ess: float = 80.0,
    min_tail_ess: float = 15.0,
    stop_when_ess_met: bool = False,
    verify_selector_determinism: bool = True,
):
    """Run individual selective tests for SHAP-selected features in user data.

    The default event is equality of the unordered observed top-k set. Each
    selected feature still has its own response path and therefore its own test.
    Set ``target_rule='uniform_from_selected'`` to test one feature chosen by a
    fixed auxiliary draw; ``same_target`` then conditions on reproduction of
    the complete selector-to-target algorithm's output.
    ``sigma`` must be externally supplied and known under the stated model.
    The default ``conditional_mc`` method returns a finite-sample-valid,
    conservative conditional Monte Carlo rank p-value. ``inference_method='ais'``
    retains the self-normalized importance-sampling estimate for exploratory
    work, but that estimate is not an exact finite-sample p-value. Custom
    selectors are checked on repeated identical inputs unless
    ``verify_selector_determinism=False`` is requested.
    """
    X, y, k_select, sigma = _validate_data(X, y, k_select, sigma)
    _validate_selection_event(selection_event)
    _validate_target_rule(target_rule)
    if selection_event == "same_target" and target_rule != "uniform_from_selected":
        raise ValueError("same_target requires target_rule='uniform_from_selected'.")
    if multiplicity not in {"none", "bonferroni", "holm"}:
        raise ValueError("multiplicity must be 'none', 'bonferroni', or 'holm'.")
    if inference_method not in {"conditional_mc", "ais"}:
        raise ValueError("inference_method must be 'conditional_mc' or 'ais'.")
    if (
        selection_event == "feature_inclusion"
        and multiplicity == "holm"
        and target_rule == "all_selected"
    ):
        raise ValueError(
            "Holm adjustment is not implemented for feature_inclusion because "
            "the featurewise p-values condition on different inclusion events. "
            "Use multiplicity='bonferroni' or a common exact_set/exact_ranking event."
        )
    for name, value in {
        "pilot_iters": pilot_iters,
        "pilot_samples": pilot_samples,
        "final_batch_size": final_batch_size,
        "max_final_samples": max_final_samples,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
    if pilot_iters < 0 or pilot_samples < 1:
        raise ValueError("pilot_iters must be nonnegative and pilot_samples positive.")
    if final_batch_size < 1 or max_final_samples < 1:
        raise ValueError("Final AIS sample counts must be positive.")
    for name, value in {
        "min_denominator_ess": min_denominator_ess,
        "min_tail_ess": min_tail_ess,
    }.items():
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if (
        isinstance(ais_seed, (bool, np.bool_))
        or not isinstance(ais_seed, (int, np.integer))
        or ais_seed < 0
    ):
        raise ValueError("ais_seed must be a nonnegative integer.")
    if (
        isinstance(target_seed, (bool, np.bool_))
        or not isinstance(target_seed, (int, np.integer))
        or target_seed < 0
    ):
        raise ValueError("target_seed must be a nonnegative integer.")
    if not isinstance(stop_when_ess_met, (bool, np.bool_)):
        raise TypeError("stop_when_ess_met must be boolean.")
    if inference_method == "conditional_mc" and stop_when_ess_met:
        raise ValueError(
            "stop_when_ess_met is available only for exploratory AIS; "
            "conditional_mc always uses its fixed proposal budget."
        )
    if inference_method == "ais" and (
        min_denominator_ess > max_final_samples
        or min_tail_ess > max_final_samples
    ):
        raise ValueError(
            "AIS ESS targets cannot exceed max_final_samples; such targets are "
            "mathematically unattainable."
        )
    if not isinstance(verify_selector_determinism, (bool, np.bool_)):
        raise TypeError("verify_selector_determinism must be boolean.")

    resolved_selector = make_selector(
        estimator=estimator,
        selector=selector,
        selection_decimals=selection_decimals,
    )
    check_repeated_calls = verify_selector_determinism and not isinstance(
        resolved_selector, ShapSelector
    )

    def select_response(response):
        first = _validate_selection_result(
            resolved_selector.select(X, response, k_select), X.shape[1], k_select
        )
        if check_repeated_calls:
            for _ in range(2):
                repeated = _validate_selection_result(
                    resolved_selector.select(X, response, k_select),
                    X.shape[1],
                    k_select,
                )
                if not (
                    np.array_equal(first.selected_features, repeated.selected_features)
                    and np.array_equal(first.ranking, repeated.ranking)
                    and np.array_equal(first.importance, repeated.importance)
                ):
                    raise ValueError(
                        "Custom selector is not deterministic for repeated identical "
                        "inputs. Fix and condition on its randomness before selective "
                        "inference."
                    )
        return first

    observed = select_response(y)
    observed_selected = tuple(int(feature) for feature in observed.selected_features)
    realized_u = None
    observed_target = None
    if target_rule == "uniform_from_selected":
        if target_feature is not None and (
            isinstance(target_feature, (bool, np.bool_))
            or not isinstance(target_feature, (int, np.integer))
        ):
            raise TypeError("target_feature must be an integer or None.")
        if auxiliary_u is None:
            realized_u = float(np.random.default_rng(int(target_seed)).random())
        else:
            if not np.isscalar(auxiliary_u) or not np.isfinite(auxiliary_u):
                raise ValueError("auxiliary_u must be a finite scalar in [0, 1).")
            realized_u = float(auxiliary_u)
        computed_target = target_from_selected_set(observed_selected, realized_u)
        if target_feature is not None and int(target_feature) != computed_target:
            raise ValueError(
                "target_feature does not equal the target induced by observed "
                "selection and auxiliary_u."
            )
        observed_target = computed_target
        tested_features = (computed_target,)
    else:
        if auxiliary_u is not None or target_feature is not None:
            raise ValueError(
                "auxiliary_u and target_feature require "
                "target_rule='uniform_from_selected'."
            )
        tested_features = observed_selected
    rank_by_feature = np.empty(X.shape[1], dtype=int)
    rank_by_feature[observed.ranking] = np.arange(1, X.shape[1] + 1)
    importance_table = pd.DataFrame(
        {
            "feature": np.arange(X.shape[1]),
            "shap_importance": observed.importance,
            "shap_rank": rank_by_feature,
            "selected": np.isin(np.arange(X.shape[1]), observed.selected_features),
        }
    ).sort_values("shap_rank", ignore_index=True)
    importance_table["selection_event"] = selection_event

    if selection_event == "exact_set":
        message = (
            "Exact-set conditioning can make the event rare. Conditional Monte "
            "Carlo remains finite-sample valid but can be conservative when few "
            "draws reproduce selection; inspect selected_samples."
            if inference_method == "conditional_mc"
            else
            "Exact-set conditioning can make the event rare; exploratory AIS rows "
            "are estimates only and require ESS/Monte Carlo diagnostics."
        )
        warnings.warn(message, UserWarning, stacklevel=2)

    feature_seeds = np.random.SeedSequence(int(ais_seed)).spawn(len(tested_features))
    rows = []
    for test_index, feature in enumerate(tested_features):
        feature = int(feature)
        position = int(rank_by_feature[feature])
        basis = _spline_effect_basis(X[:, feature])
        test_rank = int(basis.shape[1])
        t_obs, projected = _chi_statistic(y, basis, sigma=sigma)
        projected_norm = float(np.linalg.norm(projected))
        zero_tolerance = np.sqrt(np.finfo(float).eps) * max(
            1.0, float(np.linalg.norm(y))
        )
        if projected_norm <= zero_tolerance:
            raise FloatingPointError(
                f"Feature {feature} has a numerically undefined effect direction."
            )
        orthogonal = y - projected
        direction = projected / projected_norm
        cache: dict[float, bool] = {}

        def is_selected(z):
            z = float(z)
            if z not in cache:
                candidate = orthogonal + sigma * direction * z
                candidate_result = select_response(candidate)
                cache[z] = selection_event_holds(
                    candidate_result.selected_features,
                    observed_selected,
                    feature,
                    selection_event,
                    auxiliary_u=realized_u,
                    ranking=candidate_result.ranking,
                    observed_ranking=observed.ranking,
                )
            return cache[z]

        if not is_selected(t_obs):
            raise RuntimeError(
                "z=T_obs did not reproduce the observed selection event for "
                f"feature {feature}."
            )
        feature_rng = np.random.default_rng(feature_seeds[test_index])
        if inference_method == "conditional_mc":
            p_value, diagnostics = _run_conditional_mc(
                t_obs,
                test_rank,
                is_selected,
                feature_rng,
                batch_size=final_batch_size,
                n_proposals=max_final_samples,
            )
        else:
            p_value, diagnostics = _run_ais(
                t_obs,
                test_rank,
                is_selected,
                feature_rng,
                pilot_iters=pilot_iters,
                pilot_samples=pilot_samples,
                final_batch_size=final_batch_size,
                max_final_samples=max_final_samples,
                min_denominator_ess=min_denominator_ess,
                min_tail_ess=min_tail_ess,
                stop_when_ess_met=bool(stop_when_ess_met),
            )
        rows.append(
            {
                "feature": feature,
                "selection_position": position,
                "shap_importance": float(observed.importance[feature]),
                "shap_rank": int(rank_by_feature[feature]),
                "test_rank": test_rank,
                "t_obs": float(t_obs),
                "unadjusted_p_value": float(stats.chi.sf(t_obs, df=test_rank)),
                "raw_selective_p_value": p_value,
                "selection_event": selection_event,
                "target_rule": target_rule,
                "auxiliary_u": realized_u,
                "selection_event_definition": selection_event_definition(
                    selection_event, feature
                ),
                **diagnostics,
            }
        )

    feature_results = pd.DataFrame(rows)
    # ``p_value`` is retained as a concise compatibility alias; the explicit
    # name prevents confusing individual raw values with adjusted values.
    feature_results["p_value"] = feature_results["raw_selective_p_value"]
    feature_results["multiplicity_method"] = multiplicity
    feature_results["variance_method"] = "known_user_supplied"
    feature_results["sigma"] = sigma
    feature_results["adjusted_selective_p_value"] = adjust_p_values(
        feature_results["raw_selective_p_value"].to_numpy(), multiplicity
    )
    selector_settings = dict(resolved_selector.get_settings())
    settings = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "k_select": k_select,
        "sigma": sigma,
        "variance_method": "known_user_supplied",
        "effect_confidence_intervals_implemented": False,
        "selection_event": selection_event,
        "target_rule": target_rule,
        "target_seed": int(target_seed),
        "auxiliary_u": realized_u,
        "observed_target_feature": observed_target,
        "multiplicity": multiplicity,
        "inference_method": inference_method,
        "ais_seed": int(ais_seed),
        "monte_carlo_seed": int(ais_seed),
        "pilot_iters": pilot_iters,
        "pilot_samples": pilot_samples,
        "final_batch_size": final_batch_size,
        "max_final_samples": max_final_samples,
        "min_denominator_ess": min_denominator_ess,
        "min_tail_ess": min_tail_ess,
        "stop_when_ess_met": bool(stop_when_ess_met),
        "verify_selector_determinism": bool(verify_selector_determinism),
        "determinism_check_repetitions": 3 if check_repeated_calls else 1,
        **selector_settings,
    }
    return {
        "observed_selected_features": np.asarray(observed_selected, dtype=int),
        "shap_importance": observed.importance.copy(),
        "shap_ranking": observed.ranking.copy(),
        "importance_table": importance_table,
        "feature_results": feature_results,
        "inference_diagnostics": feature_results.copy(),
        "ais_diagnostics": feature_results.copy(),
        "selection_event": selection_event,
        "target_rule": target_rule,
        "observed_target_feature": observed_target,
        "auxiliary_u": realized_u,
        "hypothesis": HYPOTHESIS,
        "assumptions": ASSUMPTIONS,
        "effect_confidence_intervals_implemented": False,
        "settings": settings,
    }
