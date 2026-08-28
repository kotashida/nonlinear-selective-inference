"""Paired global-null calibration for alternative selection events."""

from __future__ import annotations

import itertools
import os
import warnings
from collections.abc import Sequence

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from .api import _validate_selection_result, selective_inference
from .selection import (
    _BUILTIN_SELECTOR_TYPES,
    _MemoizedSelector,
    _resolve_rf_params,
    _validate_selection_event,
    make_selector,
    target_from_selected_set,
)
from .simulation import (
    _generate_gaussian_design,
    _validate_feature_correlation,
    _validate_inputs,
    _validate_positive_finite,
    _validate_seed,
)


DEFAULT_NULL_EVENTS = ("feature_inclusion", "exact_set", "same_target")


def _validate_null_inputs(
    n_iters,
    n_samples,
    n_features,
    k_select,
    sigma,
    selection_events,
    alpha_levels,
):
    _validate_inputs(n_iters, n_samples, n_features, k_select, 0.05)
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive scalar.")
    if isinstance(selection_events, str):
        raise TypeError("selection_events must be a sequence, not one string.")
    events = tuple(selection_events)
    if not events or len(set(events)) != len(events):
        raise ValueError("selection_events must contain at least one unique event.")
    for event in events:
        _validate_selection_event(event)

    if np.isscalar(alpha_levels):
        raise TypeError("alpha_levels must be a sequence, not one scalar.")
    levels = tuple(float(alpha) for alpha in alpha_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("alpha_levels must contain unique values.")
    if any(not np.isfinite(alpha) or not 0.0 < alpha < 1.0 for alpha in levels):
        raise ValueError("Every alpha level must lie strictly between 0 and 1.")
    return events, tuple(sorted(levels))


def _alpha_label(alpha: float) -> str:
    return format(float(alpha), ".12g")


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        return np.nan, np.nan
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(0.025, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(0.975, successes + 1, trials - successes))
    )
    return lower, upper


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _calibration_summary(
    event_results: pd.DataFrame,
    *,
    n_iters: int,
    alpha_levels: Sequence[float],
) -> dict[str, float | int | str]:
    if len(event_results) != n_iters:
        raise ValueError(
            "event_results must contain exactly one row per requested iteration."
        )
    if event_results.empty or event_results["selection_event"].nunique() != 1:
        raise ValueError("event_results must contain exactly one selection event.")
    event = str(event_results["selection_event"].iloc[0])
    p_values = event_results["p_value"].to_numpy(dtype=float)
    finite_values = p_values[np.isfinite(p_values)]
    if np.any(np.isinf(p_values)) or np.any(
        (finite_values < 0.0) | (finite_values > 1.0)
    ):
        raise ValueError("p-values must lie in [0, 1] or be NaN.")
    finite = np.isfinite(p_values)
    converged = p_values[finite]
    failures = int(np.sum(~finite))
    strict_available = failures == 0
    summary: dict[str, float | int | str] = {
        "selection_event": event,
        "calibration_target": (
            "super_uniform_finite_sample"
            if "finite_sample_valid" in event_results
            and bool(event_results["finite_sample_valid"].fillna(False).all())
            else "exploratory_estimate"
        ),
        "n_iterations": n_iters,
        "n_converged": int(converged.size),
        "n_failed": failures,
        "failure_rate": failures / n_iters,
        "mean_p_value": (
            float(np.mean(converged)) if converged.size and strict_available else np.nan
        ),
        "converged_mean_p_value": (
            float(np.mean(converged)) if converged.size else np.nan
        ),
        "median_p_value": (
            float(np.median(converged)) if converged.size and strict_available else np.nan
        ),
        "converged_median_p_value": (
            float(np.median(converged)) if converged.size else np.nan
        ),
        "variance_p_value": (
            float(np.var(converged, ddof=1))
            if converged.size > 1 and strict_available else np.nan
        ),
        "converged_variance_p_value": (
            float(np.var(converged, ddof=1)) if converged.size > 1 else np.nan
        ),
        "uniform_ks_statistic": (
            float(stats.kstest(converged, "uniform").statistic)
            if converged.size and strict_available
            else np.nan
        ),
        "converged_uniform_ks_statistic": (
            float(stats.kstest(converged, "uniform").statistic)
            if converged.size
            else np.nan
        ),
        "mean_denominator_ess": _finite_mean(event_results["denominator_ess"]),
        "mean_tail_ess": _finite_mean(event_results["tail_ess"]),
        "mean_mc_se": _finite_mean(event_results["mc_se"]),
    }
    for probability in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        summary[f"p_value_quantile_{_alpha_label(probability)}"] = (
            float(np.quantile(converged, probability))
            if converged.size
            else np.nan
        )
    for alpha in alpha_levels:
        label = _alpha_label(alpha)
        rejections = int(np.sum(converged < alpha))
        converged_rate = rejections / converged.size if converged.size else np.nan
        ci_lower, ci_upper = _clopper_pearson(rejections, int(converged.size))
        summary.update(
            {
                f"rejection_rate_{label}": (
                    converged_rate if strict_available else np.nan
                ),
                f"converged_rejection_rate_{label}": converged_rate,
                f"rejection_ci_95_lower_{label}": (
                    ci_lower if strict_available else np.nan
                ),
                f"rejection_ci_95_upper_{label}": (
                    ci_upper if strict_available else np.nan
                ),
                f"rejection_rate_lower_bound_{label}": rejections / n_iters,
                f"rejection_rate_upper_bound_{label}": (
                    rejections + failures
                )
                / n_iters,
            }
        )
    return summary


def _paired_rejection_comparisons(
    p_value_results: pd.DataFrame,
    selection_events: Sequence[str],
    alpha_levels: Sequence[float],
) -> pd.DataFrame:
    rows = []
    for first_event, second_event in itertools.combinations(selection_events, 2):
        paired = p_value_results[
            p_value_results["selection_event"].isin((first_event, second_event))
        ].pivot(index="iteration", columns="selection_event", values="p_value")
        for alpha in alpha_levels:
            complete = np.isfinite(paired[first_event]) & np.isfinite(
                paired[second_event]
            )
            first_rejected = paired.loc[complete, first_event] < alpha
            second_rejected = paired.loc[complete, second_event] < alpha
            differences = second_rejected.astype(float) - first_rejected.astype(float)
            difference = float(differences.mean()) if len(differences) else np.nan
            standard_error = (
                float(differences.std(ddof=1) / np.sqrt(len(differences)))
                if len(differences) > 1
                else np.nan
            )
            critical = (
                float(stats.t.ppf(0.975, df=len(differences) - 1))
                if len(differences) > 1
                else np.nan
            )
            all_complete = bool(np.all(complete))
            rows.append(
                {
                    "first_event": first_event,
                    "second_event": second_event,
                    "alpha": float(alpha),
                    "rejection_rate_difference": (
                        difference if all_complete else np.nan
                    ),
                    "converged_rejection_rate_difference": difference,
                    "paired_simulation_se": standard_error,
                    "ci_95_lower": (
                        max(-1.0, difference - critical * standard_error)
                        if np.isfinite(critical) and all_complete else np.nan
                    ),
                    "ci_95_upper": (
                        min(1.0, difference + critical * standard_error)
                        if np.isfinite(critical) and all_complete else np.nan
                    ),
                    "converged_ci_95_lower": (
                        max(-1.0, difference - critical * standard_error)
                        if np.isfinite(critical) else np.nan
                    ),
                    "converged_ci_95_upper": (
                        min(1.0, difference + critical * standard_error)
                        if np.isfinite(critical) else np.nan
                    ),
                    "n_complete_pairs": int(np.sum(complete)),
                    "first_only_rejections": int(
                        np.sum(first_rejected & ~second_rejected)
                    ),
                    "second_only_rejections": int(
                        np.sum(~first_rejected & second_rejected)
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_selection_event_null_calibration(
    n_iters: int,
    n_samples: int,
    n_features: int,
    k_select: int,
    *,
    sigma: float = 1.0,
    feature_correlation: float = 0.0,
    selection_events: Sequence[str] = DEFAULT_NULL_EVENTS,
    alpha_levels: Sequence[float] = (0.01, 0.05, 0.10),
    seed: int = 123,
    iteration_start: int = 0,
    design_seed: int | None = None,
    fixed_auxiliary_u: float | None = None,
    selection_decimals: int = 10,
    pilot_iters: int = 3,
    pilot_samples: int = 40,
    final_batch_size: int = 80,
    max_final_samples: int = 800,
    min_denominator_ess: float = 80.0,
    min_tail_ess: float = 15.0,
    selection_method: str | None = None,
    rf_params=None,
    estimator=None,
    selector=None,
    inference_method: str = "conditional_mc",
    stop_when_ess_met: bool = False,
):
    """Evaluate paired single-target selective p-values under the global null.

    By default, a fresh auxiliary target-selection draw is generated in every
    iteration.  ``fixed_auxiliary_u`` instead reuses one value across all null
    responses, which evaluates calibration conditional on that realized target
    randomization and matches the fixed-seed ``same_target`` workflow.
    """
    events, levels = _validate_null_inputs(
        n_iters,
        n_samples,
        n_features,
        k_select,
        sigma,
        selection_events,
        alpha_levels,
    )
    feature_correlation = _validate_feature_correlation(feature_correlation)
    for name, value in {
        "seed": seed,
        "iteration_start": iteration_start,
        "pilot_iters": pilot_iters,
        "pilot_samples": pilot_samples,
        "final_batch_size": final_batch_size,
        "max_final_samples": max_final_samples,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
    if iteration_start < 0:
        raise ValueError("iteration_start must be nonnegative.")
    if design_seed is not None and (
        isinstance(design_seed, (bool, np.bool_))
        or not isinstance(design_seed, (int, np.integer))
    ):
        raise TypeError("design_seed must be an integer or None.")
    if fixed_auxiliary_u is not None:
        if (
            not np.isscalar(fixed_auxiliary_u)
            or not np.isfinite(fixed_auxiliary_u)
            or not 0.0 <= float(fixed_auxiliary_u) < 1.0
        ):
            raise ValueError("fixed_auxiliary_u must be a finite scalar in [0, 1).")
        fixed_auxiliary_u = float(fixed_auxiliary_u)
    if pilot_iters < 0 or pilot_samples < 1:
        raise ValueError("pilot_iters must be nonnegative and pilot_samples positive.")
    if final_batch_size < 1 or max_final_samples < 1:
        raise ValueError("Final AIS sample counts must be positive.")
    _validate_positive_finite("min_denominator_ess", min_denominator_ess)
    _validate_positive_finite("min_tail_ess", min_tail_ess)
    if not isinstance(stop_when_ess_met, (bool, np.bool_)):
        raise TypeError("stop_when_ess_met must be boolean.")
    seed = _validate_seed("seed", seed)
    if design_seed is not None:
        design_seed = _validate_seed("design_seed", design_seed)
    if inference_method not in {"conditional_mc", "ais"}:
        raise ValueError("inference_method must be 'conditional_mc' or 'ais'.")
    if inference_method == "conditional_mc" and stop_when_ess_met:
        raise ValueError("stop_when_ess_met requires inference_method='ais'.")
    if inference_method == "ais" and (
        min_denominator_ess > max_final_samples
        or min_tail_ess > max_final_samples
    ):
        raise ValueError(
            "AIS ESS targets cannot exceed max_final_samples; such targets are "
            "mathematically unattainable."
        )
    if sum(value is not None for value in (rf_params, estimator, selector)) > 1:
        raise ValueError("Pass only one of rf_params, estimator, or selector.")
    if (
        selection_method is not None
        and selection_method != "shap"
        and rf_params is not None
    ):
        raise ValueError("rf_params are available only for SHAP selection.")
    if k_select == 1:
        warnings.warn(
            "For k_select=1, feature_inclusion, exact_set, and same_target "
            "define the same selection event.",
            UserWarning,
            stacklevel=2,
        )

    root_design_seed, iteration_parent = np.random.SeedSequence(int(seed)).spawn(2)
    resolved_design_seed = (
        int(root_design_seed.generate_state(1, dtype=np.uint32)[0])
        if design_seed is None
        else int(design_seed)
    )
    X = _generate_gaussian_design(
        np.random.default_rng(resolved_design_seed),
        n_samples,
        n_features,
        feature_correlation,
    )
    resolved_rf_params = (
        _resolve_rf_params(rf_params)
        if (selection_method is None or selection_method == "shap")
        and estimator is None
        and selector is None
        else None
    )
    resolved_selector = make_selector(
        selection_method=selection_method,
        estimator=estimator,
        selector=selector,
        selection_decimals=selection_decimals,
        rf_params=resolved_rf_params,
    )

    records = []
    iteration_seeds = iteration_parent.spawn(iteration_start + n_iters)[
        iteration_start:
    ]
    for iteration, iteration_seed in enumerate(
        tqdm(iteration_seeds, desc="Calibrating selection events"),
        start=iteration_start + 1,
    ):
        response_seed, target_seed, ais_seed = iteration_seed.spawn(3)
        response = (
            np.random.default_rng(response_seed).standard_normal(n_samples) * sigma
        )
        shared_target_seed = int(
            target_seed.generate_state(1, dtype=np.uint32)[0]
        )
        shared_ais_seed = int(ais_seed.generate_state(1, dtype=np.uint32)[0])
        iteration_selector = (
            _MemoizedSelector(resolved_selector)
            if isinstance(resolved_selector, _BUILTIN_SELECTOR_TYPES)
            else resolved_selector
        )
        observed = _validate_selection_result(
            iteration_selector.select(X, response, k_select),
            n_features,
            k_select,
        )
        observed_selected = tuple(
            int(feature) for feature in observed.selected_features
        )
        auxiliary_u = (
            float(np.random.default_rng(shared_target_seed).random())
            if fixed_auxiliary_u is None
            else fixed_auxiliary_u
        )
        target_feature = target_from_selected_set(observed_selected, auxiliary_u)

        for event in events:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Exact-set conditioning can make the event rare"
                )
                result = selective_inference(
                    X,
                    response,
                    k_select=k_select,
                    sigma=sigma,
                    selector=iteration_selector,
                    selection_event=event,
                    target_rule="uniform_from_selected",
                    target_seed=shared_target_seed,
                    auxiliary_u=auxiliary_u,
                    target_feature=target_feature,
                    multiplicity="none",
                    inference_method=inference_method,
                    ais_seed=shared_ais_seed,
                    pilot_iters=pilot_iters,
                    pilot_samples=pilot_samples,
                    final_batch_size=final_batch_size,
                    max_final_samples=max_final_samples,
                    min_denominator_ess=min_denominator_ess,
                    min_tail_ess=min_tail_ess,
                    stop_when_ess_met=stop_when_ess_met,
                )
            returned_selected = tuple(
                int(feature) for feature in result["observed_selected_features"]
            )
            frame = result["feature_results"]
            if (
                returned_selected != observed_selected
                or result["observed_target_feature"] != target_feature
                or result["auxiliary_u"] != auxiliary_u
                or len(frame) != 1
                or int(frame.iloc[0]["feature"]) != target_feature
            ):
                raise RuntimeError(
                    "Selection events did not use the same observed set, target, "
                    "and auxiliary randomness."
                )
            row = frame.iloc[0].to_dict()
            p_value = float(row["raw_selective_p_value"])
            records.append(
                {
                    "iteration": iteration,
                    "selection_event": event,
                    "target_feature": target_feature,
                    "observed_selected_features": observed_selected,
                    "auxiliary_u": auxiliary_u,
                    "target_seed": shared_target_seed,
                    "ais_seed": shared_ais_seed,
                    "p_value": p_value,
                    "failed": not np.isfinite(p_value),
                    **row,
                }
            )

    p_value_results = pd.DataFrame(records)
    summary = pd.DataFrame(
        [
            _calibration_summary(
                p_value_results[p_value_results["selection_event"] == event],
                n_iters=n_iters,
                alpha_levels=levels,
            )
            for event in events
        ]
    )
    comparisons = _paired_rejection_comparisons(
        p_value_results, events, levels
    )
    return {
        "calibration_summary": summary,
        "summary": summary,
        "p_value_results": p_value_results,
        "paired_rejection_comparisons": comparisons,
        "fixed_design": X.copy(),
        "settings": {
            "n_iters": n_iters,
            "iteration_start": iteration_start,
            "n_samples": n_samples,
            "n_features": n_features,
            "k_select": k_select,
            "sigma": float(sigma),
            "feature_correlation": feature_correlation,
            "variance_method": "known_simulation_sigma",
            "selection_events": events,
            "target_rule": "uniform_from_selected",
            "auxiliary_randomization_mode": (
                "redrawn_each_iteration"
                if fixed_auxiliary_u is None
                else "fixed_across_iterations"
            ),
            "fixed_auxiliary_u": fixed_auxiliary_u,
            "multiplicity": "none",
            "inference_method": inference_method,
            "alpha_levels": levels,
            "seed": int(seed),
            "design_seed": resolved_design_seed,
            "fixed_design": True,
            "selection_decimals": selection_decimals,
            "selection_method": dict(resolved_selector.get_settings()).get(
                "selection_method", "custom"
            ),
            "rf_params": (
                None if resolved_rf_params is None else resolved_rf_params.copy()
            ),
            "selector_settings": dict(resolved_selector.get_settings()),
            "pilot_iters": pilot_iters,
            "pilot_samples": pilot_samples,
            "final_batch_size": final_batch_size,
            "max_final_samples": max_final_samples,
            "min_denominator_ess": min_denominator_ess,
            "min_tail_ess": min_tail_ess,
            "stop_when_ess_met": bool(stop_when_ess_met),
        },
    }
