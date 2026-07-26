"""Paired power comparisons for alternative SHAP selection events."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .api import selective_inference
from .selection import (
    _resolve_rf_params,
    _validate_selection_event,
    make_selector,
)
from .simulation import SIMULATION_SIGMA, _validate_inputs


DEFAULT_SELECTION_EVENTS = ("exact_set", "feature_inclusion")


def _validate_power_inputs(
    n_features: int,
    signal_features: Sequence[int],
    signal_strength: float,
    selection_events: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Validate and normalize inputs specific to the power experiment."""
    try:
        normalized_features = tuple(int(feature) for feature in signal_features)
    except (TypeError, ValueError) as error:
        raise TypeError("signal_features must be a sequence of integers.") from error
    if not normalized_features:
        raise ValueError("signal_features must contain at least one feature.")
    if any(
        isinstance(feature, (bool, np.bool_))
        or not isinstance(feature, (int, np.integer))
        for feature in signal_features
    ):
        raise TypeError("signal_features must be a sequence of integers.")
    if len(set(normalized_features)) != len(normalized_features):
        raise ValueError("signal_features must not contain duplicates.")
    if any(not 0 <= feature < n_features for feature in normalized_features):
        raise ValueError("Every signal feature must lie in [0, n_features).")
    if (
        not np.isscalar(signal_strength)
        or not np.isfinite(signal_strength)
        or signal_strength <= 0
    ):
        raise ValueError("signal_strength must be a finite positive scalar.")

    if isinstance(selection_events, str):
        raise TypeError("selection_events must be a sequence, not one string.")
    normalized_events = tuple(selection_events)
    if len(normalized_events) < 2:
        raise ValueError("selection_events must contain at least two events.")
    if len(set(normalized_events)) != len(normalized_events):
        raise ValueError("selection_events must not contain duplicates.")
    for selection_event in normalized_events:
        _validate_selection_event(selection_event)
    return normalized_features, normalized_events


def _nonlinear_effect(x: np.ndarray) -> np.ndarray:
    """Return a centered, unit-SD smooth nonlinear effect."""
    raw_effect = x + 0.5 * (np.square(x) - 1.0)
    centered = raw_effect - np.mean(raw_effect)
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise FloatingPointError("Could not standardize the simulated signal.")
    return centered / scale


def _generate_power_dataset(
    rng,
    n_samples: int,
    n_features: int,
    signal_features: Sequence[int],
    signal_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent Gaussian features with smooth nonlinear signals.

    Each signal feature contributes a centered effect with empirical standard
    deviation ``signal_strength``; the Gaussian noise standard deviation is 1.
    """
    X = rng.standard_normal((n_samples, n_features))
    mean_response = np.zeros(n_samples)
    for feature in signal_features:
        mean_response += signal_strength * _nonlinear_effect(X[:, feature])
    response = mean_response + rng.standard_normal(n_samples) * SIMULATION_SIGMA
    return X, response


def _summarize_event_power(
    signal_results: pd.DataFrame,
    feature_results: pd.DataFrame,
    *,
    n_iters: int,
    n_signal_features: int,
    alpha: float,
) -> dict[str, float | int | str]:
    """Summarize selection, power, null-feature rejections, and AIS failures."""
    selection_event = str(signal_results["selection_event"].iloc[0])
    selected_signal = signal_results["selected"].to_numpy(dtype=bool)
    failed_signal = signal_results["failed"].to_numpy(dtype=bool)
    rejected_signal = signal_results["rejected"].to_numpy(dtype=bool)
    n_selected_signal = int(np.sum(selected_signal))
    n_failed_signal = int(np.sum(failed_signal))

    iteration_complete = signal_results.groupby("iteration")["failed"].apply(
        lambda failed: not bool(np.any(failed))
    )
    complete_iterations = iteration_complete[iteration_complete].index
    complete_signal = signal_results[
        signal_results["iteration"].isin(complete_iterations)
    ]
    iteration_power = complete_signal.groupby("iteration")["rejected"].mean()
    converged_power = (
        float(iteration_power.mean()) if not iteration_power.empty else np.nan
    )
    converged_simulation_se = (
        float(iteration_power.std(ddof=1) / np.sqrt(iteration_power.size))
        if iteration_power.size > 1
        else np.nan
    )
    strict_complete = n_failed_signal == 0

    converged_selected = signal_results[selected_signal & ~failed_signal]
    converged_conditional_power = (
        float(converged_selected["rejected"].mean())
        if not converged_selected.empty
        else np.nan
    )

    null_results = feature_results[~feature_results["is_signal"]]
    failed_null = int(null_results["failed"].sum()) if not null_results.empty else 0
    converged_null = null_results[~null_results["failed"]]
    converged_null_rejection_rate = (
        float(converged_null["rejected"].mean())
        if not converged_null.empty
        else np.nan
    )

    return {
        "selection_event": selection_event,
        "power": converged_power if strict_complete else np.nan,
        "simulation_se": converged_simulation_se if strict_complete else np.nan,
        "converged_power": converged_power,
        "converged_simulation_se": converged_simulation_se,
        "conditional_power": (
            converged_conditional_power if strict_complete else np.nan
        ),
        "converged_conditional_power": converged_conditional_power,
        "signal_selection_rate": float(np.mean(selected_signal)),
        "signal_test_failure_rate": (
            n_failed_signal / n_selected_signal if n_selected_signal else 0.0
        ),
        "converged_null_rejection_rate": converged_null_rejection_rate,
        "null_test_failure_rate": (
            failed_null / len(null_results) if len(null_results) else 0.0
        ),
        "n_iterations": n_iters,
        "n_complete_iterations": int(iteration_complete.sum()),
        "n_signal_features": n_signal_features,
        "n_selected_signal_tests": n_selected_signal,
        "n_converged_signal_tests": n_selected_signal - n_failed_signal,
        "n_selected_null_tests": int(len(null_results)),
        "alpha": alpha,
    }


def _paired_comparisons(
    signal_results: pd.DataFrame,
    summary: pd.DataFrame,
    selection_events: Sequence[str],
) -> pd.DataFrame:
    """Return paired power differences relative to the first event."""
    baseline = selection_events[0]
    summary_by_event = summary.set_index("selection_event")
    rows = []
    for comparison_event in selection_events[1:]:
        paired = signal_results[
            signal_results["selection_event"].isin((baseline, comparison_event))
        ].pivot(
            index=["iteration", "feature"],
            columns="selection_event",
            values=["rejected", "failed"],
        )
        pair_complete = ~(
            paired["failed"][baseline].astype(bool)
            | paired["failed"][comparison_event].astype(bool)
        )
        complete_by_iteration = pair_complete.groupby(level="iteration").all()
        complete_iterations = complete_by_iteration[complete_by_iteration].index
        complete_paired = paired.loc[
            paired.index.get_level_values("iteration").isin(complete_iterations),
            "rejected",
        ]
        iteration_difference = (
            complete_paired[comparison_event].astype(float)
            - complete_paired[baseline].astype(float)
        ).groupby(level="iteration").mean()
        converged_difference = (
            float(iteration_difference.mean())
            if not iteration_difference.empty
            else np.nan
        )
        paired_se = (
            float(
                iteration_difference.std(ddof=1)
                / np.sqrt(iteration_difference.size)
            )
            if iteration_difference.size > 1
            else np.nan
        )
        baseline_power = summary_by_event.loc[baseline, "power"]
        comparison_power = summary_by_event.loc[comparison_event, "power"]
        strict_difference = (
            float(comparison_power - baseline_power)
            if np.isfinite(baseline_power) and np.isfinite(comparison_power)
            else np.nan
        )
        rows.append(
            {
                "baseline_event": baseline,
                "comparison_event": comparison_event,
                "power_difference": strict_difference,
                "converged_power_difference": converged_difference,
                "paired_simulation_se": paired_se,
                "ci_95_lower": converged_difference - 1.96 * paired_se,
                "ci_95_upper": converged_difference + 1.96 * paired_se,
                "n_complete_pairs": int(iteration_difference.size),
            }
        )
    return pd.DataFrame(rows)


def compare_selection_event_power(
    n_iters: int,
    n_samples: int,
    n_features: int,
    k_select: int,
    *,
    signal_features: Sequence[int] = (0,),
    signal_strength: float = 1.0,
    selection_events: Sequence[str] = DEFAULT_SELECTION_EVENTS,
    alpha: float = 0.05,
    seed: int = 123,
    selection_decimals: int = 10,
    pilot_iters: int = 3,
    pilot_samples: int = 40,
    final_batch_size: int = 80,
    max_final_samples: int = 800,
    min_denominator_ess: float = 80.0,
    min_tail_ess: float = 15.0,
    rf_params=None,
    estimator=None,
    selector=None,
    multiplicity: str = "none",
):
    """Compare conditioning-event power on identical alternative datasets.

    ``power`` is the probability that a true signal is selected and rejected.
    The same generated data, selector, and AIS seed are used for every event in
    each iteration, making event differences paired. If any selected-signal AIS
    test fails, strict ``power`` is unavailable and explicitly labeled
    ``converged_power`` is also reported from complete iterations.
    """
    _validate_inputs(n_iters, n_samples, n_features, k_select, alpha)
    signal_features, selection_events = _validate_power_inputs(
        n_features, signal_features, signal_strength, selection_events
    )
    if k_select == 1 and {
        "exact_set",
        "feature_inclusion",
    }.issubset(selection_events):
        warnings.warn(
            "For k_select=1, exact_set and feature_inclusion define the same "
            "selection event, so their exact power is identical.",
            UserWarning,
            stacklevel=2,
        )
    if not isinstance(selection_decimals, int) or selection_decimals < 0:
        raise ValueError("selection_decimals must be a nonnegative integer.")
    if pilot_iters < 0 or pilot_samples < 1:
        raise ValueError("pilot_iters must be nonnegative and pilot_samples positive.")
    if final_batch_size < 1 or max_final_samples < 1:
        raise ValueError("Final AIS sample counts must be positive.")
    if min_denominator_ess <= 0 or min_tail_ess <= 0:
        raise ValueError("AIS ESS thresholds must be positive.")
    if multiplicity not in {"none", "bonferroni", "holm"}:
        raise ValueError("multiplicity must be 'none', 'bonferroni', or 'holm'.")
    if sum(value is not None for value in (rf_params, estimator, selector)) > 1:
        raise ValueError("Pass only one of rf_params, estimator, or selector.")

    resolved_rf_params = (
        _resolve_rf_params(rf_params)
        if estimator is None and selector is None
        else None
    )
    resolved_selector = make_selector(
        estimator=estimator,
        selector=selector,
        selection_decimals=selection_decimals,
        rf_params=resolved_rf_params,
    )
    p_value_column = (
        "raw_selective_p_value"
        if multiplicity == "none"
        else "adjusted_selective_p_value"
    )
    signal_feature_set = set(signal_features)
    feature_records = []
    signal_records = []

    iteration_seeds = np.random.SeedSequence(seed).spawn(n_iters)
    for iteration, iteration_seed in enumerate(
        tqdm(iteration_seeds, desc="Comparing selection events"), start=1
    ):
        data_seed, ais_seed = iteration_seed.spawn(2)
        X, response = _generate_power_dataset(
            np.random.default_rng(data_seed),
            n_samples,
            n_features,
            signal_features,
            signal_strength,
        )
        shared_ais_seed = int(ais_seed.generate_state(1, dtype=np.uint32)[0])
        observed_selected = None
        event_results = {}
        for selection_event in selection_events:
            inference_result = selective_inference(
                X,
                response,
                k_select=k_select,
                sigma=SIMULATION_SIGMA,
                selector=resolved_selector,
                selection_event=selection_event,
                multiplicity=multiplicity,
                ais_seed=shared_ais_seed,
                pilot_iters=pilot_iters,
                pilot_samples=pilot_samples,
                final_batch_size=final_batch_size,
                max_final_samples=max_final_samples,
                min_denominator_ess=min_denominator_ess,
                min_tail_ess=min_tail_ess,
            )
            selected = tuple(
                int(feature)
                for feature in inference_result["observed_selected_features"]
            )
            if observed_selected is None:
                observed_selected = selected
            elif selected != observed_selected:
                raise RuntimeError(
                    "The deterministic selector returned different observed "
                    "features across conditioning events."
                )

            event_frame = inference_result["feature_results"].copy()
            event_frame.insert(0, "iteration", iteration)
            event_frame["is_signal"] = event_frame["feature"].isin(
                signal_feature_set
            )
            event_frame["p_value_used"] = event_frame[p_value_column]
            event_frame["failed"] = ~np.isfinite(event_frame["p_value_used"])
            event_frame["rejected"] = (
                event_frame["p_value_used"] < alpha
            ).fillna(False)
            feature_records.extend(event_frame.to_dict(orient="records"))
            event_results[selection_event] = event_frame.set_index("feature")

        for selection_event, event_frame in event_results.items():
            for feature in signal_features:
                selected = feature in event_frame.index
                if selected:
                    row = event_frame.loc[feature]
                    p_value = float(row["p_value_used"])
                    failed = bool(row["failed"])
                    rejected = bool(row["rejected"])
                else:
                    p_value = np.nan
                    failed = False
                    rejected = False
                signal_records.append(
                    {
                        "iteration": iteration,
                        "selection_event": selection_event,
                        "feature": feature,
                        "selected": selected,
                        "p_value": p_value,
                        "failed": failed,
                        "rejected": rejected,
                    }
                )

    feature_results = pd.DataFrame(feature_records)
    signal_results = pd.DataFrame(signal_records)
    summary = pd.DataFrame(
        [
            _summarize_event_power(
                signal_results[
                    signal_results["selection_event"] == selection_event
                ],
                feature_results[
                    feature_results["selection_event"] == selection_event
                ],
                n_iters=n_iters,
                n_signal_features=len(signal_features),
                alpha=alpha,
            )
            for selection_event in selection_events
        ]
    )
    comparisons = _paired_comparisons(
        signal_results, summary, selection_events
    )
    return {
        "summary": summary,
        "comparisons": comparisons,
        "signal_results": signal_results,
        "feature_results": feature_results,
        "alpha": alpha,
        "settings": {
            "n_iters": n_iters,
            "n_samples": n_samples,
            "n_features": n_features,
            "k_select": k_select,
            "signal_features": signal_features,
            "signal_strength": float(signal_strength),
            "selection_events": selection_events,
            "alpha": alpha,
            "seed": seed,
            "selection_decimals": selection_decimals,
            "multiplicity": multiplicity,
            "variance_method": "known_simulation_sigma",
            "sigma": SIMULATION_SIGMA,
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
        },
    }
