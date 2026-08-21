"""Orchestration for selective-inference simulations."""

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
from tqdm.auto import tqdm

from .api import adjust_p_values, selective_inference
from .inference import _chi_statistic, _spline_effect_basis
from .selection import _resolve_rf_params, _validate_selection_event, make_selector


SIMULATION_SIGMA = 1.0


def _validate_seed(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _validate_positive_finite(name, value):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def _validate_feature_correlation(value):
    if (
        not np.isscalar(value)
        or not np.isfinite(value)
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError("feature_correlation must lie in [0, 1).")
    return float(value)


def _generate_gaussian_design(rng, n_samples, n_features, feature_correlation=0.0):
    """Generate an independent or AR(1)-correlated Gaussian design."""
    correlation = _validate_feature_correlation(feature_correlation)
    independent = rng.standard_normal((n_samples, n_features))
    if correlation == 0.0:
        return independent
    indices = np.arange(n_features)
    covariance = correlation ** np.abs(indices[:, None] - indices[None, :])
    return independent @ np.linalg.cholesky(covariance).T


def _generate_null_dataset(
    rng, n_samples, n_features, feature_correlation=0.0
):
    """Generate one global-null data set from a NumPy random generator."""
    X = _generate_gaussian_design(
        rng, n_samples, n_features, feature_correlation
    )
    response = rng.standard_normal(n_samples)
    return X, response


def _validate_inputs(n_iters, n_samples, n_features, k_select, alpha):
    counts = {
        "n_iters": n_iters,
        "n_samples": n_samples,
        "n_features": n_features,
        "k_select": k_select,
    }
    for name, value in counts.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
    if n_iters < 1:
        raise ValueError("n_iters must be at least 1.")
    if n_samples <= 4:
        raise ValueError("n_samples must exceed 4 for a three-dimensional spline effect.")
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if not 1 <= k_select <= n_features:
        raise ValueError("k_select must satisfy 1 <= k_select <= n_features.")
    if (
        not np.isscalar(alpha)
        or not np.isfinite(alpha)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("alpha must lie strictly between 0 and 1.")


def _method_summary(name, p_values_by_iteration, alpha, require_complete=True):
    complete = np.array([np.all(np.isfinite(row)) for row in p_values_by_iteration])
    failure_rate = float(1.0 - np.mean(complete))
    converged_rows = [
        np.asarray(row)
        for row, ok in zip(p_values_by_iteration, complete)
        if ok
    ]
    converged_p_values = (
        np.concatenate(converged_rows) if converged_rows else np.array([])
    )
    converged_fpr = (
        float(np.mean(converged_p_values < alpha))
        if converged_p_values.size
        else np.nan
    )
    uniform_ks_statistic = (
        float(stats.kstest(converged_p_values, "uniform").statistic)
        if converged_p_values.size
        else np.nan
    )
    converged_familywise_error_rate = (
        float(np.mean([np.any(row < alpha) for row in converged_rows]))
        if converged_rows
        else np.nan
    )
    converged_mean_rejections = (
        float(np.mean([np.sum(row < alpha) for row in converged_rows]))
        if converged_rows
        else np.nan
    )
    converged_iteration_rates = np.asarray(
        [np.mean(row < alpha) for row in converged_rows], dtype=float
    )
    if converged_iteration_rates.size > 1:
        critical_value = float(
            stats.t.ppf(0.975, df=converged_iteration_rates.size - 1)
        )
        converged_rate_se = float(
            np.std(converged_iteration_rates, ddof=1)
            / np.sqrt(converged_iteration_rates.size)
        )
        converged_fpr_ci = (
            max(0.0, converged_fpr - critical_value * converged_rate_se),
            min(1.0, converged_fpr + critical_value * converged_rate_se),
        )
    else:
        converged_fpr_ci = (np.nan, np.nan)
    n_complete = len(converged_rows)
    if n_complete:
        familywise_successes = int(
            np.sum([np.any(row < alpha) for row in converged_rows])
        )
        converged_fwer_ci = (
            0.0
            if familywise_successes == 0
            else float(
                stats.beta.ppf(
                    0.025,
                    familywise_successes,
                    n_complete - familywise_successes + 1,
                )
            ),
            1.0
            if familywise_successes == n_complete
            else float(
                stats.beta.ppf(
                    0.975,
                    familywise_successes + 1,
                    n_complete - familywise_successes,
                )
            ),
        )
    else:
        converged_fwer_ci = (np.nan, np.nan)

    if require_complete and failure_rate > 0:
        fpr = np.nan
        simulation_se = np.nan
        familywise_error_rate = np.nan
        mean_rejections = np.nan
        fpr_ci = (np.nan, np.nan)
        fwer_ci = (np.nan, np.nan)
    else:
        iteration_rates = np.array(
            [
                np.mean(np.asarray(row) < alpha)
                for row, ok in zip(p_values_by_iteration, complete)
                if ok
            ]
        )
        fpr = float(np.mean(iteration_rates))
        simulation_se = (
            float(np.std(iteration_rates, ddof=1) / np.sqrt(iteration_rates.size))
            if iteration_rates.size > 1
            else np.nan
        )
        familywise_error_rate = converged_familywise_error_rate
        mean_rejections = converged_mean_rejections
        fpr_ci = converged_fpr_ci
        fwer_ci = converged_fwer_ci

    return {
        "method": name,
        "fpr": fpr,
        "simulation_se": simulation_se,
        "converged_fpr": converged_fpr,
        "fpr_ci_95_lower": fpr_ci[0],
        "fpr_ci_95_upper": fpr_ci[1],
        "converged_fpr_ci_95_lower": converged_fpr_ci[0],
        "converged_fpr_ci_95_upper": converged_fpr_ci[1],
        "familywise_error_rate": familywise_error_rate,
        "converged_familywise_error_rate": converged_familywise_error_rate,
        "familywise_error_ci_95_lower": fwer_ci[0],
        "familywise_error_ci_95_upper": fwer_ci[1],
        "converged_familywise_error_ci_95_lower": converged_fwer_ci[0],
        "converged_familywise_error_ci_95_upper": converged_fwer_ci[1],
        # Under the global null, FDP is one exactly when there is any rejection,
        # so FDR equals FWER.
        "false_discovery_rate": familywise_error_rate,
        "converged_false_discovery_rate": converged_familywise_error_rate,
        "mean_rejections": mean_rejections,
        "converged_mean_rejections": converged_mean_rejections,
        "failure_rate": failure_rate,
        "n_pvalues": int(converged_p_values.size),
        # Diagnostic only: selected-feature p-values within an iteration can be
        # dependent, so no iid KS-test p-value is reported.
        "uniform_ks_statistic": uniform_ks_statistic,
    }, converged_p_values


def run_simulation(
    n_iters,
    n_samples,
    n_features,
    k_select,
    alpha=0.05,
    seed=123,
    selection_decimals=10,
    pilot_iters=3,
    pilot_samples=40,
    final_batch_size=80,
    max_final_samples=800,
    min_denominator_ess=80.0,
    min_tail_ess=15.0,
    rf_params=None,
    estimator=None,
    selector=None,
    selection_event="exact_set",
    multiplicity="none",
    inference_method="conditional_mc",
    stop_when_ess_met=False,
    selection_method=None,
):
    """Run the global-null experiment documented in ``docs/``.

    ``rf_params`` may override any default ``RandomForestRegressor`` parameter.
    """
    _validate_inputs(n_iters, n_samples, n_features, k_select, alpha)
    if (
        isinstance(selection_decimals, (bool, np.bool_))
        or not isinstance(selection_decimals, (int, np.integer))
        or selection_decimals < 0
    ):
        raise ValueError("selection_decimals must be a nonnegative integer.")
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
    _validate_positive_finite("min_denominator_ess", min_denominator_ess)
    _validate_positive_finite("min_tail_ess", min_tail_ess)
    if sum(value is not None for value in (rf_params, estimator, selector)) > 1:
        raise ValueError("Pass only one of rf_params, estimator, or selector.")
    if (
        selection_method is not None
        and selection_method != "shap"
        and rf_params is not None
    ):
        raise ValueError("rf_params are available only for SHAP selection.")
    seed = _validate_seed("seed", seed)
    if not isinstance(stop_when_ess_met, (bool, np.bool_)):
        raise TypeError("stop_when_ess_met must be boolean.")
    _validate_selection_event(selection_event)
    if selection_event == "same_target":
        raise ValueError(
            "run_simulation does not define a randomized single-target rule for "
            "same_target; use compare_selection_event_null_calibration or call "
            "selective_inference with target_rule='uniform_from_selected'."
        )
    if multiplicity not in {"none", "bonferroni", "holm"}:
        raise ValueError("multiplicity must be 'none', 'bonferroni', or 'holm'.")
    if selection_event == "feature_inclusion" and multiplicity == "holm":
        raise ValueError(
            "Holm adjustment is not implemented for feature_inclusion because "
            "the featurewise p-values condition on different inclusion events. "
            "Use multiplicity='bonferroni' or a common exact_set/exact_ranking event."
        )
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
    selector_display = {
        "mutual_information": "Mutual information",
        "marginal_screening": "Marginal screening",
    }.get(selection_method, "SHAP")

    random_p_values = []
    unadjusted_p_values = []
    selective_p_values = []
    ais_diagnostics = []

    if selection_event == "exact_set":
        message = (
            "Exact-set conditioning can be rare. Conditional Monte Carlo remains "
            "valid but may be conservative when few proposal radii reproduce the "
            "event; inspect selected_samples and selection_probability_estimate."
            if inference_method == "conditional_mc"
            else
            "Exact-set conditioning can be rare. Exploratory AIS estimates can "
            "fail when too little selected or tail weight is sampled; inspect "
            "status, denominator_ess, and tail_ess."
        )
        warnings.warn(
            message,
            UserWarning,
            stacklevel=2,
        )

    iteration_seeds = np.random.SeedSequence(seed).spawn(n_iters)
    for iteration, iteration_seed in enumerate(
        tqdm(iteration_seeds, desc="Running simulation"), start=1
    ):
        data_seed, random_selection_seed, ais_seed = iteration_seed.spawn(3)
        data_rng = np.random.default_rng(data_seed)
        random_selection_rng = np.random.default_rng(random_selection_seed)

        X, response = _generate_null_dataset(data_rng, n_samples, n_features)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Exact-set conditioning can make the event rare"
            )
            inference_result = selective_inference(
                X,
                response,
                k_select=k_select,
                sigma=SIMULATION_SIGMA,
                selector=resolved_selector,
                selection_event=selection_event,
                multiplicity=multiplicity,
                inference_method=inference_method,
                ais_seed=int(ais_seed.generate_state(1, dtype=np.uint32)[0]),
                pilot_iters=pilot_iters,
                pilot_samples=pilot_samples,
                final_batch_size=final_batch_size,
                max_final_samples=max_final_samples,
                min_denominator_ess=min_denominator_ess,
                min_tail_ess=min_tail_ess,
                stop_when_ess_met=stop_when_ess_met,
            )
        random_selected = random_selection_rng.choice(
            n_features, size=k_select, replace=False
        )

        random_iteration = []
        for feature in random_selected:
            basis = _spline_effect_basis(X[:, feature])
            statistic, _ = _chi_statistic(response, basis)
            random_iteration.append(stats.chi.sf(statistic, df=basis.shape[1]))
        random_p_values.append(
            adjust_p_values(np.asarray(random_iteration), multiplicity)
        )

        feature_results = inference_result["feature_results"].copy()
        feature_results.insert(0, "iteration", iteration)
        feature_results["rank"] = feature_results["test_rank"]
        ais_diagnostics.extend(feature_results.to_dict(orient="records"))
        unadjusted_iteration = adjust_p_values(
            feature_results["unadjusted_p_value"].to_numpy(), multiplicity
        )
        selective_column = (
            "raw_selective_p_value"
            if multiplicity == "none"
            else "adjusted_selective_p_value"
        )
        selective_iteration = feature_results[selective_column].to_numpy()

        unadjusted_p_values.append(np.asarray(unadjusted_iteration))
        selective_p_values.append(np.asarray(selective_iteration))

    random_summary, random_flat = _method_summary(
        "Random", random_p_values, alpha, require_complete=True
    )
    unadjusted_summary, unadjusted_flat = _method_summary(
        f"Unadjusted {selector_display}",
        unadjusted_p_values,
        alpha,
        require_complete=True,
    )
    selective_method_name = (
        f"Selective {selector_display} (conditional MC)"
        if inference_method == "conditional_mc"
        else f"Selective {selector_display} (approximate AIS)"
    )
    selective_summary, selective_flat = _method_summary(
        selective_method_name,
        selective_p_values,
        alpha,
        require_complete=True,
    )

    summary = pd.DataFrame(
        [random_summary, unadjusted_summary, selective_summary]
    )
    summary["selection_event"] = selection_event
    summary["multiplicity"] = multiplicity
    summary["p_value_scale"] = (
        "raw" if multiplicity == "none" else "multiplicity_adjusted"
    )
    diagnostics = pd.DataFrame(ais_diagnostics)
    return {
        "summary": summary,
        "p_values": {
            "Random": random_flat,
            f"Unadjusted {selector_display}": unadjusted_flat,
            selective_method_name: selective_flat,
        },
        "inference_diagnostics": diagnostics,
        "ais_diagnostics": diagnostics,
        "alpha": alpha,
        "settings": {
            "n_iters": n_iters,
            "n_samples": n_samples,
            "n_features": n_features,
            "k_select": k_select,
            "seed": seed,
            "selection_decimals": selection_decimals,
            "selection_method": (
                "shap" if selection_method is None else selection_method
            ),
            "rf_params": None if resolved_rf_params is None else resolved_rf_params.copy(),
            "selection_event": selection_event,
            "multiplicity": multiplicity,
            "inference_method": inference_method,
            "variance_method": "known_simulation_sigma",
            "sigma": SIMULATION_SIGMA,
            "selector_settings": dict(resolved_selector.get_settings()),
            "pilot_iters": pilot_iters,
            "pilot_samples": pilot_samples,
            "final_batch_size": final_batch_size,
            "max_final_samples": max_final_samples,
            "min_denominator_ess": min_denominator_ess,
            "min_tail_ess": min_tail_ess,
            "p_value_scale": (
                "raw" if multiplicity == "none" else "multiplicity_adjusted"
            ),
            "stop_when_ess_met": bool(stop_when_ess_met),
        },
    }
