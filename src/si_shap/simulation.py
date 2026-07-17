"""Orchestration for the SHAP selective-inference simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from .inference import TRUE_SIGMA, _chi_statistic, _run_ais, _spline_effect_basis
from .selection import _select_features, _top_k, _tree_shap_importance


def _validate_inputs(n_iters, n_samples, n_features, k_select, alpha):
    if n_iters < 1:
        raise ValueError("n_iters must be at least 1.")
    if n_samples <= 4:
        raise ValueError("n_samples must exceed 4 for a three-dimensional spline effect.")
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if not 1 <= k_select <= n_features:
        raise ValueError("k_select must satisfy 1 <= k_select <= n_features.")
    if not 0.0 < alpha < 1.0:
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

    if require_complete and failure_rate > 0:
        fpr = np.nan
        simulation_se = np.nan
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

    return {
        "method": name,
        "fpr": fpr,
        "simulation_se": simulation_se,
        "converged_fpr": converged_fpr,
        "failure_rate": failure_rate,
        "n_pvalues": int(converged_p_values.size),
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
):
    """Run the global-null experiment documented in ``docs/``."""
    _validate_inputs(n_iters, n_samples, n_features, k_select, alpha)
    if not isinstance(selection_decimals, int) or selection_decimals < 0:
        raise ValueError("selection_decimals must be a nonnegative integer.")
    if pilot_iters < 0 or pilot_samples < 1:
        raise ValueError("pilot_iters must be nonnegative and pilot_samples positive.")
    if final_batch_size < 1 or max_final_samples < 1:
        raise ValueError("Final AIS sample counts must be positive.")
    if min_denominator_ess <= 0 or min_tail_ess <= 0:
        raise ValueError("AIS ESS thresholds must be positive.")

    random_p_values = []
    naive_p_values = []
    selective_p_values = []
    ais_diagnostics = []

    iteration_seeds = np.random.SeedSequence(seed).spawn(n_iters)
    for iteration, iteration_seed in enumerate(
        tqdm(iteration_seeds, desc="Running simulation"), start=1
    ):
        data_seed, random_selection_seed, ais_seed = iteration_seed.spawn(3)
        data_rng = np.random.default_rng(data_seed)
        random_selection_rng = np.random.default_rng(random_selection_seed)

        X = data_rng.standard_normal((n_samples, n_features))
        response = data_rng.standard_normal(n_samples)

        shap_importance = _tree_shap_importance(X, response, selection_decimals)
        shap_selected = _top_k(shap_importance, k_select)
        random_selected = random_selection_rng.choice(
            n_features, size=k_select, replace=False
        )

        random_iteration = []
        for feature in random_selected:
            basis = _spline_effect_basis(X[:, feature])
            statistic, _ = _chi_statistic(response, basis)
            random_iteration.append(stats.chi.sf(statistic, df=basis.shape[1]))
        random_p_values.append(np.asarray(random_iteration))

        naive_iteration = []
        selective_iteration = []
        feature_seeds = ais_seed.spawn(k_select)
        for position, feature in enumerate(shap_selected):
            basis = _spline_effect_basis(X[:, feature])
            rank = basis.shape[1]
            statistic, projected = _chi_statistic(response, basis)
            naive_iteration.append(stats.chi.sf(statistic, df=rank))

            orthogonal = response - projected
            projected_norm = np.linalg.norm(projected)
            zero_tolerance = np.sqrt(np.finfo(float).eps) * max(
                1.0, np.linalg.norm(response)
            )
            if projected_norm <= zero_tolerance:
                raise FloatingPointError(
                    "The observed effect direction is numerically undefined."
                )
            direction = projected / projected_norm

            def is_selected(z):
                candidate = orthogonal + TRUE_SIGMA * direction * z
                selected = _select_features(
                    X, candidate, k_select, selection_decimals
                )
                return bool(feature in selected)

            if not is_selected(statistic):
                raise RuntimeError(
                    "The observed response was not reconstructed inside its "
                    "selection event."
                )

            p_value, diagnostics = _run_ais(
                statistic,
                rank,
                is_selected,
                np.random.default_rng(feature_seeds[position]),
                pilot_iters=pilot_iters,
                pilot_samples=pilot_samples,
                final_batch_size=final_batch_size,
                max_final_samples=max_final_samples,
                min_denominator_ess=min_denominator_ess,
                min_tail_ess=min_tail_ess,
            )
            diagnostics.update(
                {
                    "iteration": iteration,
                    "feature": int(feature),
                    "rank": rank,
                    "t_obs": statistic,
                }
            )
            ais_diagnostics.append(diagnostics)
            selective_iteration.append(p_value)

        naive_p_values.append(np.asarray(naive_iteration))
        selective_p_values.append(np.asarray(selective_iteration))

    random_summary, random_flat = _method_summary(
        "Random", random_p_values, alpha, require_complete=True
    )
    naive_summary, naive_flat = _method_summary(
        "Naive SHAP", naive_p_values, alpha, require_complete=True
    )
    selective_summary, selective_flat = _method_summary(
        "Selective SHAP (AIS)",
        selective_p_values,
        alpha,
        require_complete=True,
    )

    summary = pd.DataFrame([random_summary, naive_summary, selective_summary])
    diagnostics = pd.DataFrame(ais_diagnostics)
    return {
        "summary": summary,
        "p_values": {
            "Random": random_flat,
            "Naive SHAP": naive_flat,
            "Selective SHAP (AIS)": selective_flat,
        },
        "ais_diagnostics": diagnostics,
        "alpha": alpha,
        "settings": {
            "n_iters": n_iters,
            "n_samples": n_samples,
            "n_features": n_features,
            "k_select": k_select,
            "seed": seed,
            "selection_decimals": selection_decimals,
        },
    }
