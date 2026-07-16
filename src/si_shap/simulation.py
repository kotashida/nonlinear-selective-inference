"""Simulation utilities for SHAP-based selective inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from patsy import dmatrix
from scipy import stats
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm


RF_PARAMS = {"n_estimators": 50, "max_depth": 5, "random_state": 42}


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


def _top_k(scores, k):
    """Select by decreasing score, breaking ties by increasing feature index."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("SHAP importance scores must be a finite one-dimensional array.")
    return np.lexsort((np.arange(scores.size), -scores))[:k]


def _tree_shap_importance(X, response, selection_decimals):
    # A fixed RF seed and common rounding make the selection map deterministic.
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


def _spline_effect_basis(x):
    """Return an orthonormal basis for the centered cubic B-spline effect."""
    design = np.asarray(
        dmatrix(
            "bs(x, df=3, degree=3, include_intercept=False) - 1",
            {"x": np.asarray(x)},
            return_type="dataframe",
        )
    )
    centered = design - design.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        raise ValueError("The centered spline design has rank zero.")
    tolerance = max(centered.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        raise ValueError("The centered spline design has numerical rank zero.")
    return left[:, :rank]


def _f_statistic(response, basis):
    """Return the partial-F statistic after removing an unknown intercept."""
    mean_component = np.full_like(response, np.mean(response), dtype=float)
    centered_response = response - mean_component
    coordinates = basis.T @ centered_response
    projected = basis @ coordinates
    orthogonal = centered_response - projected
    numerator_df = basis.shape[1]
    denominator_df = response.size - numerator_df - 1
    residual_sum_squares = float(orthogonal @ orthogonal)
    if denominator_df <= 0 or residual_sum_squares <= 0.0:
        raise FloatingPointError(
            "The residual variance needed for the F statistic is undefined."
        )
    effect_sum_squares = float(coordinates @ coordinates)
    statistic = (effect_sum_squares / numerator_df) / (
        residual_sum_squares / denominator_df
    )
    return statistic, projected, orthogonal, mean_component, denominator_df


def _truncated_normal_logpdf(z, mean, sd):
    z = np.asarray(z)
    return stats.norm.logpdf(z, loc=mean, scale=sd) - stats.norm.logcdf(mean / sd)


def _sample_truncated_normal(rng, mean, sd, size):
    lower = -mean / sd
    return stats.truncnorm.rvs(
        lower,
        np.inf,
        loc=mean,
        scale=sd,
        size=size,
        random_state=rng,
    )


def _effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0 or np.sum(weights) == 0:
        return 0.0
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def _adapt_proposal(
    t_obs,
    numerator_df,
    denominator_df,
    is_selected,
    rng,
    pilot_iters,
    pilot_samples,
):
    mean = float(t_obs)
    sd = max(1.0, 0.25 * t_obs + 0.5)

    for _ in range(pilot_iters):
        z_values = _sample_truncated_normal(rng, mean, sd, pilot_samples)
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values),
            dtype=bool,
            count=pilot_samples,
        )
        if not np.any(selected):
            sd *= 1.5
            continue

        selected_z = z_values[selected]
        log_weights = (
            stats.f.logpdf(selected_z, dfn=numerator_df, dfd=denominator_df)
            - _truncated_normal_logpdf(selected_z, mean, sd)
        )
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            sd *= 1.5
            continue

        selected_z = selected_z[finite]
        weights = np.exp(log_weights[finite] - np.max(log_weights[finite]))
        new_mean = float(np.average(selected_z, weights=weights))
        new_variance = float(np.average((selected_z - new_mean) ** 2, weights=weights))

        # Damping and a scale floor prevent collapse after a small pilot sample.
        mean = 0.5 * mean + 0.5 * new_mean
        sd = max(0.25, 0.5 * sd + 0.5 * np.sqrt(max(new_variance, 0.0)))

    return mean, sd


def _sample_defensive_mixture(
    rng,
    size,
    numerator_df,
    denominator_df,
    t_obs,
    adapted_mean,
    adapted_sd,
):
    mixture_weights = np.array([0.25, 0.375, 0.375])
    counts = rng.multinomial(size, mixture_weights)
    observed_sd = max(0.75, 0.25 * t_obs + 0.5)

    samples = [
        stats.f.rvs(
            dfn=numerator_df,
            dfd=denominator_df,
            size=counts[0],
            random_state=rng,
        ),
        _sample_truncated_normal(rng, t_obs, observed_sd, counts[1]),
        _sample_truncated_normal(rng, adapted_mean, adapted_sd, counts[2]),
    ]
    z_values = np.concatenate(samples)
    rng.shuffle(z_values)

    component_logpdf = np.vstack(
        [
            np.log(mixture_weights[0])
            + stats.f.logpdf(z_values, dfn=numerator_df, dfd=denominator_df),
            np.log(mixture_weights[1])
            + _truncated_normal_logpdf(z_values, t_obs, observed_sd),
            np.log(mixture_weights[2])
            + _truncated_normal_logpdf(z_values, adapted_mean, adapted_sd),
        ]
    )
    return z_values, logsumexp(component_logpdf, axis=0)


def _run_ais(
    t_obs,
    numerator_df,
    denominator_df,
    is_selected,
    rng,
    pilot_iters=3,
    pilot_samples=40,
    final_batch_size=80,
    max_final_samples=800,
    min_denominator_ess=80.0,
    min_tail_ess=15.0,
):
    adapted_mean, adapted_sd = _adapt_proposal(
        t_obs,
        numerator_df,
        denominator_df,
        is_selected,
        rng,
        pilot_iters,
        pilot_samples,
    )

    selected_z_batches = []
    selected_log_weight_batches = []
    proposals = 0

    while proposals < max_final_samples:
        batch_size = min(final_batch_size, max_final_samples - proposals)
        z_values, proposal_logpdf = _sample_defensive_mixture(
            rng,
            batch_size,
            numerator_df,
            denominator_df,
            t_obs,
            adapted_mean,
            adapted_sd,
        )
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values),
            dtype=bool,
            count=batch_size,
        )
        if np.any(selected):
            selected_z = z_values[selected]
            log_weights = (
                stats.f.logpdf(
                    selected_z, dfn=numerator_df, dfd=denominator_df
                )
                - proposal_logpdf[selected]
            )
            finite = np.isfinite(log_weights)
            if np.any(finite):
                selected_z_batches.append(selected_z[finite])
                selected_log_weight_batches.append(log_weights[finite])
        proposals += batch_size

        if not selected_log_weight_batches:
            continue
        all_z = np.concatenate(selected_z_batches)
        all_log_weights = np.concatenate(selected_log_weight_batches)
        weights = np.exp(all_log_weights - np.max(all_log_weights))
        tail = all_z >= t_obs
        denominator_ess = _effective_sample_size(weights)
        tail_ess = _effective_sample_size(weights[tail])
        if denominator_ess >= min_denominator_ess and tail_ess >= min_tail_ess:
            break

    if not selected_log_weight_batches:
        return np.nan, {
            "status": "no_selected_samples",
            "proposals": proposals,
            "selected_samples": 0,
            "tail_samples": 0,
            "denominator_ess": 0.0,
            "tail_ess": 0.0,
            "mc_se": np.nan,
        }

    all_z = np.concatenate(selected_z_batches)
    all_log_weights = np.concatenate(selected_log_weight_batches)
    weights = np.exp(all_log_weights - np.max(all_log_weights))
    tail = all_z >= t_obs
    denominator_ess = _effective_sample_size(weights)
    tail_ess = _effective_sample_size(weights[tail])
    converged = denominator_ess >= min_denominator_ess and tail_ess >= min_tail_ess

    if not converged:
        return np.nan, {
            "status": "insufficient_ess",
            "proposals": proposals,
            "selected_samples": int(all_z.size),
            "tail_samples": int(np.sum(tail)),
            "denominator_ess": denominator_ess,
            "tail_ess": tail_ess,
            "mc_se": np.nan,
        }

    p_value = float(np.sum(weights * tail) / np.sum(weights))
    mc_variance = np.sum(weights**2 * (tail.astype(float) - p_value) ** 2) / np.sum(weights) ** 2
    return np.clip(p_value, 0.0, 1.0), {
        "status": "ok",
        "proposals": proposals,
        "selected_samples": int(all_z.size),
        "tail_samples": int(np.sum(tail)),
        "denominator_ess": denominator_ess,
        "tail_ess": tail_ess,
        "mc_se": float(np.sqrt(mc_variance)),
    }


def _method_summary(name, p_values_by_iteration, alpha, require_complete=True):
    complete = np.array([np.all(np.isfinite(row)) for row in p_values_by_iteration])
    failure_rate = float(1.0 - np.mean(complete))
    converged_rows = [np.asarray(row) for row, ok in zip(p_values_by_iteration, complete) if ok]
    converged_p_values = np.concatenate(converged_rows) if converged_rows else np.array([])
    converged_fpr = (
        float(np.mean(converged_p_values < alpha)) if converged_p_values.size else np.nan
    )

    if require_complete and failure_rate > 0:
        fpr = np.nan
        simulation_se = np.nan
    else:
        iteration_rates = np.array(
            [np.mean(np.asarray(row) < alpha) for row, ok in zip(p_values_by_iteration, complete) if ok]
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
    noise_scale=1.0,
):
    """Run a global-null simulation with sigma unknown to every test."""
    _validate_inputs(n_iters, n_samples, n_features, k_select, alpha)
    if not isinstance(selection_decimals, int) or selection_decimals < 0:
        raise ValueError("selection_decimals must be a nonnegative integer.")
    if pilot_iters < 0 or pilot_samples < 1:
        raise ValueError("pilot_iters must be nonnegative and pilot_samples positive.")
    if final_batch_size < 1 or max_final_samples < 1:
        raise ValueError("Final AIS sample counts must be positive.")
    if min_denominator_ess <= 0 or min_tail_ess <= 0:
        raise ValueError("AIS ESS thresholds must be positive.")
    if (
        not np.isscalar(noise_scale)
        or not np.isfinite(noise_scale)
        or noise_scale <= 0
    ):
        raise ValueError("noise_scale must be a finite positive scalar.")

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

        # [1] グローバル帰無仮説下のデータ生成
        X = data_rng.standard_normal((n_samples, n_features))
        # noise_scale belongs only to the data-generating process.  No test
        # receives its true value; each F statistic self-normalizes by its
        # orthogonal residual mean square.
        response = noise_scale * data_rng.standard_normal(n_samples)

        # [2] 固定した Random Forest 手続きによる SHAP 重要度
        shap_importance = _tree_shap_importance(X, response, selection_decimals)

        # [3] 決定的な SHAP 上位 k 選択と応答に独立なランダム選択
        shap_selected = _top_k(shap_importance, k_select)
        random_selected = random_selection_rng.choice(
            n_features, size=k_select, replace=False
        )

        # [4] ランダム選択後の未知分散 F 検定
        random_iteration = []
        for feature in random_selected:
            basis = _spline_effect_basis(X[:, feature])
            statistic, _, _, _, denominator_df = _f_statistic(response, basis)
            random_iteration.append(
                stats.f.sf(statistic, dfn=basis.shape[1], dfd=denominator_df)
            )
        random_p_values.append(np.asarray(random_iteration))

        # [5] SHAP 選択後の2種類の推論
        naive_iteration = []
        selective_iteration = []
        feature_seeds = ais_seed.spawn(k_select)
        for position, feature in enumerate(shap_selected):
            # [5b-A] [5a] と共有する中心化 B スプライン効果基底
            basis = _spline_effect_basis(X[:, feature])
            rank = basis.shape[1]

            # [5b-B] [5a] と共有する効果空間への射影と F 統計量
            (
                statistic,
                projected,
                orthogonal,
                mean_component,
                denominator_df,
            ) = _f_statistic(response, basis)

            # [5a] 同じ統計量を使い、選択事象だけを無視する Naive 検定
            naive_iteration.append(
                stats.f.sf(statistic, dfn=rank, dfd=denominator_df)
            )

            # [5b] 選択事象で条件付けた推論
            # [5b-C] 直交成分と効果方向を固定した候補応答
            projected_norm = np.linalg.norm(projected)
            orthogonal_norm = np.linalg.norm(orthogonal)
            centered_norm = np.linalg.norm(response - mean_component)
            zero_tolerance = np.sqrt(np.finfo(float).eps) * max(
                1.0, centered_norm
            )
            if projected_norm <= zero_tolerance or orthogonal_norm <= zero_tolerance:
                raise FloatingPointError(
                    "An observed direction needed for selective inference is "
                    "numerically undefined."
                )
            effect_direction = projected / projected_norm
            residual_direction = orthogonal / orthogonal_norm

            def is_selected(z):
                # Conditional on the sample mean, centered-response norm, and
                # both directions, F is the only random coordinate.  This
                # spherical path eliminates the unknown common noise scale.
                effect_fraction = (rank * z) / (denominator_df + rank * z)
                effect_fraction = np.clip(effect_fraction, 0.0, 1.0)
                candidate = mean_component + centered_norm * (
                    np.sqrt(effect_fraction) * effect_direction
                    + np.sqrt(1.0 - effect_fraction) * residual_direction
                )
                selected = _select_features(
                    X, candidate, k_select, selection_decimals
                )
                return bool(feature in selected)

            if not is_selected(statistic):
                raise RuntimeError(
                    "The observed response was not reconstructed inside its "
                    "selection event."
                )

            # [5b-D] Pilot 適応と診断付き最終 AIS を分離
            p_value, diagnostics = _run_ais(
                statistic,
                rank,
                denominator_df,
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
                    "denominator_df": denominator_df,
                    "t_obs": statistic,
                }
            )
            ais_diagnostics.append(diagnostics)
            selective_iteration.append(p_value)

        naive_p_values.append(np.asarray(naive_iteration))
        selective_p_values.append(np.asarray(selective_iteration))

    # [6] グローバル帰無仮説下の選択仮説当たり棄却率
    random_summary, random_flat = _method_summary(
        "Random", random_p_values, alpha, require_complete=True
    )
    naive_summary, naive_flat = _method_summary(
        "Naive SHAP", naive_p_values, alpha, require_complete=True
    )
    selective_summary, selective_flat = _method_summary(
        "Selective SHAP (AIS)", selective_p_values, alpha, require_complete=True
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
            "noise_scale": float(noise_scale),
            "variance_assumption": "unknown",
        },
    }


def plot_results(results):
    """Plot p-value histograms; failed SI estimates are excluded and reported separately."""
    alpha = results["alpha"]
    summary = results["summary"].set_index("method")
    colors = {"Random": "gray", "Naive SHAP": "blue", "Selective SHAP (AIS)": "green"}

    plt.figure(figsize=(12, 6))
    for method, p_values in results["p_values"].items():
        if p_values.size == 0:
            continue
        fpr = summary.loc[method, "fpr"]
        label_fpr = "unavailable" if not np.isfinite(fpr) else f"{fpr:.3f}"
        plt.hist(
            p_values,
            bins=20,
            range=(0, 1),
            alpha=0.45,
            color=colors[method],
            edgecolor="black",
            label=f"{method} (FPR: {label_fpr})",
        )
    plt.axvline(alpha, color="red", linestyle="--", linewidth=2, label=f"alpha={alpha}")
    plt.title("Unknown-variance F tests: random, naive, and selective inference")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
