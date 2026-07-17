"""Test statistics and adaptive importance sampling for SI."""

from __future__ import annotations

import numpy as np
from patsy import dmatrix
from scipy import stats
from scipy.special import logsumexp


TRUE_SIGMA = 1.0


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


def _chi_statistic(response, basis, sigma=TRUE_SIGMA):
    coordinates = basis.T @ response
    projected = basis @ coordinates
    statistic = np.linalg.norm(coordinates) / sigma
    return statistic, projected


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
    rank,
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
            stats.chi.logpdf(selected_z, df=rank)
            - _truncated_normal_logpdf(selected_z, mean, sd)
        )
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            sd *= 1.5
            continue

        selected_z = selected_z[finite]
        weights = np.exp(log_weights[finite] - np.max(log_weights[finite]))
        new_mean = float(np.average(selected_z, weights=weights))
        new_variance = float(
            np.average((selected_z - new_mean) ** 2, weights=weights)
        )

        mean = 0.5 * mean + 0.5 * new_mean
        sd = max(0.25, 0.5 * sd + 0.5 * np.sqrt(max(new_variance, 0.0)))

    return mean, sd


def _sample_defensive_mixture(rng, size, rank, t_obs, adapted_mean, adapted_sd):
    mixture_weights = np.array([0.25, 0.375, 0.375])
    counts = rng.multinomial(size, mixture_weights)
    observed_sd = max(0.75, 0.25 * t_obs + 0.5)

    samples = [
        stats.chi.rvs(df=rank, size=counts[0], random_state=rng),
        _sample_truncated_normal(rng, t_obs, observed_sd, counts[1]),
        _sample_truncated_normal(rng, adapted_mean, adapted_sd, counts[2]),
    ]
    z_values = np.concatenate(samples)
    rng.shuffle(z_values)

    component_logpdf = np.vstack(
        [
            np.log(mixture_weights[0]) + stats.chi.logpdf(z_values, df=rank),
            np.log(mixture_weights[1])
            + _truncated_normal_logpdf(z_values, t_obs, observed_sd),
            np.log(mixture_weights[2])
            + _truncated_normal_logpdf(z_values, adapted_mean, adapted_sd),
        ]
    )
    return z_values, logsumexp(component_logpdf, axis=0)


def _run_ais(
    t_obs,
    rank,
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
        rank,
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
            rank,
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
                stats.chi.logpdf(selected_z, df=rank) - proposal_logpdf[selected]
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
    mc_variance = (
        np.sum(weights**2 * (tail.astype(float) - p_value) ** 2)
        / np.sum(weights) ** 2
    )
    return np.clip(p_value, 0.0, 1.0), {
        "status": "ok",
        "proposals": proposals,
        "selected_samples": int(all_z.size),
        "tail_samples": int(np.sum(tail)),
        "denominator_ess": denominator_ess,
        "tail_ess": tail_ess,
        "mc_se": float(np.sqrt(mc_variance)),
    }
