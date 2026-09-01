"""Test statistics and adaptive importance sampling for SI."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
from patsy import dmatrix
from scipy import stats
from scipy.special import logsumexp


DEFENSIVE_MIXTURE_WEIGHTS = (0.25, 0.375, 0.375)


def _clopper_pearson_interval(successes, trials, confidence=0.95):
    """Return an exact binomial interval, or NaNs when there are no trials."""
    if trials == 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


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


def _chi_statistic(response, basis, sigma=1.0):
    coordinates = basis.T @ response
    projected = basis @ coordinates
    statistic = np.linalg.norm(coordinates) / sigma
    return statistic, projected


def _quadratic_nonnegative_intervals(coefficients, *, lower=0.0):
    """Return intervals where every supplied quadratic is nonnegative."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] != 3:
        raise ValueError("coefficients must have shape (n_inequalities, 3).")
    if not np.all(np.isfinite(coefficients)) or not np.isfinite(lower):
        raise ValueError("coefficients and lower must be finite.")

    roots = []
    for quadratic, linear, constant in coefficients:
        scale = max(1.0, abs(quadratic), abs(linear), abs(constant))
        tolerance = 64.0 * np.finfo(float).eps * scale
        if abs(quadratic) <= tolerance:
            if abs(linear) > tolerance:
                root = -constant / linear
                if root > lower:
                    roots.append(float(root))
            continue
        discriminant = linear * linear - 4.0 * quadratic * constant
        discriminant_tolerance = 128.0 * np.finfo(float).eps * max(
            1.0, linear * linear, abs(4.0 * quadratic * constant)
        )
        if discriminant < -discriminant_tolerance:
            continue
        square_root = np.sqrt(max(0.0, discriminant))
        q_value = -0.5 * (linear + np.copysign(square_root, linear))
        if q_value == 0.0:
            candidate_roots = (-linear / (2.0 * quadratic),)
        else:
            candidate_roots = (q_value / quadratic, constant / q_value)
        roots.extend(float(root) for root in candidate_roots if root > lower)

    roots.sort()
    unique_roots = []
    for root in roots:
        if not unique_roots or not np.isclose(
            root, unique_roots[-1], rtol=1e-10, atol=1e-12
        ):
            unique_roots.append(root)

    boundaries = [float(lower), *unique_roots, np.inf]
    intervals = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        midpoint = (
            0.5 * (left + right)
            if np.isfinite(right)
            else left + max(1.0, abs(left))
        )
        values = (
            coefficients[:, 0] * midpoint * midpoint
            + coefficients[:, 1] * midpoint
            + coefficients[:, 2]
        )
        scales = np.maximum(
            1.0,
            np.abs(coefficients[:, 0]) * midpoint * midpoint
            + np.abs(coefficients[:, 1]) * midpoint
            + np.abs(coefficients[:, 2]),
        )
        if np.all(values >= -256.0 * np.finfo(float).eps * scales):
            if intervals and np.isclose(
                intervals[-1][1], left, rtol=1e-10, atol=1e-12
            ):
                intervals[-1] = (intervals[-1][0], right)
            else:
                intervals.append((left, right))
    return tuple(intervals)


def _spline_exact_set_intervals(bases, orthogonal, direction, sigma, selected):
    """Compute the exact top-k spline-screening event along a response ray."""
    orthogonal = np.asarray(orthogonal, dtype=float)
    direction = np.asarray(direction, dtype=float)
    selected = tuple(int(feature) for feature in selected)
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("selected must contain unique feature indices.")

    score_coefficients = []
    for basis in bases:
        if basis is None:
            score_coefficients.append(np.zeros(3, dtype=float))
            continue
        offset = np.asarray(basis, dtype=float).T @ orthogonal
        slope = float(sigma) * (np.asarray(basis, dtype=float).T @ direction)
        score_coefficients.append(
            np.array(
                [
                    float(slope @ slope),
                    float(2.0 * (offset @ slope)),
                    float(offset @ offset),
                ]
            )
        )
    score_coefficients = np.asarray(score_coefficients)
    unselected = [
        feature for feature in range(len(bases)) if feature not in selected_set
    ]
    inequalities = [
        score_coefficients[selected_feature] - score_coefficients[other_feature]
        for selected_feature in selected
        for other_feature in unselected
    ]
    if not inequalities:
        return ((0.0, np.inf),)
    return _quadratic_nonnegative_intervals(inequalities, lower=0.0)


def _chi_interval_probability(lower, upper, rank):
    """Evaluate chi probability on one interval without avoidable cancellation."""
    lower = max(0.0, float(lower))
    upper = float(upper)
    if upper <= lower:
        return 0.0
    if np.isposinf(upper):
        return float(stats.chi.sf(lower, df=rank))
    median = float(stats.chi.ppf(0.5, df=rank))
    if lower >= median:
        probability = stats.chi.sf(lower, df=rank) - stats.chi.sf(upper, df=rank)
    else:
        probability = stats.chi.cdf(upper, df=rank) - stats.chi.cdf(lower, df=rank)
    return float(np.clip(probability, 0.0, 1.0))


def _run_exact_spline(t_obs, rank, intervals):
    """Return the exact truncated-chi p-value for spline exact-set selection."""
    denominator = sum(
        _chi_interval_probability(lower, upper, rank)
        for lower, upper in intervals
    )
    numerator = sum(
        _chi_interval_probability(max(lower, t_obs), upper, rank)
        for lower, upper in intervals
        if upper >= t_obs
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("The exact spline selection event has zero probability.")
    p_value = float(np.clip(numerator / denominator, 0.0, 1.0))
    return p_value, {
        "status": "ok",
        "resolution_status": "exact_selection_region",
        "proposals": 0,
        "selected_samples": np.nan,
        "tail_samples": np.nan,
        "denominator_ess": np.inf,
        "tail_ess": np.inf,
        "mc_se": 0.0,
        "mc_ci_95_lower": p_value,
        "mc_ci_95_upper": p_value,
        "tail_probability_mc_ci_95_lower": p_value,
        "tail_probability_mc_ci_95_upper": p_value,
        "selection_probability_estimate": float(denominator),
        "selection_interval_count": len(intervals),
        "sampling_mode": "exact_truncated_chi",
        "p_value_method": "exact_spline_selection_region",
        "finite_sample_valid": True,
    }


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


def _defensive_mixture_logpdf(z, rank, t_obs, adapted_mean, adapted_sd):
    """Evaluate the proposal density used by the final AIS stage."""
    z = np.asarray(z)
    observed_sd = max(0.75, 0.25 * t_obs + 0.5)
    component_logpdf = np.vstack(
        [
            np.log(DEFENSIVE_MIXTURE_WEIGHTS[0])
            + stats.chi.logpdf(z, df=rank),
            np.log(DEFENSIVE_MIXTURE_WEIGHTS[1])
            + _truncated_normal_logpdf(z, t_obs, observed_sd),
            np.log(DEFENSIVE_MIXTURE_WEIGHTS[2])
            + _truncated_normal_logpdf(z, adapted_mean, adapted_sd),
        ]
    )
    return logsumexp(component_logpdf, axis=0)


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
    mixture_weights = np.asarray(DEFENSIVE_MIXTURE_WEIGHTS)
    counts = rng.multinomial(size, mixture_weights)
    observed_sd = max(0.75, 0.25 * t_obs + 0.5)

    samples = [
        stats.chi.rvs(df=rank, size=counts[0], random_state=rng),
        _sample_truncated_normal(rng, t_obs, observed_sd, counts[1]),
        _sample_truncated_normal(rng, adapted_mean, adapted_sd, counts[2]),
    ]
    z_values = np.concatenate(samples)
    rng.shuffle(z_values)

    proposal_logpdf = _defensive_mixture_logpdf(
        z_values,
        rank,
        t_obs,
        adapted_mean,
        adapted_sd,
    )
    return z_values, proposal_logpdf


def _run_conditional_mc(
    t_obs,
    rank,
    is_selected,
    rng,
    *,
    batch_size=80,
    n_proposals=800,
):
    """Return a finite-sample-valid conditional Monte Carlo rank p-value.

    Proposals are i.i.d. from the null chi distribution.  Conditional on the
    number that reproduce the selection event, the retained radii and the
    observed radius are exchangeable under the selective null.  Therefore

        (1 + number of retained radii >= t_obs) / (1 + retained radii)

    is a (possibly conservative) exact Monte Carlo p-value.  A rare event does
    not create a missing p-value: no retained draws gives the valid value one.
    """
    selected_samples = 0
    tail_samples = 0
    proposals = 0
    while proposals < n_proposals:
        current_size = min(batch_size, n_proposals - proposals)
        z_values = stats.chi.rvs(df=rank, size=current_size, random_state=rng)
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values),
            dtype=bool,
            count=current_size,
        )
        retained = z_values[selected]
        selected_samples += int(retained.size)
        tail_samples += int(np.sum(retained >= t_obs))
        proposals += current_size

    p_value = float((tail_samples + 1.0) / (selected_samples + 1.0))
    minimum_attainable_p_value = float(1.0 / (selected_samples + 1.0))
    if selected_samples:
        tail_fraction = tail_samples / selected_samples
        mc_se = float(
            np.sqrt(tail_fraction * (1.0 - tail_fraction) / selected_samples)
        )
    else:
        mc_se = np.nan
    ci_lower, ci_upper = _clopper_pearson_interval(
        tail_samples, selected_samples
    )
    return p_value, {
        "status": "ok",
        "resolution_status": (
            "resolved" if selected_samples else "no_selected_mc_draws_p_equals_one"
        ),
        "proposals": proposals,
        "selected_samples": selected_samples,
        "tail_samples": tail_samples,
        "minimum_attainable_p_value": minimum_attainable_p_value,
        "denominator_ess": float(selected_samples),
        "tail_ess": float(tail_samples),
        "mc_se": mc_se,
        "mc_ci_95_lower": ci_lower,
        "mc_ci_95_upper": ci_upper,
        "tail_probability_mc_ci_95_lower": ci_lower,
        "tail_probability_mc_ci_95_upper": ci_upper,
        "selection_probability_estimate": selected_samples / proposals,
        "sampling_mode": "fixed_budget_conditional_mc",
        "p_value_method": "conditional_monte_carlo_rank",
        "finite_sample_valid": True,
    }


def _mcmc_transition(
    initial,
    rank,
    is_selected,
    rng,
    *,
    n_steps,
    step_scales,
    global_jump_probability,
):
    """Apply a reversible kernel targeting chi(rank) restricted to selection."""
    current = float(initial)
    current_log_density = float(stats.chi.logpdf(current, df=rank))
    accepted = 0
    for _ in range(n_steps):
        if rng.random() < global_jump_probability:
            proposal = float(stats.chi.rvs(df=rank, random_state=rng))
            if is_selected(proposal):
                current = proposal
                current_log_density = float(
                    stats.chi.logpdf(current, df=rank)
                )
                accepted += 1
            continue

        scale = float(step_scales[rng.integers(len(step_scales))])
        proposal = abs(float(current + rng.normal(scale=scale)))
        if not is_selected(proposal):
            continue
        proposal_log_density = float(stats.chi.logpdf(proposal, df=rank))
        if np.log(rng.random()) <= min(
            0.0, proposal_log_density - current_log_density
        ):
            current = proposal
            current_log_density = proposal_log_density
            accepted += 1
    return current, accepted


def _run_mcmc_rank(
    t_obs,
    rank,
    is_selected,
    rng,
    *,
    n_replicates=199,
    n_steps=20,
    step_scales=(0.25, 0.5, 1.0, 2.0),
    global_jump_probability=0.1,
):
    """Return an exact parallel reversible-MCMC rank p-value.

    Let ``K`` be the reversible transition implemented above and let
    ``H ~ K^L(t_obs, .)``.  Reversibility implies that, under the selective
    null, ``t_obs | H`` also has law ``K^L(H, .)``.  Replicates independently
    drawn from that same law are therefore exchangeable with the observation
    conditional on ``H``.  Their conservative upper rank is a finite-sample
    valid p-value even when the selection event is extremely rare under the
    unconditional chi law.  Mixing affects power, not validity.
    """
    if (
        isinstance(n_replicates, (bool, np.bool_))
        or not isinstance(n_replicates, (int, np.integer))
        or n_replicates < 1
    ):
        raise ValueError("n_replicates must be a positive integer.")
    if (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, (int, np.integer))
        or n_steps < 1
    ):
        raise ValueError("n_steps must be a positive integer.")
    scales = np.asarray(step_scales, dtype=float)
    if scales.ndim != 1 or scales.size == 0 or not np.all(
        np.isfinite(scales) & (scales > 0.0)
    ):
        raise ValueError("step_scales must contain finite positive values.")
    if (
        not np.isscalar(global_jump_probability)
        or not np.isfinite(global_jump_probability)
        or not 0.0 <= float(global_jump_probability) <= 1.0
    ):
        raise ValueError("global_jump_probability must lie in [0, 1].")
    if not is_selected(float(t_obs)):
        raise ValueError("t_obs must satisfy the selection event.")

    transition_kwargs = {
        "n_steps": int(n_steps),
        "step_scales": scales,
        "global_jump_probability": float(global_jump_probability),
    }
    hub, hub_accepted = _mcmc_transition(
        t_obs,
        rank,
        is_selected,
        rng,
        **transition_kwargs,
    )
    replicates = np.empty(int(n_replicates), dtype=float)
    replicate_acceptances = 0
    for index in range(int(n_replicates)):
        replicates[index], accepted = _mcmc_transition(
            hub,
            rank,
            is_selected,
            rng,
            **transition_kwargs,
        )
        replicate_acceptances += accepted

    tail_samples = int(np.sum(replicates >= t_obs))
    p_value = float((tail_samples + 1.0) / (n_replicates + 1.0))
    tail_fraction = tail_samples / n_replicates
    mc_se = float(
        np.sqrt(tail_fraction * (1.0 - tail_fraction) / n_replicates)
    )
    ci_lower, ci_upper = _clopper_pearson_interval(
        tail_samples, int(n_replicates)
    )
    moved_replicas = int(np.sum(replicates != hub))
    total_steps = int((n_replicates + 1) * n_steps)
    return p_value, {
        "status": "ok",
        "resolution_status": (
            "resolved" if moved_replicas else "no_moved_mcmc_replicas_p_equals_one"
        ),
        "proposals": total_steps,
        "selected_samples": int(n_replicates),
        "tail_samples": tail_samples,
        "minimum_attainable_p_value": float(1.0 / (n_replicates + 1.0)),
        "denominator_ess": float(n_replicates),
        "tail_ess": float(tail_samples),
        "mc_se": mc_se,
        "mc_ci_95_lower": ci_lower,
        "mc_ci_95_upper": ci_upper,
        "tail_probability_mc_ci_95_lower": ci_lower,
        "tail_probability_mc_ci_95_upper": ci_upper,
        "selection_probability_estimate": np.nan,
        "sampling_mode": "parallel_reversible_mcmc",
        "p_value_method": "parallel_reversible_mcmc_rank",
        "finite_sample_valid": True,
        "mcmc_hub": float(hub),
        "mcmc_steps": int(n_steps),
        "mcmc_replicates": int(n_replicates),
        "mcmc_hub_acceptance_rate": float(hub_accepted / n_steps),
        "mcmc_replicate_acceptance_rate": float(
            replicate_acceptances / (n_replicates * n_steps)
        ),
        "mcmc_moved_replicas": moved_replicas,
        "mcmc_unique_replicas": int(np.unique(replicates).size),
        "mcmc_global_jump_probability": float(global_jump_probability),
        "mcmc_step_scales": tuple(float(value) for value in scales),
    }


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
    stop_when_ess_met=False,
):
    """Estimate the selected chi-tail ratio with a defensive proposal.

    This exploratory estimator uses the full, predeclared final-sample budget.
    ``stop_when_ess_met=True`` retains the former exploratory early-stopping
    behavior, but its data-dependent sample size should not be used as evidence
    that the resulting estimate is an exact finite-sample p-value.
    """
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
        if (
            stop_when_ess_met
            and denominator_ess >= min_denominator_ess
            and tail_ess >= min_tail_ess
        ):
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
            "mc_ci_95_lower": np.nan,
            "mc_ci_95_upper": np.nan,
            "sampling_mode": (
                "ess_early_stopping" if stop_when_ess_met else "fixed_budget"
            ),
            "p_value_method": "self_normalized_importance_sampling_estimate",
            "finite_sample_valid": False,
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
            "mc_ci_95_lower": np.nan,
            "mc_ci_95_upper": np.nan,
            "sampling_mode": (
                "ess_early_stopping" if stop_when_ess_met else "fixed_budget"
            ),
            "p_value_method": "self_normalized_importance_sampling_estimate",
            "finite_sample_valid": False,
        }

    p_value = float(np.sum(weights * tail) / np.sum(weights))
    mc_variance = (
        np.sum(weights**2 * (tail.astype(float) - p_value) ** 2)
        / np.sum(weights) ** 2
    )
    p_value = float(np.clip(p_value, 0.0, 1.0))
    mc_se = float(np.sqrt(mc_variance))
    return p_value, {
        "status": "ok",
        "proposals": proposals,
        "selected_samples": int(all_z.size),
        "tail_samples": int(np.sum(tail)),
        "denominator_ess": denominator_ess,
        "tail_ess": tail_ess,
        "mc_se": mc_se,
        # This is a normal-approximation Monte Carlo interval, not a formal
        # selective-inference confidence bound for a self-normalized ratio.
        "mc_ci_95_lower": float(max(0.0, p_value - 1.96 * mc_se)),
        "mc_ci_95_upper": float(min(1.0, p_value + 1.96 * mc_se)),
        "sampling_mode": (
            "ess_early_stopping" if stop_when_ess_met else "fixed_budget"
        ),
        "p_value_method": "self_normalized_importance_sampling_estimate",
        "finite_sample_valid": False,
    }
