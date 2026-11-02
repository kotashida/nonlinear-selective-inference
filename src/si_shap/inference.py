"""Test statistics and adaptive importance sampling for SI."""

from __future__ import annotations

import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
from patsy import dmatrix
from scipy import optimize, stats
from scipy.special import logsumexp


DEFENSIVE_MIXTURE_WEIGHTS = (0.25, 0.375, 0.375)
ABLATION_DENOMINATOR_MIXTURE_WEIGHTS = (0.25, 0.30, 0.20, 0.25)
ABLATION_CLUSTER_UNIFORM_MASS = 0.35
ROBUST_CLUSTER_UNIFORM_MASS = 0.20
ROBUST_MAX_COMPONENTS = 64
ROBUST_MIN_NONSELECTION_RUN = 3
ROBUST_COMPONENT_DRAW_FLOOR = 128
REDESIGNED_DENOMINATOR_MIXTURE_WEIGHTS = (0.25, 0.20, 0.375, 0.175)
AIS_STRATEGIES = (
    "baseline",
    "multimodal_qadapt",
    "raised_sd_floor",
    "adaptive_mixture_weights",
    "reallocated_pilot_exploration",
    "ablation_a",
    "ablation_b",
    "ablation_c",
    "ablation_ab",
    "ablation_ac",
    "ablation_bc",
    "ablation_abc",
    "a_robust_consolidation",
    "a_robust_draw_floor",
    "a_robust_draw_floor_pilot15",
    "a_cdf_interval",
    "a_robust_floor_multimodal_denominator",
    "a_full_stack_optimized",
    "a_robust_draw_floor_tail_stable",
)

AIS_ABLATION_MECHANISMS = {
    "ablation_a": (True, False, False),
    "ablation_b": (False, True, False),
    "ablation_c": (False, False, True),
    "ablation_ab": (True, True, False),
    "ablation_ac": (True, False, True),
    "ablation_bc": (False, True, True),
    "ablation_abc": (True, True, True),
}

AIS_REFINEMENT_STRATEGIES = {
    "a_robust_draw_floor_tail_stable": {
        "draw_floor": ROBUST_COMPONENT_DRAW_FLOOR,
        "pilot15": False, "interval": False, "denominator": False,
        "optimize": False, "tail_stable": True,
    },
    "a_robust_consolidation": {
        "draw_floor": 0,
        "pilot15": False,
        "interval": False,
        "denominator": False,
        "optimize": False,
    },
    "a_robust_draw_floor": {
        "draw_floor": ROBUST_COMPONENT_DRAW_FLOOR,
        "pilot15": False,
        "interval": False,
        "denominator": False,
        "optimize": False,
    },
    "a_robust_draw_floor_pilot15": {
        "draw_floor": ROBUST_COMPONENT_DRAW_FLOOR,
        "pilot15": True,
        "interval": False,
        "denominator": False,
        "optimize": False,
    },
    "a_cdf_interval": {
        "draw_floor": 0,
        "pilot15": False,
        "interval": True,
        "denominator": False,
        "optimize": False,
    },
    "a_robust_floor_multimodal_denominator": {
        "draw_floor": ROBUST_COMPONENT_DRAW_FLOOR,
        "pilot15": False,
        "interval": False,
        "denominator": True,
        "optimize": False,
    },
    "a_full_stack_optimized": {
        "draw_floor": ROBUST_COMPONENT_DRAW_FLOOR,
        "pilot15": True,
        "interval": False,
        "denominator": True,
        "optimize": True,
    },
}


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
    sd_floor=0.25,
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
        sd = max(sd_floor, 0.5 * sd + 0.5 * np.sqrt(max(new_variance, 0.0)))

    return mean, sd


def _pilot_proposal_logpdf(z, components):
    """Density of a weighted truncated-normal/chi pilot mixture."""
    terms = []
    for kind, weight, mean, sd, rank in components:
        log_density = (
            stats.chi.logpdf(z, df=rank)
            if kind == "chi"
            else _truncated_normal_logpdf(z, mean, sd)
        )
        terms.append(np.log(weight) + log_density)
    return logsumexp(np.vstack(terms), axis=0)


def _adapt_multimodal_proposal(
    t_obs, rank, is_selected, rng, pilot_iters, pilot_samples
):
    """Fit a gap-clustered truncated-normal mixture to all accepted pilots."""
    mean = float(t_obs)
    sd = max(1.0, 0.25 * t_obs + 0.5)
    accepted_values = []
    accepted_log_weights = []
    for _ in range(pilot_iters):
        z_values = _sample_truncated_normal(rng, mean, sd, pilot_samples)
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values), bool, pilot_samples
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
        log_weights = log_weights[finite]
        local_weights = np.exp(log_weights - np.max(log_weights))
        new_mean = float(np.average(selected_z, weights=local_weights))
        variance = float(np.average((selected_z - new_mean) ** 2, weights=local_weights))
        mean = 0.5 * mean + 0.5 * new_mean
        sd = max(0.25, 0.5 * sd + 0.5 * np.sqrt(max(variance, 0.0)))
        accepted_values.append(selected_z)
        accepted_log_weights.append(log_weights)

    if not accepted_values:
        return ((mean, sd, 1.0),)
    values = np.concatenate(accepted_values)
    log_weights = np.concatenate(accepted_log_weights)
    order = np.argsort(values)
    values = values[order]
    log_weights = log_weights[order]
    gaps = np.diff(values)
    positive_gaps = gaps[gaps > 0.0]
    typical_gap = float(np.median(positive_gaps)) if positive_gaps.size else 0.0
    gap_threshold = max(0.15, 6.0 * typical_gap)
    groups = list(np.split(np.arange(values.size), np.flatnonzero(gaps > gap_threshold) + 1))
    # Limit evaluation cost while preserving the largest empirical gaps.
    while len(groups) > 8:
        join = int(np.argmin([values[b[0]] - values[a[-1]] for a, b in zip(groups, groups[1:])]))
        groups[join] = np.concatenate((groups[join], groups[join + 1]))
        del groups[join + 1]

    shifted = np.exp(log_weights - np.max(log_weights))
    fitted = []
    masses = []
    for group in groups:
        group_weights = shifted[group]
        group_values = values[group]
        cluster_mean = float(np.average(group_values, weights=group_weights))
        variance = float(
            np.average((group_values - cluster_mean) ** 2, weights=group_weights)
        )
        fitted.append((cluster_mean, max(0.25, np.sqrt(max(variance, 0.0)))))
        masses.append(float(np.sum(group_weights)))
    masses = np.asarray(masses) / np.sum(masses)
    return tuple((mean, sd, float(weight)) for (mean, sd), weight in zip(fitted, masses))


def _adapt_reallocated_pilot(
    t_obs, rank, is_selected, rng, pilot_iters, pilot_samples
):
    """Reuse the fixed pilot budget across local, chi, and lower-z exploration."""
    mean = float(t_obs)
    sd = max(1.0, 0.25 * t_obs + 0.5)
    for _ in range(pilot_iters):
        n_local = int(round(0.4 * pilot_samples))
        n_chi = int(round(0.3 * pilot_samples))
        n_lower = pilot_samples - n_local - n_chi
        lower_centers = np.geomspace(max(0.25, min(1.0, t_obs / 8.0)), max(0.3, 0.8 * t_obs), 3)
        lower_counts = np.full(3, n_lower // 3, dtype=int)
        lower_counts[: n_lower % 3] += 1
        components = [
            ("tn", n_local / pilot_samples, mean, sd, rank),
            ("chi", n_chi / pilot_samples, 0.0, 1.0, rank),
        ]
        chunks = [_sample_truncated_normal(rng, mean, sd, n_local)]
        chunks.append(stats.chi.rvs(df=rank, size=n_chi, random_state=rng))
        for center, count in zip(lower_centers, lower_counts):
            lower_sd = max(0.75, 0.25 * float(center) + 0.5)
            components.append(("tn", count / pilot_samples, float(center), lower_sd, rank))
            chunks.append(_sample_truncated_normal(rng, float(center), lower_sd, int(count)))
        z_values = np.concatenate(chunks)
        rng.shuffle(z_values)
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values), bool, pilot_samples
        )
        if not np.any(selected):
            sd *= 1.5
            continue
        selected_z = z_values[selected]
        log_weights = (
            stats.chi.logpdf(selected_z, df=rank)
            - _pilot_proposal_logpdf(selected_z, components)
        )
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            sd *= 1.5
            continue
        selected_z = selected_z[finite]
        weights = np.exp(log_weights[finite] - np.max(log_weights[finite]))
        new_mean = float(np.average(selected_z, weights=weights))
        variance = float(np.average((selected_z - new_mean) ** 2, weights=weights))
        mean = 0.5 * mean + 0.5 * new_mean
        sd = max(0.25, 0.5 * sd + 0.5 * np.sqrt(max(variance, 0.0)))
    return mean, sd


def _randomized_chi_strata(rng, size, rank):
    """Draw one randomized point from every equal-probability chi stratum."""
    if size < 1:
        return np.array([], dtype=float)
    uniforms = (np.arange(size, dtype=float) + rng.random(size)) / size
    rng.shuffle(uniforms)
    return _finite_chi_ppf(uniforms, rank)


def _finite_chi_ppf(probabilities, rank):
    """Evaluate chi quantiles strictly inside (0, 1) and require finiteness."""
    epsilon = np.finfo(float).eps
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon
    )
    values = stats.chi.ppf(probabilities, df=rank)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Chi inverse-CDF sampling produced a nonfinite draw.")
    return values


def _weighted_location_scale(values, log_weights, *, sd_floor=0.05):
    """Return stable importance-weighted truncated-normal parameters."""
    values = np.asarray(values, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError("At least one finite weighted value is required.")
    values = values[finite]
    log_weights = log_weights[finite]
    weights = np.exp(log_weights - np.max(log_weights))
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, max(float(sd_floor), np.sqrt(max(variance, 0.0)))


def _pilot_island_components(
    z_values,
    selected,
    selected_log_weights,
    rank,
    *,
    uniform_mass=ABLATION_CLUSTER_UNIFORM_MASS,
):
    """Fit one component per pilot-discovered island in chi-CDF space.

    A nonselected pilot point between selected points certifies a discovered
    separation. Large unobserved CDF gaps are also split so that sparse pilot
    coverage cannot spuriously join distant islands.
    """
    z_values = np.asarray(z_values, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    selected_log_weights = np.asarray(selected_log_weights, dtype=float)
    if z_values.shape != selected.shape or z_values.shape != selected_log_weights.shape:
        raise ValueError("Pilot values, states, and log weights must have equal shape.")
    finite = np.isfinite(z_values)
    if not np.any(finite & selected & np.isfinite(selected_log_weights)):
        return ()

    z_values = z_values[finite]
    selected = selected[finite]
    selected_log_weights = selected_log_weights[finite]
    cdf_values = stats.chi.cdf(z_values, df=rank)
    order = np.argsort(cdf_values, kind="stable")
    z_values = z_values[order]
    cdf_values = cdf_values[order]
    selected = selected[order]
    selected_log_weights = selected_log_weights[order]

    positive_gaps = np.diff(cdf_values)
    positive_gaps = positive_gaps[positive_gaps > 0.0]
    typical_gap = float(np.median(positive_gaps)) if positive_gaps.size else 0.0
    gap_threshold = max(0.01, 8.0 * typical_gap)
    selected_indices = np.flatnonzero(selected & np.isfinite(selected_log_weights))
    groups = []
    start = 0
    for offset in range(1, selected_indices.size):
        left = selected_indices[offset - 1]
        right = selected_indices[offset]
        separated_by_state = np.any(~selected[left + 1 : right])
        separated_by_gap = cdf_values[right] - cdf_values[left] > gap_threshold
        if separated_by_state or separated_by_gap:
            groups.append(selected_indices[start:offset])
            start = offset
    groups.append(selected_indices[start:])

    fitted = []
    log_masses = []
    for group in groups:
        group_values = z_values[group]
        group_log_weights = selected_log_weights[group]
        mean, sd = _weighted_location_scale(
            group_values, group_log_weights, sd_floor=0.05
        )
        if group_values.size == 1:
            index = int(group[0])
            neighbor_distances = []
            if index:
                neighbor_distances.append(z_values[index] - z_values[index - 1])
            if index + 1 < z_values.size:
                neighbor_distances.append(z_values[index + 1] - z_values[index])
            if neighbor_distances:
                sd = max(sd, 0.5 * min(neighbor_distances))
        fitted.append((mean, sd))
        log_masses.append(float(logsumexp(group_log_weights)))

    masses = np.exp(np.asarray(log_masses) - logsumexp(log_masses))
    masses = (1.0 - uniform_mass) * masses + uniform_mass / len(fitted)
    return tuple(
        (mean, sd, float(mass))
        for (mean, sd), mass in zip(fitted, masses)
    )


def _robust_pilot_islands(
    z_values,
    selected,
    selected_log_weights,
    rank,
    *,
    max_components=ROBUST_MAX_COMPONENTS,
    min_nonselection_run=ROBUST_MIN_NONSELECTION_RUN,
    uniform_mass=ROBUST_CLUSTER_UNIFORM_MASS,
    sort_by_z=False,
    tail_stable=False,
):
    """Consolidate pilot states into stable chi-CDF selection islands."""
    z_values = np.asarray(z_values, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    selected_log_weights = np.asarray(selected_log_weights, dtype=float)
    finite = np.isfinite(z_values)
    z_values = z_values[finite]
    selected = selected[finite]
    selected_log_weights = selected_log_weights[finite]
    if not np.any(selected & np.isfinite(selected_log_weights)):
        return (), 0

    u_values = stats.chi.cdf(z_values, df=rank)
    # CDF saturation creates ties at one; sorting those ties by input order
    # scrambles the tail and can reverse boundary brackets. z remains ordered.
    order = np.argsort(z_values if sort_by_z or tail_stable else u_values, kind="stable")
    z_values = z_values[order]
    u_values = u_values[order]
    selected = selected[order]
    selected_log_weights = selected_log_weights[order]
    sf_values = stats.chi.sf(z_values, df=rank) if tail_stable else None
    selected_indices = np.flatnonzero(selected & np.isfinite(selected_log_weights))
    pre_merge_count = 1
    robust_groups = []
    start = 0
    minimum_gap = max(0.002, 2.0 / max(1, u_values.size))
    for offset in range(1, selected_indices.size):
        left = int(selected_indices[offset - 1])
        right = int(selected_indices[offset])
        nonselected = int(np.sum(~selected[left + 1 : right]))
        if nonselected:
            pre_merge_count += 1
        gap = (sf_values[left] - sf_values[right]
               if tail_stable and u_values[left] > 0.5
               else u_values[right] - u_values[left])
        sustained = (
            nonselected >= min_nonselection_run
            and gap >= minimum_gap
        )
        if sustained:
            robust_groups.append(selected_indices[start:offset])
            start = offset
    robust_groups.append(selected_indices[start:])

    # Merge adjacent clusters whose fitted proposal cores substantially overlap.
    merged = []
    for group in robust_groups:
        if not merged:
            merged.append(group)
            continue
        left = merged[-1]
        left_mean, left_sd = _weighted_location_scale(
            z_values[left], selected_log_weights[left], sd_floor=0.05
        )
        right_mean, right_sd = _weighted_location_scale(
            z_values[group], selected_log_weights[group], sd_floor=0.05
        )
        if right_mean - left_mean <= 0.5 * (left_sd + right_sd):
            merged[-1] = np.concatenate((left, group))
        else:
            merged.append(group)
    robust_groups = merged

    # Cap fragmentation by merging the closest adjacent islands first.
    while len(robust_groups) > max_components:
        gaps = [
            u_values[right[0]] - u_values[left[-1]]
            for left, right in zip(robust_groups, robust_groups[1:])
        ]
        join = int(np.argmin(gaps))
        robust_groups[join] = np.concatenate(
            (robust_groups[join], robust_groups[join + 1])
        )
        del robust_groups[join + 1]

    islands = []
    log_masses = []
    for group in robust_groups:
        first = int(group[0])
        last = int(group[-1])
        lower_candidates = np.flatnonzero(~selected[:first])
        upper_candidates = np.flatnonzero(~selected[last + 1 :])
        lower_u = (
            0.5 * (u_values[lower_candidates[-1]] + u_values[first])
            if lower_candidates.size
            else max(0.0, u_values[first] - minimum_gap)
        )
        upper_u = (
            0.5 * (u_values[last] + u_values[last + 1 + upper_candidates[0]])
            if upper_candidates.size
            else min(1.0, u_values[last] + minimum_gap)
        )
        group_log_weights = selected_log_weights[group]
        mean, sd = _weighted_location_scale(
            z_values[group], group_log_weights, sd_floor=0.05
        )
        epsilon = np.finfo(float).eps
        lower_z = (
            0.5 * (z_values[lower_candidates[-1]] + z_values[first])
            if lower_candidates.size else max(0.0, z_values[first] - sd)
        )
        upper_z = (
            0.5 * (z_values[last] + z_values[last + 1 + upper_candidates[0]])
            if upper_candidates.size else z_values[last] + sd
        )
        islands.append(
            {
                "lower_u": float(np.clip(lower_u, epsilon, 1.0 - epsilon)),
                "upper_u": float(np.clip(upper_u, epsilon, 1.0 - epsilon)),
                "mean": mean,
                "sd": sd,
                "lower_z": float(lower_z),
                "upper_z": float(upper_z),
            }
        )
        log_masses.append(float(logsumexp(group_log_weights)))

    masses = np.exp(np.asarray(log_masses) - logsumexp(log_masses))
    masses = (1.0 - uniform_mass) * masses + uniform_mass / len(islands)
    for island, mass in zip(islands, masses):
        island["mass"] = float(mass)
    return tuple(islands), pre_merge_count


def _boundary_pilot_draws(rng, size, rank, is_selected, boundaries):
    """Sample only around pilot-bracketed transitions with known proposal density."""
    if size < 1:
        empty = np.array([], dtype=float)
        return empty, np.array([], dtype=bool), empty
    intervals = []
    for boundary in boundaries:
        lower = max(1e-9, float(boundary) - 0.005)
        upper = min(1.0 - 1e-9, float(boundary) + 0.005)
        if upper > lower:
            intervals.append((lower, upper))
    if not intervals:
        intervals = [(1e-9, 1.0 - 1e-9)]

    labels = np.arange(size, dtype=int) % len(intervals)
    rng.shuffle(labels)
    uniforms = np.empty(size, dtype=float)
    for label, (lower, upper) in enumerate(intervals):
        mask = labels == label
        uniforms[mask] = rng.uniform(lower, upper, int(np.sum(mask)))
    proposal_u = np.zeros(size, dtype=float)
    for lower, upper in intervals:
        proposal_u += (
            ((uniforms >= lower) & (uniforms <= upper)).astype(float)
            / (len(intervals) * (upper - lower))
        )
    z_values = _finite_chi_ppf(uniforms, rank)
    selected = np.fromiter(
        (is_selected(float(z)) for z in z_values), bool, size
    )
    log_weights = -np.log(proposal_u)
    log_weights[~selected | ~np.isfinite(log_weights)] = np.nan
    return z_values, selected, log_weights


def _prepare_refinement_pilot(
    t_obs,
    rank,
    is_selected,
    rng,
    pilot_iters,
    pilot_samples,
    *,
    pilot15,
):
    """Build the A pilot, reserving 1.5x extras for refinement/validation."""
    discovery_iters = min(10, pilot_iters) if pilot15 else pilot_iters
    pilot = _adapt_ablation_pilot(
        t_obs,
        rank,
        is_selected,
        rng,
        discovery_iters,
        pilot_samples,
        stratified=False,
    )
    pilot["fit_mask"] = np.ones(pilot["z_values"].size, dtype=bool)
    pilot["validation_mask"] = np.zeros(pilot["z_values"].size, dtype=bool)
    if not pilot15 or pilot_iters <= discovery_iters:
        return pilot

    preliminary, _ = _robust_pilot_islands(
        pilot["z_values"], pilot["selected"], pilot["log_weights"], rank
    )
    boundaries = [
        boundary
        for island in preliminary
        for boundary in (island["lower_u"], island["upper_u"])
        if 0.0 < boundary < 1.0
    ]
    extra = (pilot_iters - discovery_iters) * pilot_samples
    refinement_size = extra // 2
    validation_size = extra - refinement_size
    refinement = _boundary_pilot_draws(
        rng, refinement_size, rank, is_selected, boundaries
    )
    validation = _boundary_pilot_draws(
        rng, validation_size, rank, is_selected, boundaries
    )
    pilot["z_values"] = np.concatenate(
        (pilot["z_values"], refinement[0], validation[0])
    )
    pilot["selected"] = np.concatenate(
        (pilot["selected"], refinement[1], validation[1])
    )
    pilot["log_weights"] = np.concatenate(
        (pilot["log_weights"], refinement[2], validation[2])
    )
    pilot["fit_mask"] = np.concatenate(
        (
            pilot["fit_mask"],
            np.ones(refinement_size, dtype=bool),
            np.zeros(validation_size, dtype=bool),
        )
    )
    pilot["validation_mask"] = np.concatenate(
        (
            pilot["validation_mask"],
            np.zeros(refinement_size, dtype=bool),
            np.ones(validation_size, dtype=bool),
        )
    )
    pilot["sampling_scheme"] += "_boundary_refinement_validation"
    return pilot


def _adapt_ablation_pilot(
    t_obs,
    rank,
    is_selected,
    rng,
    pilot_iters,
    pilot_samples,
    *,
    stratified,
):
    """Run the common reallocated pilot and retain every evaluated draw."""
    mean = float(t_obs)
    sd = max(1.0, 0.25 * t_obs + 0.5)
    if stratified:
        n_local = int(round(0.4 * pilot_samples))
        stratified_draws = _randomized_chi_strata(
            rng, pilot_iters * (pilot_samples - n_local), rank
        )
    else:
        stratified_draws = np.array([], dtype=float)

    all_values = []
    all_selected = []
    all_log_weights = []
    stratified_offset = 0
    for _ in range(pilot_iters):
        n_local = int(round(0.4 * pilot_samples))
        local_values = _sample_truncated_normal(rng, mean, sd, n_local)
        if stratified:
            n_chi = pilot_samples - n_local
            chi_values = stratified_draws[
                stratified_offset : stratified_offset + n_chi
            ]
            stratified_offset += n_chi
            components = [
                ("tn", n_local / pilot_samples, mean, sd, rank),
                ("chi", n_chi / pilot_samples, 0.0, 1.0, rank),
            ]
            z_values = np.concatenate((local_values, chi_values))
        else:
            n_chi = int(round(0.3 * pilot_samples))
            n_lower = pilot_samples - n_local - n_chi
            lower_centers = np.geomspace(
                max(0.25, min(1.0, t_obs / 8.0)), max(0.3, 0.8 * t_obs), 3
            )
            lower_counts = np.full(3, n_lower // 3, dtype=int)
            lower_counts[: n_lower % 3] += 1
            components = [
                ("tn", n_local / pilot_samples, mean, sd, rank),
                ("chi", n_chi / pilot_samples, 0.0, 1.0, rank),
            ]
            chunks = [
                local_values,
                stats.chi.rvs(df=rank, size=n_chi, random_state=rng),
            ]
            for center, count in zip(lower_centers, lower_counts):
                lower_sd = max(0.75, 0.25 * float(center) + 0.5)
                components.append(
                    ("tn", count / pilot_samples, float(center), lower_sd, rank)
                )
                chunks.append(
                    _sample_truncated_normal(
                        rng, float(center), lower_sd, int(count)
                    )
                )
            z_values = np.concatenate(chunks)
        rng.shuffle(z_values)
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values), bool, pilot_samples
        )
        proposal_logpdf = _pilot_proposal_logpdf(z_values, components)
        log_weights = stats.chi.logpdf(z_values, df=rank) - proposal_logpdf
        log_weights[~selected | ~np.isfinite(log_weights)] = np.nan
        all_values.append(z_values)
        all_selected.append(selected)
        all_log_weights.append(log_weights)

        usable = selected & np.isfinite(log_weights)
        if not np.any(usable):
            sd *= 1.5
            continue
        new_mean, new_sd = _weighted_location_scale(
            z_values[usable], log_weights[usable], sd_floor=0.25
        )
        mean = 0.5 * mean + 0.5 * new_mean
        sd = max(0.25, 0.5 * sd + 0.5 * new_sd)

    retained_values = (
        np.concatenate(all_values) if all_values else np.array([], dtype=float)
    )
    retained_selected = (
        np.concatenate(all_selected) if all_selected else np.array([], dtype=bool)
    )
    retained_log_weights = (
        np.concatenate(all_log_weights)
        if all_log_weights
        else np.array([], dtype=float)
    )
    return {
        "mean": mean,
        "sd": sd,
        "z_values": retained_values,
        "selected": retained_selected,
        "log_weights": retained_log_weights,
        "sampling_scheme": (
            "reallocated_randomized_chi_cdf_strata"
            if stratified
            else "reallocated_local_chi_lower_mixture"
        ),
    }


def _ablation_final_components(
    t_obs,
    rank,
    pilot,
    *,
    multimodal,
    denominator_targeted,
):
    """Construct a fixed defensive deterministic-mixture proposal."""
    if multimodal:
        adapted = _pilot_island_components(
            pilot["z_values"],
            pilot["selected"],
            pilot["log_weights"],
            rank,
        )
    else:
        adapted = ((pilot["mean"], pilot["sd"], 1.0),)
    if not adapted:
        adapted = ((pilot["mean"], pilot["sd"], 1.0),)

    if denominator_targeted:
        usable_low = (
            pilot["selected"]
            & np.isfinite(pilot["log_weights"])
            & (pilot["z_values"] < t_obs)
        )
        if np.any(usable_low):
            denominator_mean, denominator_sd = _weighted_location_scale(
                pilot["z_values"][usable_low],
                pilot["log_weights"][usable_low],
                sd_floor=0.10,
            )
        else:
            denominator_mean = min(float(stats.chi.median(df=rank)), 0.5 * t_obs)
            denominator_sd = max(0.5, 0.25 * denominator_mean + 0.25)
        group_weights = ABLATION_DENOMINATOR_MIXTURE_WEIGHTS
    else:
        denominator_mean = np.nan
        denominator_sd = np.nan
        group_weights = (*DEFENSIVE_MIXTURE_WEIGHTS, 0.0)

    base_weight, observed_weight, adapted_weight, denominator_weight = group_weights
    observed_sd = max(0.75, 0.25 * t_obs + 0.5)
    components = [
        ("chi", 0.0, 1.0, float(base_weight)),
        ("tn", float(t_obs), observed_sd, float(observed_weight)),
    ]
    components.extend(
        ("tn", float(mean), float(sd), float(adapted_weight * weight))
        for mean, sd, weight in adapted
    )
    if denominator_targeted:
        components.append(
            ("tn", denominator_mean, denominator_sd, float(denominator_weight))
        )
    return tuple(components), adapted, (denominator_mean, denominator_sd), group_weights


def _component_mixture_logpdf(z, rank, components):
    """Evaluate an arbitrary chi/truncated-normal mixture stably."""
    z = np.asarray(z, dtype=float)
    result = np.full(z.shape, -np.inf, dtype=float)
    for kind, mean, sd, weight in components:
        if weight <= 0.0:
            continue
        if kind == "chi":
            log_density = stats.chi.logpdf(z, df=rank)
        elif kind == "cdf_interval":
            cdf = stats.chi.cdf(z, df=rank)
            inside = (cdf >= mean) & (cdf <= sd)
            log_density = np.where(
                inside,
                stats.chi.logpdf(z, df=rank) - np.log(sd - mean),
                -np.inf,
            )
        elif kind == "sf_interval":
            survival = stats.chi.sf(z, df=rank)
            log_density = np.where(
                (survival >= mean) & (survival <= sd),
                stats.chi.logpdf(z, df=rank) - np.log(sd - mean),
                -np.inf,
            )
        else:
            log_density = _truncated_normal_logpdf(z, mean, sd)
        result = np.logaddexp(result, np.log(weight) + log_density)
    return result


def _deterministic_component_schedule(rng, size, components):
    """Allocate the declared total draw count to mixture components exactly."""
    weights = np.asarray([component[3] for component in components], dtype=float)
    weights = weights / np.sum(weights)
    expected = size * weights
    counts = np.floor(expected).astype(int)
    remainder = size - int(np.sum(counts))
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    labels = np.repeat(np.arange(len(components), dtype=int), counts)
    rng.shuffle(labels)
    return labels


def _sample_component_labels(rng, labels, rank, components):
    """Sample values for a fixed vector of deterministic-mixture labels."""
    values = np.empty(labels.size, dtype=float)
    for index, (kind, mean, sd, _) in enumerate(components):
        positions = np.flatnonzero(labels == index)
        if not positions.size:
            continue
        if kind == "chi":
            draws = stats.chi.rvs(df=rank, size=positions.size, random_state=rng)
        elif kind == "cdf_interval":
            epsilon = np.finfo(float).eps
            lower = float(np.clip(mean, epsilon, 1.0 - epsilon))
            upper = float(np.clip(sd, epsilon, 1.0 - epsilon))
            if upper <= lower:
                raise ValueError("Chi-CDF interval components require lower < upper.")
            uniforms = rng.uniform(lower, upper, positions.size)
            draws = _finite_chi_ppf(uniforms, rank)
        elif kind == "sf_interval":
            lower = max(float(mean), np.nextafter(0.0, 1.0))
            upper = min(float(sd), np.nextafter(1.0, 0.0))
            if upper <= lower:
                raise ValueError("Chi survival intervals require lower < upper.")
            draws = stats.chi.isf(rng.uniform(lower, upper, positions.size), df=rank)
        else:
            draws = _sample_truncated_normal(rng, mean, sd, positions.size)
        for _ in range(5):
            nonfinite = ~np.isfinite(draws)
            if not np.any(nonfinite):
                break
            if kind == "chi":
                draws[nonfinite] = stats.chi.rvs(
                    df=rank, size=int(np.sum(nonfinite)), random_state=rng
                )
            elif kind == "sf_interval":
                draws[nonfinite] = stats.chi.isf(
                    rng.uniform(lower, upper, int(np.sum(nonfinite))), df=rank
                )
            elif kind == "cdf_interval":
                uniforms = rng.uniform(lower, upper, int(np.sum(nonfinite)))
                draws[nonfinite] = _finite_chi_ppf(uniforms, rank)
            else:
                draws[nonfinite] = _sample_truncated_normal(
                    rng, mean, sd, int(np.sum(nonfinite))
                )
        if not np.all(np.isfinite(draws)):
            raise FloatingPointError(
                f"Proposal component {index} ({kind}) produced a nonfinite draw."
            )
        values[positions] = draws
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("AIS proposal sampling produced a nonfinite draw.")
    return values


def _refinement_components(
    t_obs, rank, pilot, *, interval, denominator, draw_floor, max_final_samples,
    tail_stable=False,
):
    """Create consolidated A components and optional island-specific low-z terms."""
    fit = pilot["fit_mask"]
    islands, pre_merge_count = _robust_pilot_islands(
        pilot["z_values"][fit], pilot["selected"][fit],
        pilot["log_weights"][fit], rank, sort_by_z=interval,
        tail_stable=tail_stable,
    )
    if not islands:
        fallback_u = float(stats.chi.cdf(pilot["mean"], df=rank))
        epsilon = np.finfo(float).eps
        islands = ({
            "lower_u": max(epsilon, fallback_u - 0.05),
            "upper_u": min(1.0 - epsilon, fallback_u + 0.05),
            "mean": float(pilot["mean"]), "sd": float(pilot["sd"]), "mass": 1.0,
        },)

    base_weight, observed_weight, adapted_weight = DEFENSIVE_MIXTURE_WEIGHTS
    denominator_weight = 0.0
    if denominator:
        base_weight, observed_weight, adapted_weight, denominator_weight = (
            REDESIGNED_DENOMINATOR_MIXTURE_WEIGHTS
        )
    components = [
        ("chi", 0.0, 1.0, float(base_weight)),
        ("tn", float(t_obs), max(0.75, 0.25 * t_obs + 0.5), float(observed_weight)),
    ]
    roles = ["base", "observed"]
    internal_masses = [1.0, 1.0]
    for island in islands:
        if interval:
            if island["upper_u"] <= island["lower_u"]:
                lower_sf = float(stats.chi.sf(island["upper_z"], df=rank))
                upper_sf = float(stats.chi.sf(island["lower_z"], df=rank))
                component = (
                    "sf_interval", lower_sf, upper_sf,
                    adapted_weight * island["mass"],
                )
            else:
                component = (
                    "cdf_interval", island["lower_u"], island["upper_u"],
                    adapted_weight * island["mass"],
                )
        else:
            component = (
                "tn", island["mean"], island["sd"],
                adapted_weight * island["mass"],
            )
        components.append(component)
        roles.append("adapted")
        internal_masses.append(island["mass"])

    if denominator:
        threshold_u = float(stats.chi.cdf(t_obs, df=rank))
        low = [island for island in islands if island["lower_u"] < threshold_u]
        if not low:
            low = [min(islands, key=lambda item: item["lower_u"])]
        low_masses = np.asarray([item["mass"] for item in low], dtype=float)
        low_masses /= np.sum(low_masses)
        for island, mass in zip(low, low_masses):
            components.append((
                "tn", island["mean"], max(0.10, island["sd"]),
                denominator_weight * float(mass),
            ))
            roles.append("denominator")
            internal_masses.append(float(mass))

    if draw_floor:
        floored = [
            index for index, role in enumerate(roles)
            if role in {"adapted", "denominator"}
        ]
        requested_floor = draw_floor / max_final_samples
        available = max(0.0, components[1][3] - 0.05)
        low_floor, high_floor = 0.0, requested_floor
        for _ in range(60):
            candidate = 0.5 * (low_floor + high_floor)
            needed = sum(
                max(0.0, candidate - components[index][3])
                for index in floored
            )
            if needed <= available:
                low_floor = candidate
            else:
                high_floor = candidate
        floor_weight = low_floor
        deficit = sum(
            max(0.0, floor_weight - components[index][3]) for index in floored
        )
        for index in floored:
            kind, mean, sd, weight = components[index]
            components[index] = (kind, mean, sd, max(weight, floor_weight))
        kind, mean, sd, weight = components[1]
        components[1] = (kind, mean, sd, weight - deficit)

    for role in ("adapted", "denominator"):
        indices = [index for index, value in enumerate(roles) if value == role]
        if indices:
            total = sum(components[index][3] for index in indices)
            for index in indices:
                internal_masses[index] = components[index][3] / total

    return (
        tuple(components), tuple(roles), tuple(internal_masses), islands,
        pre_merge_count,
    )


def _optimize_refinement_group_weights(
    components, roles, internal_masses, rank, pilot, draw_floor,
    max_final_samples,
):
    """Optimize role-level weights on held-out pilots, then freeze them."""
    mask = (
        pilot["validation_mask"] & pilot["selected"]
        & np.isfinite(pilot["log_weights"])
    )
    if np.sum(mask) < 20:
        return components, False
    role_order = tuple(dict.fromkeys(roles))
    role_indices = {
        role: [index for index, value in enumerate(roles) if value == role]
        for role in role_order
    }
    initial = np.asarray([
        sum(components[index][3] for index in role_indices[role])
        for role in role_order
    ])
    floor_weight = draw_floor / max_final_samples if draw_floor else 0.0
    lower, upper = [], []
    for role in role_order:
        if role == "base":
            lower.append(0.25); upper.append(0.25)
        elif role == "observed":
            lower.append(0.05); upper.append(0.60)
        else:
            masses = [internal_masses[index] for index in role_indices[role]]
            lower.append(max(floor_weight / mass for mass in masses))
            upper.append(0.70)
    z_values = pilot["z_values"][mask]
    log_weights = pilot["log_weights"][mask]
    log_p = stats.chi.logpdf(z_values, df=rank)

    def rebuild(group_weights):
        lookup = dict(zip(role_order, group_weights))
        rebuilt = []
        for component, role, mass in zip(components, roles, internal_masses):
            kind, mean, sd, _ = component
            role_mass = 1.0 if role in {"base", "observed"} else mass
            rebuilt.append((kind, mean, sd, lookup[role] * role_mass))
        return tuple(rebuilt)

    def objective(group_weights):
        log_q = _component_mixture_logpdf(
            z_values, rank, rebuild(group_weights)
        )
        terms = log_weights + log_p - log_q
        return float(logsumexp(terms) - np.log(terms.size))

    result = optimize.minimize(
        objective, initial, method="SLSQP", bounds=list(zip(lower, upper)),
        constraints={"type": "eq", "fun": lambda values: np.sum(values) - 1.0},
        options={"maxiter": 100, "ftol": 1e-8},
    )
    return (rebuild(result.x), True) if result.success else (components, False)


def _weight_gini(weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights >= 0.0)]
    if not weights.size or np.sum(weights) <= 0.0:
        return np.nan
    ordered = np.sort(weights)
    index = np.arange(1, ordered.size + 1, dtype=float)
    return float(
        np.sum((2.0 * index - ordered.size - 1.0) * ordered)
        / (ordered.size * np.sum(ordered))
    )


def _psis_khat(weights):
    """Return a raw generalized-Pareto tail-shape diagnostic."""
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if weights.size < 20:
        return np.nan
    tail_size = max(10, min(weights.size // 5, int(3.0 * np.sqrt(weights.size))))
    ordered = np.sort(weights)
    threshold = ordered[-tail_size - 1]
    positive = ordered[-tail_size:] - threshold
    positive = positive[positive > 0.0]
    if positive.size < 5:
        return -np.inf
    try:
        shape, _, _ = stats.genpareto.fit(positive, floc=0.0)
    except (FloatingPointError, ValueError):
        return np.nan
    return float(shape)


def _component_diagnostics_json(
    components, roles, islands, schedule, proposals, selected_labels, weights, tail
):
    records = []
    labels = (
        np.concatenate(selected_labels) if selected_labels
        else np.array([], dtype=int)
    )
    for index, ((kind, mean, sd, mixture_weight), role) in enumerate(
        zip(components, roles)
    ):
        local = labels == index
        local_weights = weights[local] if weights.size else np.array([], dtype=float)
        local_tail = tail[local] if tail.size else np.array([], dtype=bool)
        island = None
        if role == "adapted":
            adapted_index = sum(value == "adapted" for value in roles[:index])
            island = islands[adapted_index]
        records.append({
            "index": index, "role": role, "kind": kind,
            "mixture_weight": float(mixture_weight),
            "mean": None if kind in {"cdf_interval", "sf_interval"} else float(mean),
            "scale": None if kind in {"cdf_interval", "sf_interval"} else float(sd),
            "lower_sf": float(mean) if kind == "sf_interval" else None,
            "upper_sf": float(sd) if kind == "sf_interval" else None,
            "lower_u": float(mean) if kind == "cdf_interval" else (island or {}).get("lower_u"),
            "upper_u": float(sd) if kind == "cdf_interval" else (island or {}).get("upper_u"),
            "draws_planned": int(np.sum(schedule == index)),
            "draws_allocated": int(np.sum(schedule[:proposals] == index)),
            "draws_selected": int(np.sum(local)),
            "denominator_ess": float(_effective_sample_size(local_weights)),
            "tail_ess": float(_effective_sample_size(local_weights[local_tail])),
        })
    return json.dumps(records, separators=(",", ":"), allow_nan=False)


def _general_mixture_logpdf(z, rank, t_obs, adapted_components, mixture_weights):
    base_weight, observed_weight, adapted_weight = mixture_weights
    observed_sd = max(0.75, 0.25 * t_obs + 0.5)
    terms = [
        np.log(base_weight) + stats.chi.logpdf(z, df=rank),
        np.log(observed_weight) + _truncated_normal_logpdf(z, t_obs, observed_sd),
    ]
    for mean, sd, component_weight in adapted_components:
        terms.append(
            np.log(adapted_weight * component_weight)
            + _truncated_normal_logpdf(z, mean, sd)
        )
    return logsumexp(np.vstack(terms), axis=0)


def _sample_general_mixture(
    rng, size, rank, t_obs, adapted_components, mixture_weights
):
    component_weights = [mixture_weights[0], mixture_weights[1]] + [
        mixture_weights[2] * item[2] for item in adapted_components
    ]
    counts = rng.multinomial(size, component_weights)
    observed_sd = max(0.75, 0.25 * t_obs + 0.5)
    chunks = [
        stats.chi.rvs(df=rank, size=counts[0], random_state=rng),
        _sample_truncated_normal(rng, t_obs, observed_sd, counts[1]),
    ]
    for count, (mean, sd, _) in zip(counts[2:], adapted_components):
        chunks.append(_sample_truncated_normal(rng, mean, sd, count))
    z_values = np.concatenate(chunks)
    rng.shuffle(z_values)
    return z_values


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


def _run_ais_refinement(
    t_obs,
    rank,
    is_selected,
    rng,
    *,
    pilot_iters,
    pilot_samples,
    final_batch_size,
    max_final_samples,
    min_denominator_ess,
    min_tail_ess,
    stop_when_ess_met,
    ais_strategy,
):
    """Run a robust A refinement with frozen deterministic-mixture weights."""
    options = AIS_REFINEMENT_STRATEGIES[ais_strategy]
    pilot = _prepare_refinement_pilot(
        t_obs, rank, is_selected, rng, pilot_iters, pilot_samples,
        pilot15=options["pilot15"],
    )
    components, roles, internal_masses, islands, pre_merge_count = (
        _refinement_components(
            t_obs, rank, pilot, interval=options["interval"],
            denominator=options["denominator"],
            draw_floor=options["draw_floor"],
            max_final_samples=max_final_samples,
            tail_stable=options.get("tail_stable", False),
        )
    )
    optimized = False
    if options["optimize"]:
        components, optimized = _optimize_refinement_group_weights(
            components, roles, internal_masses, rank, pilot,
            options["draw_floor"], max_final_samples,
        )
    schedule = _deterministic_component_schedule(rng, max_final_samples, components)
    allocation_weights = np.bincount(
        schedule, minlength=len(components)
    ).astype(float) / max_final_samples
    components = tuple(
        (kind, mean, sd, float(weight))
        for (kind, mean, sd, _), weight in zip(components, allocation_weights)
    )
    selected_z_batches = []
    selected_log_weight_batches = []
    selected_label_batches = []
    proposals = 0

    def current_weighted_sample():
        if not selected_z_batches:
            return np.array([]), np.array([]), np.array([], dtype=bool)
        all_z = np.concatenate(selected_z_batches)
        log_weights = np.concatenate(selected_log_weight_batches)
        finite = np.isfinite(log_weights)
        all_z = all_z[finite]
        log_weights = log_weights[finite]
        if not log_weights.size:
            return all_z, np.array([]), np.array([], dtype=bool)
        weights = np.exp(log_weights - np.max(log_weights))
        return all_z, weights, all_z >= t_obs

    while proposals < max_final_samples:
        batch_size = min(final_batch_size, max_final_samples - proposals)
        labels = schedule[proposals : proposals + batch_size]
        z_values = _sample_component_labels(rng, labels, rank, components)
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values), bool, batch_size
        )
        if np.any(selected):
            selected_z = z_values[selected]
            selected_z_batches.append(selected_z)
            selected_label_batches.append(labels[selected])
            selected_log_weight_batches.append(
                stats.chi.logpdf(selected_z, df=rank)
                - _component_mixture_logpdf(selected_z, rank, components)
            )
        proposals += batch_size
        _, weights, tail = current_weighted_sample()
        if weights.size and (
            stop_when_ess_met
            and _effective_sample_size(weights) >= min_denominator_ess
            and _effective_sample_size(weights[tail]) >= min_tail_ess
        ):
            break

    all_z, weights, tail = current_weighted_sample()
    denominator_ess = _effective_sample_size(weights)
    tail_ess = _effective_sample_size(weights[tail])
    normalized = weights / np.sum(weights) if np.sum(weights) else weights
    pilot_selected = pilot["selected"] & np.isfinite(pilot["log_weights"])
    role_weights = {
        role: float(sum(c[3] for c, value in zip(components, roles) if value == role))
        for role in set(roles)
    }
    common_diagnostics = {
        "ais_strategy": ais_strategy,
        "pilot_evaluations": int(pilot_iters * pilot_samples),
        "aj_evaluations": int(pilot_iters * pilot_samples + proposals),
        "pilot_sampling_scheme": pilot["sampling_scheme"],
        "pilot_selected_samples": int(np.sum(pilot_selected)),
        "pilot_low_selected_samples": int(
            np.sum(pilot_selected & (pilot["z_values"] < t_obs))
        ),
        "pilot_validation_samples": int(np.sum(pilot["validation_mask"])),
        "pre_merge_cluster_count": int(pre_merge_count),
        "post_merge_cluster_count": int(len(islands)),
        "adaptive_component_count": int(len(islands)),
        "minimum_adapted_component_weight": float(
            min(c[3] for c, role in zip(components, roles) if role == "adapted")
        ),
        "absolute_component_draw_floor": int(options["draw_floor"]),
        "denominator_component_count": int(sum(role == "denominator" for role in roles)),
        "final_base_weight": role_weights.get("base", 0.0),
        "final_observed_weight": role_weights.get("observed", 0.0),
        "final_adapted_weight": role_weights.get("adapted", 0.0),
        "final_denominator_weight": role_weights.get("denominator", 0.0),
        "final_component_count": len(components),
        "final_allocation_method": (
            "heldout_ess_optimized_deterministic_mixture"
            if optimized else "deterministic_mixture_total_budget"
        ),
        "mixture_optimization_succeeded": bool(optimized),
        "max_normalized_weight": float(np.max(normalized)) if normalized.size else np.nan,
        "weight_gini": _weight_gini(weights),
        "psis_khat": _psis_khat(weights),
        "tail_psis_khat": _psis_khat(weights[tail]),
        "component_diagnostics_json": _component_diagnostics_json(
            components, roles, islands, schedule, proposals,
            selected_label_batches, weights, tail,
        ),
    }
    converged = denominator_ess >= min_denominator_ess and tail_ess >= min_tail_ess
    if not weights.size or not converged:
        return np.nan, {
            "status": "no_selected_samples" if not weights.size else "insufficient_ess",
            "proposals": proposals,
            "selected_samples": int(all_z.size),
            "tail_samples": int(np.sum(tail)),
            "denominator_ess": denominator_ess,
            "tail_ess": tail_ess,
            "mc_se": np.nan,
            "mc_ci_95_lower": np.nan,
            "mc_ci_95_upper": np.nan,
            "sampling_mode": "ess_early_stopping" if stop_when_ess_met else "fixed_budget",
            "p_value_method": "self_normalized_importance_sampling_estimate",
            "finite_sample_valid": False,
            **common_diagnostics,
        }

    p_value = float(np.sum(weights * tail) / np.sum(weights))
    mc_variance = (
        np.sum(weights**2 * (tail.astype(float) - p_value) ** 2)
        / np.sum(weights) ** 2
    )
    mc_se = float(np.sqrt(mc_variance))
    p_value = float(np.clip(p_value, 0.0, 1.0))
    return p_value, {
        "status": "ok", "proposals": proposals,
        "selected_samples": int(all_z.size), "tail_samples": int(np.sum(tail)),
        "denominator_ess": denominator_ess, "tail_ess": tail_ess,
        "mc_se": mc_se,
        "mc_ci_95_lower": float(max(0.0, p_value - 1.96 * mc_se)),
        "mc_ci_95_upper": float(min(1.0, p_value + 1.96 * mc_se)),
        "sampling_mode": "ess_early_stopping" if stop_when_ess_met else "fixed_budget",
        "p_value_method": "self_normalized_importance_sampling_estimate",
        "finite_sample_valid": False,
        **common_diagnostics,
    }


def _run_ais_ablation(
    t_obs,
    rank,
    is_selected,
    rng,
    *,
    pilot_iters,
    pilot_samples,
    final_batch_size,
    max_final_samples,
    min_denominator_ess,
    min_tail_ess,
    stop_when_ess_met,
    ais_strategy,
):
    """Run one A/B/C ablation with a frozen deterministic-mixture proposal."""
    multimodal, denominator_targeted, stratified = AIS_ABLATION_MECHANISMS[
        ais_strategy
    ]
    pilot = _adapt_ablation_pilot(
        t_obs,
        rank,
        is_selected,
        rng,
        pilot_iters,
        pilot_samples,
        stratified=stratified,
    )
    components, adapted, denominator_parameters, _ = (
        _ablation_final_components(
            t_obs,
            rank,
            pilot,
            multimodal=multimodal,
            denominator_targeted=denominator_targeted,
        )
    )
    schedule = _deterministic_component_schedule(
        rng, max_final_samples, components
    )
    allocation_weights = np.bincount(
        schedule, minlength=len(components)
    ).astype(float) / max_final_samples
    components = tuple(
        (kind, mean, sd, float(weight))
        for (kind, mean, sd, _), weight in zip(components, allocation_weights)
    )
    selected_z_batches = []
    selected_log_weight_batches = []
    proposals = 0

    def current_weighted_sample():
        if not selected_z_batches:
            return np.array([]), np.array([]), np.array([], dtype=bool)
        all_z = np.concatenate(selected_z_batches)
        log_weights = np.concatenate(selected_log_weight_batches)
        finite = np.isfinite(log_weights)
        all_z = all_z[finite]
        log_weights = log_weights[finite]
        if not log_weights.size:
            return all_z, np.array([]), np.array([], dtype=bool)
        weights = np.exp(log_weights - np.max(log_weights))
        return all_z, weights, all_z >= t_obs

    while proposals < max_final_samples:
        batch_size = min(final_batch_size, max_final_samples - proposals)
        labels = schedule[proposals : proposals + batch_size]
        z_values = _sample_component_labels(rng, labels, rank, components)
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values),
            dtype=bool,
            count=batch_size,
        )
        if np.any(selected):
            selected_z = z_values[selected]
            selected_z_batches.append(selected_z)
            selected_log_weight_batches.append(
                stats.chi.logpdf(selected_z, df=rank)
                - _component_mixture_logpdf(selected_z, rank, components)
            )
        proposals += batch_size

        if not selected_z_batches:
            continue
        _, weights, tail = current_weighted_sample()
        denominator_ess = _effective_sample_size(weights)
        tail_ess = _effective_sample_size(weights[tail])
        if (
            stop_when_ess_met
            and denominator_ess >= min_denominator_ess
            and tail_ess >= min_tail_ess
        ):
            break

    selected_pilot = pilot["selected"] & np.isfinite(pilot["log_weights"])
    base_weight = components[0][3]
    observed_weight = components[1][3]
    adapted_weight = float(
        sum(component[3] for component in components[2 : 2 + len(adapted)])
    )
    denominator_weight = (
        components[-1][3] if denominator_targeted else 0.0
    )
    common_diagnostics = {
        "ais_strategy": ais_strategy,
        "pilot_evaluations": int(pilot_iters * pilot_samples),
        "aj_evaluations": int(pilot_iters * pilot_samples + proposals),
        "pilot_sampling_scheme": pilot["sampling_scheme"],
        "pilot_selected_samples": int(np.sum(selected_pilot)),
        "pilot_low_selected_samples": int(
            np.sum(selected_pilot & (pilot["z_values"] < t_obs))
        ),
        "adaptive_component_count": len(adapted),
        "minimum_adapted_component_weight": float(
            min(component[2] for component in adapted)
        ),
        "denominator_component_count": int(denominator_targeted),
        "denominator_component_mean": float(denominator_parameters[0]),
        "denominator_component_sd": float(denominator_parameters[1]),
        "final_base_weight": float(base_weight),
        "final_observed_weight": float(observed_weight),
        "final_adapted_weight": float(adapted_weight),
        "final_denominator_weight": float(denominator_weight),
        "final_component_count": len(components),
        "final_allocation_method": "deterministic_mixture_total_budget",
    }
    if not selected_z_batches:
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
            **common_diagnostics,
        }

    all_z, weights, tail = current_weighted_sample()
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
            **common_diagnostics,
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
        "mc_ci_95_lower": float(max(0.0, p_value - 1.96 * mc_se)),
        "mc_ci_95_upper": float(min(1.0, p_value + 1.96 * mc_se)),
        "sampling_mode": (
            "ess_early_stopping" if stop_when_ess_met else "fixed_budget"
        ),
        "p_value_method": "self_normalized_importance_sampling_estimate",
        "finite_sample_valid": False,
        **common_diagnostics,
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
    ais_strategy="baseline",
):
    """Estimate the selected chi-tail ratio with a defensive proposal.

    This exploratory estimator uses the full, predeclared final-sample budget.
    ``stop_when_ess_met=True`` retains the former exploratory early-stopping
    behavior, but its data-dependent sample size should not be used as evidence
    that the resulting estimate is an exact finite-sample p-value.
    """
    if ais_strategy not in AIS_STRATEGIES:
        raise ValueError(f"ais_strategy must be one of {AIS_STRATEGIES}.")
    if ais_strategy in AIS_REFINEMENT_STRATEGIES:
        return _run_ais_refinement(
            t_obs,
            rank,
            is_selected,
            rng,
            pilot_iters=pilot_iters,
            pilot_samples=pilot_samples,
            final_batch_size=final_batch_size,
            max_final_samples=max_final_samples,
            min_denominator_ess=min_denominator_ess,
            min_tail_ess=min_tail_ess,
            stop_when_ess_met=stop_when_ess_met,
            ais_strategy=ais_strategy,
        )
    if ais_strategy in AIS_ABLATION_MECHANISMS:
        return _run_ais_ablation(
            t_obs,
            rank,
            is_selected,
            rng,
            pilot_iters=pilot_iters,
            pilot_samples=pilot_samples,
            final_batch_size=final_batch_size,
            max_final_samples=max_final_samples,
            min_denominator_ess=min_denominator_ess,
            min_tail_ess=min_tail_ess,
            stop_when_ess_met=stop_when_ess_met,
            ais_strategy=ais_strategy,
        )
    if ais_strategy == "multimodal_qadapt":
        adapted_components = _adapt_multimodal_proposal(
            t_obs, rank, is_selected, rng, pilot_iters, pilot_samples
        )
    elif ais_strategy == "reallocated_pilot_exploration":
        adapted_mean, adapted_sd = _adapt_reallocated_pilot(
            t_obs, rank, is_selected, rng, pilot_iters, pilot_samples
        )
        adapted_components = ((adapted_mean, adapted_sd, 1.0),)
    else:
        adapted_mean, adapted_sd = _adapt_proposal(
            t_obs,
            rank,
            is_selected,
            rng,
            pilot_iters,
            pilot_samples,
            sd_floor=0.75 if ais_strategy == "raised_sd_floor" else 0.25,
        )
        adapted_components = ((adapted_mean, adapted_sd, 1.0),)

    selected_z_batches = []
    proposal_history = []
    mixture_weights = np.asarray(DEFENSIVE_MIXTURE_WEIGHTS, dtype=float)
    proposals = 0

    def current_weighted_sample():
        if not selected_z_batches:
            return np.array([]), np.array([]), np.array([], dtype=bool)
        all_z = np.concatenate(selected_z_batches)
        if ais_strategy == "adaptive_mixture_weights":
            log_densities = np.vstack(
                [
                    _general_mixture_logpdf(
                        all_z, rank, t_obs, adapted_components, historic_weights
                    )
                    for historic_weights in proposal_history
                ]
            )
            proposal_logpdf = logsumexp(log_densities, axis=0) - np.log(
                len(proposal_history)
            )
        else:
            proposal_logpdf = _general_mixture_logpdf(
                all_z, rank, t_obs, adapted_components, mixture_weights
            )
        log_weights = stats.chi.logpdf(all_z, df=rank) - proposal_logpdf
        finite = np.isfinite(log_weights)
        all_z = all_z[finite]
        log_weights = log_weights[finite]
        if not log_weights.size:
            return all_z, np.array([]), np.array([], dtype=bool)
        weights = np.exp(log_weights - np.max(log_weights))
        return all_z, weights, all_z >= t_obs

    while proposals < max_final_samples:
        batch_size = min(final_batch_size, max_final_samples - proposals)
        batch_weights = mixture_weights.copy()
        z_values = _sample_general_mixture(
            rng,
            batch_size,
            rank,
            t_obs,
            adapted_components,
            batch_weights,
        )
        selected = np.fromiter(
            (is_selected(float(z)) for z in z_values),
            dtype=bool,
            count=batch_size,
        )
        if np.any(selected):
            selected_z_batches.append(z_values[selected])
        proposal_history.append(batch_weights)
        proposals += batch_size

        if not selected_z_batches:
            continue
        all_z, weights, tail = current_weighted_sample()
        denominator_ess = _effective_sample_size(weights)
        tail_ess = _effective_sample_size(weights[tail])
        if (
            stop_when_ess_met
            and denominator_ess >= min_denominator_ess
            and tail_ess >= min_tail_ess
        ):
            break

        if ais_strategy == "adaptive_mixture_weights":
            denominator_progress = denominator_ess / min_denominator_ess
            tail_progress = tail_ess / min_tail_ess
            if denominator_progress < tail_progress:
                donor = 1 if mixture_weights[1] > 0.10 else 2
                transfer = min(0.05, mixture_weights[donor] - 0.10)
                mixture_weights[donor] -= transfer
                mixture_weights[0] += transfer
            else:
                donor = 0 if mixture_weights[0] > 0.10 else 2
                transfer = min(0.05, mixture_weights[donor] - 0.10)
                mixture_weights[donor] -= transfer
                mixture_weights[1] += transfer

    common_diagnostics = {
        "ais_strategy": ais_strategy,
        "pilot_evaluations": int(pilot_iters * pilot_samples),
        "aj_evaluations": int(pilot_iters * pilot_samples + proposals),
        "adaptive_component_count": len(adapted_components),
        "final_base_weight": float(mixture_weights[0]),
        "final_observed_weight": float(mixture_weights[1]),
        "final_adapted_weight": float(mixture_weights[2]),
    }
    if not selected_z_batches:
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
            **common_diagnostics,
        }

    all_z, weights, tail = current_weighted_sample()
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
            **common_diagnostics,
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
        **common_diagnostics,
    }
