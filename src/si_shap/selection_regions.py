"""Numerical selection regions for SHAP-selected features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .inference import (
    TRUE_SIGMA,
    _adapt_proposal,
    _chi_statistic,
    _spline_effect_basis,
)
from .selection import (
    _resolve_rf_params,
    _select_features,
)
from .simulation import _generate_null_dataset


@dataclass(frozen=True)
class SelectionRegionResult:
    """Selection-region information for one generated data set."""

    dataset_number: int
    seed: int
    selected_feature: int
    rank: int
    t_obs: float
    z_max: float
    intervals: tuple[tuple[float, float], ...]
    selection_probability: float
    omitted_tail_probability: float
    observed_proposal_sd: float
    adapted_proposal_mean: float
    adapted_proposal_sd: float
    adaptation_seed: int
    selection_position: int = 1
    k_select: int = 1


def _refine_transition(
    is_selected: Callable[[float], bool],
    left: float,
    right: float,
    left_state: bool,
    boundary_tol: float,
) -> float:
    """Locate one selection-state transition by bisection."""
    while right - left > boundary_tol:
        midpoint = 0.5 * (left + right)
        if bool(is_selected(midpoint)) == left_state:
            left = midpoint
        else:
            right = midpoint
    return 0.5 * (left + right)


def find_selection_intervals(
    is_selected: Callable[[float], bool],
    z_max: float,
    *,
    grid_size: int = 201,
    boundary_tol: float = 1e-5,
    anchor_points: Iterable[float] = (),
) -> tuple[tuple[float, float], ...]:
    """Approximate ``{z in [0, z_max]: is_selected(z)}`` as intervals.

    A regular grid discovers state changes and bisection refines each detected
    boundary.  Consequently, intervals narrower than one grid cell can be
    missed; ``grid_size`` controls that numerical resolution.
    """
    if not np.isfinite(z_max) or z_max <= 0.0:
        raise ValueError("z_max must be finite and positive.")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2.")
    if not np.isfinite(boundary_tol) or boundary_tol <= 0.0:
        raise ValueError("boundary_tol must be finite and positive.")

    anchors = np.asarray(tuple(anchor_points), dtype=float)
    if anchors.size and (
        not np.all(np.isfinite(anchors))
        or np.any(anchors < 0.0)
        or np.any(anchors > z_max)
    ):
        raise ValueError("anchor_points must be finite and lie in [0, z_max].")
    grid = np.unique(
        np.concatenate((np.linspace(0.0, z_max, grid_size), anchors))
    )
    states = np.fromiter(
        (bool(is_selected(float(z))) for z in grid),
        dtype=bool,
        count=grid.size,
    )
    transition_indices = np.flatnonzero(states[:-1] != states[1:])
    transitions = [
        _refine_transition(
            is_selected,
            float(grid[index]),
            float(grid[index + 1]),
            bool(states[index]),
            boundary_tol,
        )
        for index in transition_indices
    ]

    boundaries = [0.0, *transitions, float(z_max)]
    intervals: list[tuple[float, float]] = []
    state = bool(states[0])
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if state:
            intervals.append((left, right))
        state = not state
    return tuple(intervals)


def selection_probability(
    intervals: Iterable[tuple[float, float]], rank: int
) -> float:
    """Integrate a chi distribution over a collection of intervals."""
    if rank < 1:
        raise ValueError("rank must be positive.")
    probability = sum(
        stats.chi.cdf(right, df=rank) - stats.chi.cdf(left, df=rank)
        for left, right in intervals
    )
    return float(np.clip(probability, 0.0, 1.0))


def compute_selection_regions(
    *,
    dataset_seeds: Iterable[int],
    n_samples: int = 100,
    n_features: int = 20,
    k_select: int = 1,
    selection_decimals: int = 10,
    grid_size: int = 201,
    boundary_tol: float = 1e-5,
    tail_probability: float = 1e-8,
    proposal_pilot_iters: int = 3,
    proposal_pilot_samples: int = 40,
    rf_params=None,
) -> list[SelectionRegionResult]:
    """Generate null data sets and compute top-k SHAP selection regions.

    ``dataset_seeds`` is required so callers choose the generated data sets at
    runtime rather than implicitly relying on a package-level seed sequence.

    One result is returned for every feature in the observed top-k set.  Each
    region conditions on that feature remaining included in the top-k set.

    For each selected feature, the observed response is decomposed as
    ``y = y_perp + sigma * direction * T_obs``.  Feature selection is then
    repeated along ``y(z) = y_perp + sigma * direction * z`` for ``z >= 0``.
    The scan ends at a chi quantile whose omitted null-tail mass is at most
    ``tail_probability`` (and always includes ``T_obs``).

    ``rf_params`` may override any default ``RandomForestRegressor`` parameter.
    """
    seeds = tuple(int(seed) for seed in dataset_seeds)
    if not seeds:
        raise ValueError("dataset_seeds must contain at least one seed.")
    if n_samples <= 4:
        raise ValueError("n_samples must exceed 4.")
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    if not isinstance(k_select, (int, np.integer)) or not 1 <= k_select <= n_features:
        raise ValueError("k_select must satisfy 1 <= k_select <= n_features.")
    if not isinstance(selection_decimals, int) or selection_decimals < 0:
        raise ValueError("selection_decimals must be a nonnegative integer.")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must lie strictly between 0 and 1.")
    if proposal_pilot_iters < 0 or proposal_pilot_samples < 1:
        raise ValueError(
            "proposal_pilot_iters must be nonnegative and "
            "proposal_pilot_samples positive."
        )
    resolved_rf_params = _resolve_rf_params(rf_params)

    results: list[SelectionRegionResult] = []
    for dataset_number, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        X, response = _generate_null_dataset(rng, n_samples, n_features)

        selected_features = _select_features(
            X,
            response,
            k_select,
            selection_decimals,
            rf_params=resolved_rf_params,
        )
        for selection_position, selected_feature in enumerate(selected_features, start=1):
            feature = int(selected_feature)
            basis = _spline_effect_basis(X[:, feature])
            rank = int(basis.shape[1])
            t_obs, projected = _chi_statistic(response, basis)
            projected_norm = float(np.linalg.norm(projected))
            zero_tolerance = np.sqrt(np.finfo(float).eps) * max(
                1.0, float(np.linalg.norm(response))
            )
            if projected_norm <= zero_tolerance:
                raise FloatingPointError(
                    f"Data set {dataset_number}, feature {feature} has an "
                    "undefined effect direction."
                )

            direction = projected / projected_norm
            orthogonal = response - projected
            cache: dict[float, bool] = {}

            def is_selected(z: float) -> bool:
                z = float(z)
                if z not in cache:
                    candidate = orthogonal + TRUE_SIGMA * direction * z
                    selected = _select_features(
                        X,
                        candidate,
                        k_select=k_select,
                        selection_decimals=selection_decimals,
                        rf_params=resolved_rf_params,
                    )
                    cache[z] = bool(feature in selected)
                return cache[z]

            if not is_selected(t_obs):
                raise RuntimeError(
                    "The observed response was not reconstructed inside its "
                    f"selection event for data set {dataset_number}, feature {feature}."
                )

            chi_quantile = float(stats.chi.ppf(1.0 - tail_probability, df=rank))
            z_max = max(chi_quantile, 1.05 * float(t_obs))
            intervals = find_selection_intervals(
                is_selected,
                z_max,
                grid_size=grid_size,
                boundary_tol=boundary_tol,
                anchor_points=(t_obs,),
            )
            probability = selection_probability(intervals, rank)
            adaptation_seed = seed + 10_000 + selection_position - 1
            adapted_mean, adapted_sd = _adapt_proposal(
                t_obs,
                rank,
                is_selected,
                np.random.default_rng(adaptation_seed),
                pilot_iters=proposal_pilot_iters,
                pilot_samples=proposal_pilot_samples,
            )
            observed_sd = max(0.75, 0.25 * float(t_obs) + 0.5)

            results.append(
                SelectionRegionResult(
                    dataset_number=dataset_number,
                    seed=seed,
                    selected_feature=feature,
                    rank=rank,
                    t_obs=float(t_obs),
                    z_max=z_max,
                    intervals=intervals,
                    selection_probability=probability,
                    omitted_tail_probability=float(stats.chi.sf(z_max, df=rank)),
                    observed_proposal_sd=observed_sd,
                    adapted_proposal_mean=adapted_mean,
                    adapted_proposal_sd=adapted_sd,
                    adaptation_seed=adaptation_seed,
                    selection_position=selection_position,
                    k_select=k_select,
                )
            )
    return results


def selection_regions_frame(
    results: Iterable[SelectionRegionResult],
) -> pd.DataFrame:
    """Return a compact tabular summary suitable for CSV output."""
    records = []
    for result in results:
        record = asdict(result)
        record["intervals"] = " U ".join(
            f"[{left:.6f}, {right:.6f}]" for left, right in result.intervals
        )
        records.append(record)
    return pd.DataFrame.from_records(records)
