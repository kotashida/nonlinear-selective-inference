"""Numerical selection regions after interchangeable feature selection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

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
    _adapt_proposal,
    _chi_statistic,
    _spline_effect_basis,
)
from .selection import (
    _BUILTIN_SELECTOR_TYPES,
    _resolve_rf_params,
    _top_k,
    _tree_shap_importance,
    _validate_selection_event,
    _validate_target_rule,
    make_selector,
    selection_event_definition,
    selection_event_holds,
    target_from_selected_set,
)
from .simulation import (
    _generate_null_dataset,
    _validate_positive_finite,
    _validate_seed,
)


SIMULATION_SIGMA = 1.0


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
    selection_event: str = "exact_set"
    selection_event_definition: str = ""
    observed_selected_features: tuple[int, ...] = ()
    selection_probability_upper_bound: float = np.nan
    relative_omitted_tail_bound: float = np.nan
    effective_grid_size: int = 0
    grid_refinements: int = 0
    target_rule: str = "all_selected"
    auxiliary_u: float | None = None
    observed_target_feature: int | None = None
    target_seed: int | None = None
    regions_certified: bool = False
    region_method: str = "finite_grid_with_bisection_diagnostic"


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
    grid_refinements: int = 1,
) -> tuple[tuple[float, float], ...]:
    """Approximate ``{z in [0, z_max]: is_selected(z)}`` as intervals.

    A regular grid discovers state changes and bisection refines each detected
    boundary. ``grid_refinements`` repeatedly inserts cell midpoints before
    transition detection. Components narrower than the final cell can still be
    missed, so the returned intervals remain a numerical diagnostic.
    """
    if not np.isfinite(z_max) or z_max <= 0.0:
        raise ValueError("z_max must be finite and positive.")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2.")
    if not isinstance(grid_refinements, int) or grid_refinements < 0:
        raise ValueError("grid_refinements must be a nonnegative integer.")
    if not np.isfinite(boundary_tol) or boundary_tol <= 0.0:
        raise ValueError("boundary_tol must be finite and positive.")

    anchors = np.asarray(tuple(anchor_points), dtype=float)
    if anchors.size and (
        not np.all(np.isfinite(anchors))
        or np.any(anchors < 0.0)
        or np.any(anchors > z_max)
    ):
        raise ValueError("anchor_points must be finite and lie in [0, z_max].")
    effective_grid_size = (grid_size - 1) * (2**grid_refinements) + 1
    grid = np.unique(
        np.concatenate(
            (np.linspace(0.0, z_max, effective_grid_size), anchors)
        )
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
    if (
        isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or rank < 1
    ):
        raise ValueError("rank must be a positive integer.")
    intervals = tuple(intervals)
    previous_right = 0.0
    for index, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError("Every interval must contain a left and right endpoint.")
        left, right = interval
        if (
            not np.isfinite(left)
            or (not np.isfinite(right) and right != np.inf)
            or left < 0.0
            or right < left
        ):
            raise ValueError(
                "Interval endpoints must be nonnegative and ordered; only the "
                "right endpoint may be positive infinity."
            )
        if index and left < previous_right:
            raise ValueError("Selection intervals must be sorted and non-overlapping.")
        previous_right = right
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
    grid_refinements: int = 1,
    boundary_tol: float = 1e-5,
    tail_probability: float = 1e-8,
    proposal_pilot_iters: int = 3,
    proposal_pilot_samples: int = 40,
    selection_method: str | None = None,
    rf_params=None,
    estimator=None,
    selector=None,
    selection_event: str = "exact_set",
    target_rule: str = "all_selected",
    target_seed: int = 123,
) -> list[SelectionRegionResult]:
    """Generate null data sets and compute top-k selection regions.

    ``dataset_seeds`` is required so callers choose the generated data sets at
    runtime rather than implicitly relying on a package-level seed sequence.

    One result is returned for every feature in the observed top-k set. The
    default event conditions on equality of the unordered observed top-k set.

    For each selected feature, the observed response is decomposed as
    ``y = y_perp + sigma * direction * T_obs``.  Feature selection is then
    repeated along ``y(z) = y_perp + sigma * direction * z`` for ``z >= 0``.
    The scan ends at a chi quantile whose omitted null-tail mass is at most
    ``tail_probability`` (and always includes ``T_obs``).

    ``rf_params`` may override any default ``RandomForestRegressor`` parameter.
    """
    raw_seeds = tuple(dataset_seeds)
    seeds = tuple(_validate_seed("dataset seed", seed) for seed in raw_seeds)
    if not seeds:
        raise ValueError("dataset_seeds must contain at least one seed.")
    for name, value in {
        "n_samples": n_samples,
        "n_features": n_features,
        "k_select": k_select,
        "grid_size": grid_size,
        "grid_refinements": grid_refinements,
        "proposal_pilot_iters": proposal_pilot_iters,
        "proposal_pilot_samples": proposal_pilot_samples,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
    if n_samples <= 4:
        raise ValueError("n_samples must exceed 4.")
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    if not isinstance(k_select, (int, np.integer)) or not 1 <= k_select <= n_features:
        raise ValueError("k_select must satisfy 1 <= k_select <= n_features.")
    if (
        isinstance(selection_decimals, (bool, np.bool_))
        or not isinstance(selection_decimals, (int, np.integer))
        or selection_decimals < 0
    ):
        raise ValueError("selection_decimals must be a nonnegative integer.")
    if not np.isscalar(tail_probability) or not np.isfinite(tail_probability) or not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must lie strictly between 0 and 1.")
    _validate_positive_finite("boundary_tol", boundary_tol)
    if proposal_pilot_iters < 0 or proposal_pilot_samples < 1:
        raise ValueError(
            "proposal_pilot_iters must be nonnegative and "
            "proposal_pilot_samples positive."
        )
    _validate_selection_event(selection_event)
    _validate_target_rule(target_rule)
    if selection_event == "same_target" and target_rule != "uniform_from_selected":
        raise ValueError("same_target requires target_rule='uniform_from_selected'.")
    target_seed = _validate_seed("target_seed", target_seed)
    if sum(value is not None for value in (rf_params, estimator, selector)) > 1:
        raise ValueError("Pass only one of rf_params, estimator, or selector.")
    if (
        selection_method is not None
        and selection_method != "shap"
        and rf_params is not None
    ):
        raise ValueError("rf_params are available only for SHAP selection.")
    resolved_rf_params = (
        _resolve_rf_params(rf_params)
        if (selection_method is None or selection_method == "shap")
        else None
    )
    resolved_selector = None
    if selection_method is not None or estimator is not None or selector is not None:
        resolved_selector = make_selector(
            selection_method=selection_method,
            estimator=estimator,
            selector=selector,
            selection_decimals=selection_decimals,
            rf_params=rf_params,
        )

    def select_response(X, response):
        if resolved_selector is not None:
            first_result = resolved_selector.select(X, response, k_select)
            first = np.asarray(first_result.selected_features, dtype=int)
            ranking = np.asarray(first_result.ranking, dtype=int)
            if not isinstance(resolved_selector, _BUILTIN_SELECTOR_TYPES):
                second_result = resolved_selector.select(X, response, k_select)
                second = np.asarray(second_result.selected_features, dtype=int)
                second_ranking = np.asarray(second_result.ranking, dtype=int)
                if not (
                    np.array_equal(first, second)
                    and np.array_equal(ranking, second_ranking)
                ):
                    raise ValueError(
                        "Custom selector is not deterministic for repeated identical "
                        "inputs. Fix and condition on its randomness before computing "
                        "selection regions."
                    )
            if (
                first.shape != (k_select,)
                or np.unique(first).size != k_select
                or np.any(first < 0)
                or np.any(first >= X.shape[1])
                or ranking.shape != (X.shape[1],)
                or set(ranking.tolist()) != set(range(X.shape[1]))
                or not np.array_equal(first, ranking[:k_select])
            ):
                raise ValueError(
                    "selector must return a complete ranking whose first k_select "
                    "entries are distinct valid selected feature indices."
                )
            return first, ranking
        importance = _tree_shap_importance(
            X,
            response,
            selection_decimals,
            rf_params=resolved_rf_params,
        )
        ranking = _top_k(importance, importance.size)
        return ranking[:k_select].copy(), ranking

    results: list[SelectionRegionResult] = []
    for dataset_number, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        X, response = _generate_null_dataset(rng, n_samples, n_features)

        selected_features, observed_ranking = select_response(X, response)
        observed_selected = tuple(int(value) for value in selected_features)
        auxiliary_u = None
        observed_target = None
        if target_rule == "uniform_from_selected":
            target_rng = np.random.default_rng(
                np.random.SeedSequence([int(target_seed), int(seed)])
            )
            auxiliary_u = float(target_rng.random())
            observed_target = target_from_selected_set(
                observed_selected, auxiliary_u
            )
            tested_features = (observed_target,)
        else:
            tested_features = observed_selected
        position_by_feature = {
            int(feature): position
            for position, feature in enumerate(selected_features, start=1)
        }
        for selected_feature in tested_features:
            feature = int(selected_feature)
            selection_position = position_by_feature[feature]
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
                    candidate = orthogonal + SIMULATION_SIGMA * direction * z
                    selected, candidate_ranking = select_response(X, candidate)
                    cache[z] = selection_event_holds(
                        selected,
                        observed_selected,
                        feature,
                        selection_event,
                        auxiliary_u=auxiliary_u,
                        ranking=candidate_ranking,
                        observed_ranking=observed_ranking,
                    )
                return cache[z]

            if not is_selected(t_obs):
                raise RuntimeError(
                    "The observed response was not reconstructed inside its "
                    f"selection event for data set {dataset_number}, feature {feature}."
                )

            chi_quantile = float(stats.chi.isf(tail_probability, df=rank))
            if not np.isfinite(chi_quantile):
                raise FloatingPointError(
                    "Could not compute a finite chi upper-tail cutoff; increase "
                    "tail_probability."
                )
            z_max = max(chi_quantile, 1.05 * float(t_obs))
            intervals = find_selection_intervals(
                is_selected,
                z_max,
                grid_size=grid_size,
                boundary_tol=boundary_tol,
                anchor_points=(t_obs,),
                grid_refinements=grid_refinements,
            )
            probability = selection_probability(intervals, rank)
            omitted_tail_probability = float(stats.chi.sf(z_max, df=rank))
            probability_upper_bound = float(
                min(1.0, probability + omitted_tail_probability)
            )
            relative_omitted_tail_bound = (
                omitted_tail_probability / probability
                if probability > 0.0
                else np.inf
            )
            adaptation_seed = int(
                np.random.SeedSequence(
                    [seed, selection_position, 10_000]
                ).generate_state(1, dtype=np.uint32)[0]
            )
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
                    omitted_tail_probability=omitted_tail_probability,
                    observed_proposal_sd=observed_sd,
                    adapted_proposal_mean=adapted_mean,
                    adapted_proposal_sd=adapted_sd,
                    adaptation_seed=adaptation_seed,
                    selection_position=selection_position,
                    k_select=k_select,
                    selection_event=selection_event,
                    selection_event_definition=selection_event_definition(
                        selection_event, feature
                    ),
                    observed_selected_features=observed_selected,
                    selection_probability_upper_bound=probability_upper_bound,
                    relative_omitted_tail_bound=relative_omitted_tail_bound,
                    effective_grid_size=(grid_size - 1)
                    * (2**grid_refinements)
                    + 1,
                    grid_refinements=grid_refinements,
                    target_rule=target_rule,
                    auxiliary_u=auxiliary_u,
                    observed_target_feature=observed_target,
                    target_seed=(
                        int(target_seed)
                        if target_rule == "uniform_from_selected"
                        else None
                    ),
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
