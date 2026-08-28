"""Audit target-event calibration when threshold selection has variable size.

This experiment is a controlled counterexample to the claim that
``feature_inclusion`` always represents a uniformly randomized target.  The
null statistic ``T`` has a chi distribution.  Feature 0 is always selected,
while features 1 through ``k - 1`` pass a threshold only for non-extreme values
of ``T``::

    M(T) = {0, ..., k - 1},  T <= c
           {0},             T > c.

The target is drawn uniformly from ``M(T)``.  Conditional on observing target
0, the null density of ``T`` is therefore proportional to

    f_chi(T) * 1{0 in M(T)} / |M(T)|.

Feature inclusion drops the inverse-set-size factor.  The other methods below
use the correct marginalized likelihood, condition on the realized auxiliary
draw, or condition on the exact selected set.  The construction deliberately
makes extreme values more likely to produce a singleton set, so unweighted
feature inclusion is anti-conservative.  It is an audit counterexample, not a
model of every threshold selector.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from si_shap.selection import target_from_selected_set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "variable_size_selection_event_counterexample"
)
METHODS = (
    "feature_inclusion",
    "weighted_feature_inclusion",
    "same_target",
    "exact_set",
)


def _validate_inputs(
    n_iters, rank, split_probability, large_set_size, alpha_levels, seed
):
    for name, value in {
        "n_iters": n_iters,
        "rank": rank,
        "large_set_size": large_set_size,
        "seed": seed,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
    if n_iters < 1:
        raise ValueError("n_iters must be positive.")
    if rank < 1:
        raise ValueError("rank must be positive.")
    if large_set_size < 2:
        raise ValueError("large_set_size must be at least two.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    if (
        not np.isscalar(split_probability)
        or not np.isfinite(split_probability)
        or not 0.0 < float(split_probability) < 1.0
    ):
        raise ValueError("split_probability must lie strictly between 0 and 1.")
    levels = np.asarray(alpha_levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("alpha_levels must be a nonempty one-dimensional sequence.")
    if not np.all(np.isfinite(levels)) or np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("Every alpha level must lie strictly between 0 and 1.")
    if np.unique(levels).size != levels.size:
        raise ValueError("alpha_levels must not contain duplicates.")
    return (
        int(n_iters),
        int(rank),
        float(split_probability),
        int(large_set_size),
        tuple(levels),
        int(seed),
    )


def selected_features(statistic, cutoff, large_set_size=2):
    """Return the variable-size set selected by the audit threshold rule."""
    statistic = float(statistic)
    cutoff = float(cutoff)
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("statistic must be finite and nonnegative.")
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    if isinstance(large_set_size, (bool, np.bool_)) or not isinstance(
        large_set_size, (int, np.integer)
    ):
        raise TypeError("large_set_size must be an integer.")
    if large_set_size < 2:
        raise ValueError("large_set_size must be at least two.")
    return tuple(range(int(large_set_size))) if statistic <= cutoff else (0,)


def inverse_size_weight(statistic, cutoff, large_set_size=2):
    """Return P(target=0 | statistic) under uniform target selection."""
    return 1.0 / len(selected_features(statistic, cutoff, large_set_size))


def expected_feature_inclusion_rejection_rate(
    alpha, split_probability, large_set_size=2
):
    """Return the exact inclusion rejection rate conditional on target 0."""
    alpha = float(alpha)
    split_probability = float(split_probability)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if not 0.0 < split_probability < 1.0:
        raise ValueError("split_probability must lie strictly between 0 and 1.")
    if isinstance(large_set_size, (bool, np.bool_)) or not isinstance(
        large_set_size, (int, np.integer)
    ):
        raise TypeError("large_set_size must be an integer.")
    if large_set_size < 2:
        raise ValueError("large_set_size must be at least two.")
    high_tail_mass = 1.0 - split_probability
    target_probability = high_tail_mass + split_probability / large_set_size
    if alpha <= high_tail_mass:
        weighted_tail_mass = alpha
    else:
        weighted_tail_mass = high_tail_mass + (
            alpha - high_tail_mass
        ) / large_set_size
    return weighted_tail_mass / target_probability


def event_p_values(
    statistic, auxiliary_u, *, rank, split_probability, large_set_size=2
):
    """Compute all four p-values for an observation whose target is feature 0."""
    statistic = float(statistic)
    auxiliary_u = float(auxiliary_u)
    cutoff = float(stats.chi.ppf(split_probability, df=rank))
    selected = selected_features(statistic, cutoff, large_set_size)
    if target_from_selected_set(selected, auxiliary_u) != 0:
        raise ValueError("The observed randomized target must be feature 0.")

    null_tail = float(stats.chi.sf(statistic, df=rank))
    null_cdf = float(stats.chi.cdf(statistic, df=rank))
    target_probability = (
        1.0 - split_probability + split_probability / large_set_size
    )

    if statistic <= cutoff:
        marginalized_tail_mass = (
            (split_probability - null_cdf) / large_set_size
            + 1.0
            - split_probability
        )
    else:
        marginalized_tail_mass = null_tail
    weighted = marginalized_tail_mass / target_probability

    if auxiliary_u < 1.0 / large_set_size:
        fixed_u = null_tail
    else:
        # Above the first k-bin, target 0 occurs only for the singleton set.
        fixed_u = null_tail / (1.0 - split_probability)

    if len(selected) == large_set_size:
        exact_set = (split_probability - null_cdf) / split_probability
    else:
        exact_set = null_tail / (1.0 - split_probability)

    return {
        "feature_inclusion": float(np.clip(null_tail, 0.0, 1.0)),
        "weighted_feature_inclusion": float(np.clip(weighted, 0.0, 1.0)),
        "same_target": float(np.clip(fixed_u, 0.0, 1.0)),
        "exact_set": float(np.clip(exact_set, 0.0, 1.0)),
    }


def _clopper_pearson(successes, trials):
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


def _summarize(results, *, alpha_levels, split_probability, large_set_size):
    rows = []
    for method in METHODS:
        values = results.loc[results["method"] == method, "p_value"].to_numpy(
            dtype=float
        )
        row = {
            "method": method,
            "n_iterations": int(values.size),
            "mean_p_value": float(np.mean(values)),
            "median_p_value": float(np.median(values)),
            "uniform_ks_statistic": float(stats.kstest(values, "uniform").statistic),
            "uniform_ks_p_value": float(stats.kstest(values, "uniform").pvalue),
            "calibration_target": {
                "feature_inclusion": "incorrect_unweighted_inclusion",
                "weighted_feature_inclusion": (
                    "target_inclusion_weighted_by_inverse_set_size"
                ),
                "same_target": "fixed_auxiliary_randomness",
                "exact_set": "exact_selected_set_auxiliary_randomness_unfixed",
            }[method],
        }
        for alpha in alpha_levels:
            label = f"{alpha:g}"
            rejections = int(np.sum(values < alpha))
            lower, upper = _clopper_pearson(rejections, values.size)
            row[f"rejections_{label}"] = rejections
            row[f"rejection_rate_{label}"] = rejections / values.size
            row[f"rejection_ci_95_lower_{label}"] = lower
            row[f"rejection_ci_95_upper_{label}"] = upper
            row[f"expected_rejection_rate_{label}"] = (
                expected_feature_inclusion_rejection_rate(
                    alpha, split_probability, large_set_size
                )
                if method == "feature_inclusion"
                else float(alpha)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(
    *,
    n_iters=10_000,
    rank=3,
    split_probability=0.5,
    large_set_size=2,
    alpha_levels=(0.01, 0.05, 0.10),
    seed=123,
):
    """Simulate p-values conditional on the randomized target being feature 0."""
    (
        n_iters,
        rank,
        split_probability,
        large_set_size,
        alpha_levels,
        seed,
    ) = _validate_inputs(
        n_iters, rank, split_probability, large_set_size, alpha_levels, seed
    )
    rng = np.random.default_rng(seed)
    cutoff = float(stats.chi.ppf(split_probability, df=rank))
    records = []
    attempts = 0
    accepted = 0
    while accepted < n_iters:
        attempts += 1
        statistic = float(stats.chi.rvs(df=rank, random_state=rng))
        auxiliary_u = float(rng.random())
        selected = selected_features(statistic, cutoff, large_set_size)
        target = target_from_selected_set(selected, auxiliary_u)
        if target != 0:
            continue
        accepted += 1
        values = event_p_values(
            statistic,
            auxiliary_u,
            rank=rank,
            split_probability=split_probability,
            large_set_size=large_set_size,
        )
        for method, p_value in values.items():
            records.append(
                {
                    "iteration": accepted,
                    "attempt": attempts,
                    "method": method,
                    "target_feature": target,
                    "statistic": statistic,
                    "selected_features": selected,
                    "selected_set_size": len(selected),
                    "auxiliary_u": auxiliary_u,
                    "inverse_set_size_weight": 1.0 / len(selected),
                    "p_value": p_value,
                    "rejection_event_definition": {
                        "feature_inclusion": (
                            "target in selected set; ignores 1/|M|"
                        ),
                        "weighted_feature_inclusion": (
                            "target inclusion weighted by 1/|M|; auxiliary_u marginalized"
                        ),
                        "same_target": (
                            "target(selected set, observed auxiliary_u) == observed target"
                        ),
                        "exact_set": (
                            "selected(T) == observed selected set; auxiliary_u unfixed"
                        ),
                    }[method],
                    "conditions_on_auxiliary_u": method == "same_target",
                }
            )

    results = pd.DataFrame(records)
    summary = _summarize(
        results,
        alpha_levels=alpha_levels,
        split_probability=split_probability,
        large_set_size=large_set_size,
    )
    settings = {
        "n_iters": n_iters,
        "rank": rank,
        "split_probability": split_probability,
        "large_set_size": large_set_size,
        "statistic_cutoff": cutoff,
        "alpha_levels": alpha_levels,
        "seed": seed,
        "methods": METHODS,
        "attempts_until_n_target_zero": attempts,
        "empirical_target_zero_probability": n_iters / attempts,
        "theoretical_target_zero_probability": (
            1.0 - split_probability + split_probability / large_set_size
        ),
        "selector": (
            "M(T)={0,...,k-1} for T<=cutoff and M(T)={0} for T>cutoff"
        ),
        "target_rule": "uniform_from_selected",
        "conditioning_target": 0,
    }
    return {
        "summary": summary,
        "p_value_results": results,
        "settings": settings,
    }


def _plot_ecdf_difference(result, output_path):
    figure, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4))
    grid = np.linspace(0.0, 1.0, 501)
    for axis, method in zip(np.atleast_1d(axes), METHODS):
        values = np.sort(
            result["p_value_results"]
            .loc[result["p_value_results"]["method"] == method, "p_value"]
            .to_numpy(dtype=float)
        )
        ecdf = np.searchsorted(values, grid, side="right") / values.size
        axis.plot(grid, ecdf - grid)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(title=method, xlabel="p", ylabel="ECDF(p) - p", xlim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_rejection_rates(result, output_path):
    summary = result["summary"]
    levels = result["settings"]["alpha_levels"]
    plot_methods = tuple(
        method for method in METHODS if method != "weighted_feature_inclusion"
    )
    plot_summary = summary.set_index("method").loc[list(plot_methods)]
    figure, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4))
    for axis, alpha in zip(np.atleast_1d(axes), levels):
        label = f"{alpha:g}"
        rates = plot_summary[f"rejection_rate_{label}"].to_numpy(dtype=float)
        expected = plot_summary[f"expected_rejection_rate_{label}"].to_numpy(
            dtype=float
        )
        positions = np.arange(len(plot_methods))
        axis.scatter(positions, rates, label="Observed", zorder=3)
        axis.scatter(positions, expected, marker="x", label="Expected", zorder=3)
        axis.axhline(alpha, color="black", linestyle="--", label="Nominal")
        axis.set_xticks(positions, plot_methods, rotation=20, ha="right")
        axis.set(
            title=(
                f"alpha = {alpha:g}, large-set k = "
                f"{result['settings']['large_set_size']}"
            ),
            ylabel="Null rejection rate",
        )
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_results(result, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(output_dir / "calibration_summary.csv", index=False)
    result["p_value_results"].to_csv(output_dir / "p_value_results.csv", index=False)
    with (output_dir / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(result["settings"], file, indent=2, sort_keys=True)
    _plot_ecdf_difference(result, output_dir / "ecdf_difference.png")
    _plot_rejection_rates(result, output_dir / "rejection_rates.png")
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iters", type=int, default=10_000)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--split-probability", type=float, default=0.5)
    parser.add_argument("--large-set-size", type=int, default=2)
    parser.add_argument(
        "--alpha-levels", type=float, nargs="+", default=[0.01, 0.05, 0.10]
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_experiment(
        n_iters=args.n_iters,
        rank=args.rank,
        split_probability=args.split_probability,
        large_set_size=args.large_set_size,
        alpha_levels=args.alpha_levels,
        seed=args.seed,
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            DEFAULT_OUTPUT_DIR
            if args.large_set_size == 2
            else Path(f"{DEFAULT_OUTPUT_DIR}_k{args.large_set_size}")
        )
    output_dir = write_results(result, output_dir)
    print(result["summary"].to_string(index=False))
    print(f"\nSaved results to {output_dir}")
    return result


if __name__ == "__main__":
    main()
