"""Audit fixed-u target inference when the selected-set size is constant.

This experiment complements ``compare_variable_size_target_events.py``.  The
null statistic ``T`` has a chi distribution and the selector always returns
``k`` features, but the position of feature ``k - 1`` in the sorted selected
set changes at a cutoff ``c``::

    M(T) = {0, ..., k - 1},      T <= c
           {k - 1, ..., 2k - 2}, T > c.

The target is chosen uniformly by applying one auxiliary draw ``U`` to the
sorted selected set.  Feature ``k - 1`` is always included, so conditioning only on
its inclusion leaves the marginal null distribution of ``T`` unchanged.
However, for fixed ``U < 1/k``, that target is selected only when ``T > c``;
for fixed ``U >= 1 - 1/k``, it is selected only when ``T <= c``.  Thus
inclusion is uniform only after the target randomization is marginalized.
``same_target`` is uniform both marginally and within the two target-compatible
fixed-U strata.  ``exact_set`` is also uniform without conditioning on the
auxiliary draw because the exact selected set itself determines which side of
the cutoff contains ``T``.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fixed_k_target_event_counterexample"
METHODS = ("feature_inclusion", "same_target", "exact_set")
STRATA = ("pooled", "u_first_target_interval", "u_last_target_interval")
METHOD_LABELS = {
    "feature_inclusion": "feature_inclusion",
    "same_target": "same_target (fixed U)",
    "exact_set": "exact_set (U unfixed)",
}
STRATUM_LABELS = {
    "pooled": "pooled",
    "u_first_target_interval": "U < 1/k",
    "u_last_target_interval": "U >= 1 - 1/k",
}


def _validate_inputs(n_iters, rank, k, split_probability, alpha_levels, seed):
    for name, value in {
        "n_iters": n_iters,
        "rank": rank,
        "k": k,
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
    if k < 2:
        raise ValueError("k must be at least two.")
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
        int(k),
        float(split_probability),
        tuple(levels),
        int(seed),
    )


def selected_features(statistic, cutoff, k=2):
    """Return one of two selected sets whose size is always ``k``."""
    statistic = float(statistic)
    cutoff = float(cutoff)
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("statistic must be finite and nonnegative.")
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")
    if k < 2:
        raise ValueError("k must be at least two.")
    return tuple(range(k)) if statistic <= cutoff else tuple(range(k - 1, 2 * k - 1))


def event_p_values(statistic, auxiliary_u, *, rank, split_probability, k=2):
    """Compute p-values when the randomized target is feature ``k - 1``."""
    statistic = float(statistic)
    auxiliary_u = float(auxiliary_u)
    cutoff = float(stats.chi.ppf(split_probability, df=rank))
    selected = selected_features(statistic, cutoff, k)
    target_feature = k - 1
    if target_from_selected_set(selected, auxiliary_u) != target_feature:
        raise ValueError(
            f"The observed randomized target must be feature {target_feature}."
        )

    null_tail = float(stats.chi.sf(statistic, df=rank))
    if auxiliary_u < 1.0 / k:
        fixed_u_tail = null_tail / (1.0 - split_probability)
    else:
        null_cdf = float(stats.chi.cdf(statistic, df=rank))
        fixed_u_tail = (split_probability - null_cdf) / split_probability

    if statistic > cutoff:
        exact_set_tail = null_tail / (1.0 - split_probability)
    else:
        null_cdf = float(stats.chi.cdf(statistic, df=rank))
        exact_set_tail = (split_probability - null_cdf) / split_probability

    return {
        "feature_inclusion": float(np.clip(null_tail, 0.0, 1.0)),
        "same_target": float(np.clip(fixed_u_tail, 0.0, 1.0)),
        # This branch uses only the selected set, not auxiliary_u.  It agrees
        # numerically with same_target in this particular construction.
        "exact_set": float(np.clip(exact_set_tail, 0.0, 1.0)),
    }


def expected_rejection_rate(method, stratum, alpha, split_probability):
    """Return the exact conditional null rejection probability."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}.")
    if stratum not in STRATA:
        raise ValueError(f"stratum must be one of {STRATA}.")
    alpha = float(alpha)
    split_probability = float(split_probability)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if not 0.0 < split_probability < 1.0:
        raise ValueError("split_probability must lie strictly between 0 and 1.")
    if method in {"same_target", "exact_set"} or stratum == "pooled":
        return alpha
    high_tail_mass = 1.0 - split_probability
    if stratum == "u_first_target_interval":
        return min(alpha / high_tail_mass, 1.0)
    return max(alpha - high_tail_mass, 0.0) / split_probability


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


def _stratum_mask(results, stratum):
    if stratum == "pooled":
        return np.ones(len(results), dtype=bool)
    return results["auxiliary_u_stratum"].to_numpy(dtype=str) == stratum


def _summarize(results, *, alpha_levels, split_probability):
    rows = []
    for method in METHODS:
        method_results = results.loc[results["method"] == method].reset_index(drop=True)
        for stratum in STRATA:
            values = method_results.loc[
                _stratum_mask(method_results, stratum), "p_value"
            ].to_numpy(dtype=float)
            ks_result = stats.kstest(values, "uniform")
            row = {
                "method": method,
                "auxiliary_u_stratum": stratum,
                "n_iterations": int(values.size),
                "mean_p_value": float(np.mean(values)),
                "median_p_value": float(np.median(values)),
                "uniform_ks_statistic": float(ks_result.statistic),
                "uniform_ks_p_value": float(ks_result.pvalue),
                "calibration_target": {
                    "feature_inclusion": (
                        "marginalized_target_randomness"
                        if stratum == "pooled"
                        else "fixed_auxiliary_randomness"
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
                row[f"expected_rejection_rate_{label}"] = expected_rejection_rate(
                    method, stratum, alpha, split_probability
                )
            rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(
    *,
    n_iters=10_000,
    rank=3,
    k=2,
    split_probability=0.5,
    alpha_levels=(0.01, 0.05, 0.10),
    seed=123,
):
    """Simulate p-values conditional on target feature ``k - 1``."""
    n_iters, rank, k, split_probability, alpha_levels, seed = _validate_inputs(
        n_iters, rank, k, split_probability, alpha_levels, seed
    )
    rng = np.random.default_rng(seed)
    cutoff = float(stats.chi.ppf(split_probability, df=rank))
    records = []
    attempts = 0
    accepted = 0
    target_feature = k - 1
    while accepted < n_iters:
        attempts += 1
        statistic = float(stats.chi.rvs(df=rank, random_state=rng))
        auxiliary_u = float(rng.random())
        selected = selected_features(statistic, cutoff, k)
        target = target_from_selected_set(selected, auxiliary_u)
        if target != target_feature:
            continue
        accepted += 1
        values = event_p_values(
            statistic,
            auxiliary_u,
            rank=rank,
            split_probability=split_probability,
            k=k,
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
                    "auxiliary_u_stratum": (
                        "u_first_target_interval"
                        if auxiliary_u < 1.0 / k
                        else "u_last_target_interval"
                    ),
                    "p_value": p_value,
                    "selection_event_definition": {
                        "feature_inclusion": (
                            f"{target_feature} in selected(T); auxiliary_u marginalized"
                        ),
                        "same_target": (
                            "target(sorted(selected(T)), observed auxiliary_u) "
                            f"== {target_feature}"
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
    )
    settings = {
        "experiment": "B_fixed_k_fixed_u_counterexample",
        "n_iters": n_iters,
        "rank": rank,
        "k": k,
        "split_probability": split_probability,
        "statistic_cutoff": cutoff,
        "alpha_levels": alpha_levels,
        "seed": seed,
        "methods": METHODS,
        "method_labels": METHOD_LABELS,
        "strata": STRATA,
        "attempts_until_n_conditioning_target": attempts,
        "empirical_conditioning_target_probability": n_iters / attempts,
        "theoretical_conditioning_target_probability": 1.0 / k,
        "selector": (
            "M(T)={0,...,k-1} for T<=cutoff and "
            "M(T)={k-1,...,2k-2} for T>cutoff"
        ),
        "selected_set_size": k,
        "target_rule": "uniform_from_selected",
        "conditioning_target": target_feature,
    }
    return {
        "summary": summary,
        "p_value_results": results,
        "settings": settings,
    }


def _plot_ecdf_difference(result, output_path):
    figure, axes = plt.subplots(
        len(METHODS), len(STRATA), figsize=(5 * len(STRATA), 4 * len(METHODS))
    )
    grid = np.linspace(0.0, 1.0, 501)
    for row, method in enumerate(METHODS):
        method_results = (
            result["p_value_results"]
            .loc[result["p_value_results"]["method"] == method]
            .reset_index(drop=True)
        )
        for column, stratum in enumerate(STRATA):
            axis = axes[row, column]
            values = np.sort(
                method_results.loc[
                    _stratum_mask(method_results, stratum), "p_value"
                ].to_numpy(dtype=float)
            )
            ecdf = np.searchsorted(values, grid, side="right") / values.size
            axis.plot(grid, ecdf - grid)
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
            axis.set(
                title=f"{METHOD_LABELS[method]}: {stratum}",
                xlabel="p",
                ylabel="ECDF(p) - p",
                xlim=(0, 1),
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_rejection_rates(result, output_path):
    summary = result["summary"]
    levels = result["settings"]["alpha_levels"]
    figure, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4))
    for axis, alpha in zip(np.atleast_1d(axes), levels):
        label = f"{alpha:g}"
        positions = np.arange(len(STRATA), dtype=float)
        offsets = np.linspace(-0.18, 0.18, len(METHODS))
        for offset, method in zip(offsets, METHODS):
            method_summary = summary.loc[summary["method"] == method].set_index(
                "auxiliary_u_stratum"
            )
            rates = method_summary.loc[
                list(STRATA), f"rejection_rate_{label}"
            ].to_numpy(dtype=float)
            expected = method_summary.loc[
                list(STRATA), f"expected_rejection_rate_{label}"
            ].to_numpy(dtype=float)
            observed = axis.scatter(
                positions + offset,
                rates,
                label=METHOD_LABELS[method],
                zorder=3,
            )
            axis.scatter(
                positions + offset,
                expected,
                color=observed.get_facecolor(),
                marker="x",
                zorder=4,
            )
        axis.axhline(alpha, color="black", linestyle="--", label="Nominal")
        axis.set_xticks(positions, [STRATUM_LABELS[stratum] for stratum in STRATA])
        axis.set(title=f"alpha = {alpha:g}", ylabel="Null rejection rate")
        axis.legend(fontsize=7, title="dot: observed; x: expected", title_fontsize=7)
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
    _plot_ecdf_difference(result, output_dir / "ecdf_difference_by_u_stratum.png")
    _plot_rejection_rates(result, output_dir / "rejection_rates_by_u_stratum.png")
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iters", type=int, default=10_000)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--split-probability", type=float, default=0.5)
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
        k=args.k,
        split_probability=args.split_probability,
        alpha_levels=args.alpha_levels,
        seed=args.seed,
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            DEFAULT_OUTPUT_DIR
            if args.k == 2
            else DEFAULT_OUTPUT_DIR.with_name(f"{DEFAULT_OUTPUT_DIR.name}_k{args.k}")
        )
    output_dir = write_results(result, output_dir)
    print(result["summary"].to_string(index=False))
    print(f"\nSaved results to {output_dir}")
    return result


if __name__ == "__main__":
    main()
