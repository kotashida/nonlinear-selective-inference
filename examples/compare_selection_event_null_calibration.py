"""Compare selective-p-value calibration for three selection events."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np

from si_shap import compare_selection_event_null_calibration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "selection_event_null_calibration"
PACKAGE_DISTRIBUTIONS = (
    "si-shap",
    "numpy",
    "pandas",
    "scipy",
    "patsy",
    "scikit-learn",
    "shap",
    "matplotlib",
    "tqdm",
)


def _runtime_metadata():
    """Return dependency and source metadata used to validate pooled shards."""
    package_versions = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None

    git_commit = None
    git_dirty = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            git_commit = commit.stdout.strip()
        if status.returncode == 0:
            git_dirty = bool(status.stdout.strip())
    except OSError:
        pass

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _parse_rf_parameter(argument):
    name, separator, raw_value = argument.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("RF parameters must use NAME=VALUE syntax.")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return name, value


def _rf_parameters(arguments):
    parameters = {}
    for name, value in arguments:
        if name in parameters:
            raise ValueError(f"Random Forest parameter {name!r} was repeated.")
        parameters[name] = value
    return parameters


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--k-select", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--feature-correlation", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--design-seed", type=int)
    parser.add_argument(
        "--fixed-auxiliary-u",
        type=float,
        help=(
            "reuse one target-selection draw in [0,1) across all iterations; "
            "omit to redraw it independently in each iteration"
        ),
    )
    parser.add_argument(
        "--selection-events",
        nargs="+",
        choices=("feature_inclusion", "exact_set", "same_target", "exact_ranking"),
        default=["feature_inclusion", "exact_set", "same_target"],
    )
    parser.add_argument(
        "--alpha-levels", type=float, nargs="+", default=[0.01, 0.05, 0.10]
    )
    parser.add_argument("--selection-decimals", type=int, default=10)
    parser.add_argument(
        "--selection-method",
        choices=("shap", "mutual_information", "marginal_screening"),
        default="shap",
    )
    parser.add_argument("--pilot-iters", type=int, default=3)
    parser.add_argument("--pilot-samples", type=int, default=40)
    parser.add_argument("--final-batch-size", type=int, default=80)
    parser.add_argument("--max-final-samples", type=int, default=800)
    parser.add_argument("--min-denominator-ess", type=float, default=80.0)
    parser.add_argument("--min-tail-ess", type=float, default=15.0)
    parser.add_argument("--stop-when-ess-met", action="store_true")
    parser.add_argument(
        "--inference-method",
        choices=("conditional_mc", "ais"),
        default="conditional_mc",
    )
    parser.add_argument(
        "--rf-param",
        action="append",
        default=[],
        type=_parse_rf_parameter,
        metavar="NAME=VALUE",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _event_p_values(result, event):
    frame = result["p_value_results"]
    values = frame.loc[frame["selection_event"] == event, "p_value"].to_numpy(
        dtype=float
    )
    return values[np.isfinite(values)]


def _plot_histograms(result, output_path):
    events = result["settings"]["selection_events"]
    figure, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 4), squeeze=False)
    for axis, event in zip(axes[0], events):
        values = _event_p_values(result, event)
        axis.hist(values, bins=np.linspace(0.0, 1.0, 11), density=True, alpha=0.75)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set(title=event, xlabel="Selective p-value", ylabel="Density")
        axis.set_xlim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_uniform_qq(result, output_path):
    events = result["settings"]["selection_events"]
    figure, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 4), squeeze=False)
    for axis, event in zip(axes[0], events):
        observed = np.sort(_event_p_values(result, event))
        expected = (np.arange(observed.size) + 0.5) / observed.size if observed.size else np.array([])
        axis.scatter(expected, observed, s=12)
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.set(title=event, xlabel="Uniform quantile", ylabel="Observed quantile")
        axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_ecdf(result, output_path):
    events = result["settings"]["selection_events"]
    figure, axis = plt.subplots(figsize=(7, 5))
    grid = np.linspace(0.0, 1.0, 501)
    for event in events:
        values = np.sort(_event_p_values(result, event))
        if values.size:
            ecdf = np.searchsorted(values, grid, side="right") / values.size
            axis.plot(grid, ecdf, label=event)
    axis.plot(grid, grid, color="black", linestyle="--", label="Uniform CDF")
    axis.set(xlabel="p", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_ecdf_difference(result, output_path):
    events = result["settings"]["selection_events"]
    figure, axes = plt.subplots(
        1, len(events), figsize=(5 * len(events), 4), squeeze=False
    )
    grid = np.linspace(0.0, 1.0, 501)
    for axis, event in zip(axes[0], events):
        raw_values = result["p_value_results"].loc[
            result["p_value_results"]["selection_event"] == event, "p_value"
        ].to_numpy(dtype=float)
        values = np.sort(raw_values[np.isfinite(raw_values)])
        if values.size:
            difference = (
                np.searchsorted(values, grid, side="right") / values.size - grid
            )
            if np.all(np.isfinite(raw_values)):
                dkw = np.sqrt(np.log(1.0 / 0.05) / (2.0 * values.size))
                axis.axhline(
                    dkw,
                    color="gray",
                    alpha=0.7,
                    label="95% super-uniform upper band",
                )
            else:
                axis.text(
                    0.02,
                    0.98,
                    "Diagnostic only: failed p-values omitted",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=8,
                )
            axis.plot(grid, difference, label="ECDF - Uniform CDF")
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(title=event, xlabel="p", ylabel="ECDF(p) - p", xlim=(0, 1))
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_results(result, output_dir):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result["calibration_summary"].to_csv(
        output_dir / "calibration_summary.csv", index=False
    )
    result["p_value_results"].to_csv(
        output_dir / "p_value_results.csv", index=False
    )
    result["paired_rejection_comparisons"].to_csv(
        output_dir / "paired_rejection_comparisons.csv", index=False
    )
    np.save(output_dir / "fixed_design.npy", result["fixed_design"])
    with (output_dir / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(result["settings"], file, indent=2, sort_keys=True)
    _plot_histograms(result, output_dir / "p_value_histograms.png")
    _plot_uniform_qq(result, output_dir / "uniform_qq_plots.png")
    _plot_ecdf(result, output_dir / "empirical_cdf.png")
    _plot_ecdf_difference(result, output_dir / "ecdf_difference.png")


def main(argv=None):
    args = parse_args(argv)
    rf_params = _rf_parameters(args.rf_param) or None
    result = compare_selection_event_null_calibration(
        n_iters=args.n_iters,
        n_samples=args.n_samples,
        n_features=args.n_features,
        k_select=args.k_select,
        sigma=args.sigma,
        feature_correlation=args.feature_correlation,
        selection_events=args.selection_events,
        alpha_levels=args.alpha_levels,
        seed=args.seed,
        design_seed=args.design_seed,
        fixed_auxiliary_u=args.fixed_auxiliary_u,
        selection_decimals=args.selection_decimals,
        pilot_iters=args.pilot_iters,
        pilot_samples=args.pilot_samples,
        final_batch_size=args.final_batch_size,
        max_final_samples=args.max_final_samples,
        min_denominator_ess=args.min_denominator_ess,
        min_tail_ess=args.min_tail_ess,
        selection_method=args.selection_method,
        rf_params=rf_params,
        inference_method=args.inference_method,
        stop_when_ess_met=args.stop_when_ess_met,
    )
    result["settings"]["runtime_metadata"] = _runtime_metadata()
    _write_results(result, args.output_dir)
    print(result["calibration_summary"].to_string(index=False))
    print(f"\nSaved results to {args.output_dir.resolve()}")
    return result


if __name__ == "__main__":
    main()
