"""Compare exact-set and feature-inclusion power on paired simulations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from si_shap import compare_selection_event_power


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "selection_event_power"
MIN_FALLBACK_COMPLETE_ITERATIONS = 20
MIN_FALLBACK_COMPLETE_FRACTION = 0.5
PRESETS = {
    "quick": {
        "n_iters": 10,
        "signal_strength": 0.3,
        "pilot_samples": 40,
        "max_final_samples": 800,
    },
    "calibrated": {
        "n_iters": 100,
        "signal_strength": 0.3,
        "pilot_samples": 40,
        "max_final_samples": 1600,
    },
    "calibrated_plus": {
        "n_iters": 100,
        "signal_strength": 0.3,
        "pilot_samples": 200,
        "max_final_samples": 6400,
    },
    "regulated": {
        "n_iters": 100,
        "signal_strength": 0.3,
        "pilot_samples": 40,
        "pilot_iters": 4,
        "max_final_samples": 6400,
    },
    "improved": {
            "n_iters": 100,
            "signal_strength": 0.3,
            "pilot_samples": 100,
            "pilot_iters": 5,
            "max_final_samples": 6400,
        },
    "stress": {
        "n_iters": 100,
        "signal_strength": 1.0,
        "pilot_samples": 40,
        "max_final_samples": 6400,
    },
}
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


def _parse_rf_parameter(argument):
    """Parse one NAME=VALUE Random Forest parameter."""
    name, separator, raw_value = argument.partition("=")
    name = name.strip()
    if not separator or not name:
        raise argparse.ArgumentTypeError(
            "Random Forest parameters must use NAME=VALUE syntax."
        )
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return name, value


def _rf_parameters(arguments):
    """Return unique Random Forest overrides from parsed NAME=VALUE pairs."""
    parameters = {}
    for name, value in arguments:
        if name in parameters:
            raise ValueError(f"Random Forest parameter {name!r} was repeated.")
        parameters[name] = value
    return parameters


def _apply_preset(args):
    """Fill unset experiment controls from the selected named preset."""
    preset = PRESETS[args.preset]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="calibrated",
        help=(
            "named defaults for iterations, signal strength, proposal pilot "
            "size, and AIS budget"
        ),
    )
    parser.add_argument("--n-iters", type=int, help="simulation iterations")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--k-select", type=int, default=2)
    parser.add_argument("--signal-features", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--signal-strength",
        type=float,
        help="per-feature empirical signal SD",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--selection-decimals", type=int, default=10)
    parser.add_argument(
        "--selection-events",
        nargs="+",
        choices=("exact_set", "feature_inclusion", "exact_ranking"),
        default=["exact_set", "feature_inclusion"],
    )
    parser.add_argument(
        "--multiplicity",
        choices=("none", "holm", "bonferroni"),
        default="none",
    )
    parser.add_argument("--pilot-iters", type=int, default=3)
    parser.add_argument(
        "--pilot-samples",
        type=int,
        help="proposal samples per pilot iteration",
    )
    parser.add_argument("--final-batch-size", type=int, default=80)
    parser.add_argument(
        "--max-final-samples",
        type=int,
        help="maximum final AIS proposals per selected-feature test",
    )
    parser.add_argument("--min-denominator-ess", type=float, default=80.0)
    parser.add_argument("--min-tail-ess", type=float, default=15.0)
    parser.add_argument(
        "--stop-when-ess-met",
        action="store_true",
        help=(
            "enable exploratory ESS-based early stopping; the default uses the "
            "full fixed final-sample budget"
        ),
    )
    parser.add_argument(
        "--rf-param",
        action="append",
        default=[],
        type=_parse_rf_parameter,
        metavar="NAME=VALUE",
        help="override a RandomForestRegressor parameter; may be repeated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "directory for CSV, JSON, and PNG outputs "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    return _apply_preset(parser.parse_args(argv))


def _power_plot_data(summary):
    """Return guarded values and diagnostics used by the power plot."""
    strict = summary["power"].to_numpy(dtype=float)
    converged = summary["converged_power"].to_numpy(dtype=float)
    n_complete = summary["n_complete_iterations"].to_numpy(dtype=int)
    n_iterations = summary["n_iterations"].to_numpy(dtype=int)
    minimum_complete = np.maximum(
        MIN_FALLBACK_COMPLETE_ITERATIONS,
        np.ceil(MIN_FALLBACK_COMPLETE_FRACTION * n_iterations).astype(int),
    )
    fallback_used = (
        ~np.isfinite(strict)
        & np.isfinite(converged)
        & (n_complete >= minimum_complete)
    )
    values = np.where(np.isfinite(strict), strict, np.nan)
    values = np.where(fallback_used, converged, values)
    errors = summary["converged_simulation_se"].to_numpy(dtype=float)
    errors = np.where(np.isfinite(values) & np.isfinite(errors), errors, 0.0)
    return values, errors, fallback_used, n_complete, n_iterations


def _plot_power_comparison(result, output_path):
    """Save power estimates without promoting tiny complete-case subsets."""
    summary = result["summary"]
    values, errors, fallback_used, n_complete, n_iterations = _power_plot_data(
        summary
    )
    plotted_values = np.where(np.isfinite(values), values, 0.0)

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    bars = axis.bar(
        summary["selection_event"].str.replace("_", "-"),
        plotted_values,
        yerr=errors,
        capsize=5,
        color=("tab:blue", "tab:green", "tab:orange")[: len(summary)],
        alpha=0.8,
    )
    for bar, value, plotted_value, used_fallback, complete, total in zip(
        bars,
        values,
        plotted_values,
        fallback_used,
        n_complete,
        n_iterations,
    ):
        if np.isfinite(value):
            label = f"{value:.3f}"
            if used_fallback:
                label += f"*\n({complete}/{total} complete)"
        else:
            label = f"NA\n({complete}/{total} complete)"
        label_inside_top = plotted_value >= 0.95
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            plotted_value - 0.015 if label_inside_top else plotted_value + 0.015,
            label,
            ha="center",
            va="top" if label_inside_top else "bottom",
        )
    axis.axhline(result["alpha"], color="gray", linestyle=":", linewidth=1)
    axis.set_ylim(
        0.0,
        min(1.0, max(0.15, float(np.max(plotted_values + errors)) + 0.12)),
    )
    axis.set_ylabel("Overall power: P(signal selected and rejected)")
    axis.set_title(
        "Paired comparison of SHAP selection-conditioning events", pad=12
    )
    if np.any(fallback_used):
        figure.text(
            0.5,
            0.01,
            "* converged iterations only; complete-case count shown",
            ha="center",
            fontsize=9,
        )
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _runtime_metadata():
    """Return environment and source-control metadata for reproducibility."""
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


def _validate_output_location(output_dir):
    """Fail before simulation if the output location cannot be written."""
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    probe = tempfile.NamedTemporaryFile(dir=output_dir.parent, delete=False)
    probe_path = Path(probe.name)
    probe.close()
    probe_path.unlink()
    return output_dir


def _write_results_to_directory(result, output_dir):
    """Stage a complete result bundle and replace the destination directory."""
    output_dir = output_dir.resolve()
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    backup_dir = None
    try:
        if output_dir.exists():
            shutil.copytree(output_dir, staging_dir, dirs_exist_ok=True)
        result["summary"].to_csv(staging_dir / "power_summary.csv", index=False)
        result["comparisons"].to_csv(
            staging_dir / "paired_power_comparison.csv", index=False
        )
        result["signal_results"].to_csv(
            staging_dir / "signal_results.csv", index=False
        )
        result["feature_results"].to_csv(
            staging_dir / "feature_results.csv", index=False
        )
        with (staging_dir / "settings.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(result["settings"], file, indent=2, sort_keys=True)
        _plot_power_comparison(result, staging_dir / "power_comparison.png")

        if output_dir.exists():
            backup_dir = output_dir.with_name(
                f".{output_dir.name}.backup-{uuid4().hex}"
            )
            os.replace(output_dir, backup_dir)
        try:
            os.replace(staging_dir, output_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists():
                os.replace(backup_dir, output_dir)
            raise
        if backup_dir is not None:
            try:
                shutil.rmtree(backup_dir)
            except OSError as error:
                warnings.warn(
                    f"Results were saved, but backup cleanup failed: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def main(argv=None):
    args = parse_args(argv)
    output_dir = _validate_output_location(args.output_dir)
    rf_params = _rf_parameters(args.rf_param)
    result = compare_selection_event_power(
        n_iters=args.n_iters,
        n_samples=args.n_samples,
        n_features=args.n_features,
        k_select=args.k_select,
        signal_features=args.signal_features,
        signal_strength=args.signal_strength,
        selection_events=args.selection_events,
        alpha=args.alpha,
        seed=args.seed,
        selection_decimals=args.selection_decimals,
        pilot_iters=args.pilot_iters,
        pilot_samples=args.pilot_samples,
        final_batch_size=args.final_batch_size,
        max_final_samples=args.max_final_samples,
        min_denominator_ess=args.min_denominator_ess,
        min_tail_ess=args.min_tail_ess,
        rf_params=rf_params,
        multiplicity=args.multiplicity,
        stop_when_ess_met=args.stop_when_ess_met,
    )
    result["settings"]["run_preset"] = args.preset
    result["settings"]["runtime_metadata"] = _runtime_metadata()
    _write_results_to_directory(result, output_dir)

    display_columns = [
        "selection_event",
        "power",
        "simulation_se",
        "converged_power",
        "conditional_power",
        "signal_selection_rate",
        "signal_test_failure_rate",
        "n_complete_iterations",
        "n_iterations",
        "n_converged_signal_tests",
        "n_selected_signal_tests",
    ]
    print("\nSelection-event power comparison:")
    print(result["summary"][display_columns].to_string(index=False))
    print("\nPaired differences (comparison minus baseline):")
    print(result["comparisons"].to_string(index=False))
    print(f"\nSaved results to {output_dir}")
    return result


if __name__ == "__main__":
    main()
