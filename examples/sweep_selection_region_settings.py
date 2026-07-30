"""Compare SHAP selection regions across conditioning, top-k, and forest settings.

The quick preset is intended as a smoke test.  The recommended preset runs
the compact comparison matrix described in the project README.  Every
experiment gets its own directory so plots from different settings are not
overwritten.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

# This command is a headless batch job.  A GUI backend such as TkAgg can leave
# Tk objects to be finalized by Random Forest worker threads and abort Python.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from si_shap import (
    compute_selection_regions,
    plot_selection_regions,
    selection_regions_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


RF_CONFIGS = {
    "baseline": {
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_leaf": 1,
    },
    "shallow": {
        "n_estimators": 100,
        "max_depth": 3,
        "min_samples_leaf": 2,
    },
    "medium": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 2,
    },
    "flexible": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 1,
    },
}

PRESETS = {
    "quick": {
        "k_select": (1, 2),
        "rf_configs": ("baseline", "medium"),
        "selection_events": ("exact_set",),
    },
    "recommended": {
        "k_select": (1, 2, 5, 10),
        "rf_configs": ("baseline", "medium"),
        "selection_events": ("exact_set",),
    },
    "comprehensive": {
        "k_select": (1, 2, 5, 10),
        "rf_configs": tuple(RF_CONFIGS),
        "selection_events": ("exact_set", "feature_inclusion"),
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_seeds",
        metavar="SEED",
        type=int,
        nargs="+",
        help="one or more random seeds for the generated data sets",
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="quick",
        help="experiment matrix to use when settings are not supplied",
    )
    parser.add_argument(
        "--k-select",
        type=int,
        nargs="+",
        help="top-k values; overrides the selected preset",
    )
    parser.add_argument(
        "--rf-config",
        choices=RF_CONFIGS,
        nargs="+",
        help="named forest configurations; overrides the selected preset",
    )
    parser.add_argument(
        "--selection-event",
        choices=("exact_set", "feature_inclusion", "same_target", "exact_ranking"),
        nargs="+",
        help="conditioning modes; overrides the selected preset",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=1001,
        help="selection-region scan size (default: 1001)",
    )
    parser.add_argument(
        "--grid-refinements",
        type=int,
        default=1,
        help="number of deterministic midpoint-refinement passes (default: 1)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="parallel jobs used by each forest (default: all processors)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "selection_region_sweep",
        help="root directory for sweep results",
    )
    return parser.parse_args(argv)


def resolve_experiments(args):
    """Return requested top-k values, forests, and conditioning modes."""
    preset = PRESETS[args.preset]
    k_values = tuple(args.k_select or preset["k_select"])
    rf_names = tuple(args.rf_config or preset["rf_configs"])
    selection_events = tuple(
        args.selection_event or preset["selection_events"]
    )
    if any(not 1 <= k <= 20 for k in k_values):
        raise ValueError("Every k-select value must lie between 1 and 20.")
    return k_values, rf_names, selection_events


def run_experiment(args, k_select, rf_name, selection_event):
    """Run and save one cell of the comparison matrix."""
    rf_params = {
        **RF_CONFIGS[rf_name],
        "random_state": 42,
        "n_jobs": args.n_jobs,
    }
    experiment_name = f"k{k_select}_{rf_name}_{selection_event}"
    experiment_directory = args.output_dir / experiment_name
    experiment_directory.mkdir(parents=True, exist_ok=True)

    print(f"Running {experiment_name}: {rf_params}", flush=True)
    results = compute_selection_regions(
        dataset_seeds=args.dataset_seeds,
        n_samples=100,
        n_features=20,
        k_select=k_select,
        selection_decimals=10,
        grid_size=args.grid_size,
        grid_refinements=args.grid_refinements,
        boundary_tol=1e-8,
        tail_probability=1e-8,
        rf_params=rf_params,
        selection_event=selection_event,
        target_rule=(
            "uniform_from_selected"
            if selection_event == "same_target"
            else "all_selected"
        ),
    )

    summary = selection_regions_frame(results)
    summary.insert(0, "experiment", experiment_name)
    summary.insert(1, "rf_config", rf_name)
    summary.insert(2, "rf_params", json.dumps(rf_params, sort_keys=True))
    # The region scan can be long-running, so ensure the output directory still
    # exists at the point of writing (for example, after external cleanup).
    experiment_directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(experiment_directory / "selection_regions.csv", index=False)

    figure = plot_selection_regions(
        results,
        experiment_directory / "selection_regions.png",
    )
    plt.close(figure)
    return summary


def main():
    # scikit-learn 1.9's parallel wrapper warns once per task when the process
    # starts with an empty warnings filter list, even though Random Forest uses
    # sklearn's matching Parallel/delayed helpers internally.
    if not warnings.filters:
        warnings.simplefilter("default")

    args = parse_args()
    k_values, rf_names, selection_events = resolve_experiments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_experiment(args, k_select, rf_name, selection_event)
        for k_select in k_values
        for rf_name in rf_names
        for selection_event in selection_events
    ]
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.output_dir / "all_selection_regions.csv", index=False)
    print(f"Saved {len(summaries)} experiments to {args.output_dir}")


if __name__ == "__main__":
    main()
