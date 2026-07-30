"""Generate a SHAP selection-region figure for user-supplied data seeds."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from si_shap import (
    compute_selection_regions,
    selection_regions_frame,
    plot_selection_regions,
)


def _parse_rf_parameter(argument):
    """Parse one NAME=VALUE Random Forest parameter from the command line."""
    name, separator, raw_value = argument.partition("=")
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
    """Return unique Random Forest overrides from NAME=VALUE pairs."""
    parameters = {}
    for name, value in arguments:
        if name in parameters:
            raise ValueError(f"Random Forest parameter {name!r} was repeated.")
        parameters[name] = value
    return parameters


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
        "--k-select",
        type=int,
        default=1,
        help="number of top-SHAP features to select per data set (default: 1)",
    )
    parser.add_argument(
        "--selection-event",
        choices=("exact_set", "feature_inclusion", "exact_ranking"),
        default="exact_set",
        help="conditioning event used for every feature-specific path",
    )
    parser.add_argument(
        "--rf-param",
        action="append",
        default=[],
        type=_parse_rf_parameter,
        metavar="NAME=VALUE",
        help=(
            "override a RandomForestRegressor parameter; repeat for multiple "
            "parameters, using JSON syntax for typed values"
        ),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    rf_params = _rf_parameters(args.rf_param)
    output_directory = Path("outputs") / "shap_selection_regions"
    output_directory.mkdir(parents=True, exist_ok=True)

    results = compute_selection_regions(
        dataset_seeds=args.dataset_seeds,
        n_samples=100,
        n_features=20,
        k_select=args.k_select,
        selection_decimals=10,
        grid_size=1001,
        grid_refinements=1,
        boundary_tol=1e-8,
        tail_probability=1e-8,
        rf_params=rf_params,
        selection_event=args.selection_event,
    )
    summary = selection_regions_frame(results)
    summary.to_csv(output_directory / "shap_selection_regions.csv", index=False)

    figure = plot_selection_regions(
        results,
        output_directory / "shap_selection_regions.png",
    )
    plt.close(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
