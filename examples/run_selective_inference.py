"""Run SHAP selective inference with adaptive importance sampling.

This executable example generates data under the project's global-null model,
selects the top-k features by mean absolute Tree SHAP importance, and estimates
one selective p-value for every selected feature with AIS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from si_shap import run_simulation


def _parse_rf_parameter(argument):
    """Parse one NAME=VALUE Random Forest parameter."""
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
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--k-select", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--selection-decimals", type=int, default=10)
    parser.add_argument(
        "--selection-event",
        choices=("exact_set", "feature_inclusion", "exact_ranking"),
        default="exact_set",
    )
    parser.add_argument(
        "--multiplicity",
        choices=("none", "holm", "bonferroni"),
        default="none",
    )
    parser.add_argument("--pilot-iters", type=int, default=3)
    parser.add_argument("--pilot-samples", type=int, default=40)
    parser.add_argument("--final-batch-size", type=int, default=80)
    parser.add_argument("--max-final-samples", type=int, default=800)
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
        default=Path("outputs") / "selective_inference_ais",
    )
    return parser.parse_args(argv)


def _p_values_frame(result):
    """Return all converged p-values in long form."""
    records = [
        {"method": method, "p_value": float(p_value)}
        for method, p_values in result["p_values"].items()
        for p_value in p_values
    ]
    return pd.DataFrame.from_records(records, columns=["method", "p_value"])


def main(argv=None):
    args = parse_args(argv)
    rf_params = _rf_parameters(args.rf_param)

    result = run_simulation(
        n_iters=args.n_iters,
        n_samples=args.n_samples,
        n_features=args.n_features,
        k_select=args.k_select,
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
        selection_event=args.selection_event,
        multiplicity=args.multiplicity,
        stop_when_ess_met=args.stop_when_ess_met,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(args.output_dir / "summary.csv", index=False)
    result["ais_diagnostics"].to_csv(
        args.output_dir / "selective_inference.csv", index=False
    )
    _p_values_frame(result).to_csv(
        args.output_dir / "converged_p_values.csv", index=False
    )
    with (args.output_dir / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **result["settings"],
                "alpha": args.alpha,
                "pilot_iters": args.pilot_iters,
                "pilot_samples": args.pilot_samples,
                "final_batch_size": args.final_batch_size,
                "max_final_samples": args.max_final_samples,
                "min_denominator_ess": args.min_denominator_ess,
                "min_tail_ess": args.min_tail_ess,
                "selection_event": args.selection_event,
                "multiplicity": args.multiplicity,
                "stop_when_ess_met": args.stop_when_ess_met,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    reported_p_value_column = (
        "raw_selective_p_value"
        if args.multiplicity == "none"
        else "adjusted_selective_p_value"
    )
    if reported_p_value_column not in result["ais_diagnostics"]:
        reported_p_value_column = "p_value"
    result["ais_diagnostics"]["reported_p_value"] = result["ais_diagnostics"][
        reported_p_value_column
    ]
    columns = [
        "iteration",
        "feature",
        "rank",
        "t_obs",
        "reported_p_value",
        "status",
        "proposals",
        "selected_samples",
        "tail_samples",
        "denominator_ess",
        "tail_ess",
        "mc_se",
    ]
    print("\nSelective-inference results (AIS):")
    print(result["ais_diagnostics"][columns].to_string(index=False))
    print(f"\nSaved results to {args.output_dir}")
    return result


if __name__ == "__main__":
    main()
