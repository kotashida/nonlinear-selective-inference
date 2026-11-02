"""Small fixed-design null/power experiment for exact-set linear Shapley SI.

Run: python examples/run_linear_shapley_simulation.py --iterations 100
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from si_shap import LinearShapleySelector, selective_inference


def run_experiment(iterations=100, n_samples=60, n_features=4, k_select=2,
                   n_proposals=1000, seed=2026):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    selector = LinearShapleySelector()
    records = []
    # At effect=0 every tested hypothesis is null. At effect>0 count feature
    # zero's rejections, both conditional on selection and across all trials.
    for effect in (0.0, 0.5, 1.0):
        for iteration in range(iterations):
            y = effect * X[:, 0] + rng.normal(size=n_samples)
            with warnings.catch_warnings():
                # Retained-draw counts are reported in the experiment summary.
                warnings.filterwarnings("ignore", message="Exact-set conditioning can make the event rare.*", category=UserWarning)
                result = selective_inference(
                    X, y, k_select=k_select, sigma=1, selector=selector,
                    selection_event="exact_set", target_rule="all_selected",
                    max_final_samples=n_proposals, ais_seed=seed + iteration,
                )
            for row in result["feature_results"].to_dict("records"):
                records.append({"effect": effect, "iteration": iteration,
                                "feature": row["feature"], "p_value": row["p_value"],
                                "rejected": row["p_value"] <= 0.05,
                                "selected_samples": row["selected_samples"]})
    rows = pd.DataFrame(records)
    summary = []
    for effect, group in rows.groupby("effect"):
        target = group[group.feature == 0]
        summary.append({"effect": effect,
                        "null_featurewise_rejection_rate": group.rejected.mean() if effect == 0 else np.nan,
                        "signal_selection_rate": len(target) / iterations,
                        "signal_conditional_rejection_rate": target.rejected.mean(),
                        "signal_selection_and_rejection_rate": target.rejected.sum() / iterations,
                        "mean_retained_draws": group.selected_samples.mean()})
    return rows, pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--proposals", type=int, default=1000)
    parser.add_argument("--output", help="Optional CSV path for replicate results")
    args = parser.parse_args()
    rows, summary = run_experiment(iterations=args.iterations, n_proposals=args.proposals)
    print(summary.to_string(index=False))
    if args.output:
        rows.to_csv(args.output, index=False)
