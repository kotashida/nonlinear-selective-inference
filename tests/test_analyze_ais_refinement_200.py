import numpy as np
import pandas as pd

from examples.analyze_ais_refinement_200 import (
    _mean_ci,
    paired_cost_comparisons,
    paired_failure_comparisons,
)


def _detail():
    rows = []
    conditions = [
        "baseline_control", "a_multimodal_retention",
        "a_robust_consolidation", "a_robust_draw_floor",
        "a_robust_draw_floor_pilot15", "a_cdf_interval",
        "a_robust_floor_multimodal_denominator", "a_full_stack_optimized",
    ]
    for condition_index, condition in enumerate(conditions):
        for iteration in range(4):
            rows.append({
                "condition": condition, "scope": "power", "shard": "shard_0",
                "iteration": iteration, "is_signal": iteration < 2,
                "failed": iteration < max(0, 2 - condition_index),
                "aj_evaluations": 4000 + 800 * iteration + 100 * condition_index,
            })
    return pd.DataFrame(rows)


def test_mean_ci_contains_sample_mean():
    mean, lower, upper = _mean_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert lower < mean < upper


def test_paired_reports_compare_every_new_arm_to_both_references():
    detail = _detail()
    failures = paired_failure_comparisons(detail)
    costs = paired_cost_comparisons(detail)
    assert len(failures) == 6 * 2 * 2
    assert len(costs) == 6 * 2
    assert set(costs["reference"]) == {
        "baseline_control", "a_multimodal_retention"
    }
    assert np.isfinite(costs["mean_paired_cost_difference"]).all()
