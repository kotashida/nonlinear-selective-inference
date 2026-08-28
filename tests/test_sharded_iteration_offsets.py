import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import pandas as pd

from si_shap.null_calibration import compare_selection_event_null_calibration
from si_shap.power import compare_selection_event_power


COMMON = {
    "n_samples": 20,
    "n_features": 4,
    "k_select": 2,
    "selection_events": ("same_target", "exact_set"),
    "seed": 77,
    "selection_method": "marginal_screening",
    "pilot_iters": 1,
    "pilot_samples": 2,
    "final_batch_size": 5,
    "max_final_samples": 5,
    "min_denominator_ess": 1,
    "min_tail_ess": 1,
}


def test_null_iteration_offsets_reproduce_one_contiguous_run():
    full = compare_selection_event_null_calibration(
        n_iters=2, design_seed=91, **COMMON
    )["p_value_results"]
    shards = [
        compare_selection_event_null_calibration(
            n_iters=1, iteration_start=index, design_seed=91, **COMMON
        )["p_value_results"]
        for index in range(2)
    ]
    pd.testing.assert_frame_equal(
        full.reset_index(drop=True), pd.concat(shards, ignore_index=True)
    )


def test_power_iteration_offsets_reproduce_one_contiguous_run():
    settings = {**COMMON, "signal_features": (0,), "signal_strength": 0.2}
    full = compare_selection_event_power(n_iters=2, **settings)
    shards = [
        compare_selection_event_power(n_iters=1, iteration_start=index, **settings)
        for index in range(2)
    ]
    for key in ("target_results", "feature_results"):
        pd.testing.assert_frame_equal(
            full[key].reset_index(drop=True),
            pd.concat([shard[key] for shard in shards], ignore_index=True),
        )
