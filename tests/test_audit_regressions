import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pandas as pd
import pytest

from examples import compare_selection_event_power as power_example
from examples import summarize_null_calibration_shards as shard_summary
from si_shap import selection_regions
from si_shap.api import adjust_p_values, selective_inference
from si_shap.selection import SelectionResult
from si_shap.selection_regions import selection_probability
from si_shap.simulation import run_simulation


class FixedSelector:
    def select(self, X, response, k_select):
        ranking = np.arange(X.shape[1])
        return SelectionResult(
            selected_features=ranking[:k_select],
            importance=np.arange(X.shape[1], 0, -1, dtype=float),
            ranking=ranking,
        )

    def get_settings(self):
        return {"selector": "fixed-audit-selector"}


def test_adjust_p_values_handles_multidimensional_families_by_flat_index():
    values = np.array([[0.01, 0.04], [0.2, 0.8]])

    np.testing.assert_allclose(
        adjust_p_values(values, "bonferroni"),
        [[0.04, 0.16], [0.8, 1.0]],
    )
    np.testing.assert_allclose(
        adjust_p_values(values, "holm"),
        [[0.04, 0.12], [0.4, 0.8]],
    )


def test_adjust_p_values_rejects_infinity_instead_of_hiding_it_as_failure():
    with pytest.raises(ValueError, match="infinite"):
        adjust_p_values([0.2, np.inf], "bonferroni")


def test_run_simulation_rejects_unsupported_same_target_before_running():
    with pytest.raises(ValueError, match="does not define a randomized single-target"):
        run_simulation(
            1,
            10,
            2,
            1,
            selector=FixedSelector(),
            selection_event="same_target",
        )


def test_run_simulation_rejects_holm_with_distinct_inclusion_events():
    with pytest.raises(ValueError, match="Holm adjustment is not implemented"):
        run_simulation(
            1,
            10,
            2,
            2,
            selector=FixedSelector(),
            selection_event="feature_inclusion",
            multiplicity="holm",
        )


def test_impossible_ais_ess_target_is_rejected_before_selection():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)

    with pytest.raises(ValueError, match="mathematically unattainable"):
        selective_inference(
            X,
            y,
            k_select=1,
            sigma=1.0,
            selector=FixedSelector(),
            inference_method="ais",
            max_final_samples=10,
            min_denominator_ess=11,
        )


def test_power_presets_propagate_pilot_iterations_and_allow_override():
    assert power_example.parse_args(["--preset", "regulated"]).pilot_iters == 4
    assert power_example.parse_args(["--preset", "improved"]).pilot_iters == 5
    assert power_example.parse_args(
        ["--preset", "improved", "--pilot-iters", "7"]
    ).pilot_iters == 7


def test_selection_probability_rejects_overlaps_hidden_by_clipping():
    with pytest.raises(ValueError, match="non-overlapping"):
        selection_probability(((0.0, 3.0), (2.0, 4.0)), rank=3)
    assert selection_probability(((0.0, np.inf),), rank=3) == 1.0


def test_selection_region_extreme_tail_cutoff_remains_finite(monkeypatch):
    monkeypatch.setattr(
        selection_regions,
        "find_selection_intervals",
        lambda is_selected, z_max, **kwargs: ((0.0, z_max),),
    )
    monkeypatch.setattr(
        selection_regions,
        "_adapt_proposal",
        lambda t_obs, rank, is_selected, rng, **kwargs: (t_obs, 1.0),
    )

    result = selection_regions.compute_selection_regions(
        dataset_seeds=[9],
        n_samples=10,
        n_features=2,
        k_select=1,
        selector=FixedSelector(),
        tail_probability=1e-20,
    )[0]

    assert np.isfinite(result.z_max)
    assert result.omitted_tail_probability <= 1.01e-20


def _shard_settings(seed, *, selector_name="fixed"):
    return {
        "n_iters": 1,
        "n_samples": 10,
        "n_features": 2,
        "k_select": 1,
        "sigma": 1.0,
        "selection_events": ["exact_set", "same_target"],
        "target_rule": "uniform_from_selected",
        "multiplicity": "none",
        "inference_method": "conditional_mc",
        "alpha_levels": [0.05],
        "design_seed": 4,
        "fixed_design": True,
        "selection_decimals": 10,
        "pilot_iters": 3,
        "pilot_samples": 40,
        "final_batch_size": 20,
        "max_final_samples": 20,
        "min_denominator_ess": 8.0,
        "min_tail_ess": 2.0,
        "stop_when_ess_met": False,
        "variance_method": "known_simulation_sigma",
        "rf_params": None,
        "selector_settings": {"selector": selector_name},
        "seed": seed,
    }


def test_shard_validation_rejects_duplicate_random_seeds(tmp_path):
    shard_dirs = [tmp_path / "shard_0", tmp_path / "shard_1"]
    for path in shard_dirs:
        path.mkdir()

    with pytest.raises(ValueError, match="duplicate seed"):
        shard_summary._validate_shards(
            shard_dirs,
            [_shard_settings(11), _shard_settings(11)],
        )


def test_shard_validation_rejects_selector_mismatch(tmp_path):
    shard_dirs = [tmp_path / "shard_0", tmp_path / "shard_1"]
    for path in shard_dirs:
        path.mkdir()

    with pytest.raises(ValueError, match="selector_settings"):
        shard_summary._validate_shards(
            shard_dirs,
            [_shard_settings(11), _shard_settings(12, selector_name="other")],
        )


def test_shard_loader_rejects_missing_iteration_event_rows(tmp_path):
    shard_dir = tmp_path / "shard_0"
    shard_dir.mkdir()
    settings = _shard_settings(11)
    (shard_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "iteration": 1,
                "selection_event": "exact_set",
                "p_value": 0.5,
                "denominator_ess": 10.0,
                "tail_ess": 5.0,
                "mc_se": 0.1,
                "finite_sample_valid": True,
            }
        ]
    ).to_csv(shard_dir / "p_value_results.csv", index=False)

    with pytest.raises(ValueError, match="event rows do not match"):
        shard_summary._load_pooled_results([shard_dir], [settings])
