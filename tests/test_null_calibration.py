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

from examples import compare_selection_event_null_calibration as example
from examples import summarize_null_calibration_shards as shard_summary
from si_shap import null_calibration
from si_shap.selection import SelectionResult


class DummySelector:
    def select(self, X, response, k_select):
        ranking = np.arange(X.shape[1])
        return SelectionResult(
            selected_features=ranking[:k_select],
            importance=np.arange(X.shape[1], 0, -1, dtype=float),
            ranking=ranking,
        )

    def get_settings(self):
        return {"selector": "dummy"}


def _diagnostic_frame(event, feature, p_value):
    return pd.DataFrame(
        {
            "feature": [feature],
            "selection_event": [event],
            "raw_selective_p_value": [p_value],
            "p_value": [p_value],
            "denominator_ess": [100.0],
            "tail_ess": [25.0],
            "mc_se": [0.01],
            "status": ["ok"],
        }
    )


def test_null_calibration_pairs_fixed_design_target_and_randomness(monkeypatch):
    calls = []

    def fake_inference(X, y, **kwargs):
        event = kwargs["selection_event"]
        target = kwargs["target_feature"]
        calls.append(
            {
                "X": X.copy(),
                "y": y.copy(),
                "event": event,
                "target": target,
                "u": kwargs["auxiliary_u"],
                "target_seed": kwargs["target_seed"],
                "ais_seed": kwargs["ais_seed"],
                "multiplicity": kwargs["multiplicity"],
            }
        )
        p_value = {
            "feature_inclusion": 0.1,
            "exact_set": 0.2,
            "same_target": 0.3,
        }[event]
        return {
            "observed_selected_features": np.array([0, 1]),
            "observed_target_feature": target,
            "auxiliary_u": kwargs["auxiliary_u"],
            "feature_results": _diagnostic_frame(event, target, p_value),
        }

    monkeypatch.setattr(null_calibration, "selective_inference", fake_inference)
    result = null_calibration.compare_selection_event_null_calibration(
        n_iters=2,
        n_samples=20,
        n_features=4,
        k_select=2,
        selector=DummySelector(),
    )

    assert len(result["p_value_results"]) == 6
    assert result["settings"]["fixed_design"] is True
    assert result["settings"]["multiplicity"] == "none"
    for call in calls[1:]:
        np.testing.assert_array_equal(calls[0]["X"], call["X"])
        assert call["multiplicity"] == "none"
    assert not np.array_equal(calls[0]["y"], calls[3]["y"])
    for offset in (0, 3):
        group = calls[offset : offset + 3]
        assert len({call["target"] for call in group}) == 1
        assert len({call["u"] for call in group}) == 1
        assert len({call["target_seed"] for call in group}) == 1
        assert len({call["ais_seed"] for call in group}) == 1


def test_null_calibration_can_hold_auxiliary_randomness_fixed(monkeypatch):
    auxiliary_values = []

    def fake_inference(X, y, **kwargs):
        auxiliary_values.append(kwargs["auxiliary_u"])
        event = kwargs["selection_event"]
        target = kwargs["target_feature"]
        return {
            "observed_selected_features": np.array([0, 1]),
            "observed_target_feature": target,
            "auxiliary_u": kwargs["auxiliary_u"],
            "feature_results": _diagnostic_frame(event, target, 0.5),
        }

    monkeypatch.setattr(null_calibration, "selective_inference", fake_inference)
    result = null_calibration.compare_selection_event_null_calibration(
        n_iters=3,
        n_samples=20,
        n_features=4,
        k_select=2,
        selector=DummySelector(),
        selection_events=("feature_inclusion", "same_target"),
        fixed_auxiliary_u=0.75,
    )

    assert auxiliary_values == [0.75] * 6
    assert result["settings"]["fixed_auxiliary_u"] == 0.75
    assert (
        result["settings"]["auxiliary_randomization_mode"]
        == "fixed_across_iterations"
    )


@pytest.mark.parametrize("value", [-0.1, 1.0, np.nan, np.inf, "invalid"])
def test_null_calibration_rejects_invalid_fixed_auxiliary_u(value):
    with pytest.raises((TypeError, ValueError)):
        null_calibration.compare_selection_event_null_calibration(
            n_iters=1,
            n_samples=20,
            n_features=4,
            k_select=2,
            selector=DummySelector(),
            selection_events=("feature_inclusion", "same_target"),
            fixed_auxiliary_u=value,
        )


def test_calibration_summary_reports_failure_bounds():
    frame = pd.DataFrame(
        {
            "selection_event": ["same_target"] * 3,
            "p_value": [0.01, np.nan, 0.2],
            "denominator_ess": [100.0, 0.0, 90.0],
            "tail_ess": [20.0, 0.0, 15.0],
            "mc_se": [0.01, np.nan, 0.02],
        }
    )

    summary = null_calibration._calibration_summary(
        frame, n_iters=3, alpha_levels=(0.05,)
    )

    assert np.isnan(summary["rejection_rate_0.05"])
    assert summary["converged_rejection_rate_0.05"] == 0.5
    assert summary["rejection_rate_lower_bound_0.05"] == pytest.approx(1 / 3)
    assert summary["rejection_rate_upper_bound_0.05"] == pytest.approx(2 / 3)
    assert summary["failure_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("events", "levels", "error"),
    [
        ("exact_set", (0.05,), TypeError),
        ((), (0.05,), ValueError),
        (("exact_set", "exact_set"), (0.05,), ValueError),
        (("exact_set", "same_target"), 0.05, TypeError),
        (("exact_set", "same_target"), (0.0,), ValueError),
    ],
)
def test_null_calibration_validates_events_and_alpha_levels(events, levels, error):
    with pytest.raises(error):
        null_calibration._validate_null_inputs(
            2, 20, 4, 2, 1.0, events, levels
        )


def test_example_writes_calibration_bundle(tmp_path):
    events = ("feature_inclusion", "exact_set", "same_target")
    rows = [
        {"selection_event": event, "p_value": value}
        for event, value in zip(events, (0.1, 0.2, 0.3))
    ]
    result = {
        "calibration_summary": pd.DataFrame(
            {"selection_event": events, "failure_rate": [0.0] * 3}
        ),
        "p_value_results": pd.DataFrame(rows),
        "paired_rejection_comparisons": pd.DataFrame(
            {"first_event": [events[0]], "second_event": [events[1]]}
        ),
        "fixed_design": np.ones((5, 2)),
        "settings": {"selection_events": events},
    }

    example._write_results(result, tmp_path)

    expected = {
        "calibration_summary.csv",
        "p_value_results.csv",
        "paired_rejection_comparisons.csv",
        "fixed_design.npy",
        "settings.json",
        "p_value_histograms.png",
        "uniform_qq_plots.png",
        "empirical_cdf.png",
        "ecdf_difference.png",
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())


def test_null_cli_accepts_builtin_selection_methods():
    for method in (
        "shap",
        "mutual_information",
        "marginal_screening",
        "spline_screening",
    ):
        parsed = example.parse_args(["--selection-method", method])
        assert parsed.selection_method == method


def test_shard_summary_pools_iterations_and_writes_artifacts(tmp_path):
    events = ["exact_set", "feature_inclusion", "same_target"]
    settings = {
        "n_samples": 5,
        "n_features": 2,
        "k_select": 1,
        "sigma": 1.0,
        "selection_events": events,
        "target_rule": "uniform_from_selected",
        "multiplicity": "none",
        "inference_method": "conditional_mc",
        "alpha_levels": [0.05, 0.1],
        "design_seed": 42,
        "fixed_design": True,
        "max_final_samples": 20,
        "min_denominator_ess": 8.0,
        "min_tail_ess": 2.0,
        "stop_when_ess_met": False,
    }
    design = np.ones((5, 2))
    input_dir = tmp_path / "shards"
    for shard_index, p_values in enumerate(((0.01, 0.2, 0.3), (0.4, 0.5, 0.6))):
        shard_dir = input_dir / f"shard_{shard_index}"
        shard_dir.mkdir(parents=True)
        shard_settings = {**settings, "seed": 123 + shard_index, "n_iters": 1}
        (shard_dir / "settings.json").write_text(json.dumps(shard_settings))
        np.save(shard_dir / "fixed_design.npy", design)
        rows = [
            {
                "iteration": 1,
                "selection_event": event,
                "p_value": p_value,
                "denominator_ess": 10.0,
                "tail_ess": 4.0,
                "mc_se": 0.1,
                "finite_sample_valid": True,
            }
            for event, p_value in zip(events, p_values)
        ]
        pd.DataFrame(rows).to_csv(shard_dir / "p_value_results.csv", index=False)

    output_dir = tmp_path / "summary"
    result = shard_summary.summarize_shards(input_dir, output_dir, dpi=40)

    assert result["metadata"]["n_iterations"] == 2
    assert result["p_value_results"]["iteration"].nunique() == 2
    assert set(result["p_value_results"]["shard"]) == {"shard_0", "shard_1"}
    expected = {
        "pooled_p_value_results.csv",
        "pooled_calibration_summary.csv",
        "pooled_paired_rejection_comparisons.csv",
        "pooled_metadata.json",
        "pooled_p_value_histograms.png",
        "pooled_uniform_qq_plots.png",
        "pooled_ecdf_difference.png",
        "pooled_rejection_rates.png",
        "pooled_ess_diagnostics.png",
    }
    assert expected.issubset(path.name for path in output_dir.iterdir())
