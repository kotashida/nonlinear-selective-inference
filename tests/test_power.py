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
from si_shap import power
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


def test_example_defaults_to_repository_outputs_folder():
    args = power_example.parse_args([])

    assert args.output_dir == (
        power_example.PROJECT_ROOT / "outputs" / "selection_event_power"
    )
    assert args.preset == "calibrated"
    assert args.n_iters == 100
    assert args.signal_strength == 0.3
    assert args.pilot_samples == 40
    assert args.max_final_samples == 1600


def test_power_cli_accepts_builtin_selection_methods():
    for method in (
        "shap",
        "lime",
        "mutual_information",
        "marginal_screening",
        "spline_screening",
    ):
        assert power_example.parse_args(
            ["--selection-method", method]
        ).selection_method == method


def test_calibrated_plus_uses_larger_pilot_and_final_budgets():
    args = power_example.parse_args(["--preset", "calibrated_plus"])

    assert args.n_iters == 100
    assert args.signal_strength == 0.3
    assert args.pilot_samples == 200
    assert args.max_final_samples == 6400


def test_explicit_power_controls_override_preset():
    args = power_example.parse_args(
        [
            "--preset",
            "quick",
            "--n-iters",
            "7",
            "--signal-strength",
            "0.4",
            "--pilot-samples",
            "75",
            "--max-final-samples",
            "1200",
        ]
    )

    assert args.n_iters == 7
    assert args.signal_strength == 0.4
    assert args.pilot_samples == 75
    assert args.max_final_samples == 1200


def test_duplicate_rf_parameters_are_rejected():
    with pytest.raises(ValueError, match="was repeated"):
        power_example._rf_parameters(
            [("max_depth", 3), ("max_depth", 5)]
        )


def test_power_plot_never_promotes_converged_only_iterations():
    summary = pd.DataFrame(
        {
            "power": [np.nan, np.nan, 0.4],
            "converged_power": [1.0, 0.6, 0.4],
            "converged_simulation_se": [0.0, 0.1, 0.05],
            "n_complete_iterations": [2, 60, 100],
            "n_iterations": [100, 100, 100],
        }
    )

    values, errors, complete, total = power_example._power_plot_data(
        summary
    )

    assert np.isnan(values[0])
    assert np.isnan(values[1])
    assert values[2] == 0.4
    np.testing.assert_allclose(errors, [0.0, 0.0, 0.05])
    np.testing.assert_array_equal(complete, [2, 60, 100])
    np.testing.assert_array_equal(total, [100, 100, 100])


def test_power_plot_keeps_labels_inside_axes(monkeypatch, tmp_path):
    summary = pd.DataFrame(
        {
            "selection_event": ["exact_set", "feature_inclusion"],
            "power": [np.nan, 1.0],
            "converged_power": [1.0, 1.0],
            "converged_simulation_se": [0.0, 0.0],
            "n_complete_iterations": [2, 100],
            "n_iterations": [100, 100],
        }
    )
    captured = {}
    real_subplots = power_example.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axis = real_subplots(*args, **kwargs)
        captured["axis"] = axis
        return figure, axis

    monkeypatch.setattr(power_example.plt, "subplots", capture_subplots)
    output_path = tmp_path / "power.png"
    power_example._plot_power_comparison(
        {"summary": summary, "alpha": 0.05}, output_path
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    label_positions = [text.get_position()[1] for text in captured["axis"].texts]
    assert label_positions
    assert all(0.0 <= position <= 1.0 for position in label_positions)


def _fake_power_result():
    return {
        "summary": pd.DataFrame({"selection_event": ["exact_set"]}),
        "comparisons": pd.DataFrame({"power_difference": [0.1]}),
        "signal_results": pd.DataFrame({"iteration": [1]}),
        "feature_results": pd.DataFrame({"feature": [0]}),
        "settings": {"seed": 123},
    }


def test_result_bundle_is_staged_without_preserving_stale_files(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "power"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

    def fake_plot(_result, path):
        path.write_bytes(b"png")

    monkeypatch.setattr(power_example, "_plot_power_comparison", fake_plot)
    power_example._write_results_to_directory(
        _fake_power_result(), output_dir
    )

    assert not (output_dir / "keep.txt").exists()
    assert (output_dir / "power_summary.csv").is_file()
    assert (output_dir / "paired_power_comparison.csv").is_file()
    assert (output_dir / "signal_results.csv").is_file()
    assert (output_dir / "target_results.csv").is_file()
    assert (output_dir / "feature_results.csv").is_file()
    assert (output_dir / "settings.json").is_file()
    assert (output_dir / "power_comparison.png").read_bytes() == b"png"


def test_staging_failure_leaves_existing_outputs_unchanged(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "power"
    output_dir.mkdir()
    old_summary = output_dir / "power_summary.csv"
    old_summary.write_text("old", encoding="utf-8")

    def fail_plot(_result, _path):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(power_example, "_plot_power_comparison", fail_plot)
    with pytest.raises(RuntimeError, match="plot failed"):
        power_example._write_results_to_directory(
            _fake_power_result(), output_dir
        )

    assert old_summary.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".power.staging-*"))


def test_invalid_output_path_fails_before_simulation(monkeypatch, tmp_path):
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("occupied", encoding="utf-8")

    def should_not_run(**_kwargs):
        raise AssertionError("simulation should not run")

    monkeypatch.setattr(
        power_example, "compare_selection_event_power", should_not_run
    )
    with pytest.raises(NotADirectoryError):
        power_example.main(["--output-dir", str(output_path)])


def test_main_records_preset_metadata_and_passes_resolved_controls(
    monkeypatch, tmp_path
):
    received = {}
    written = {}
    fake_result = {
        "summary": pd.DataFrame(
            {
                "selection_event": ["exact_set"],
                "power": [0.5],
                "simulation_se": [0.1],
                "converged_power": [0.5],
                "conditional_power_given_signal_target": [0.5],
                "target_signal_rate": [1.0],
                "signal_target_test_failure_rate": [0.0],
                "n_complete_iterations": [1],
                "n_iterations": [1],
                "n_converged_signal_targets": [1],
                "n_signal_targets": [1],
            }
        ),
        "comparisons": pd.DataFrame(),
        "signal_results": pd.DataFrame(),
        "feature_results": pd.DataFrame(),
        "settings": {},
    }

    def fake_compare(**kwargs):
        received.update(kwargs)
        return fake_result

    def fake_write(result, output_dir):
        written["result"] = result
        written["output_dir"] = output_dir

    monkeypatch.setattr(
        power_example, "compare_selection_event_power", fake_compare
    )
    monkeypatch.setattr(
        power_example, "_runtime_metadata", lambda: {"python_version": "test"}
    )
    monkeypatch.setattr(
        power_example, "_write_results_to_directory", fake_write
    )

    output_dir = tmp_path / "power"
    power_example.main(
        [
            "--preset",
            "quick",
            "--n-iters",
            "1",
            "--rf-param",
            "max_depth=3",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert received["n_iters"] == 1
    assert received["signal_strength"] == 0.3
    assert received["max_final_samples"] == 800
    assert received["rf_params"] == {"max_depth": 3}
    assert written["result"]["settings"]["run_preset"] == "quick"
    assert written["result"]["settings"]["runtime_metadata"] == {
        "python_version": "test"
    }
    assert written["output_dir"] == output_dir.resolve()


def test_generate_power_dataset_is_reproducible():
    first = power._generate_power_dataset(
        np.random.default_rng(123), 30, 4, (0, 2), 0.75
    )
    second = power._generate_power_dataset(
        np.random.default_rng(123), 30, 4, (0, 2), 0.75
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_power_dataset_can_return_the_fixed_design_mean():
    X, response, mean = power._generate_power_dataset(
        np.random.default_rng(123),
        30,
        4,
        (0,),
        0.75,
        return_mean=True,
    )

    assert X.shape == (30, 4)
    assert response.shape == mean.shape == (30,)
    assert not np.allclose(mean, 0.0)


@pytest.mark.parametrize(
    ("signal_features", "signal_strength", "selection_events", "error"),
    [
        ((), 1.0, ("exact_set", "feature_inclusion"), ValueError),
        ((0, 0), 1.0, ("exact_set", "feature_inclusion"), ValueError),
        ((4,), 1.0, ("exact_set", "feature_inclusion"), ValueError),
        ((0,), 0.0, ("exact_set", "feature_inclusion"), ValueError),
        ((0,), 1.0, (), ValueError),
        ((0,), 1.0, ("exact_set", "exact_set"), ValueError),
    ],
)
def test_validate_power_inputs_rejects_invalid_settings(
    signal_features, signal_strength, selection_events, error
):
    with pytest.raises(error):
        power._validate_power_inputs(
            4, signal_features, signal_strength, selection_events
        )


def test_comparison_pairs_data_and_reports_power_difference(monkeypatch):
    calls = []

    def fake_selective_inference(X, y, **kwargs):
        selection_event = kwargs["selection_event"]
        target_feature = kwargs["target_feature"]
        calls.append(
            {
                "X": X.copy(),
                "y": y.copy(),
                "selection_event": selection_event,
                "ais_seed": kwargs["ais_seed"],
                "target_seed": kwargs["target_seed"],
                "target_feature": target_feature,
                "auxiliary_u": kwargs["auxiliary_u"],
            }
        )
        signal_p_value = 0.08 if selection_event == "exact_set" else 0.02
        frame = pd.DataFrame(
            {
                "feature": [target_feature],
                "selection_event": [selection_event],
                "raw_selective_p_value": [signal_p_value],
                "adjusted_selective_p_value": [signal_p_value],
                "resolution_status": ["resolved"],
                "minimum_attainable_p_value": [0.01],
            }
        )
        return {
            "observed_selected_features": np.array([0, 1]),
            "feature_results": frame,
        }

    monkeypatch.setattr(power, "selective_inference", fake_selective_inference)
    result = power.compare_selection_event_power(
        n_iters=2,
        n_samples=20,
        n_features=4,
        k_select=2,
        signal_features=(0, 1),
        signal_strength=1.0,
        selector=DummySelector(),
    )

    inclusion, exact = result["summary"].set_index("selection_event").loc[
        ["feature_inclusion", "exact_set"]
    ].to_dict(orient="records")
    assert exact["power"] == 0.0
    assert inclusion["power"] == 1.0
    assert exact["target_signal_rate"] == 1.0
    assert inclusion["target_signal_rate"] == 1.0
    assert result["comparisons"].loc[0, "power_difference"] == -1.0
    assert result["comparisons"].loc[0, "n_complete_pairs"] == 2
    assert "converged_non_signal_rejection_rate" in result["summary"]
    assert "converged_fixed_design_null_rejection_rate" in result["summary"]
    assert "fixed_design_null" in result["feature_results"]
    assert "null_projection_norm" in result["feature_results"]
    assert not result["target_results"]["resolution_limited"].any()

    for offset in range(0, len(calls), 3):
        group = calls[offset : offset + 3]
        for call in group[1:]:
            np.testing.assert_array_equal(group[0]["X"], call["X"])
            np.testing.assert_array_equal(group[0]["y"], call["y"])
            assert group[0]["ais_seed"] == call["ais_seed"]
            assert group[0]["target_seed"] == call["target_seed"]
            assert group[0]["target_feature"] == call["target_feature"]
            assert group[0]["auxiliary_u"] == call["auxiliary_u"]
    assert calls[0]["target_seed"] != calls[3]["target_seed"]


def test_failed_signal_test_makes_strict_power_unavailable(monkeypatch):
    def fake_selective_inference(X, y, **kwargs):
        selection_event = kwargs["selection_event"]
        p_value = np.nan if selection_event == "exact_set" else 0.01
        return {
            "observed_selected_features": np.array([0]),
            "feature_results": pd.DataFrame(
                {
                    "feature": [0],
                    "selection_event": [selection_event],
                    "raw_selective_p_value": [p_value],
                    "adjusted_selective_p_value": [p_value],
                }
            ),
        }

    monkeypatch.setattr(power, "selective_inference", fake_selective_inference)
    with pytest.warns(UserWarning, match="define the same selection event"):
        result = power.compare_selection_event_power(
            n_iters=1,
            n_samples=20,
            n_features=2,
            k_select=1,
            selector=DummySelector(),
        )
    summary = result["summary"].set_index("selection_event")

    assert np.isnan(summary.loc["exact_set", "power"])
    assert summary.loc["exact_set", "signal_target_test_failure_rate"] == 1.0
    assert summary.loc["feature_inclusion", "power"] == 1.0


def test_resolution_limited_signal_tests_produce_power_bounds(monkeypatch):
    def fake_selective_inference(X, y, **kwargs):
        return {
            "observed_selected_features": np.array([0]),
            "feature_results": pd.DataFrame(
                {
                    "feature": [0],
                    "selection_event": [kwargs["selection_event"]],
                    "raw_selective_p_value": [1.0],
                    "adjusted_selective_p_value": [1.0],
                    "resolution_status": ["no_selected_mc_draws_p_equals_one"],
                    "minimum_attainable_p_value": [1.0],
                }
            ),
        }

    monkeypatch.setattr(power, "selective_inference", fake_selective_inference)
    result = power.compare_selection_event_power(
        n_iters=1,
        n_samples=20,
        n_features=2,
        k_select=1,
        selector=DummySelector(),
        selection_events=("same_target",),
    )

    summary = result["summary"].iloc[0]
    assert summary["conditional_power_resolution_lower_bound"] == 0.0
    assert summary["conditional_power_resolution_upper_bound"] == 1.0
    assert summary["n_resolution_limited_signal_targets"] == 1
    assert result["target_results"].iloc[0]["resolution_limited"]
