import os
import shutil

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import matplotlib.pyplot as plt
import pandas as pd

from examples import sweep_selection_region_settings as sweep


parse_args = sweep.parse_args
resolve_experiments = sweep.resolve_experiments


def test_quick_preset_uses_small_comparison_matrix():
    args = parse_args(["101", "202"])

    assert resolve_experiments(args) == (
        (1, 2),
        ("baseline", "medium"),
        ("exact_set",),
    )


def test_explicit_settings_override_preset():
    args = parse_args(
        [
            "101",
            "--preset",
            "recommended",
            "--k-select",
            "2",
            "5",
            "--rf-config",
            "shallow",
            "flexible",
            "--selection-event",
            "feature_inclusion",
            "exact_ranking",
        ]
    )

    assert resolve_experiments(args) == (
        (2, 5),
        ("shallow", "flexible"),
        ("feature_inclusion", "exact_ranking"),
    )


def test_run_experiment_recreates_output_directory_before_saving(
    monkeypatch, tmp_path
):
    args = parse_args(["101", "--output-dir", str(tmp_path / "sweep")])
    experiment_directory = args.output_dir / "k1_baseline_exact_set"
    received = {}

    def compute_and_remove_output(**kwargs):
        received.update(kwargs)
        assert experiment_directory.is_dir()
        shutil.rmtree(experiment_directory)
        return [object()]

    monkeypatch.setattr(sweep, "compute_selection_regions", compute_and_remove_output)
    monkeypatch.setattr(
        sweep,
        "selection_regions_frame",
        lambda _results: pd.DataFrame({"seed": [101]}),
    )
    monkeypatch.setattr(
        sweep,
        "plot_selection_regions",
        lambda _results, _path: plt.figure(),
    )

    sweep.run_experiment(args, 1, "baseline", "exact_set")

    assert (experiment_directory / "selection_regions.csv").is_file()
    assert received["selection_event"] == "exact_set"
    assert received["target_rule"] == "all_selected"


def test_conditioning_modes_use_distinct_output_directories(monkeypatch, tmp_path):
    args = parse_args(["101", "--output-dir", str(tmp_path)])
    directories = []

    monkeypatch.setattr(sweep, "compute_selection_regions", lambda **_kwargs: [])
    monkeypatch.setattr(
        sweep,
        "selection_regions_frame",
        lambda _results: pd.DataFrame({"seed": [101]}),
    )

    def record_plot(_results, output_path):
        directories.append(output_path.parent.name)
        return plt.figure()

    monkeypatch.setattr(sweep, "plot_selection_regions", record_plot)

    sweep.run_experiment(args, 1, "baseline", "exact_set")
    sweep.run_experiment(args, 1, "baseline", "feature_inclusion")

    assert directories == [
        "k1_baseline_exact_set",
        "k1_baseline_feature_inclusion",
    ]
