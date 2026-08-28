import json
from pathlib import Path

import pandas as pd
import pytest

from examples import analyze_selection_method_validation_clean_shards as analysis


def test_power_experiment_discovery_includes_every_shared_strength(tmp_path):
    shards = [tmp_path / "shard_0", tmp_path / "shard_1"]
    for shard in shards:
        for label, strength in (("0p3", 0.3), ("1p0", 1.0)):
            directory = shard / "shap" / "baseline" / f"power_feature_0_beta_{label}"
            directory.mkdir(parents=True)
            (directory / "settings.json").write_text(
                json.dumps({"signal_strength": strength, "signal_features": [0]}),
                encoding="utf-8",
            )

    experiments = analysis.discover_power_experiments(shards)

    assert list(experiments) == [
        "power_beta_0.3_feature_0",
        "power_beta_1.0_feature_0",
    ]


def test_paired_exact_interval_is_not_degenerate_without_discordances():
    lower, upper = analysis.paired_binary_exact_interval(0, 0, 5000)

    assert lower < 0 < upper
    assert lower == pytest.approx(-upper)


def test_settings_validation_checks_keys_beyond_legacy_subset(tmp_path):
    shards = [tmp_path / "shard_0", tmp_path / "shard_1"]
    settings = [
        {"seed": 1, "signal_strength": 0.3, "alpha": 0.05},
        {"seed": 2, "signal_strength": 0.5, "alpha": 0.05},
    ]

    with pytest.raises(ValueError, match="signal_strength"):
        analysis._validate_compatible_settings(shards, settings, label="power")


def test_shard_loader_rejects_duplicate_iteration_event_grid(tmp_path):
    shard = tmp_path / "shard_0"
    relative = Path("results.csv")
    shard.mkdir()
    pd.DataFrame(
        {
            "iteration": [1, 1, 2, 2],
            "selection_event": [
                "feature_inclusion",
                "feature_inclusion",
                "feature_inclusion",
                "same_target",
            ],
            "p_value": [0.1, 0.2, 0.3, 0.4],
            "failed": [False] * 4,
        }
    ).to_csv(shard / relative, index=False)

    with pytest.raises(ValueError, match="exactly one row"):
        analysis.read_shard_csvs(
            [shard],
            relative,
            expected_iters=2,
            events=analysis.EVENTS,
            required_columns={"iteration", "selection_event", "p_value", "failed"},
        )
