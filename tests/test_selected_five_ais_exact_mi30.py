"""Verify matched replay planning and report assembly without AIS simulation."""
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
spec = importlib.util.spec_from_file_location("selected_five_replay", ROOT / "examples/run_selected_five_ais_exact_mi30.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture(scope="module")
def prepared():
    config = replay.read(ROOT / "examples/selected_five_ais_robust_draw_floor.json")
    return config, replay.prepare(config)


def test_both_plans_match_historical_settings(prepared):
    config, study = prepared
    assert len(study[-1]) == 640
    other = replay.read(ROOT / "examples/selected_five_ais_robust_floor_multimodal_denominator.json")
    other_study = replay.prepare(other)
    for a, b in zip(study[-2], other_study[-2]):
        excluded = {"AIS_STRATEGY", "OUTPUT_ROOT"}
        assert {k: v for k, v in a["env"].items() if k not in excluded} == {
            k: v for k, v in b["env"].items() if k not in excluded}
    assert study[-2][1]["env"]["LIME_NUM_EXPLANATIONS"] == "40"
    assert study[-2][2]["env"]["MI_NEIGHBORS"] == "30"


def test_audits_reject_seed_and_selector_drift(prepared):
    _, study = prepared
    original = study[2]
    new = original[original.selection_method.eq("lime")].copy()
    replay.pair_trials(original, new, "lime")
    new.loc[new.index[0], "target_seed"] += 1
    with pytest.raises(AssertionError):
        replay.pair_trials(original, new, "lime")
    with pytest.raises(ValueError, match="Setting drift"):
        replay.compare_settings({"selector_settings": {"n_neighbors": 30}},
                                {"selector_settings": {"n_neighbors": 12}})


def test_finalize_preserves_exact_rows_and_historical_report_format(prepared, tmp_path, monkeypatch):
    config, study = prepared
    reference, _, original, manifest, jobs, audit = study
    output = tmp_path / "matched_report"
    jobs = [{**job, "env": {**job["env"], "OUTPUT_ROOT": str(output / "runs" / job["method"])}}
            for job in jobs]
    real_read = replay.read

    def fake_read(path):
        path = Path(path)
        if path.is_relative_to(output / "runs"):
            method = path.relative_to(output / "runs").parts[0]
            job = next(j for j in jobs if j["method"] == method)
            if path.name == "validation_settings.json":
                return {**job["settings"], "methods": [method],
                        "method_inference_methods": {method: "ais"},
                        "ais_strategy": config["ais_strategy"]}
            suffix = path.relative_to(output / "runs" / method)
            return {**real_read(job["source"] / suffix), "ais_strategy": config["ais_strategy"]}
        return real_read(path)

    def fake_raw(run, kind, settings, expected_shards):
        method = settings["methods"][0]
        data = original[original.selection_method.eq(method)].copy()
        keys = ["selection_method", "validation_shard", "iteration", "selection_event", "signal_strength"]
        added = ["feature", "t_obs", "proposals", "denominator_ess", "tail_ess"]
        n = data[data.scope.eq("null_calibration")].copy()
        t = data[data.scope.eq("power")].copy()
        return n, t.drop(columns=added), t[keys + added]

    monkeypatch.setattr(replay, "read", fake_read)
    monkeypatch.setattr(replay, "_read_raw", fake_raw)
    monkeypatch.setattr(replay.analysis, "BASE", tmp_path)
    replay.finalize(config, (reference, output, original, manifest, jobs, [dict(r) for r in audit]))
    result = pd.read_csv(output / "unified_tests.csv")
    pd.testing.assert_frame_equal(
        result[result.selection_method.eq("spline_screening_exact")].reset_index(drop=True)[original.columns],
        original[original.selection_method.eq("spline_screening_exact")].reset_index(drop=True),
        check_dtype=False)
    for filename in ("power_summary.csv", "calibration_summary.csv", "sampling_cost_summary.csv", "signal_outcomes.csv"):
        assert pd.read_csv(output / filename).columns.tolist() == pd.read_csv(reference / filename).columns.tolist()
    assert list(result.columns[:len(original.columns)]) == list(original.columns)
    assert len(result) == 5000
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == 7
    assert (output / "COMPLETED").exists()
