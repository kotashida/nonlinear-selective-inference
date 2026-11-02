import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exact_pipeline_benchmark", ROOT / "benchmarks/run_exact_pipeline_benchmark.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_archived_configuration_is_reproduced():
    config = benchmark.configuration(
        sample_size=100,
        repetitions=1000,
        repetition_start=0,
        datasets=benchmark.DATASET_ORDER,
        methods=benchmark.METHOD_ORDER,
    )
    benchmark.audit_reference_sources(config)
    assert config["reference_methods"] == list(benchmark.METHOD_ORDER)
    assert config["common_arguments"]["selection_event"] == "exact_set"
    assert config["common_arguments"]["target_rule"] == "uniform_from_selected"
    assert config["method_specific"]["ais"] == {
        "ais_strategy": "a_robust_draw_floor",
        "stop_when_ess_met": True,
    }
    assert config["method_specific"]["mutual_information"]["estimator"] == "kde"
    assert config["method_specific"]["mutual_information"]["bandwidth"] == 2.5


def test_dataset_loading_and_paired_seeds():
    datasets = benchmark.load_datasets()
    assert tuple(datasets) == benchmark.DATASET_ORDER
    for X, y in datasets.values():
        assert X.shape[0] == y.shape[0]
        assert X.dtype == y.dtype == np.dtype(float)
        assert np.isfinite(X).all() and np.isfinite(y).all()
    assert benchmark.seed_bundle(0, 100, 0) == benchmark.seed_bundle(0, 100, 0)
    assert benchmark.seed_bundle(0, 100, 0) != benchmark.seed_bundle(0, 100, 1)
    shards = [
        benchmark.assigned_repetitions(0, 1000, 32, index) for index in range(32)
    ]
    flattened = [value for shard in shards for value in shard]
    assert sorted(flattened) == list(range(1000))
    assert len(flattened) == len(set(flattened))


def test_exact_spline_smoke_uses_archived_target_and_event():
    datasets = benchmark.load_datasets()
    X, y = datasets["heating_load"]
    seeds = benchmark.seed_bundle(0, 100, 0)
    rows = np.sort(
        np.random.default_rng(seeds["subset_seed"]).choice(len(y), 100, replace=False)
    )
    result = benchmark.selective_inference(
        X[rows],
        y[rows],
        **benchmark.COMMON,
        target_seed=seeds["target_seed"],
        ais_seed=seeds["ais_seed"],
        **benchmark.method_arguments("spline_screening_exact"),
    )
    assert result["selection_event"] == "exact_set"
    assert result["target_rule"] == "uniform_from_selected"
    assert len(result["observed_selected_features"]) == 2
    assert len(result["feature_results"]) == 1
    assert result["settings"]["inference_method"] == "exact_spline"
    assert result["settings"]["pilot_iters"] == 10
    assert result["settings"]["max_final_samples"] == 51_200
