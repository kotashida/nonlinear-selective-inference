import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import json

from si_shap.inference import (
    AIS_ABLATION_MECHANISMS,
    AIS_REFINEMENT_STRATEGIES,
    AIS_STRATEGIES,
    DEFENSIVE_MIXTURE_WEIGHTS,
    _adapt_multimodal_proposal,
    _adapt_proposal,
    _deterministic_component_schedule,
    _defensive_mixture_logpdf,
    _general_mixture_logpdf,
    _pilot_island_components,
    _finite_chi_ppf,
    _robust_pilot_islands,
    _randomized_chi_strata,
    _run_ais,
    _sample_defensive_mixture,
    _sample_component_labels,
    _refinement_components,
    _sample_general_mixture,
)


def test_general_single_component_preserves_baseline_sampler():
    old_z, old_logpdf = _sample_defensive_mixture(
        np.random.default_rng(17), 800, 3, 8.0, 5.0, 0.7
    )
    components = ((5.0, 0.7, 1.0),)
    weights = np.asarray(DEFENSIVE_MIXTURE_WEIGHTS)
    new_z = _sample_general_mixture(
        np.random.default_rng(17), 800, 3, 8.0, components, weights
    )
    new_logpdf = _general_mixture_logpdf(new_z, 3, 8.0, components, weights)
    assert np.array_equal(old_z, new_z)
    assert np.allclose(old_logpdf, new_logpdf, rtol=0.0, atol=1e-14)
    assert np.allclose(
        new_logpdf,
        _defensive_mixture_logpdf(new_z, 3, 8.0, 5.0, 0.7),
        rtol=0.0,
        atol=1e-14,
    )


def test_raised_floor_and_multimodal_components():
    always = lambda z: True
    _, sd = _adapt_proposal(
        2.0, 3, always, np.random.default_rng(1), 8, 200, sd_floor=0.75
    )
    assert sd >= 0.75
    separated = lambda z: 0.5 < z < 1.0 or 2.5 < z < 3.0
    components = _adapt_multimodal_proposal(
        2.7, 3, separated, np.random.default_rng(3), 10, 320
    )
    assert len(components) >= 2
    assert np.isclose(sum(component[2] for component in components), 1.0)
    assert all(component[1] >= 0.25 for component in components)


def test_all_strategies_keep_fixed_pilot_budget_and_valid_diagnostics():
    event = lambda z: 0.5 < z < 1.0 or 2.5 < z < 3.0 or z > 5.0
    for strategy in AIS_STRATEGIES:
        _, diagnostics = _run_ais(
            2.7,
            3,
            event,
            np.random.default_rng(123),
            pilot_iters=3,
            pilot_samples=40,
            final_batch_size=80,
            max_final_samples=320,
            min_denominator_ess=2,
            min_tail_ess=2,
            stop_when_ess_met=True,
            ais_strategy=strategy,
        )
        assert diagnostics["pilot_evaluations"] == 120
        assert diagnostics["aj_evaluations"] == 120 + diagnostics["proposals"]
        assert diagnostics["ais_strategy"] == strategy
        assert np.isclose(
            diagnostics["final_base_weight"]
            + diagnostics["final_observed_weight"]
            + diagnostics["final_adapted_weight"],
            1.0 - diagnostics.get("final_denominator_weight", 0.0),
        )


def test_randomized_chi_strata_cover_every_equal_probability_cell():
    from scipy import stats

    draws = _randomized_chi_strata(np.random.default_rng(9), 100, 3)
    cells = np.floor(100 * stats.chi.cdf(draws, df=3)).astype(int)
    assert np.array_equal(np.sort(cells), np.arange(100))


def test_pilot_island_components_use_cdf_separations_and_mass_floor():
    from scipy import stats

    z = stats.chi.ppf(np.linspace(0.01, 0.99, 99), df=3)
    selected = (
        ((stats.chi.cdf(z, df=3) > 0.10) & (stats.chi.cdf(z, df=3) < 0.20))
        | ((stats.chi.cdf(z, df=3) > 0.45) & (stats.chi.cdf(z, df=3) < 0.55))
        | ((stats.chi.cdf(z, df=3) > 0.80) & (stats.chi.cdf(z, df=3) < 0.90))
    )
    log_weights = np.where(selected, 0.0, np.nan)
    components = _pilot_island_components(z, selected, log_weights, 3)
    assert len(components) == 3
    assert np.isclose(sum(component[2] for component in components), 1.0)
    assert min(component[2] for component in components) >= 0.35 / 3


def test_deterministic_component_schedule_honors_total_allocation():
    components = (
        ("chi", 0.0, 1.0, 0.25),
        ("tn", 2.0, 1.0, 0.30),
        ("tn", 1.0, 0.5, 0.20),
        ("tn", 0.8, 0.3, 0.25),
    )
    labels = _deterministic_component_schedule(
        np.random.default_rng(12), 800, components
    )
    assert np.array_equal(np.bincount(labels), [200, 240, 160, 200])


def test_chi_cdf_interval_sampling_never_emits_endpoint_infinities():
    endpoint_values = _finite_chi_ppf(np.array([0.0, 1.0]), 3)
    assert np.isfinite(endpoint_values).all()
    labels = np.zeros(10000, dtype=int)
    components = (("cdf_interval", 0.95, 1.0, 1.0),)
    draws = _sample_component_labels(
        np.random.default_rng(901), labels, 3, components
    )
    assert np.isfinite(draws).all()
    assert np.all(draws >= 0.0)


def test_saturated_tail_islands_use_survival_interval_coordinates():
    from scipy import stats

    z = np.array([11.0, 11.2, 11.4, 11.6, 11.8])
    assert np.all(stats.chi.cdf(z, df=3) == 1.0)
    pilot = {
        "z_values": z, "selected": np.array([False, True, True, True, False]),
        "log_weights": np.array([np.nan, 0.0, 0.0, 0.0, np.nan]),
        "fit_mask": np.ones(5, dtype=bool), "mean": 11.4, "sd": 0.2,
    }
    # Real pilots arrive shuffled, including points whose CDFs all round to 1.
    permutation = np.array([4, 2, 0, 3, 1])
    for field in ("z_values", "selected", "log_weights", "fit_mask"):
        pilot[field] = pilot[field][permutation]
    components, _, _, _, _ = _refinement_components(
        11.4, 3, pilot, interval=True, denominator=False,
        draw_floor=0, max_final_samples=51200,
    )
    assert components[2][0] == "sf_interval"
    draws = _sample_component_labels(
        np.random.default_rng(7), np.full(1000, 2), 3, components
    )
    assert np.isfinite(draws).all()
    assert np.all((draws >= 11.1) & (draws <= 11.7))


def test_ablation_strategy_flags_and_diagnostics_are_distinct():
    event = lambda z: 0.4 < z < 1.0 or 1.5 < z < 2.0 or z > 2.5
    assert len(AIS_ABLATION_MECHANISMS) == 7
    for strategy, expected in AIS_ABLATION_MECHANISMS.items():
        _, diagnostics = _run_ais(
            2.7,
            3,
            event,
            np.random.default_rng(321),
            pilot_iters=3,
            pilot_samples=40,
            final_batch_size=80,
            max_final_samples=320,
            min_denominator_ess=2,
            min_tail_ess=2,
            stop_when_ess_met=True,
            ais_strategy=strategy,
        )
        multimodal, denominator, stratified = expected
        assert (diagnostics["adaptive_component_count"] > 1) == multimodal
        assert diagnostics["denominator_component_count"] == int(denominator)
        assert ("strata" in diagnostics["pilot_sampling_scheme"]) == stratified
        assert diagnostics["pilot_evaluations"] == 120
        assert diagnostics["final_allocation_method"] == (
            "deterministic_mixture_total_budget"
        )


def test_robust_islands_require_sustained_nonselection_and_cap_components():
    from scipy import stats

    u = np.linspace(0.01, 0.99, 99)
    z = stats.chi.ppf(u, df=3)
    selected = np.ones(99, dtype=bool)
    selected[[20, 50, 51, 52, 80]] = False
    log_weights = np.where(selected, 0.0, np.nan)
    islands, pre_merge = _robust_pilot_islands(
        z, selected, log_weights, 3, max_components=2
    )
    assert pre_merge == 4
    assert len(islands) == 2
    assert np.isclose(sum(item["mass"] for item in islands), 1.0)


def test_refinement_strategies_log_forward_component_diagnostics():
    event = lambda z: 0.5 < z < 1.0 or 2.5 < z < 3.0 or z > 5.0
    for strategy, options in AIS_REFINEMENT_STRATEGIES.items():
        pilot_iters = 15 if options["pilot15"] else 10
        _, diagnostics = _run_ais(
            2.7,
            3,
            event,
            np.random.default_rng(123),
            pilot_iters=pilot_iters,
            pilot_samples=80,
            final_batch_size=200,
            max_final_samples=3200,
            min_denominator_ess=10,
            min_tail_ess=3,
            stop_when_ess_met=True,
            ais_strategy=strategy,
        )
        assert diagnostics["pilot_evaluations"] == pilot_iters * 80
        assert diagnostics["pre_merge_cluster_count"] >= diagnostics["post_merge_cluster_count"]
        assert diagnostics["max_normalized_weight"] >= 0.0
        assert 0.0 <= diagnostics["weight_gini"] <= 1.0
        assert "psis_khat" in diagnostics
        components = json.loads(diagnostics["component_diagnostics_json"])
        assert len(components) == diagnostics["final_component_count"]
        assert all("draws_planned" in component for component in components)
        assert all("denominator_ess" in component for component in components)
        if options["interval"]:
            assert any(component["kind"] == "cdf_interval" for component in components)
        if options["denominator"]:
            assert diagnostics["denominator_component_count"] >= 1
            assert np.isclose(diagnostics["final_adapted_weight"], 0.375, atol=1e-12) or options["optimize"]
        if options["draw_floor"]:
            retained = [
                component for component in components
                if component["role"] in {"adapted", "denominator"}
            ]
            assert min(component["draws_planned"] for component in retained) >= 128
