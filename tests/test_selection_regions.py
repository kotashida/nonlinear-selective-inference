import inspect
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from si_shap.plotting import plot_selection_regions
from si_shap.selection_regions import (
    SelectionRegionResult,
    find_selection_intervals,
    selection_probability,
)


def test_compute_selection_regions_requires_dataset_seeds():
    from si_shap.selection_regions import compute_selection_regions

    parameter = inspect.signature(compute_selection_regions).parameters[
        "dataset_seeds"
    ]

    assert parameter.default is inspect.Parameter.empty


def test_compute_selection_regions_returns_each_top_k_feature(monkeypatch):
    from si_shap import selection_regions

    received_rf_params = []
    monkeypatch.setattr(
        selection_regions,
        "_tree_shap_importance",
        lambda X, response, selection_decimals, rf_params: (
            received_rf_params.append(rf_params) or np.array([1.0, 3.0, 2.0])
        ),
    )
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

    results = selection_regions.compute_selection_regions(
        dataset_seeds=[101],
        n_samples=10,
        n_features=3,
        k_select=2,
        rf_params={"n_estimators": 12},
    )

    assert [result.selected_feature for result in results] == [1, 2]
    assert [result.selection_position for result in results] == [1, 2]
    assert all(result.k_select == 2 for result in results)
    assert all(
        result.selection_probability
        <= result.selection_probability_upper_bound
        <= 1.0
        for result in results
    )
    assert all(result.relative_omitted_tail_bound >= 0.0 for result in results)
    assert all(result.regions_certified is False for result in results)
    assert received_rf_params
    assert all(params["n_estimators"] == 12 for params in received_rf_params)
    assert all(params["max_depth"] == 5 for params in received_rf_params)

    target_results = selection_regions.compute_selection_regions(
        dataset_seeds=[101],
        n_samples=10,
        n_features=3,
        k_select=2,
        selection_event="same_target",
        target_rule="uniform_from_selected",
        target_seed=17,
    )
    assert len(target_results) == 1
    assert target_results[0].observed_target_feature in {1, 2}
    assert target_results[0].selected_feature == (
        target_results[0].observed_target_feature
    )
    assert 0.0 <= target_results[0].auxiliary_u < 1.0


def test_find_selection_intervals_refines_detected_boundaries():
    def is_selected(z):
        return z <= 1.0 or 2.0 <= z <= 3.0

    intervals = find_selection_intervals(
        is_selected,
        z_max=4.0,
        grid_size=18,
        boundary_tol=1e-7,
    )

    np.testing.assert_allclose(intervals, [(0.0, 1.0), (2.0, 3.0)], atol=1e-6)


def test_anchor_point_preserves_a_narrow_observed_component():
    intervals = find_selection_intervals(
        lambda z: 1.001 <= z <= 1.002,
        z_max=2.0,
        grid_size=11,
        boundary_tol=1e-7,
        anchor_points=(1.0015,),
    )

    np.testing.assert_allclose(intervals, [(1.001, 1.002)], atol=1e-6)


def test_midpoint_refinement_can_detect_a_component_missed_by_base_grid():
    event = lambda z: 0.36 <= z <= 0.39

    coarse = find_selection_intervals(
        event, z_max=1.0, grid_size=5, grid_refinements=0
    )
    refined = find_selection_intervals(
        event,
        z_max=1.0,
        grid_size=5,
        grid_refinements=1,
        boundary_tol=1e-7,
    )

    assert coarse == ()
    np.testing.assert_allclose(refined, [(0.36, 0.39)], atol=1e-6)


def test_selection_probability_integrates_chi_mass_over_intervals():
    intervals = ((0.0, 1.0), (2.0, 3.0))

    probability = selection_probability(intervals, rank=3)

    expected = (
        stats.chi.cdf(1.0, df=3)
        + stats.chi.cdf(3.0, df=3)
        - stats.chi.cdf(2.0, df=3)
    )
    assert probability == expected


def _selection_region_result(dataset_number=1):
    return SelectionRegionResult(
        dataset_number=dataset_number,
        seed=101,
        selected_feature=2,
        rank=3,
        t_obs=2.0,
        z_max=6.0,
        intervals=((1.5, 6.0),),
        selection_probability=selection_probability(((1.5, 6.0),), rank=3),
        omitted_tail_probability=stats.chi.sf(6.0, df=3),
        observed_proposal_sd=1.0,
        adapted_proposal_mean=2.2,
        adapted_proposal_sd=0.8,
        adaptation_seed=10101,
    )


def test_single_region_plot_uses_final_proposal_by_default():
    result = _selection_region_result()
    figure = plot_selection_regions(result)
    labels = figure.axes[0].get_legend_handles_labels()[1]

    assert any("final" in label for label in labels)
    assert not any("component" in label for label in labels)
    plt.close(figure)


def test_single_region_plot_can_show_proposal_components():
    result = _selection_region_result()
    figure = plot_selection_regions(result, show_proposal_components=True)
    labels = figure.axes[0].get_legend_handles_labels()[1]

    assert any("final" in label for label in labels)
    assert any("obs" in label and "component" in label for label in labels)
    assert any("adapt" in label and "component" in label for label in labels)
    plt.close(figure)


def test_combined_plot_accepts_multiple_regions():
    results = [_selection_region_result(1), _selection_region_result(2)]

    figure = plot_selection_regions(results)

    assert sum(axis.get_visible() for axis in figure.axes) == 2
    plt.close(figure)
