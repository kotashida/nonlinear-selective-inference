import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pytest
from scipy import stats

from examples import compare_variable_size_target_events as variable_size


def test_threshold_selector_has_variable_size_at_the_declared_cutoff():
    cutoff = stats.chi.ppf(0.5, df=3)

    assert variable_size.selected_features(cutoff, cutoff) == (0, 1)
    assert variable_size.selected_features(np.nextafter(cutoff, np.inf), cutoff) == (0,)
    assert variable_size.inverse_size_weight(cutoff, cutoff) == 0.5
    assert variable_size.inverse_size_weight(cutoff + 1.0, cutoff) == 1.0

    assert variable_size.selected_features(cutoff, cutoff, 5) == (0, 1, 2, 3, 4)
    assert variable_size.selected_features(cutoff + 1.0, cutoff, 5) == (0,)
    assert variable_size.inverse_size_weight(cutoff, cutoff, 5) == 0.2


def test_event_p_values_distinguish_marginalized_and_fixed_randomness():
    rank = 3
    split = 0.5
    cutoff = stats.chi.ppf(split, df=rank)

    low_u = variable_size.event_p_values(
        cutoff + 0.5,
        0.25,
        rank=rank,
        split_probability=split,
    )
    high_u = variable_size.event_p_values(
        cutoff + 0.5,
        0.75,
        rank=rank,
        split_probability=split,
    )

    assert low_u["feature_inclusion"] == low_u["same_target"]
    assert high_u["same_target"] == pytest.approx(
        high_u["feature_inclusion"] / (1.0 - split)
    )
    assert high_u["weighted_feature_inclusion"] == pytest.approx(
        high_u["feature_inclusion"] / (1.0 - 0.5 * split)
    )
    assert high_u["exact_set"] == pytest.approx(high_u["same_target"])

    lower_statistic = cutoff - 0.5
    pair_set = variable_size.event_p_values(
        lower_statistic,
        0.25,
        rank=rank,
        split_probability=split,
    )
    assert pair_set["exact_set"] == pytest.approx(
        (split - stats.chi.cdf(lower_statistic, df=rank)) / split
    )


def test_feature_inclusion_has_analytic_type_one_error_inflation():
    assert variable_size.expected_feature_inclusion_rejection_rate(
        0.05, 0.5
    ) == pytest.approx(1.0 / 15.0)
    assert (
        variable_size.expected_feature_inclusion_rejection_rate(0.05, 0.5)
        > 0.05
    )
    assert variable_size.expected_feature_inclusion_rejection_rate(
        0.05, 0.5, 5
    ) == pytest.approx(1.0 / 12.0)


def test_k5_event_p_values_use_the_first_fifth_of_auxiliary_randomness():
    rank = 3
    split = 0.5
    cutoff = stats.chi.ppf(split, df=rank)

    low_statistic = cutoff - 0.25
    low_u = variable_size.event_p_values(
        low_statistic,
        0.1,
        rank=rank,
        split_probability=split,
        large_set_size=5,
    )
    assert low_u["same_target"] == pytest.approx(low_u["feature_inclusion"])
    assert low_u["exact_set"] == pytest.approx(
        (split - stats.chi.cdf(low_statistic, df=rank)) / split
    )

    high_statistic = cutoff + 0.25
    high_u = variable_size.event_p_values(
        high_statistic,
        0.75,
        rank=rank,
        split_probability=split,
        large_set_size=5,
    )
    assert high_u["same_target"] == pytest.approx(
        high_u["feature_inclusion"] / (1.0 - split)
    )
    assert high_u["exact_set"] == pytest.approx(high_u["same_target"])


def test_run_experiment_is_reproducible_and_has_complete_audit_rows():
    first = variable_size.run_experiment(n_iters=40, seed=19)
    second = variable_size.run_experiment(n_iters=40, seed=19)

    assert first["settings"] == second["settings"]
    assert len(first["p_value_results"]) == 40 * len(variable_size.METHODS)
    assert set(first["p_value_results"]["method"]) == set(variable_size.METHODS)
    assert set(first["p_value_results"]["target_feature"]) == {0}
    assert set(first["p_value_results"]["selected_set_size"]) == {1, 2}
    assert first["p_value_results"]["p_value"].between(0.0, 1.0).all()
    exact_set = first["p_value_results"].loc[
        first["p_value_results"]["method"] == "exact_set"
    ]
    assert not exact_set["conditions_on_auxiliary_u"].any()
    assert exact_set["rejection_event_definition"].str.contains(
        "auxiliary_u unfixed", regex=False
    ).all()
    np.testing.assert_allclose(
        first["p_value_results"]["p_value"],
        second["p_value_results"]["p_value"],
    )
    assert len(first["summary"]) == len(variable_size.METHODS)

    k5 = variable_size.run_experiment(n_iters=40, large_set_size=5, seed=19)
    assert k5["settings"]["large_set_size"] == 5
    assert set(k5["p_value_results"]["selected_set_size"]) == {1, 5}
    assert k5["settings"]["theoretical_target_zero_probability"] == pytest.approx(
        0.6
    )


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("n_iters", 0, ValueError),
        ("rank", 0, ValueError),
        ("split_probability", 1.0, ValueError),
        ("alpha_levels", (0.05, 0.05), ValueError),
        ("seed", -1, ValueError),
        ("large_set_size", 1, ValueError),
        ("large_set_size", 2.5, TypeError),
        ("n_iters", 2.5, TypeError),
    ],
)
def test_run_experiment_rejects_invalid_audit_settings(keyword, value, error):
    arguments = {keyword: value}
    with pytest.raises(error):
        variable_size.run_experiment(**arguments)
