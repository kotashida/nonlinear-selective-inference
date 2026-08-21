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

    assert low_u["naive_feature_inclusion"] == low_u["same_target"]
    assert high_u["same_target"] == pytest.approx(
        high_u["naive_feature_inclusion"] / (1.0 - split)
    )
    assert high_u["weighted_feature_inclusion"] == pytest.approx(
        high_u["naive_feature_inclusion"] / (1.0 - 0.5 * split)
    )


def test_naive_feature_inclusion_has_analytic_type_one_error_inflation():
    assert variable_size.expected_naive_rejection_rate(0.05, 0.5) == pytest.approx(
        1.0 / 15.0
    )
    assert variable_size.expected_naive_rejection_rate(0.05, 0.5) > 0.05


def test_run_experiment_is_reproducible_and_has_complete_audit_rows():
    first = variable_size.run_experiment(n_iters=40, seed=19)
    second = variable_size.run_experiment(n_iters=40, seed=19)

    assert first["settings"] == second["settings"]
    assert len(first["p_value_results"]) == 40 * len(variable_size.METHODS)
    assert set(first["p_value_results"]["method"]) == set(variable_size.METHODS)
    assert set(first["p_value_results"]["target_feature"]) == {0}
    assert set(first["p_value_results"]["selected_set_size"]) == {1, 2}
    assert first["p_value_results"]["p_value"].between(0.0, 1.0).all()
    np.testing.assert_allclose(
        first["p_value_results"]["p_value"],
        second["p_value_results"]["p_value"],
    )
    assert len(first["summary"]) == len(variable_size.METHODS)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("n_iters", 0, ValueError),
        ("rank", 0, ValueError),
        ("split_probability", 1.0, ValueError),
        ("alpha_levels", (0.05, 0.05), ValueError),
        ("seed", -1, ValueError),
        ("n_iters", 2.5, TypeError),
    ],
)
def test_run_experiment_rejects_invalid_audit_settings(keyword, value, error):
    arguments = {keyword: value}
    with pytest.raises(error):
        variable_size.run_experiment(**arguments)
