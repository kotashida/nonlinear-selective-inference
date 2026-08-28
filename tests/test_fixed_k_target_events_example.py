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

from examples import compare_fixed_k_target_events as fixed_k


def test_selector_keeps_k_constant_while_target_position_changes():
    cutoff = stats.chi.ppf(0.5, df=3)

    assert fixed_k.selected_features(cutoff, cutoff) == (0, 1)
    assert fixed_k.selected_features(np.nextafter(cutoff, np.inf), cutoff) == (1, 2)
    assert len(fixed_k.selected_features(cutoff, cutoff)) == 2
    assert len(fixed_k.selected_features(cutoff + 1.0, cutoff)) == 2
    assert 1 in fixed_k.selected_features(cutoff, cutoff)
    assert 1 in fixed_k.selected_features(cutoff + 1.0, cutoff)

    low_k5 = fixed_k.selected_features(cutoff, cutoff, 5)
    high_k5 = fixed_k.selected_features(np.nextafter(cutoff, np.inf), cutoff, 5)
    assert low_k5 == (0, 1, 2, 3, 4)
    assert high_k5 == (4, 5, 6, 7, 8)
    assert len(low_k5) == len(high_k5) == 5
    assert 4 in low_k5 and 4 in high_k5


def test_event_p_values_condition_on_the_fixed_auxiliary_draw():
    rank = 3
    split = 0.5
    cutoff = stats.chi.ppf(split, df=rank)

    lower_u = fixed_k.event_p_values(
        cutoff + 0.5,
        0.25,
        rank=rank,
        split_probability=split,
    )
    upper_u = fixed_k.event_p_values(
        cutoff - 0.5,
        0.75,
        rank=rank,
        split_probability=split,
    )

    assert lower_u["same_target"] == pytest.approx(
        lower_u["feature_inclusion"] / (1.0 - split)
    )
    assert upper_u["same_target"] == pytest.approx(
        (upper_u["feature_inclusion"] - (1.0 - split)) / split
    )
    assert lower_u["exact_set"] == pytest.approx(lower_u["same_target"])
    assert upper_u["exact_set"] == pytest.approx(upper_u["same_target"])


def test_feature_inclusion_is_marginally_valid_but_not_fixed_u_valid():
    alpha = 0.05
    split = 0.5

    assert fixed_k.expected_rejection_rate(
        "feature_inclusion", "pooled", alpha, split
    ) == pytest.approx(alpha)
    assert fixed_k.expected_rejection_rate(
        "feature_inclusion", "u_first_target_interval", alpha, split
    ) == pytest.approx(0.10)
    assert fixed_k.expected_rejection_rate(
        "feature_inclusion", "u_last_target_interval", alpha, split
    ) == pytest.approx(0.0)
    for stratum in fixed_k.STRATA:
        assert fixed_k.expected_rejection_rate(
            "same_target", stratum, alpha, split
        ) == pytest.approx(alpha)
        assert fixed_k.expected_rejection_rate(
            "exact_set", stratum, alpha, split
        ) == pytest.approx(alpha)


def test_run_experiment_is_reproducible_and_has_complete_audit_rows():
    first = fixed_k.run_experiment(n_iters=80, seed=19)
    second = fixed_k.run_experiment(n_iters=80, seed=19)

    assert first["settings"] == second["settings"]
    assert len(first["p_value_results"]) == 80 * len(fixed_k.METHODS)
    assert set(first["p_value_results"]["method"]) == set(fixed_k.METHODS)
    assert set(first["p_value_results"]["target_feature"]) == {1}
    assert set(first["p_value_results"]["selected_set_size"]) == {2}
    assert set(first["p_value_results"]["selected_features"]) == {(0, 1), (1, 2)}
    assert first["p_value_results"]["p_value"].between(0.0, 1.0).all()
    exact_set = first["p_value_results"].loc[
        first["p_value_results"]["method"] == "exact_set"
    ]
    assert not exact_set["conditions_on_auxiliary_u"].any()
    assert exact_set["selection_event_definition"].str.contains(
        "auxiliary_u unfixed", regex=False
    ).all()
    np.testing.assert_allclose(
        first["p_value_results"]["p_value"],
        second["p_value_results"]["p_value"],
    )
    assert len(first["summary"]) == len(fixed_k.METHODS) * len(fixed_k.STRATA)


def test_run_experiment_supports_k_five():
    result = fixed_k.run_experiment(n_iters=80, k=5, seed=23)

    assert result["settings"]["k"] == 5
    assert result["settings"]["conditioning_target"] == 4
    assert result["settings"]["theoretical_conditioning_target_probability"] == 0.2
    assert set(result["p_value_results"]["target_feature"]) == {4}
    assert set(result["p_value_results"]["selected_set_size"]) == {5}
    assert set(result["p_value_results"]["selected_features"]) == {
        (0, 1, 2, 3, 4),
        (4, 5, 6, 7, 8),
    }
    first_interval = result["p_value_results"].loc[
        result["p_value_results"]["auxiliary_u_stratum"]
        == "u_first_target_interval",
        "auxiliary_u",
    ]
    last_interval = result["p_value_results"].loc[
        result["p_value_results"]["auxiliary_u_stratum"]
        == "u_last_target_interval",
        "auxiliary_u",
    ]
    assert (first_interval < 0.2).all()
    assert (last_interval >= 0.8).all()


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("n_iters", 0, ValueError),
        ("rank", 0, ValueError),
        ("k", 1, ValueError),
        ("split_probability", 1.0, ValueError),
        ("alpha_levels", (0.05, 0.05), ValueError),
        ("seed", -1, ValueError),
        ("n_iters", 2.5, TypeError),
    ],
)
def test_run_experiment_rejects_invalid_audit_settings(keyword, value, error):
    with pytest.raises(error):
        fixed_k.run_experiment(**{keyword: value})
