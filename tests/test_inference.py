import numpy as np
from scipy import stats

from si_shap.inference import (
    _chi_statistic,
    _defensive_mixture_logpdf,
    _effective_sample_size,
    _run_ais,
    _spline_effect_basis,
    _truncated_normal_logpdf,
)


def test_spline_basis_is_centered_and_orthonormal():
    x = np.linspace(-2.0, 2.0, 100)

    basis = _spline_effect_basis(x)

    assert basis.shape == (100, 3)
    np.testing.assert_allclose(basis.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)


def test_chi_statistic_returns_effect_projection():
    rng = np.random.default_rng(7)
    response = rng.standard_normal(100)
    basis = _spline_effect_basis(rng.standard_normal(100))

    statistic, projected = _chi_statistic(response, basis)

    expected_coordinates = basis.T @ response
    np.testing.assert_allclose(projected, basis @ expected_coordinates)
    np.testing.assert_allclose(statistic, np.linalg.norm(expected_coordinates))


def test_effective_sample_size_matches_equal_weight_count():
    assert _effective_sample_size(np.ones(8)) == 8.0
    assert _effective_sample_size([]) == 0.0


def test_defensive_mixture_logpdf_matches_weighted_components():
    z = np.linspace(0.1, 5.0, 20)
    t_obs = 2.0
    adapted_mean = 2.3
    adapted_sd = 0.7
    observed_sd = 1.0

    density = np.exp(
        _defensive_mixture_logpdf(z, 3, t_obs, adapted_mean, adapted_sd)
    )
    expected = (
        0.25 * stats.chi.pdf(z, df=3)
        + 0.375 * np.exp(_truncated_normal_logpdf(z, t_obs, observed_sd))
        + 0.375
        * np.exp(_truncated_normal_logpdf(z, adapted_mean, adapted_sd))
    )

    np.testing.assert_allclose(density, expected)


def test_run_ais_matches_analytic_truncated_chi_probability_with_fixed_budget():
    expected = stats.chi.sf(2.0, df=3) / stats.chi.sf(1.0, df=3)

    estimate, diagnostics = _run_ais(
        2.0,
        3,
        lambda z: z >= 1.0,
        np.random.default_rng(2026),
        pilot_iters=2,
        pilot_samples=100,
        final_batch_size=500,
        max_final_samples=4000,
        min_denominator_ess=200,
        min_tail_ess=80,
    )

    assert diagnostics["status"] == "ok"
    assert diagnostics["sampling_mode"] == "fixed_budget"
    assert diagnostics["proposals"] == 4000
    assert abs(estimate - expected) < 4.0 * diagnostics["mc_se"]
    assert diagnostics["mc_ci_95_lower"] <= estimate <= diagnostics["mc_ci_95_upper"]


def test_run_ais_early_stopping_is_explicitly_labeled():
    _, diagnostics = _run_ais(
        1.0,
        3,
        lambda _z: True,
        np.random.default_rng(11),
        pilot_iters=0,
        pilot_samples=1,
        final_batch_size=100,
        max_final_samples=1000,
        min_denominator_ess=1,
        min_tail_ess=1,
        stop_when_ess_met=True,
    )

    assert diagnostics["sampling_mode"] == "ess_early_stopping"
    assert diagnostics["proposals"] < 1000
