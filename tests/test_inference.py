import numpy as np

from si_shap.inference import (
    _chi_statistic,
    _effective_sample_size,
    _spline_effect_basis,
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
