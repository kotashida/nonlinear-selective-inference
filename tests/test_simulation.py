import numpy as np
import pytest

from si_shap.simulation import _method_summary, _validate_inputs


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 100, 10, 1, 0.05),
        (1, 4, 10, 1, 0.05),
        (1, 100, 0, 1, 0.05),
        (1, 100, 10, 0, 0.05),
        (1, 100, 10, 1, 1.0),
    ],
)
def test_validate_inputs_rejects_invalid_settings(arguments):
    with pytest.raises(ValueError):
        _validate_inputs(*arguments)


def test_method_summary_reports_failures_without_silently_estimating_fpr():
    summary, converged = _method_summary(
        "Selective SHAP (AIS)",
        [np.array([0.01]), np.array([np.nan])],
        alpha=0.05,
    )

    assert np.isnan(summary["fpr"])
    assert summary["failure_rate"] == 0.5
    np.testing.assert_array_equal(converged, [0.01])
