import numpy as np
import pytest

from si_shap.selection import _top_k


def test_top_k_breaks_ties_by_feature_index():
    selected = _top_k([1.0, 1.0, 0.5], 2)

    np.testing.assert_array_equal(selected, [0, 1])


@pytest.mark.parametrize("scores", [[1.0, np.nan], [[1.0, 2.0]]])
def test_top_k_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        _top_k(scores, 1)
