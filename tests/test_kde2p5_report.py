"""Regression tests for the selected-five KDE report labels and decisions."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from analyze_ais_method_comparison import _method_label
import plot_combined_ais_analysis as plotting


def test_method_labels_reflect_exact_inference_and_kde_settings():
    settings = {"mi_params": {"estimator": "kde", "bandwidth": 2.5}}
    assert _method_label("spline_screening", exact=True, settings=settings) == (
        "Spline screening (exact)"
    )
    assert _method_label("mutual_information", exact=False, settings=settings) == (
        "Mutual information (KDE 2.5, AIS)"
    )


def test_power_decision_requires_no_unresolved_signal_tests():
    passing = pd.Series({"signal_targets": 154, "signal_failures": 0, "ci_lower": .90})
    unresolved = pd.Series({"signal_targets": 154, "signal_failures": 3, "ci_lower": .94})
    weak = pd.Series({"signal_targets": 154, "signal_failures": 0, "ci_lower": .79})
    assert plotting._power_decision(passing) == (True, True)
    assert plotting._power_decision(unresolved) == (False, False)
    assert plotting._power_decision(weak) == (True, False)
