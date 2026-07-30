"""SHAP-based selective inference under explicit conditioning assumptions."""

from .api import adjust_p_values, selective_inference
from .null_calibration import compare_selection_event_null_calibration
from .plotting import plot_results, plot_selection_regions
from .power import compare_selection_event_power
from .selection import (
    SelectionResult,
    ShapSelector,
    selection_event_holds,
    target_from_selected_set,
)
from .selection_regions import (
    SelectionRegionResult,
    compute_selection_regions,
    selection_regions_frame,
)
from .simulation import run_simulation

__all__ = [
    "SelectionRegionResult",
    "SelectionResult",
    "ShapSelector",
    "adjust_p_values",
    "compute_selection_regions",
    "compare_selection_event_power",
    "compare_selection_event_null_calibration",
    "plot_results",
    "plot_selection_regions",
    "run_simulation",
    "selective_inference",
    "selection_event_holds",
    "target_from_selected_set",
    "selection_regions_frame",
]
