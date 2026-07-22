"""SHAP-based selective-inference simulation package."""

from .plotting import plot_results, plot_selection_regions
from .selection_regions import (
    SelectionRegionResult,
    compute_selection_regions,
    selection_regions_frame,
)
from .simulation import run_simulation

__all__ = [
    "SelectionRegionResult",
    "compute_selection_regions",
    "plot_results",
    "plot_selection_regions",
    "run_simulation",
    "selection_regions_frame",
]
