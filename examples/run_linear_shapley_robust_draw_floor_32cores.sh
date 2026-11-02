#!/usr/bin/env bash
set -euo pipefail

# Local/Linux entry point for the 300-trial matched Linear Shapley follow-up.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export MPLBACKEND=Agg

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u examples/run_linear_shapley_selected_five_matched_300.py run
