#!/usr/bin/env bash
set -euo pipefail

# Reproduce the baseline / fresh auxiliary / same_target validation slice used
# in selection_method_validation_combined_analysis, for spline screening only.
# Spline screening is single-threaded, so PARALLEL_SHARDS is the CPU-core cap.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$HOME/.venvs/nonlinear-selective-inference-py311/bin/python" ]]; then
    PYTHON_BIN="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
  elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python was not found. Set PYTHON_BIN to the Python executable." >&2
    exit 127
  fi
fi
if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 127
fi
export PYTHON_BIN

export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selection_method_validation_same_target_spline_screening_shards}"
export PRESET="comprehensive"
export METHODS="spline_screening"
export DESIGNS="baseline"
export AUXILIARY_VALUES="fresh"
export SIGNAL_STRENGTHS="0.30 0.50 0.75 1.00"
export SIGNAL_POSITIONS="first"
export SELECTION_EVENTS="same_target"
export PRIMARY_SELECTION_EVENT="same_target"

export TOTAL_NULL_ITERS="${TOTAL_NULL_ITERS:-5000}"
export TOTAL_POWER_ITERS="${TOTAL_POWER_ITERS:-5000}"
export N_SHARDS="${N_SHARDS:-32}"
export PARALLEL_SHARDS="${PARALLEL_SHARDS:-32}"
export CPU_BUDGET="${CPU_BUDGET:-32}"
export RF_JOBS="1"

if ! [[ "$CPU_BUDGET" =~ ^[1-9][0-9]*$ ]] || (( CPU_BUDGET > 32 )); then
  echo "CPU_BUDGET must be an integer from 1 to 32." >&2
  exit 2
fi
if ! [[ "$PARALLEL_SHARDS" =~ ^[1-9][0-9]*$ ]] || (( PARALLEL_SHARDS > CPU_BUDGET )); then
  echo "PARALLEL_SHARDS must be an integer from 1 to CPU_BUDGET." >&2
  exit 2
fi

# A new method-specific simulation seed keeps runs independent. DESIGN_SEED is
# identical to the existing combined analysis and reproduces its fixed design.
export BASE_SEED="${BASE_SEED:-20260830}"
export DESIGN_SEED="${DESIGN_SEED:-314159}"
export MC_PROPOSALS="${MC_PROPOSALS:-800}"
export MIN_CALIBRATION_ITERS="${MIN_CALIBRATION_ITERS:-5000}"
export MIN_SIGNAL_TARGETS="${MIN_SIGNAL_TARGETS:-300}"
export MIN_CONDITIONAL_POWER="${MIN_CONDITIONAL_POWER:-0.80}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=1

exec bash "$PROJECT_ROOT/examples/run_selection_method_validation_shards.sh"
