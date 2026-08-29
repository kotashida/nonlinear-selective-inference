#!/usr/bin/env bash
set -euo pipefail

# Run the baseline validation for Mutual Information and Marginal Correlation
# with same_target as the only conditioning event.  Each Python process is
# single-threaded, so PARALLEL_SHARDS is also the maximum CPU-core usage.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selection_method_validation_same_target_mi_mc_shards}"
export PRESET="comprehensive"
export METHODS="mutual_information marginal_screening"
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
if ! [[ "$PARALLEL_SHARDS" =~ ^[1-9][0-9]*$ ]] || (( PARALLEL_SHARDS > 32 )); then
  echo "PARALLEL_SHARDS must be an integer from 1 to 32." >&2
  exit 2
fi

export BASE_SEED="${BASE_SEED:-20260829}"
export DESIGN_SEED="${DESIGN_SEED:-314159}"
export MC_PROPOSALS="${MC_PROPOSALS:-800}"
export MIN_CALIBRATION_ITERS="${MIN_CALIBRATION_ITERS:-5000}"
export MIN_SIGNAL_TARGETS="${MIN_SIGNAL_TARGETS:-300}"
export MIN_CONDITIONAL_POWER="${MIN_CONDITIONAL_POWER:-0.80}"

# These are also set inside every Python entry point before NumPy is imported.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=1

exec bash "$PROJECT_ROOT/examples/run_selection_method_validation_shards.sh"
