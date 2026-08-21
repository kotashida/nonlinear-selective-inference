#!/usr/bin/env bash
set -euo pipefail

# Run the comprehensive validation as independent iteration shards, then pool
# every raw result before applying the global calibration and power decisions.
# Defaults mirror the proven fixed-u run: 10 processes x 3 RF workers each.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selection_method_validation_seed_20260821_shards}"
PRESET="${PRESET:-comprehensive}"
METHODS="${METHODS:-shap mutual_information marginal_screening}"
DESIGNS="${DESIGNS:-baseline correlated}"
AUXILIARY_VALUES="${AUXILIARY_VALUES:-fresh}"
SIGNAL_STRENGTHS="${SIGNAL_STRENGTHS:-0.30 0.50 0.75 1.00}"
SIGNAL_POSITIONS="${SIGNAL_POSITIONS:-first}"
TOTAL_NULL_ITERS="${TOTAL_NULL_ITERS:-5000}"
TOTAL_POWER_ITERS="${TOTAL_POWER_ITERS:-5000}"
N_SHARDS="${N_SHARDS:-10}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-10}"
CPU_BUDGET="${CPU_BUDGET:-32}"
RF_JOBS="${RF_JOBS:-3}"
BASE_SEED="${BASE_SEED:-20260821}"
DESIGN_SEED="${DESIGN_SEED:-314159}"
MC_PROPOSALS="${MC_PROPOSALS:-800}"
MIN_CALIBRATION_ITERS="${MIN_CALIBRATION_ITERS:-5000}"
MIN_SIGNAL_TARGETS="${MIN_SIGNAL_TARGETS:-300}"
MIN_CONDITIONAL_POWER="${MIN_CONDITIONAL_POWER:-0.80}"

for name in TOTAL_NULL_ITERS TOTAL_POWER_ITERS N_SHARDS PARALLEL_SHARDS CPU_BUDGET RF_JOBS; do
  value="${!name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer" >&2
    exit 2
  fi
done
if (( TOTAL_NULL_ITERS < N_SHARDS || TOTAL_POWER_ITERS < N_SHARDS )); then
  echo "Each total iteration count must be at least N_SHARDS." >&2
  exit 2
fi
if (( PARALLEL_SHARDS * RF_JOBS > CPU_BUDGET )); then
  echo "PARALLEL_SHARDS * RF_JOBS exceeds CPU_BUDGET (${CPU_BUDGET})." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
requested_config="$(mktemp)"
trap 'rm -f "$requested_config"' EXIT
{
  printf 'PRESET=%s\n' "$PRESET"
  printf 'METHODS=%s\n' "$METHODS"
  printf 'DESIGNS=%s\n' "$DESIGNS"
  printf 'AUXILIARY_VALUES=%s\n' "$AUXILIARY_VALUES"
  printf 'SIGNAL_STRENGTHS=%s\n' "$SIGNAL_STRENGTHS"
  printf 'SIGNAL_POSITIONS=%s\n' "$SIGNAL_POSITIONS"
  printf 'TOTAL_NULL_ITERS=%s\n' "$TOTAL_NULL_ITERS"
  printf 'TOTAL_POWER_ITERS=%s\n' "$TOTAL_POWER_ITERS"
  printf 'N_SHARDS=%s\n' "$N_SHARDS"
  printf 'BASE_SEED=%s\n' "$BASE_SEED"
  printf 'DESIGN_SEED=%s\n' "$DESIGN_SEED"
  printf 'MC_PROPOSALS=%s\n' "$MC_PROPOSALS"
  printf 'MIN_CALIBRATION_ITERS=%s\n' "$MIN_CALIBRATION_ITERS"
  printf 'MIN_SIGNAL_TARGETS=%s\n' "$MIN_SIGNAL_TARGETS"
  printf 'MIN_CONDITIONAL_POWER=%s\n' "$MIN_CONDITIONAL_POWER"
  printf 'RF_JOBS=%s\n' "$RF_JOBS"
} >"$requested_config"

config_path="$OUTPUT_ROOT/run_config.txt"
if [[ -f "$config_path" ]] && ! cmp -s "$requested_config" "$config_path"; then
  echo "Existing $config_path does not match this run; use a new OUTPUT_ROOT." >&2
  diff -u "$config_path" "$requested_config" || true
  exit 2
fi
cp "$requested_config" "$config_path"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=1

run_shard() {
  set -euo pipefail
  set -f
  local shard_index="$1"
  local shard_seed=$((BASE_SEED + shard_index))
  local null_base=$((TOTAL_NULL_ITERS / N_SHARDS))
  local null_extra=$((TOTAL_NULL_ITERS % N_SHARDS))
  local power_base=$((TOTAL_POWER_ITERS / N_SHARDS))
  local power_extra=$((TOTAL_POWER_ITERS % N_SHARDS))
  local null_iters="$null_base"
  local power_iters="$power_base"
  if (( shard_index < null_extra )); then null_iters=$((null_iters + 1)); fi
  if (( shard_index < power_extra )); then power_iters=$((power_iters + 1)); fi
  local shard_dir="$OUTPUT_ROOT/shard_${shard_index}"
  local log_path="$OUTPUT_ROOT/logs/shard_${shard_index}.log"

  echo "Starting shard ${shard_index}: null=${null_iters}, power=${power_iters}, seed=${shard_seed}"
  # METHODS and DESIGNS are intentionally split into their validated CLI tokens.
  # shellcheck disable=SC2086
  "$PYTHON_BIN" -u examples/validate_selection_methods.py \
    --preset "$PRESET" \
    --methods $METHODS \
    --designs $DESIGNS \
    --auxiliary-values $AUXILIARY_VALUES \
    --signal-strengths $SIGNAL_STRENGTHS \
    --signal-positions $SIGNAL_POSITIONS \
    --n-null-iters "$null_iters" \
    --n-power-iters "$power_iters" \
    --max-final-samples "$MC_PROPOSALS" \
    --minimum-calibration-iterations "$MIN_CALIBRATION_ITERS" \
    --minimum-signal-targets "$MIN_SIGNAL_TARGETS" \
    --minimum-conditional-power "$MIN_CONDITIONAL_POWER" \
    --seed "$shard_seed" \
    --design-seed "$DESIGN_SEED" \
    --rf-jobs "$RF_JOBS" \
    --output-dir "$shard_dir" \
    >"$log_path" 2>&1
  echo "Finished shard ${shard_index}; log: ${log_path}"
}

export -f run_shard
export PYTHON_BIN OUTPUT_ROOT PRESET METHODS DESIGNS AUXILIARY_VALUES
export SIGNAL_STRENGTHS SIGNAL_POSITIONS TOTAL_NULL_ITERS TOTAL_POWER_ITERS
export N_SHARDS BASE_SEED DESIGN_SEED MC_PROPOSALS MIN_CALIBRATION_ITERS
export MIN_SIGNAL_TARGETS MIN_CONDITIONAL_POWER RF_JOBS

seq 0 $((N_SHARDS - 1)) | xargs -n 1 -P "$PARALLEL_SHARDS" bash -c 'run_shard "$1"' _

"$PYTHON_BIN" -u examples/summarize_selection_method_validation_shards.py \
  "$OUTPUT_ROOT" \
  --expected-shards "$N_SHARDS" \
  --output-dir "$OUTPUT_ROOT/pooled_summary"

echo "Completed ${TOTAL_NULL_ITERS} null and ${TOTAL_POWER_ITERS} power iterations per configuration."
echo "Pooled calibration: $OUTPUT_ROOT/pooled_summary/calibration_decisions.csv"
echo "Pooled power: $OUTPUT_ROOT/pooled_summary/power_decisions.csv"
