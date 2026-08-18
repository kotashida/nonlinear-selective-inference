#!/usr/bin/env bash
set -euo pipefail

# Fixed-u counterpart of outputs/null_calibration_seed_123_shards.
# Override any setting through the environment, for example:
#   PARALLEL_SHARDS=16 N_SHARDS=20 ITERS_PER_SHARD=100 bash \
#     examples/run_fixed_u_null_calibration_shards.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/null_calibration_fixed_u_0p75_shards}"
N_SHARDS="${N_SHARDS:-10}"
ITERS_PER_SHARD="${ITERS_PER_SHARD:-100}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-1}"
BASE_SEED="${BASE_SEED:-123}"
DESIGN_SEED="${DESIGN_SEED:-1514383052}"
FIXED_U="${FIXED_U:-0.75}"
N_SAMPLES="${N_SAMPLES:-100}"
N_FEATURES="${N_FEATURES:-20}"
K_SELECT="${K_SELECT:-2}"
MC_PROPOSALS="${MC_PROPOSALS:-800}"
RF_TREES="${RF_TREES:-50}"
RF_DEPTH="${RF_DEPTH:-5}"

if ! [[ "$N_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_SHARDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$PARALLEL_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_SHARDS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_shard() {
  local shard_index="$1"
  local shard_seed=$((BASE_SEED + shard_index))
  local shard_dir="$OUTPUT_ROOT/shard_${shard_index}"
  local log_path="$OUTPUT_ROOT/logs/shard_${shard_index}.log"

  echo "Starting shard ${shard_index} with seed ${shard_seed}"
  "$PYTHON_BIN" examples/compare_selection_event_null_calibration.py \
    --n-iters "$ITERS_PER_SHARD" \
    --n-samples "$N_SAMPLES" \
    --n-features "$N_FEATURES" \
    --k-select "$K_SELECT" \
    --seed "$shard_seed" \
    --design-seed "$DESIGN_SEED" \
    --fixed-auxiliary-u "$FIXED_U" \
    --selection-events feature_inclusion exact_set same_target \
    --inference-method conditional_mc \
    --final-batch-size 80 \
    --max-final-samples "$MC_PROPOSALS" \
    --rf-param "n_estimators=${RF_TREES}" \
    --rf-param "max_depth=${RF_DEPTH}" \
    --rf-param random_state=42 \
    --rf-param n_jobs=1 \
    --output-dir "$shard_dir" \
    >"$log_path" 2>&1
  echo "Finished shard ${shard_index}; log: ${log_path}"
}

export -f run_shard
export PYTHON_BIN OUTPUT_ROOT ITERS_PER_SHARD BASE_SEED DESIGN_SEED FIXED_U
export N_SAMPLES N_FEATURES K_SELECT MC_PROPOSALS RF_TREES RF_DEPTH

seq 0 $((N_SHARDS - 1)) | xargs -n 1 -P "$PARALLEL_SHARDS" bash -c 'run_shard "$1"' _

"$PYTHON_BIN" examples/summarize_null_calibration_shards.py \
  "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/pooled_summary"

echo "Completed $((N_SHARDS * ITERS_PER_SHARD)) iterations."
echo "Pooled summary: $OUTPUT_ROOT/pooled_summary/pooled_calibration_summary.csv"
