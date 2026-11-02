#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:?REPO is required}"
PYTHON="${PYTHON:?PYTHON is required}"
OUT="${OUT:?OUT is required}"
N_SHARDS=32

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export PYTHONPATH="$REPO/src:$REPO"

mkdir -p "$OUT/shards" "$OUT/logs" "$OUT/pids"
pids=()
for shard in $(seq 0 $((N_SHARDS - 1))); do
  "$PYTHON" "$REPO/benchmarks/run_exact_pipeline_benchmark.py" run \
    --output "$OUT/shards/shard_$shard" \
    --sample-size 100 \
    --repetitions 1000 \
    --n-shards "$N_SHARDS" \
    --shard-index "$shard" \
    >"$OUT/logs/shard_$shard.log" 2>&1 &
  pid=$!
  echo "$pid" >"$OUT/pids/shard_$shard.pid"
  pids+=("$pid")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one shard failed; completed method rows are resumable." >&2
  exit 1
fi

"$PYTHON" "$REPO/benchmarks/run_exact_pipeline_benchmark.py" merge \
  --input-root "$OUT/shards" \
  --output "$OUT" \
  --n-shards "$N_SHARDS"
