#!/usr/bin/env bash
set -euo pipefail

# Reproduce the slide's paired same_target versus exact_set experiment with
# any supported built-in selector. Methods and null/power phases run
# sequentially, while each phase is split into 10 parallel shards.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/same_target_vs_exact_set}"
METHODS="${METHODS:-shap spline_screening marginal_screening}"
TOTAL_ITERS="${TOTAL_ITERS:-1000}"
N_SHARDS="${N_SHARDS:-10}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-10}"
CPU_BUDGET="${CPU_BUDGET:-32}"
RF_JOBS="${RF_JOBS:-3}"
BASE_SEED="${BASE_SEED:-20260828}"
DESIGN_SEED="${DESIGN_SEED:-144269559}"
K_SELECT="${K_SELECT:-5}"

for name in TOTAL_ITERS N_SHARDS PARALLEL_SHARDS CPU_BUDGET RF_JOBS K_SELECT; do
  value="${!name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer" >&2
    exit 2
  fi
done
if (( TOTAL_ITERS < N_SHARDS )); then
  echo "TOTAL_ITERS must be at least N_SHARDS." >&2
  exit 2
fi
if (( PARALLEL_SHARDS > N_SHARDS )); then
  echo "PARALLEL_SHARDS cannot exceed N_SHARDS." >&2
  exit 2
fi
if (( PARALLEL_SHARDS * RF_JOBS > CPU_BUDGET )); then
  echo "PARALLEL_SHARDS * RF_JOBS exceeds CPU_BUDGET (${CPU_BUDGET})." >&2
  exit 2
fi
for method in $METHODS; do
  if [[ "$method" != "shap" && "$method" != "mutual_information" && \
        "$method" != "marginal_screening" && "$method" != "spline_screening" ]]; then
    echo "Unsupported method in METHODS: $method" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT/logs"
requested_config="$(mktemp)"
trap 'rm -f "$requested_config"' EXIT
{
  printf 'METHODS=%s\n' "$METHODS"
  printf 'TOTAL_ITERS=%s\n' "$TOTAL_ITERS"
  printf 'N_SHARDS=%s\n' "$N_SHARDS"
  printf 'PARALLEL_SHARDS=%s\n' "$PARALLEL_SHARDS"
  printf 'CPU_BUDGET=%s\n' "$CPU_BUDGET"
  printf 'RF_JOBS=%s\n' "$RF_JOBS"
  printf 'BASE_SEED=%s\n' "$BASE_SEED"
  printf 'DESIGN_SEED=%s\n' "$DESIGN_SEED"
  printf 'N_SAMPLES=100\nN_FEATURES=20\nK_SELECT=%s\n' "$K_SELECT"
  printf 'SIGNAL_STRENGTH=0.2\nALPHA=0.05\n'
  printf 'MC_PROPOSALS=800\n'
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
  local method="$1"
  local experiment="$2"
  local shard="$3"
  local start=$((shard * TOTAL_ITERS / N_SHARDS))
  local end=$(((shard + 1) * TOTAL_ITERS / N_SHARDS))
  local count=$((end - start))
  local shard_dir="$OUTPUT_ROOT/${method}/shard_${shard}/${experiment}"
  local log_path="$OUTPUT_ROOT/logs/${method}_${experiment}_shard_${shard}.log"
  local selection_args=(--selection-method "$method")
  if [[ "$method" == "shap" ]]; then
    selection_args+=(--rf-param "n_jobs=${RF_JOBS}")
  fi
  mkdir -p "$shard_dir"

  if [[ "$experiment" == "null" && \
        -s "$shard_dir/settings.json" && \
        -s "$shard_dir/p_value_results.csv" ]]; then
    echo "Skipping completed ${method} ${experiment} shard ${shard}"
    return
  fi
  if [[ "$experiment" == "power" && \
        -s "$shard_dir/settings.json" && \
        -s "$shard_dir/target_results.csv" && \
        -s "$shard_dir/feature_results.csv" ]]; then
    echo "Skipping completed ${method} ${experiment} shard ${shard}"
    return
  fi

  echo "Starting ${method} ${experiment} shard ${shard}: start=${start}, count=${count}"

  if [[ "$experiment" == "null" ]]; then
    "$PYTHON_BIN" -u examples/compare_selection_event_null_calibration.py \
      --n-iters "$count" --iteration-start "$start" \
      --n-samples 100 --n-features 20 --k-select "$K_SELECT" \
      --sigma 1.0 --feature-correlation 0.0 \
      --seed "$BASE_SEED" --design-seed "$DESIGN_SEED" \
      --selection-events same_target exact_set \
      --alpha-levels 0.01 0.05 0.1 --selection-decimals 10 \
      "${selection_args[@]}" \
      --pilot-iters 3 --pilot-samples 40 --final-batch-size 80 \
      --max-final-samples 800 --min-denominator-ess 80 --min-tail-ess 15 \
      --inference-method conditional_mc --output-dir "$shard_dir" \
      >"$log_path" 2>&1
  else
    "$PYTHON_BIN" -u examples/compare_selection_event_power.py \
      --preset quick --n-iters "$count" --iteration-start "$start" \
      --n-samples 100 --n-features 20 --k-select "$K_SELECT" \
      --signal-features 0 --feature-correlation 0.0 --signal-strength 0.2 \
      --alpha 0.05 --seed "$BASE_SEED" --selection-decimals 10 \
      "${selection_args[@]}" \
      --selection-events same_target exact_set --multiplicity none \
      --inference-method conditional_mc --pilot-iters 3 --pilot-samples 40 \
      --final-batch-size 80 --max-final-samples 800 \
      --min-denominator-ess 80 --min-tail-ess 15 \
      --output-dir "$shard_dir" >"$log_path" 2>&1
  fi
  echo "Finished ${method} ${experiment} shard ${shard}; log: ${log_path}"
}

export -f run_shard
export PYTHON_BIN OUTPUT_ROOT TOTAL_ITERS N_SHARDS BASE_SEED DESIGN_SEED RF_JOBS K_SELECT

for method in $METHODS; do
  echo "Running ${method} null phase with ${PARALLEL_SHARDS} shards"
  seq 0 $((N_SHARDS - 1)) | \
    xargs -n 1 -P "$PARALLEL_SHARDS" bash -c 'run_shard "$1" null "$2"' _ "$method"

  echo "Running ${method} power phase with ${PARALLEL_SHARDS} shards"
  seq 0 $((N_SHARDS - 1)) | \
    xargs -n 1 -P "$PARALLEL_SHARDS" bash -c 'run_shard "$1" power "$2"' _ "$method"

  "$PYTHON_BIN" -u examples/summarize_same_target_exact_set_shards.py \
    "$OUTPUT_ROOT/$method" --expected-method "$method" \
    --expected-shards "$N_SHARDS" --expected-iterations "$TOTAL_ITERS" \
    --output-dir "$OUTPUT_ROOT/$method/pooled_summary"
done

"$PYTHON_BIN" -u examples/summarize_same_target_exact_set_methods.py \
  "$OUTPUT_ROOT" --methods $METHODS --output-dir "$OUTPUT_ROOT/pooled_summary"

echo "Completed ${TOTAL_ITERS} null and ${TOTAL_ITERS} power iterations for: ${METHODS}."
for method in $METHODS; do
  echo "${method}: $OUTPUT_ROOT/$method/pooled_summary/comparison_summary.csv"
done
echo "Cross-method summary: $OUTPUT_ROOT/pooled_summary/method_comparison_summary.csv"
