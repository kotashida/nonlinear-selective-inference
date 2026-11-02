#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_BASE="${OUTPUT_BASE:-outputs/selection_method_validation/exact_set/ais_mechanism_ablation_200}"
REFERENCE_CSV="${REFERENCE_CSV:-outputs/selection_method_validation/exact_set/ais_fix_followup_200/all_target_diagnostics.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

test -f "$REFERENCE_CSV" || {
  echo "Missing carried-reference data: $REFERENCE_CSV" >&2
  exit 2
}

# Match ais_fix_followup_200 exactly. The A/B/C mechanisms use the unchanged
# 10 x 320 = 3,200 pilot budget and the unchanged 51,200 final-draw cap.
export PYTHON_BIN PRESET=comprehensive METHODS=lime DESIGNS=baseline
export AUXILIARY_VALUES=fresh SIGNAL_STRENGTHS=1.00 SIGNAL_POSITIONS=first
export SELECTION_EVENTS=exact_set PRIMARY_SELECTION_EVENT=exact_set
export TOTAL_NULL_ITERS=200 TOTAL_POWER_ITERS=200 N_SHARDS=48 RF_JOBS=1
export BASE_SEED=2026091501 DESIGN_SEED=314159 MI_NEIGHBORS=12 MCMC_STEPS=20
export INFERENCE_METHOD=ais SPLINE_INFERENCE_METHOD=inherit MC_PROPOSALS=51200
export AIS_PILOT_ITERS=10 AIS_PILOT_SAMPLES=320 AIS_FINAL_BATCH_SIZE=800
export AIS_MIN_DENOMINATOR_ESS=80 AIS_MIN_TAIL_ESS=15 AIS_STOP_WHEN_ESS_MET=true
export MIN_CALIBRATION_ITERS=200 MIN_SIGNAL_TARGETS=60 MIN_CONDITIONAL_POWER=0.80
export LIME_RIDGE_ALPHA=1 LIME_KERNEL_WIDTH="" LIME_NUM_EXPLANATIONS=40

conditions=(
  a_multimodal_retention
  b_denominator_targeting
  c_stratified_exploration
  ab_multimodal_denominator
  ac_multimodal_stratified
  bc_denominator_stratified
  abc_all_mechanisms
)
strategies=(
  ablation_a
  ablation_b
  ablation_c
  ablation_ab
  ablation_ac
  ablation_bc
  ablation_abc
)
# Seven concurrent arms, with exactly 64 one-core shard workers in aggregate.
workers=(10 9 9 9 9 9 9)

mkdir -p "$OUTPUT_BASE"
pids=()
for index in "${!conditions[@]}"; do
  condition="${conditions[$index]}"
  strategy="${strategies[$index]}"
  parallel="${workers[$index]}"
  echo "Starting $condition ($strategy) with $parallel shard workers"
  (
    export AIS_STRATEGY="$strategy"
    export OUTPUT_ROOT="$OUTPUT_BASE/$condition"
    export PARALLEL_SHARDS="$parallel" CPU_BUDGET="$parallel"
    bash examples/run_selection_method_validation_shards.sh
  ) >"$OUTPUT_BASE/${condition}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Arm failed: ${conditions[$index]} (see $OUTPUT_BASE/${conditions[$index]}.log)" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

"$PYTHON_BIN" examples/analyze_ais_mechanism_ablation_200.py \
  "$OUTPUT_BASE" --reference-csv "$REFERENCE_CSV"
echo "AIS mechanism ablation completed: $OUTPUT_BASE"
