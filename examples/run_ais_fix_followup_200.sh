#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_BASE="${OUTPUT_BASE:-outputs/selection_method_validation/exact_set/ais_fix_followup_200}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Three matched arms. Each arm uses the same simulation, target, and AIS seed
# schedule so failures can be compared with paired tests.
export PYTHON_BIN PRESET=comprehensive METHODS=lime DESIGNS=baseline
export AUXILIARY_VALUES=fresh SIGNAL_STRENGTHS=1.00 SIGNAL_POSITIONS=first
export SELECTION_EVENTS=exact_set PRIMARY_SELECTION_EVENT=exact_set
export TOTAL_NULL_ITERS=200 TOTAL_POWER_ITERS=200
export N_SHARDS=48 PARALLEL_SHARDS=48 CPU_BUDGET=48 RF_JOBS=1
export BASE_SEED=2026091501 DESIGN_SEED=314159 MI_NEIGHBORS=12 MCMC_STEPS=20
export INFERENCE_METHOD=ais SPLINE_INFERENCE_METHOD=inherit MC_PROPOSALS=51200
export AIS_PILOT_ITERS=10 AIS_PILOT_SAMPLES=320 AIS_FINAL_BATCH_SIZE=800
export AIS_MIN_DENOMINATOR_ESS=80 AIS_MIN_TAIL_ESS=15 AIS_STOP_WHEN_ESS_MET=true
export MIN_CALIBRATION_ITERS=200 MIN_SIGNAL_TARGETS=60 MIN_CONDITIONAL_POWER=0.80
export LIME_RIDGE_ALPHA=1 LIME_KERNEL_WIDTH="" LIME_NUM_EXPLANATIONS=40

for condition in baseline_control multimodal_qadapt reallocated_pilot_exploration; do
  case "$condition" in
    baseline_control) AIS_STRATEGY=baseline ;;
    *) AIS_STRATEGY="$condition" ;;
  esac
  export AIS_STRATEGY OUTPUT_ROOT="$OUTPUT_BASE/$condition"
  echo "Starting matched 200-iteration AIS arm: $condition"
  bash examples/run_selection_method_validation_shards.sh
done

"$PYTHON_BIN" examples/analyze_ais_fix_followup_200.py "$OUTPUT_BASE"
echo "AIS follow-up completed: $OUTPUT_BASE"
