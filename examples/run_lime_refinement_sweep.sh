#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUT_BASE="${OUTPUT_BASE:-outputs/selection_method_validation/exact_set/lime_refinement_sweep_adaptive_pilot_200_20260910}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PRESET=comprehensive METHODS=lime DESIGNS=baseline AUXILIARY_VALUES=fresh
export SIGNAL_STRENGTHS="0.30 0.50 0.75 1.00" SIGNAL_POSITIONS=first
export SELECTION_EVENTS=exact_set PRIMARY_SELECTION_EVENT=exact_set
export TOTAL_NULL_ITERS=200 TOTAL_POWER_ITERS=200
export N_SHARDS=32 PARALLEL_SHARDS=32 CPU_BUDGET=32 RF_JOBS=1
export BASE_SEED=20260908 DESIGN_SEED=314159 MI_NEIGHBORS=12 MCMC_STEPS=20
export INFERENCE_METHOD=ais SPLINE_INFERENCE_METHOD=inherit MC_PROPOSALS=12800
export AIS_PILOT_ITERS=5 AIS_PILOT_SAMPLES=80 AIS_FINAL_BATCH_SIZE=800
export AIS_MIN_DENOMINATOR_ESS=80 AIS_MIN_TAIL_ESS=15 AIS_STOP_WHEN_ESS_MET=true
export MIN_CALIBRATION_ITERS=200 MIN_SIGNAL_TARGETS=30 MIN_CONDITIONAL_POWER=0.80

# Sequential configurations, each with at most 32 single-threaded workers.
# LIME neighborhood size remains 64 and its random_state remains 42.
for variant in explanations_30 explanations_35 explanations_45 explanations_40_ridge_0p3 explanations_40_kernel_0p6_sqrt_p; do
  export LIME_RIDGE_ALPHA=1 LIME_KERNEL_WIDTH="" LIME_NUM_EXPLANATIONS=40
  case "$variant" in
    explanations_30) LIME_NUM_EXPLANATIONS=30 ;;
    explanations_35) LIME_NUM_EXPLANATIONS=35 ;;
    explanations_45) LIME_NUM_EXPLANATIONS=45 ;;
    explanations_40_ridge_0p3) LIME_RIDGE_ALPHA=0.3 ;;
    explanations_40_kernel_0p6_sqrt_p) LIME_KERNEL_WIDTH=2.6832815729997477 ;;
  esac
  export OUTPUT_ROOT="$OUTPUT_BASE/$variant"
  echo "LIME configuration: $variant"
  bash examples/run_selection_method_validation_shards.sh
done
"$PYTHON_BIN" examples/analyze_lime_parameter_sweep.py "$OUTPUT_BASE" --refinement-200
touch "$OUTPUT_BASE/COMPLETED"
echo "LIME sweep completed: $OUTPUT_BASE"
