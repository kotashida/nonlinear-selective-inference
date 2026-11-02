#!/usr/bin/env bash
set -euo pipefail

# Run a shared-budget exact-set AIS comparison for SHAP, LIME, spline
# screening, and mutual information, followed by an exact-spline reference on
# the same simulation seeds. The default is a diagnostic pilot; override the
# iteration counts and OUTPUT_BASE for a confirmatory run.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERAL_RUNNER="$PROJECT_ROOT/examples/run_selection_method_validation_shards.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/selection_method_validation/exact_set/ais_method_comparison_pilot_20260908}"
TOTAL_NULL_ITERS="${TOTAL_NULL_ITERS:-200}"
TOTAL_POWER_ITERS="${TOTAL_POWER_ITERS:-200}"
N_SHARDS="${N_SHARDS:-32}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-32}"
CPU_BUDGET="${CPU_BUDGET:-32}"
RF_JOBS="${RF_JOBS:-1}"
BASE_SEED="${BASE_SEED:-20260908}"
DESIGN_SEED="${DESIGN_SEED:-314159}"
MIN_CALIBRATION_ITERS="${MIN_CALIBRATION_ITERS:-$TOTAL_NULL_ITERS}"
MIN_SIGNAL_TARGETS="${MIN_SIGNAL_TARGETS:-30}"

common_environment=(
  PYTHON_BIN="$PYTHON_BIN"
  PRESET=comprehensive
  DESIGNS=baseline
  AUXILIARY_VALUES=fresh
  SIGNAL_STRENGTHS="0.30 0.50 0.75 1.00"
  SIGNAL_POSITIONS=first
  SELECTION_EVENTS=exact_set
  PRIMARY_SELECTION_EVENT=exact_set
  TOTAL_NULL_ITERS="$TOTAL_NULL_ITERS"
  TOTAL_POWER_ITERS="$TOTAL_POWER_ITERS"
  N_SHARDS="$N_SHARDS"
  PARALLEL_SHARDS="$PARALLEL_SHARDS"
  CPU_BUDGET="$CPU_BUDGET"
  RF_JOBS="$RF_JOBS"
  BASE_SEED="$BASE_SEED"
  DESIGN_SEED="$DESIGN_SEED"
  MC_PROPOSALS=1600
  MI_NEIGHBORS=12
  AIS_PILOT_ITERS=3
  AIS_PILOT_SAMPLES=40
  AIS_FINAL_BATCH_SIZE=1600
  AIS_MIN_DENOMINATOR_ESS=80
  AIS_MIN_TAIL_ESS=15
  AIS_STOP_WHEN_ESS_MET=false
  MIN_CALIBRATION_ITERS="$MIN_CALIBRATION_ITERS"
  MIN_SIGNAL_TARGETS="$MIN_SIGNAL_TARGETS"
  MIN_CONDITIONAL_POWER=0.80
)

mkdir -p "$OUTPUT_BASE"

echo "Running matched AIS comparison."
env "${common_environment[@]}" \
  OUTPUT_ROOT="$OUTPUT_BASE/ais_all_methods" \
  METHODS="shap lime spline_screening mutual_information" \
  INFERENCE_METHOD=ais \
  SPLINE_INFERENCE_METHOD=inherit \
  bash "$GENERAL_RUNNER"

echo "Running exact-spline reference with the same data seeds."
env "${common_environment[@]}" \
  OUTPUT_ROOT="$OUTPUT_BASE/spline_exact_reference" \
  METHODS=spline_screening \
  INFERENCE_METHOD=ais \
  SPLINE_INFERENCE_METHOD=exact_spline \
  bash "$GENERAL_RUNNER"

echo "AIS comparison: $OUTPUT_BASE/ais_all_methods/pooled_summary"
echo "Exact-spline reference: $OUTPUT_BASE/spline_exact_reference/pooled_summary"
