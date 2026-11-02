#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUT_BASE="${OUTPUT_BASE:-outputs/selection_method_validation/exact_set/mi_parameter_tuning/estimators_histogram_kde_200_trials_20260912}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PRESET=comprehensive METHODS=mutual_information DESIGNS=baseline AUXILIARY_VALUES=fresh
export SIGNAL_STRENGTHS="0.30 0.50 0.75 1.00" SIGNAL_POSITIONS=first
export SELECTION_EVENTS=exact_set PRIMARY_SELECTION_EVENT=exact_set
export TOTAL_NULL_ITERS=200 TOTAL_POWER_ITERS=200
export N_SHARDS=32 PARALLEL_SHARDS=32 CPU_BUDGET=32 RF_JOBS=1
export BASE_SEED=20260908 DESIGN_SEED=314159 MI_NEIGHBORS=30 MI_BINS=5 MI_BANDWIDTH=1.0 MCMC_STEPS=20
export INFERENCE_METHOD=ais SPLINE_INFERENCE_METHOD=inherit MC_PROPOSALS=51200
export AIS_PILOT_ITERS=10 AIS_PILOT_SAMPLES=320 AIS_FINAL_BATCH_SIZE=800
export AIS_MIN_DENOMINATOR_ESS=80 AIS_MIN_TAIL_ESS=15 AIS_STOP_WHEN_ESS_MET=true
export MIN_CALIBRATION_ITERS=200 MIN_SIGNAL_TARGETS=30 MIN_CONDITIONAL_POWER=0.80

for estimator in histogram kde; do
  export MI_ESTIMATOR="$estimator"
  export OUTPUT_ROOT="$OUTPUT_BASE/$estimator"
  echo "Running MI estimator=$estimator with fixed combined AIS"
  bash examples/run_selection_method_validation_shards.sh
done
"$PYTHON_BIN" examples/verify_mi_estimators.py "$OUTPUT_BASE"
touch "$OUTPUT_BASE/COMPLETED"
echo "MI sweep completed: $OUTPUT_BASE"
