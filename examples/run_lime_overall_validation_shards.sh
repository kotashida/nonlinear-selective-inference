#!/usr/bin/env bash
set -euo pipefail

# Reproduce slides 10-11 with LIME as the feature-selection method:
# fixed X, n=100, p=20, k=2, same_target, 5,000 null iterations,
# 5,000 iterations at each beta, and 800 conditional-MC proposals.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/lime_overall_validation_shards}"
export PRESET="comprehensive"
export METHODS="lime"
export DESIGNS="baseline"
export AUXILIARY_VALUES="fresh"
export SIGNAL_STRENGTHS="0.30 0.50 0.75 1.00"
export SIGNAL_POSITIONS="first"
export SELECTION_EVENTS="same_target"
export PRIMARY_SELECTION_EVENT="same_target"
export TOTAL_NULL_ITERS="${TOTAL_NULL_ITERS:-5000}"
export TOTAL_POWER_ITERS="${TOTAL_POWER_ITERS:-5000}"
export MC_PROPOSALS="${MC_PROPOSALS:-800}"
export MIN_CALIBRATION_ITERS="${MIN_CALIBRATION_ITERS:-5000}"
export MIN_SIGNAL_TARGETS="${MIN_SIGNAL_TARGETS:-300}"
export MIN_CONDITIONAL_POWER="${MIN_CONDITIONAL_POWER:-0.80}"
export N_SHARDS="${N_SHARDS:-10}"
export PARALLEL_SHARDS="${PARALLEL_SHARDS:-10}"
export CPU_BUDGET="${CPU_BUDGET:-32}"
export RF_JOBS="${RF_JOBS:-3}"
export BASE_SEED="${BASE_SEED:-20260821}"
export DESIGN_SEED="${DESIGN_SEED:-314159}"

exec bash "$PROJECT_ROOT/examples/run_selection_method_validation_shards.sh"
