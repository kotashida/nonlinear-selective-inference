#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
CONTROL_ROOT=outputs/selection_method_validation/exact_set/selected_five_ais_exact_mi30_64cores
mkdir -p "$CONTROL_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1 MPLBACKEND=Agg
settings=(robust_draw_floor robust_floor_multimodal_denominator)
pids=()
for setting in "${settings[@]}"; do
  "$PYTHON_BIN" -u examples/run_selected_five_ais_exact_mi30.py \
    "examples/selected_five_ais_${setting}.json" --run --workers 32 \
    >"$CONTROL_ROOT/${setting}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Arm failed: ${settings[$index]}; see $CONTROL_ROOT/${settings[$index]}.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1
printf 'Both matched AIS arms completed.\n' > "$CONTROL_ROOT/COMPLETED"
echo 'Both arms completed; 32 one-core shard workers per arm, 64 aggregate.'
