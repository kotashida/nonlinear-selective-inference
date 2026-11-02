from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import pytest

from examples.analyze_lime_parameter_sweep import AIS_200_VARIANTS
from si_shap.inference import _run_ais


def test_larger_cap_preserves_early_success():
    # Shared inference core: no selector-specific special case.
    arguments = dict(t_obs=2.0, rank=3, is_selected=lambda z: z > 0.8,
                     pilot_iters=5, pilot_samples=80, final_batch_size=800,
                     min_denominator_ess=80, min_tail_ess=15, stop_when_ess_met=True)
    a = _run_ais(**arguments, rng=np.random.default_rng(20260908), max_final_samples=12800)
    b = _run_ais(**arguments, rng=np.random.default_rng(20260908), max_final_samples=51200)
    assert np.isfinite(a[0])
    assert a == b


def test_ais_launcher_preserves_lime_and_exports_exact_budget(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash unavailable')
    directory = tmp_path / 'examples'
    directory.mkdir()
    launch = directory / 'run_lime_ais_settings_sweep.sh'
    launch.write_bytes((root/'examples'/launch.name).read_bytes().replace(b'\r\n', b'\n'))
    mock = '''#!/usr/bin/env bash
mkdir -p "$OUTPUT_ROOT"
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "${OUTPUT_ROOT##*/}" "$LIME_RIDGE_ALPHA" "$LIME_KERNEL_WIDTH" "$LIME_NUM_EXPLANATIONS" "$TOTAL_NULL_ITERS" "$TOTAL_POWER_ITERS" "$BASE_SEED" "$DESIGN_SEED" "$MC_PROPOSALS" "$AIS_PILOT_ITERS" "$AIS_PILOT_SAMPLES" "$AIS_FINAL_BATCH_SIZE" "$AIS_MIN_DENOMINATOR_ESS" "$AIS_MIN_TAIL_ESS" "$AIS_STOP_WHEN_ESS_MET" "$METHODS"
'''
    (directory/'run_selection_method_validation_shards.sh').write_bytes(mock.encode())
    result = subprocess.run([str(bash), str(launch)], env={**os.environ, 'PYTHON_BIN':'true', 'OUTPUT_BASE':'results'},
                            capture_output=True, text=True, check=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if '|' in line]
    assert len(rows) == 5
    for row, (name, (budget, rounds, samples)) in zip(rows, AIS_200_VARIANTS.items()):
        assert row[0] == name
        assert row[1:8] == ['1', '', '40', '200', '200', '20260908', '314159']
        assert list(map(int, row[8:11])) == [budget, rounds, samples]
        assert row[11:] == ['800', '80', '15', 'true', 'lime']
