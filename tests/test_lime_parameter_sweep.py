import numpy as np
import pytest
from examples import validate_selection_methods as validation
from examples.summarize_selection_method_validation_shards import _validate_settings
from si_shap.selection import LimeSelector, make_selector


def test_additional_variants_change_one_lime_setting_only():
    from examples.analyze_lime_parameter_sweep import ADDITIONAL_VARIANTS, VARIANTS
    baseline = VARIANTS["baseline"]
    assert len(ADDITIONAL_VARIANTS) == 3
    assert not set(ADDITIONAL_VARIANTS).intersection(VARIANTS)
    for ridge, width, explanations in ADDITIONAL_VARIANTS.values():
        assert sum(a != b for a, b in zip((ridge, width, explanations), baseline)) == 1
        selector = validation._selector_arguments("lime", 12, dict(
            ridge_alpha=ridge, kernel_width=width, num_explanations=explanations), 1)["selector"]
        assert selector.num_samples == 64 and selector.random_state == 42


def test_focused_variants_preserve_40_explanation_reference():
    from examples.analyze_lime_parameter_sweep import FOCUSED_200_VARIANTS, ADDITIONAL_VARIANTS
    assert len(FOCUSED_200_VARIANTS) == 5
    reference = ADDITIONAL_VARIANTS["explanations_40"]
    assert FOCUSED_200_VARIANTS["explanations_40"] == reference
    for name, values in FOCUSED_200_VARIANTS.items():
        assert sum(a != b for a, b in zip(values, reference)) == (name != "explanations_40")
        ridge, width, explanations = values
        selector = validation._selector_arguments("lime", 12, dict(
            ridge_alpha=ridge, kernel_width=width, num_explanations=explanations), 1)["selector"]
        assert selector.num_samples == 64 and selector.random_state == 42


@pytest.mark.parametrize("family", ["focused", "refinement"])
def test_focused_shell_exports_only_requested_changes(tmp_path, family):
    from pathlib import Path
    import os
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[1]
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash unavailable')
    examples = tmp_path / 'examples'
    examples.mkdir()
    launcher = examples / ('run_lime_explanations40_sweep.sh' if family == 'focused' else 'run_lime_refinement_sweep.sh')
    launcher.write_bytes((root / 'examples' / launcher.name).read_bytes().replace(b'\r\n', b'\n'))
    # Capture the actual exported configuration without starting simulation workers.
    mock = '''#!/usr/bin/env bash
mkdir -p "$OUTPUT_ROOT"
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "${OUTPUT_ROOT##*/}" "$LIME_RIDGE_ALPHA" "$LIME_KERNEL_WIDTH" "$LIME_NUM_EXPLANATIONS" "$TOTAL_NULL_ITERS" "$TOTAL_POWER_ITERS" "$BASE_SEED" "$DESIGN_SEED" "$MC_PROPOSALS" "$AIS_PILOT_ITERS" "$AIS_PILOT_SAMPLES" "$AIS_FINAL_BATCH_SIZE" "$AIS_MIN_DENOMINATOR_ESS" "$AIS_MIN_TAIL_ESS" "$AIS_STOP_WHEN_ESS_MET"
'''
    (examples / 'run_selection_method_validation_shards.sh').write_bytes(mock.encode())
    result = subprocess.run([str(bash), str(launcher)], env={**os.environ, 'PYTHON_BIN': 'true', 'OUTPUT_BASE': 'results'},
                            capture_output=True, text=True, check=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if '|' in line]
    from examples.analyze_lime_parameter_sweep import FOCUSED_200_VARIANTS, REFINEMENT_200_VARIANTS
    variants = FOCUSED_200_VARIANTS if family == "focused" else REFINEMENT_200_VARIANTS
    assert len(rows) == 5
    for row, (name, (ridge, width, explanations)) in zip(rows, variants.items()):
        assert row[0] == name
        assert float(row[1]) == ridge
        assert (float(row[2]) if row[2] else None) == width
        assert int(row[3]) == explanations
        assert row[4:] == ['200', '200', '20260908', '314159', '12800', '5', '80', '800', '80', '15', 'true']


def test_default_lime_matches_existing_selector():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(30, 5))
    y = X[:, 0] + rng.normal(size=30)
    original = make_selector(selection_method="lime", rf_params={"n_jobs": 1})
    configured = validation._selector_arguments("lime", 12, rf_jobs=1)
    assert configured["selection_method"] is None
    actual = configured["selector"]
    assert actual.get_settings() == original.get_settings()
    np.testing.assert_array_equal(actual.select(X, y, 2).scores, original.select(X, y, 2).scores)


def test_custom_lime_parameters_and_shard_mismatch(tmp_path):
    args = validation.parse_args(["--methods", "lime", "--lime-ridge-alpha", "3",
                                  "--lime-kernel-width", "4.47", "--lime-num-explanations", "20"])
    params = dict(ridge_alpha=args.lime_ridge_alpha, kernel_width=args.lime_kernel_width,
                  num_explanations=args.lime_num_explanations)
    selector = validation._selector_arguments("lime", 12, params, 1)["selector"]
    assert isinstance(selector, LimeSelector)
    assert all(selector.get_settings()[key] == value for key, value in params.items())
    assert selector.num_samples == 64
    with pytest.raises(ValueError, match="lime_params"):
        _validate_settings([tmp_path / "shard_0", tmp_path / "shard_1"],
                           [{"lime_params": params}, {"lime_params": {**params, "ridge_alpha": 10}}])
