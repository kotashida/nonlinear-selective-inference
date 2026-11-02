"""The portable hash verifier accepts only a hash-confirmed adjacent tail float."""
from pathlib import Path
import subprocess
import sys


def test_portable_hash_reconciliation_is_bounded_and_strict():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import runpy
m = runpy.run_path('benchmarks/verify_selected_five_null_fpr_portable.py')
np = m['np']
value = np.array([[0.1, 4.253924557449169], [-0.5, 1.0]])
origin = value.copy()
origin[0, 1] = np.nextafter(origin[0, 1], np.inf)
recovered, detail = m['reconcile_tail_rounding'](value, m['array_hash'](origin))
assert m['array_hash'](recovered) == m['array_hash'](origin)
assert detail['flat_index'] == 1 and detail['ulp_steps'] == 1
np.testing.assert_array_equal(value, np.array([[0.1,4.253924557449169],[-0.5,1.0]]))
assert m['reconcile_tail_rounding'](value, m['array_hash'](value))[1] is None
for corrupted in [np.array([[.2,4.253924557449169],[-.5,1.0]]),
                  np.nextafter(origin, np.inf)]:
    try:
        m['reconcile_tail_rounding'](value, m['array_hash'](corrupted))
    except RuntimeError:
        pass
    else:
        raise AssertionError('corruption must not be accepted')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
