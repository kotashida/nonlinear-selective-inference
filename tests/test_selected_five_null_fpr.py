"""Verify experiment logic in isolation from the current si_shap package."""
from pathlib import Path
import subprocess
import sys


def test_null_design_denominators_pairing_and_archived_source():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import runpy
import numpy as np
import pandas as pd
m = runpy.run_path('benchmarks/run_selected_five_null_fpr.py')
c = m['settings'](2)
X, y, seeds = m['trial'](0, c)
X2, y2, seeds2 = m['trial'](0, c)
assert X.shape == (100, 20) and y.shape == (100,)
np.testing.assert_array_equal(X, X2)
np.testing.assert_array_equal(y, y2)
assert seeds == seeds2 and len(set(seeds.values())) == 4
assert not np.array_equal(m['trial'](1, c)[0], X)
b = m['b']
r = b.selective_inference(X, y, **b.COMMON, target_seed=seeds['target_seed'],
    ais_seed=seeds['ais_seed'], **b.method_arguments('spline_screening_exact'))
assert len(r['feature_results']) == 1 and r['selection_event'] == 'exact_set'
assert r['observed_target_feature'] in r['observed_selected_features']
assert r['settings']['sigma'] == 1.0
f = pd.DataFrame({'method':['shap_ais']*3, 'p_value':[.001,.6,np.nan],
    'naive_p_value':[.001,.02,.3], 'proposals':[800]*3,
    'denominator_ess':[80]*3, 'tail_ess':[15]*3})
s = m['summarize'](f,c)
row = s[(s.p_value_kind == 'selective') & (s.alpha == .05)].iloc[0]
assert row.n_attempted == 3 and row.n_valid == 2 and row.n_failed == 1
assert row.fpr_all_attempts == 1/3 and row.fpr_valid_only == .5
assert row.unresolved_fpr_upper == 2/3 and not row.complete_finite_evidence
f['p_value'] = np.nan
row = m['summarize'](f,c).iloc[0]
assert row.n_failed == 3 and row.unresolved_fpr_lower == 0 and row.unresolved_fpr_upper == 1
# A failed AIS pair must remain in the denominator and be marked uncomparable.
c1 = m['settings'](1)
records = []
for method in c1['methods']:
    fields = dict(dataset='iid_gaussian_null', sample_size=100, repetition=0,
        method=method, seeds=seeds, rows=np.arange(100), elapsed=0)
    row = b._result_row(r, **fields) if method == 'spline_screening_exact' else b._error_row(ValueError('synthetic failure'), **fields)
    row.update(X_sha256=m['array_hash'](X), y_sha256=m['array_hash'](y), naive_p_value=np.nan)
    records.append(row)
audit, spline = m['verify'](pd.DataFrame(records), c1)
assert len(audit) == 1 and not spline.comparable.any()
assert sorted(i for shard in range(32) for i in b.assigned_repetitions(0,10000,32,shard)) == list(range(10000))
try:
    m['require'](False, 'source drift')
except RuntimeError:
    pass
else:
    raise AssertionError('drift must be fatal')
print('Null generation, archive hashes, exact inference, denominator handling, allocation passed.')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
