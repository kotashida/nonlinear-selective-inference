"""Full AIS MI sweep: sequential configurations, at most 32 single-thread shards."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
MANIFEST = ROOT / 'examples/mi_cpu_sweep_1000.json'


def read_manifest():
    m = json.loads(MANIFEST.read_text())
    e = m['environment']
    if not 1 <= int(e['PARALLEL_SHARDS']) <= int(e['N_SHARDS']) <= 32:
        raise ValueError('Both total and parallel shard counts must be <=32.')
    if int(e['PARALLEL_SHARDS']) * int(e['RF_JOBS']) > int(e['CPU_BUDGET']) or int(e['CPU_BUDGET']) > 32:
        raise ValueError('CPU budget exceeded.')
    from examples.validate_selection_methods import DESIGNS
    if m['design'] != DESIGNS['baseline'] or m['design']['k_select'] != 2:
        raise ValueError('The baseline design must remain unchanged with k_select=2.')
    if len(m['configurations']) != 10 or len({c['name'] for c in m['configurations']}) != 10:
        raise ValueError('Expected exactly ten uniquely named MI configurations.')
    return m


def environment(m, c):
    return {**os.environ, **m['environment'], 'PYTHON_BIN': sys.executable,
            'PYTHONPATH': str(ROOT / 'src') + os.pathsep + str(ROOT),
            'MI_ESTIMATOR': c['estimator'], 'MI_NEIGHBORS': str(c['neighbors']),
            'MI_BANDWIDTH': str(c['bandwidth']), 'OUTPUT_ROOT': str(ROOT / m['output_root'] / c['name'])}


def source_hashes():
    files = [*sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'examples').glob('*.py')),
             ROOT / 'examples/run_selection_method_validation_shards.sh', MANIFEST, ROOT / 'pyproject.toml']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files}


def lock(m):
    out = ROOT / m['output_root']
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('requested_manifest.json', m), ('source_hashes.json', source_hashes())]:
        p = out / name
        if p.exists() and json.loads(p.read_text()) != value:
            raise ValueError(f'{name} changed; use a new experiment directory.')
        p.write_text(json.dumps(value, indent=2) + '\n')
    return out


def verify_variant(m, c):
    import numpy as np
    import pandas as pd
    from examples.validate_selection_methods import _seed
    from si_shap.selection import MutualInformationSelector
    root = ROOT / m['output_root'] / c['name']
    e = m['environment']
    selector = MutualInformationSelector(n_neighbors=c['neighbors'], estimator=c['estimator'], bandwidth=c['bandwidth'])
    for i in range(32):
        n = 32 if i < 8 else 31
        shard = root / f'shard_{i}'
        s = json.loads((shard / 'validation_settings.json').read_text())
        expected = dict(n_null_iters=n, n_power_iters=n, seed=20260908+i, design_seed=314159,
                        methods=['mutual_information'], designs=['baseline'], comparison_events=['exact_set'],
                        mi_neighbors=c['neighbors'], mi_params=dict(estimator=c['estimator'], bandwidth=c['bandwidth'], n_bins=5),
                        signal_strengths=[.3,.5,.75,1.], signal_positions=['first'], fixed_auxiliary_values=[None], rf_jobs=1)
        fixed = dict(inference_method='ais', max_final_samples=51200, pilot_iters=10, pilot_samples=320,
                     final_batch_size=800, min_denominator_ess=80., min_tail_ess=15.,
                     stop_when_ess_met=True, ais_strategy='a_robust_draw_floor')
        for k,v in {**expected, **fixed}.items():
            if s[k] != v: raise ValueError((shard,k,s[k],v))
        for j, condition in enumerate(['null_fresh', 'power_feature_0_beta_0p3', 'power_feature_0_beta_0p5', 'power_feature_0_beta_0p75', 'power_feature_0_beta_1p0']):
            folder = shard / 'mutual_information/baseline' / condition
            raw = json.loads((folder / 'settings.json').read_text())
            for k,v in {**fixed, **m['design'], 'n_iters':n, 'selection_events':['exact_set']}.items():
                if raw[k] != v: raise ValueError((folder,k,raw[k],v))
            expected_seed = _seed(20260908+i, 0, 0) if j == 0 else _seed(20260908+i, 10000, 0, 0, j-1)
            assert raw['seed'] == expected_seed
            if raw['selector_settings'] != dict(selector.get_settings()):
                raise ValueError(f'Wrong selector at {folder}')
            filename = 'p_value_results.csv' if j == 0 else 'target_results.csv'
            frame = pd.read_csv(folder / filename)
            assert len(frame) == n and frame.iteration.nunique() == n
            assert frame.selection_event.eq('exact_set').all()
            assert set(frame.iteration) == set(range(1,n+1))
            if j:
                assert raw['signal_features'] == [0] and raw['signal_strength'] == [.3,.5,.75,1.][j-1]
                feats=pd.read_csv(folder/'feature_results.csv')
                assert len(feats)==n and feats.iteration.nunique()==n
                joined=frame.merge(feats,on=['iteration','selection_event'],suffixes=('_target','_feature'),validate='one_to_one')
                assert (joined.target_feature == joined.feature).all()
                np.testing.assert_allclose(joined.p_value_target,joined.p_value_used,equal_nan=True)
    pooled=json.loads((root/'pooled_summary/validation_settings.json').read_text())
    assert pooled['n_shards']==32 and pooled['n_null_iters']==pooled['n_power_iters']==1000
    assert pooled['shard_root_seeds']==list(range(20260908,20260940))
    return 5000


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check','run','verify'])
    args=parser.parse_args()
    m=read_manifest()
    if args.action=='check':
        print(json.dumps(m,indent=2)); return
    out=lock(m)
    if args.action=='run':
        for c in m['configurations']:
            print('Running configuration',c,flush=True)
            subprocess.run(['bash','examples/run_selection_method_validation_shards.sh'],cwd=ROOT,env=environment(m,c),check=True)
            verify_variant(m,c)
            subprocess.run([sys.executable,'examples/analyze_mi_cpu_sweep.py','--variant',c['name']],cwd=ROOT,check=True)
    total=sum(verify_variant(m,c) for c in m['configurations'])
    assert total==50000
    if args.action=='run':
        subprocess.run([sys.executable,'examples/analyze_mi_cpu_sweep.py'],cwd=ROOT,check=True)
        (out/'COMPLETED').write_text('Verified 10 MI configurations; 50,000 AIS tests; max 32 parallel shards.\n')
    print('Verified',total,'AIS tests.',flush=True)


if __name__=='__main__': main()
