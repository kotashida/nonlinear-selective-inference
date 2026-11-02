"""Package existing diagnostic inputs without fitting a model or sampling.

Run locally once. Server deployment needs these small NPZ fixtures, not the
previous output trees or the benchmark pickle. Existing fixtures are immutable.
"""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from si_shap.power import _generate_power_dataset
from si_shap.simulation import _generate_gaussian_design

OUT = ROOT/'examples/shap_sampling_followup_inputs'
OLD = ROOT/'outputs/selection_method_validation/exact_set/shap_sampling_ten_250_20260928'
BENCH = ROOT/'outputs/benchmarks/selected_five_kde2p5_1000_cpu32_20260924'


def arrays_hash(X, y):
    return hashlib.sha256(np.asarray(X,dtype='<f8').tobytes()+
                          np.asarray(y,dtype='<f8').tobytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError('Fixtures already exist; refusing to overwrite')
    old = pd.read_csv(OLD/'unified_tests.csv', keep_default_na=False, na_values=[''])
    baseline = old[old.arm.eq('test_01')]
    manifest = json.loads((OLD/'requested_manifest.json').read_text())
    hard = pd.read_csv(OLD/'analysis/hard_trials.csv', keep_default_na=False)
    cases = []
    for h in hard.itertuples():
        row = baseline[baseline.condition.eq(h.condition)&baseline.shard.eq(h.shard)
                       &baseline.iteration.eq(h.iteration)].iloc[0]
        seed = manifest['base_seed']+int(h.shard)
        beta = None if h.condition == 'null' else float(h.condition.split('_')[1])
        if beta is None:
            _, parent = np.random.SeedSequence(seed).spawn(2)
            child = parent.spawn(int(h.iteration))[-1]
            response_seed, _, _ = child.spawn(3)
            X = _generate_gaussian_design(np.random.default_rng(manifest['design_seed']),100,20)
            y = np.random.default_rng(response_seed).standard_normal(100)
            mean = np.zeros(100)
        else:
            condition = 1+manifest['betas'].index(beta)
            child = np.random.SeedSequence(seed+condition*10000).spawn(int(h.iteration))[-1]
            data_seed, _, _ = child.spawn(3)
            X, y, mean = _generate_power_dataset(np.random.default_rng(data_seed),100,20,(0,),beta,
                                                return_mean=True)
        case_id = f"old_{h.condition}_s{h.shard}_i{h.iteration}"
        cases.append((case_id, X, y, mean, row, 'selected_difficult_synthetic', beta))
    b = pd.read_csv(BENCH/'results.csv')
    with (ROOT/'benchmarks/table_dataset.pkl').open('rb') as handle:
        datasets = pickle.load(handle)
    for dataset in ('heating_load','cooling_load'):
        g = b[b.method.eq('shap_ais')&b.dataset.eq(dataset)]
        failed = g[g.p_value.isna()]
        # One severe concentration case and one near-threshold denominator case.
        severe = failed.nlargest(1,'max_normalized_weight')
        near = failed[failed.denominator_ess.lt(80)].nlargest(1,'denominator_ess')
        targets = pd.concat([severe,near]).drop_duplicates('repetition')
        if len(targets) != 2:
            raise ValueError('Expected two distinct Energy failure examples')
        for _, row in targets.iterrows():
            full_X, full_y = (np.asarray(a,dtype=float) for a in datasets[dataset])
            indices = json.loads(row.row_indices)
            cases.append((f"energy_{dataset}_r{int(row.repetition)}",full_X[indices],full_y[indices],
                          np.full(100,np.nan),row,'selected_difficult_real_data',None))
        for _, row in targets.iterrows():
            controls = g[g.p_value.notna()&g.feature.eq(row.feature)
                         &~g.repetition.isin([c[4].repetition for c in cases if c[4].get('dataset')==dataset])]
            control = controls.iloc[np.argmin(abs(controls.t_obs-row.t_obs))]
            indices = json.loads(control.row_indices)
            cases.append((f"control_{dataset}_r{int(control.repetition)}",full_X[indices],full_y[indices],
                          np.full(100,np.nan),control,'matched_successful_real_data_control',None))
    OUT.mkdir(parents=True)
    metadata = []
    for case_id, X, y, mean, row, reason, beta in cases:
        np.savez_compressed(OUT/f'{case_id}.npz',X=X,y=y,mean=mean)
        selected = row.observed_selected_features
        if isinstance(selected,str):
            import ast
            selected = list(ast.literal_eval(selected))
        metadata.append(dict(case_id=case_id,file=f'{case_id}.npz',reason=reason,beta=beta,
                             auxiliary_u=float(row.auxiliary_u),
                             target_feature=int(row.get('target_feature',row.get('feature'))),
                             observed_selected_features=sorted(map(int,selected)),
                             t_obs=float(row.t_obs),input_hash=arrays_hash(X,y),
                             previous_p_value=None if pd.isna(row.p_value) else float(row.p_value),
                             previous_status=str(row.status)))
    # All three reference failures, both large disagreements, two severe Energy
    # failures, and one successful Energy control; fixed before any new results.
    replay = ['old_beta_1.0_s17_i2','old_beta_1.0_s19_i5','old_beta_1.0_s27_i2',
              'old_beta_0.75_s3_i1','old_null_s9_i8',
              next(c['case_id'] for c in metadata if c['case_id'].startswith('energy_heating')),
              next(c['case_id'] for c in metadata if c['case_id'].startswith('energy_cooling')),
              next(c['case_id'] for c in metadata if c['case_id'].startswith('control_'))]
    value = dict(cases=metadata,replay_cases=replay,selection_policy=
                 '20 existing hard synthetic cases; 2 failures and 2 matched successful controls per Energy task')
    (OUT/'cases.json').write_text(json.dumps(value,indent=2)+'\n')
    print(f'Packaged {len(metadata)} diagnostic inputs and {len(replay)} replay cases; no inference run.')


if __name__ == '__main__':
    main()
