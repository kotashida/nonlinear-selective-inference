"""Read-only audit of historical results; writes only a new audit report."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['heating_load','cooling_load','gas_turbine','red_wine','white_wine','abalone','concrete','housing']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()


def match_hash(x, checksum):
    if array_sha(x) == checksum:
        return 0
    # Exactly the documented cross-platform one-ULP normal-tail reconciliation.
    for j in np.flatnonzero(np.abs(x.ravel()) >= 3.6541528853610088):
        for direction in (-np.inf, np.inf):
            candidate = x.copy()
            candidate.ravel()[j] = np.nextafter(candidate.ravel()[j], direction)
            if array_sha(candidate) == checksum:
                return 1
    raise ValueError('Regenerated input does not match its originating array hash')


def audit(path):
    config = json.loads((path/'configuration.json').read_text())
    columns = pd.read_csv(path/'results.csv', nrows=0).columns
    wanted = ['dataset','sample_size','repetition','method','p_value','unadjusted_p_value',
        'row_indices','observed_selected_features','observed_target_feature','target_seed','ais_seed',
        'subset_seed','design_seed','response_seed','X_sha256','y_sha256','radial_t_obs','runtime_seconds',
        'sigma','selection_event','target_rule','multiplicity_method','test_statistic','p_value_method','finite_sample_valid','status']
    frame = pd.read_csv(path/'results.csv', usecols=[c for c in wanted if c in columns])
    methods, datasets = config['methods'], config['datasets']
    reps = config['repetitions']
    expected = {(d, i, method) for d in datasets for i in range(reps) for method in methods}
    keys = list(zip(frame.dataset, frame.repetition, frame.method))
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert frame.sample_size.eq(100).all() and frame.sigma.eq(1).all()
    assert frame.selection_event.eq('exact_set').all() and frame.target_rule.eq('uniform_from_selected').all()
    assert frame.multiplicity_method.eq('none').all() and frame.test_statistic.eq('spline').all()
    reconciliations = 0
    pair_mismatch = 0
    for (dataset, trial), group in frame.groupby(['dataset','repetition']):
        names = ['target_seed','ais_seed','row_indices']
        assert all(group[k].nunique(dropna=False) == 1 for k in names)
        if 'subset_seed' in frame:
            seed_children = np.random.SeedSequence([config['master_seed'], ORDER.index(dataset), 100, trial]).spawn(3)
            expected_seeds = dict(zip(['subset_seed','target_seed','ais_seed'], [int(s.generate_state(1)[0]) for s in seed_children]))
            # Only row count is needed; the original pickle is fixed and checked below.
            with_data = audit.data[dataset]
            rows = np.sort(np.random.default_rng(expected_seeds['subset_seed']).choice(len(with_data[1]), 100, replace=False))
            assert json.loads(group.row_indices.iloc[0]) == rows.tolist()
        else:
            seed_children = np.random.SeedSequence([config['master_seed'],100,20,trial]).spawn(4)
            expected_seeds = dict(zip(['design_seed','response_seed','target_seed','ais_seed'], [int(s.generate_state(1)[0]) for s in seed_children]))
            X = np.random.default_rng(expected_seeds['design_seed']).normal(size=(100,20))
            y = np.random.default_rng(expected_seeds['response_seed']).normal(size=100)
            for name, x in [('X',X),('y',y)]:
                assert group[name+'_sha256'].nunique() == 1
                reconciliations += match_hash(x, group[name+'_sha256'].iloc[0])
        assert all(group[name].eq(seed).all() for name, seed in expected_seeds.items())
        u = np.random.default_rng(expected_seeds['target_seed']).random()
        for row in group.itertuples():
            if pd.notna(row.observed_target_feature):
                selected = sorted(json.loads(row.observed_selected_features))
                assert len(selected) == len(set(selected)) == 2
                assert int(row.observed_target_feature) == selected[min(int(u*2),1)]
        pair = group[group.method.isin(['spline_screening_ais','spline_screening_exact'])].set_index('method')
        if len(pair) == 2 and pair.radial_t_obs.notna().all():
            left, right = pair.loc['spline_screening_ais'], pair.loc['spline_screening_exact']
            same = set(json.loads(left.observed_selected_features)) == set(json.loads(right.observed_selected_features))
            same &= left.observed_target_feature == right.observed_target_feature
            same &= np.isclose(left.radial_t_obs,right.radial_t_obs,rtol=0,atol=1e-12)
            pair_mismatch += not same
    assert pair_mismatch == 0
    summaries = []
    stored = pd.read_csv(path/'summary.csv')
    for (dataset, method), group in frame.groupby(['dataset','method']):
        p = group.p_value
        valid = p.notna()
        assert p[valid].between(0,1).all()
        v, k = int(valid.sum()), int(p.lt(.05).sum())
        if 'p_value_kind' in stored:
            old = stored[(stored.method == method) & (stored.p_value_kind == 'selective') & (stored.alpha == .05)].iloc[0]
            assert old.n_attempted == len(group) and old.n_valid == v and old.n_rejected == k
        else:
            old = stored[(stored.method == method) & (stored.dataset == dataset)].iloc[0]
            assert old.n_results == len(group) and old.n_finite == v and np.isclose(old.selective_rejection_rate,k/v)
        summaries.append(dict(dataset=dataset,method=method,attempted=len(group),finite=v,failures=len(group)-v,
            rejected_alpha05=k,valid_fraction=k/v if v else None,all_attempt_lower=k/len(group),
            all_attempt_upper=(k+len(group)-v)/len(group),median_seconds=float(group.runtime_seconds.median())))
    source_checks = {}
    for name, checksum in config['archived_core_source_hashes'].items():
        local = ROOT/'benchmarks/fpr_frozen'/Path(name).relative_to('src')
        source_checks[name] = sha(local) == checksum
    assert all(source_checks.values())
    if 'input' in config:
        assert sha(ROOT/config['input']['path']) == config['input']['sha256']
        for name, checksum in config['input']['raw_source_sha256'].items():
            assert sha(ROOT/'benchmarks/dataset'/name) == checksum
    return dict(output=path.relative_to(ROOT).as_posix(),config_sha256=sha(path/'configuration.json'),
        results_sha256=sha(path/'results.csv'),attempts=len(frame),failures=int(frame.p_value.isna().sum()),
        key_coverage_complete=True,rows_seeds_targets_matched=True,spline_pair_mismatches=pair_mismatch,
        stored_alpha05_summaries_reproduced=True,core_archived_source_matches_frozen_package=source_checks,
        one_ulp_reconciled_arrays=reconciliations,summaries=summaries,
        inference_metadata=frame.groupby(['method','p_value_method','finite_sample_valid'],dropna=False).size().reset_index(name='count').to_dict('records'))


def main():
    with (ROOT/'benchmarks/table_dataset.pkl').open('rb') as handle:
        audit.data = pickle.load(handle)
    report = dict(schema=1,audit_source_sha256=sha(Path(__file__)),runs=[])
    for path in sorted((ROOT/'outputs/benchmarks').glob('selected_five_*')):
        if (path/'results.csv').exists():
            result = audit(path)
            report['runs'].append(result)
            print(path.name, result['attempts'], 'attempts;',result['failures'],'failures; audits passed',flush=True)
    out = ROOT/'docs/publication_mcmc_previous_audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
