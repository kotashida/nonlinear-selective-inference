"""Ten paired MI-only parameter tests; replay saved datasets without inference.

Run with: .venv/Scripts/python.exe examples/run_mi_targeting_ten_tests.py
All simulation and target settings come from the reference run, including k=2.
AIS is not executed: it occurs after, and cannot affect, the tested outcomes.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from si_shap.power import _generate_power_dataset
from si_shap.selection import MutualInformationSelector, target_from_selected_set

BASE = ROOT / 'outputs/selection_method_validation/exact_set'
REFERENCE = BASE / 'selected_five_methods_robust_draw_floor_ais_exact_mi30'
CONFIGURATIONS = [
    {'name': f'knn_{k}', 'estimator': 'knn', 'n_neighbors': k}
    for k in (20, 25, 30)
] + [
    {'name': f'kde_{h:g}', 'estimator': 'kde', 'bandwidth': h}
    for h in (.25, .4, .6, .8, 1., 1.25, 1.5)
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'mi_parameter_tuning/targeting_ten_tests_20260920')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'reference': str(REFERENCE.relative_to(ROOT)), 'configurations': CONFIGURATIONS,
                'k_select': 2, 'inference_executed': False,
                'purpose': 'Paired signal inclusion and targeting; no p-values or power claims',
                'changes': 'MI estimator, neighbor count, or KDE bandwidth only',
                'validation': 'Reused reference seeds; exploratory, not independent confirmation'}
    (out / 'experiment_config.json').write_text(json.dumps(manifest, indent=2) + '\n')
    data = pd.read_csv(REFERENCE / 'unified_tests.csv')
    reference = data[data.scope.eq('power') & data.selection_method.eq('mutual_information')]
    assert len(reference) == 800
    selectors = {c['name']: MutualInformationSelector(random_state=42, **{k:v for k,v in c.items() if k != 'name'}) for c in CONFIGURATIONS}
    records, audits = [], []
    for (shard, beta), group in reference.groupby(['validation_shard', 'signal_strength']):
        folder = REFERENCE / 'runs/mutual_information' / shard / 'mutual_information/baseline' / ('power_feature_0_beta_' + str(beta).replace('.', 'p'))
        settings = json.loads((folder / 'settings.json').read_text())
        assert settings['k_select'] == 2 and settings['signal_features'] == [0]
        assert settings['selector_settings']['n_neighbors'] == 30
        raw = pd.read_csv(folder / 'feature_results.csv').set_index('iteration')
        seeds = np.random.SeedSequence(settings['seed']).spawn(settings['iteration_start'] + settings['n_iters'])
        audits.append({'shard': shard, 'beta': beta, 'settings': settings})
        for row in group.itertuples():
            ds, ts, _ = seeds[row.iteration - 1].spawn(3)
            X, y = _generate_power_dataset(np.random.default_rng(ds), settings['n_samples'], settings['n_features'], settings['signal_features'], beta, feature_correlation=settings['feature_correlation'])
            u = float(np.random.default_rng(int(ts.generate_state(1, dtype=np.uint32)[0])).random())
            assert np.isclose(u, row.auxiliary_u, rtol=0, atol=1e-15)
            digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
            for name, selector in selectors.items():
                result = selector.select(X, y, settings['k_select'])
                target = target_from_selected_set(result.selected_features, u)
                rank = int(np.flatnonzero(result.ranking == 0)[0] + 1)
                if name == 'knn_30':
                    assert target == row.target_feature
                    assert np.isclose(result.scores[target], raw.loc[row.iteration, 'importance_score'], rtol=1e-10, atol=1e-12)
                    assert int(np.flatnonzero(result.ranking == target)[0] + 1) == raw.loc[row.iteration, 'selection_rank']
                records.append({'configuration': name, 'shard': shard, 'iteration': row.iteration, 'beta': beta,
                                'dataset_sha256': digest, 'k_select': 2, 'auxiliary_u': u,
                                'selected_features': json.dumps(result.selected_features.tolist()),
                                'ranking': json.dumps(result.ranking.tolist()), 'scores': json.dumps(result.scores.tolist()),
                                'signal_rank': rank, 'signal_selected': rank <= 2, 'signal_targeted': target == 0,
                                'target_feature': target, 'signal_score': result.scores[0],
                                'selection_margin': result.scores[0] - np.sort(result.scores[1:])[-2]})
            assert hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest() == digest
        print(f'Completed {shard}, beta={beta:g}', flush=True)
    trials = pd.DataFrame(records)
    keys = ['shard', 'iteration', 'beta']
    assert len(trials) == 8000 and not trials.duplicated(['configuration'] + keys).any()
    assert (trials.groupby(keys).dataset_sha256.nunique() == 1).all()
    assert (trials.groupby(keys).auxiliary_u.nunique() == 1).all()
    trials.to_csv(out / 'trials.csv', index=False)
    (out / 'reference_settings.json').write_text(json.dumps(audits, indent=2) + '\n')
    summary = trials.groupby(['configuration', 'beta'], sort=False).agg(
        n=('signal_selected', 'size'), selected=('signal_selected', 'sum'), targeted=('signal_targeted', 'sum'),
        selection_rate=('signal_selected', 'mean'), targeting_rate=('signal_targeted', 'mean'),
        rank_one_rate=('signal_rank', lambda x: float((x == 1).mean())), mean_rank=('signal_rank', 'mean'),
        mean_margin=('selection_margin', 'mean')).reset_index()
    summary['randomization_averaged_targeting'] = summary.selection_rate / 2
    summary.to_csv(out / 'summary.csv', index=False)
    control = trials[trials.configuration.eq('knn_30')]
    pairs = []
    for name, group in trials.groupby('configuration', sort=False):
        paired = group.merge(control, on=keys, suffixes=('_new', '_control'), validate='one_to_one')
        for beta, part in paired.groupby('beta'):
            for metric in ('signal_selected', 'signal_targeted'):
                a, b = part[metric + '_new'], part[metric + '_control']
                gained, lost = int((a & ~b).sum()), int((b & ~a).sum())
                pairs.append({'configuration': name, 'beta': beta, 'metric': metric, 'gained': gained, 'lost': lost,
                              'difference': float(a.mean() - b.mean()),
                              'exploratory_p': binomtest(gained, gained + lost).pvalue if gained + lost else 1.})
    pd.DataFrame(pairs).to_csv(out / 'paired_vs_knn30.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    (out / 'COMPLETED').write_text('10 configurations x 4 betas x 200 paired trials; reference replay verified.\n')


if __name__ == '__main__':
    main()
