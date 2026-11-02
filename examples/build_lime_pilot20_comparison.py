"""Audit configurations before combining existing results; never run simulations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_matched_ais_methods as analysis
from analyze_ais_method_comparison import _read_raw
import plot_lime_pilot20_comparison as plotting

BASE = analysis.BASE
OLD = BASE / 'selected_four_methods_combined_ais_mi30'
NEW = BASE / 'lime_ais_pilot_strategy_200_20260913/pilot_rounds20'
OUT = BASE / 'selected_five_configurations_ais_mi30_lime_pilot10_vs20'


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(OLD / 'source_manifest.json')
    new_settings = read(NEW / 'pooled_summary/validation_settings.json')
    old_lime = 'lime_ais_tuning/five_settings_200_trials/ais_combined'
    def differences(a, b):
        return {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert differences(manifest[old_lime], new_settings) == {'pilot_iters': [10, 20]}
    audit = pd.read_csv(OLD / 'configuration_audit.csv')
    records = []
    for row in audit.itertuples():
        old = read(BASE / row.source_settings)
        new_path = NEW / f'shard_{row.shard}/lime/baseline/{row.condition}/settings.json'
        new = read(new_path)
        excluded = set() if row.method == 'lime' else {'selection_method', 'selector_settings', 'rf_params'}
        diff = differences({k:v for k,v in old.items() if k not in excluded},
                           {k:v for k,v in new.items() if k not in excluded})
        assert diff == {'pilot_iters': [10, 20]}, (row.method, row.shard, row.condition, diff)
        records.append(dict(method=row.method, shard=row.shard, condition=row.condition,
                            source_settings=row.source_settings, new_settings=str(new_path.relative_to(BASE)),
                            differences=json.dumps(diff), match_except_pilot_iters=True))
    for shard in range(32):
        assert differences(read(BASE / old_lime / f'shard_{shard}/validation_settings.json'),
                           read(NEW / f'shard_{shard}/validation_settings.json')) == {'pilot_iters': [10, 20]}
    # The configuration gate above completes before reading or combining test records.
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(OUT / 'configuration_audit.csv', index=False)
    report = '\n\n'.join([
        '# Configuration audit: PASS',
        'All 160 paired LIME condition manifests, all 32 shard manifests and the pooled run manifest differ only in pilot_iters: 10 versus 20. All 640 cross-method condition comparisons agree on shared settings after excluding the intentionally different selector identities, selector settings and rf_params. Runtime metadata also matches. No other LIME parameter differs.',
        'Data-generating process in the experiment implementation: independent standard Gaussian features; null response is Gaussian noise with SD 1; power response is beta times the centered, empirical-unit-SD transform x + 0.5*(x^2-1) of feature 0, plus Gaussian noise with SD 1. Matched settings: baseline design; n_samples=100; n_features=20; feature_correlation=0; sigma=1; k_select=2; signal feature 0; signal strengths 0.3, 0.5, 0.75, 1.0; 200 null trials and 200 power trials per strength; 32 shards; identical seeds and iteration allocation. Null fixed design and fresh auxiliary randomization match. Target is uniform_from_selected; selection event exact_set; multiplicity none; known simulation sigma; power alpha=0.05; calibration alpha=0.01, 0.05, 0.10.',
        'AIS: pilot samples=320; final cap=51200; final batch=800; ESS thresholds=80/15; early stopping enabled; MCMC steps=20, jump probability=0.1 and scales=[0.25,0.5,1,2] match. LIME: 40 explanations, 64 samples, ridge alpha=1, default kernel width 0.75*sqrt(p), seed=42; all neighborhood, aggregation, rounding and random-forest parameters match (50 trees, max depth 5, n_jobs=1). MI remains at 30 neighbors; the new LIME launcher records MI_NEIGHBORS=12, which is unused by LIME and also equals the old LIME launcher setting.',
        'Evidence is the saved run/condition settings and matched observed trial statistics. Git commit and dirty-state metadata are null in both runs, so a historical source-code identity cannot be independently established from these artifacts.',
        'Only display changes: fifth configuration, explicit LIME pilot labels, reordered power panels, three rows for per-method panels, and extended tradeoff x-axis when necessary to show the extra pilot cost. Original rates, failure handling, Wilson intervals, scales, colors for existing methods, PNG/PDF output and gallery structure are retained.'
    ])
    (OUT / 'configuration_audit.md').write_text(report, encoding='utf-8')
    print('Configuration gate PASS', flush=True)
    data = pd.read_csv(OLD / 'unified_tests.csv')
    n, t, f = _read_raw(NEW, 'ais', new_settings, 32)
    keys = ['selection_method','validation_shard','iteration','selection_event','signal_strength']
    t = t.merge(f[keys + ['feature','t_obs','proposals','denominator_ess','tail_ess']], on=keys, validate='one_to_one')
    frames = []
    paired = []
    for frame, scope in [(n,'null_calibration'), (t,'power')]:
        ref = data[data.selection_method.eq('lime') & data.scope.eq(scope)]
        join = ['validation_shard','iteration','selection_event'] + (['signal_strength'] if scope == 'power' else [])
        pair = ref.merge(frame, on=join, suffixes=('_old','_new'), validate='one_to_one')
        assert len(pair) == len(frame) == (200 if scope == 'null_calibration' else 800)
        for key in ['target_feature','target_seed','auxiliary_u','t_obs'] + (['target_is_signal'] if scope == 'power' else ['observed_selected_features']):
            if key in {'auxiliary_u', 't_obs'}:
                np.testing.assert_allclose(pair[key+'_old'], pair[key+'_new'], rtol=0, atol=1e-14, err_msg=key)
            else:
                np.testing.assert_array_equal(pair[key+'_old'], pair[key+'_new'], err_msg=key)
        paired.append(dict(scope=scope, matched_trials=len(pair), targets_seeds_auxiliary_statistics_match=True))
        frame = frame.copy()
        frame['scope'], frame['ais_setting'], frame['source_run'] = scope, 'combined', str(NEW.relative_to(BASE)).replace('\\','/')
        frame['selection_method'] = 'lime_pilot20'
        frame['option'] = frame['analysis_label'] = 'LIME (AIS_PILOT_ITERS = 20)'
        frame['max_final_samples'], frame['pilot_iters'], frame['pilot_samples'] = 51200, 20, 320
        frame['total_proposals'] = frame.proposals + 6400
        frames.append(frame)
    pd.DataFrame(paired).to_csv(OUT / 'trial_audit.csv', index=False)
    data.loc[data.selection_method.eq('lime'), ['option','analysis_label']] = 'LIME (AIS_PILOT_ITERS = 10)'
    data = pd.concat([data, *frames], ignore_index=True)
    assert len(data) == 5000 and not data.duplicated(['scope', *keys]).any()
    data.to_csv(OUT / 'unified_tests.csv', index=False)
    manifest[str(NEW.relative_to(BASE)).replace('\\','/')] = new_settings
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    config = read(OLD / 'selected_configuration.json')
    config['methods']['lime_pilot20'] = dict(config['methods']['lime'], pilot_iters=20)
    config['methods']['lime']['pilot_iters'] = 10
    config['ais']['pilot_iters_by_configuration'] = {'default':10, 'lime_pilot20':20}
    (OUT / 'selected_configuration.json').write_text(json.dumps(config, indent=2))
    analysis.OUT = OUT
    p, c = analysis.summarize(data)
    for name, table in [('power_summary.csv',p), ('calibration_summary.csv',c)]:
        old = pd.read_csv(OLD / name)
        actual = table[table.method.ne('lime_pilot20')].reset_index(drop=True)
        pd.testing.assert_frame_equal(old, actual, check_exact=False, rtol=1e-13, atol=1e-15)
    plotting.SOURCE = plotting.OUT = OUT
    plotting.DATA_FILENAME = 'unified_tests.csv'
    plotting.METHODS = {'shap':'Tree-SHAP', 'lime':'LIME (AIS_PILOT_ITERS = 10)',
                        'spline_screening':'Spline screening', 'mutual_information':'Mutual information (30 neighbors)',
                        'lime_pilot20':'LIME (AIS_PILOT_ITERS = 20)'}
    plotting.COLORS['lime_pilot20'] = '#bc4545'
    plotting.FIGURES.clear()
    plotting.main()
    with (OUT / 'README.md').open('a', encoding='utf-8') as handle:
        handle.write('\n\n[Configuration audit](configuration_audit.md): PASS. All 1,000 paired LIME trials match targets, target seeds, auxiliary draws and observed statistics (floating-point tolerance 1e-14 for CSV round trips). Regenerate with `.venv/Scripts/python.exe examples/build_lime_pilot20_comparison.py`.\n')


if __name__ == '__main__':
    main()
