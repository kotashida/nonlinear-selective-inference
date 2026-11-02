"""Reuse the selected-four-method plot gallery for all seven refinement folders."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_matched_ais_methods as analysis
import plot_combined_ais_analysis as plotting

BASE = analysis.BASE
SOURCE = BASE / 'ais_refinement_200'
OUT = SOURCE / 'selected_seven_settings_combined_ais_refinement_200'
LABELS = {
    'a_cdf_interval': 'CDF interval (repaired)',
    'a_cdf_interval_pre_repair_20260916_224037_162': 'CDF interval (pre-repair archive)',
    'a_full_stack_optimized': 'Full stack optimized',
    'a_robust_consolidation': 'Robust consolidation',
    'a_robust_draw_floor': 'Robust draw floor',
    'a_robust_draw_floor_pilot15': 'Robust draw floor (15 pilots)',
    'a_robust_floor_multimodal_denominator': 'Robust floor + multimodal denominator',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert {p.name for p in SOURCE.iterdir() if p.is_dir() and p != OUT} == set(LABELS)
    frames, manifest, audit = [], {}, []
    for arm in LABELS:
        run = SOURCE / arm
        settings_paths = sorted(run.glob('shard_*/validation_settings.json'))
        assert len(settings_paths) == (46 if 'pre_repair' in arm else 48), arm
        settings_list = []
        for path in settings_paths:
            settings = json.loads(path.read_text())
            assert settings['methods'] == ['lime']
            assert settings['signal_strengths'] == [1.0]
            assert settings['primary_selection_event'] == 'exact_set'
            assert settings['max_final_samples'] == 51200
            assert settings['pilot_iters'] in (10, 15)
            settings_list.append(settings)
            for scope, suffix in [
                ('null_calibration', 'null_fresh/p_value_results.csv'),
                ('power', 'power_feature_0_beta_1p0/feature_results.csv'),
            ]:
                raw_path = path.parent / 'lime/baseline' / suffix
                frame = pd.read_csv(raw_path)
                frame = frame[frame.selection_event.eq('exact_set')].copy()
                frame['failed'] = frame.failed.astype(str).str.lower().eq('true')
                if scope == 'power':
                    frame['target_is_signal'] = frame.is_signal.astype(str).str.lower().eq('true')
                frame['scope'] = scope
                frame['signal_strength'] = 1.0 if scope == 'power' else np.nan
                frame['selection_method'] = arm
                frame['ais_setting'] = 'combined'
                frame['source_run'] = str(run.relative_to(BASE)).replace('\\', '/')
                frame['validation_shard'] = path.parent.name
                for field in ['max_final_samples', 'pilot_iters', 'pilot_samples']:
                    frame[field] = settings[field]
                frame['total_proposals'] = frame.aj_evaluations
                np.testing.assert_allclose(frame.total_proposals, frame.proposals + frame.pilot_evaluations)
                frames.append(frame)
                audit.append(dict(arm=arm, scope=scope, source_file=str(raw_path.relative_to(BASE)), records=len(frame)))
        manifest[str(run.relative_to(BASE)).replace('\\', '/')] = settings_list
    data = pd.concat(frames, ignore_index=True)
    keys = ['selection_method', 'scope', 'validation_shard', 'iteration']
    assert not data.duplicated(keys).any()
    counts = data.groupby(['selection_method', 'scope']).size()
    counts.rename('records').to_csv(OUT / 'trial_audit.csv')
    assert counts.loc[[a for a in LABELS if 'pre_repair' not in a]].eq(200).all()
    paired_keys = ['scope', 'validation_shard', 'iteration']
    assert data.groupby(paired_keys).selection_method.nunique().isin([6, 7]).all()
    for field in ['feature', 'auxiliary_u']:
        assert data.groupby(paired_keys)[field].nunique(dropna=False).eq(1).all(), field
    data.to_csv(OUT / 'unified_tests.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT / 'configuration_audit.csv', index=False)
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    analysis.OUT = OUT
    analysis.summarize(data)
    plotting.SOURCE = plotting.OUT = OUT
    plotting.DATA_FILENAME = 'unified_tests.csv'
    plotting.METHODS = LABELS
    plotting.COLORS = dict(zip(LABELS, ['#2674a6', '#d58122', '#258568', '#9256a0', '#bb4f70', '#6f8f2f', '#4f7f91']))
    plotting.BETAS = [1.0]
    plotting.EXPECTED_PILOT_ITERS = (10, 15)
    plotting.ALLOW_INCOMPLETE = True
    plotting.REPORT_LABEL = 'AIS refinement (seven settings)'
    plotting.REPORT_INTRO = (
        'Seven LIME refinement folders at beta=1: six complete runs have 200 null and 200 power trials each. '
        'The pre-repair archive has only 46 of 48 shards; its plots use available records (see trial_audit.csv). '
        'Exact-set conditioning; LIME uses 40 explanations. Final cap 51,200; '
        'pilot 10 or 15 x 320; batch 800; ESS thresholds 80/15. '
        'The pre-repair CDF folder is a historical archive, not an independent replicate. '
        'Only beta=1 was simulated, so power and cost curves contain one point per setting. '
        'Sampling cost includes each setting\'s actual pilot evaluations.'
    )
    plotting.FIGURES.clear()
    plotting.main()
    note = '\n\nRegenerate with `.venv/Scripts/python.exe examples/plot_ais_refinement_200.py`. '
    note += 'Figures reuse `examples/plot_combined_ais_analysis.py`, the original reference plotting script.\n'
    with (OUT / 'README.md').open('a', encoding='utf-8') as handle:
        handle.write(note)


if __name__ == '__main__':
    main()
