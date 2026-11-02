"""Assemble the selected four-method configuration with MI-30 and combined AIS."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_matched_ais_methods as analysis
import plot_combined_ais_analysis as plotting

BASE=analysis.BASE
OUT=BASE/'selected_four_methods_combined_ais_mi30'
OPTIONS=['SHAP','LIME 40','Spline screening','MI k=30']


def main():
    OUT.mkdir(exist_ok=True)
    source=BASE/'mi_parameter_tuning/analysis'
    data=pd.read_csv(source/'unified_tests.csv')
    data=data[data.option.isin(OPTIONS)].copy()
    assert len(data)==4000 and set(data.option)==set(OPTIONS)
    keys=['selection_method','scope','validation_shard','iteration','selection_event','signal_strength']
    assert not data.duplicated(keys).any()
    assert data.ais_setting.eq('combined').all() and data.max_final_samples.eq(51200).all()
    assert data.pilot_iters.eq(10).all() and data.pilot_samples.eq(320).all()
    assert data.loc[data.selection_method.eq('mutual_information'),'mi_neighbors'].eq(30).all()
    manifests={k.replace('\\','/'):v for k,v in json.loads((source/'source_manifest.json').read_text()).items()}
    used={rel:manifests[rel] for rel in data.source_run.unique()}
    for s in used.values():
        assert s['shard_root_seeds']==list(range(20260908,20260940))
        assert s['n_null_iters']==s['n_power_iters']==200
        assert s['inference_method']=='ais' and s['spline_inference_method']=='inherit'
    audits=[]
    sources=data.groupby('selection_method').source_run.first().to_dict()
    # Verify low-level simulation settings across selectors, and explicit selector controls.
    for shard in range(32):
        for folder in ['null_fresh',*[f'power_feature_0_beta_{str(b).replace(".","p")}' for b in [.3,.5,.75,1.]]]:
            reference=None
            for method,rel in sources.items():
                path=BASE/rel/f'shard_{shard}/{method}/baseline/{folder}/settings.json'
                s=json.loads(path.read_text())
                if method=='lime':assert s['selector_settings']['num_explanations']==40
                if method=='mutual_information':assert s['selector_settings']['n_neighbors']==30
                shared={k:v for k,v in s.items() if k not in {'selection_method','selector_settings','rf_params'}}
                if reference is None:reference=shared
                else:assert reference==shared,(method,shard,folder)
                audits.append(dict(method=method,shard=shard,condition=folder,source_settings=str(path.relative_to(BASE)),shared_settings_match=True))
    pd.DataFrame(audits).to_csv(OUT/'configuration_audit.csv',index=False)
    trial_audit=[]
    for scope in ['null_calibration','power']:
        extra=['signal_strength'] if scope=='power' else []
        ref=data[data.selection_method.eq('shap')&data.scope.eq(scope)]
        for method in sources:
            other=data[data.selection_method.eq(method)&data.scope.eq(scope)]
            pair=ref.merge(other,on=['validation_shard','iteration','selection_event',*extra],suffixes=('_ref','_other'),validate='one_to_one')
            assert len(pair)==(800 if scope=='power' else 200)
            np.testing.assert_array_equal(pair.target_seed_ref,pair.target_seed_other)
            np.testing.assert_allclose(pair.auxiliary_u_ref,pair.auxiliary_u_other,rtol=0,atol=1e-15)
            trial_audit.append(dict(method=method,scope=scope,matched_trials=len(pair)))
    pd.DataFrame(trial_audit).to_csv(OUT/'trial_audit.csv',index=False)
    data.to_csv(OUT/'unified_tests.csv',index=False)
    (OUT/'source_manifest.json').write_text(json.dumps(used,indent=2))
    (OUT/'selected_configuration.json').write_text(json.dumps({
        'methods':{'shap':'original SHAP selector','lime':{'num_explanations':40,'num_samples':64,'ridge_alpha':1},
                   'spline_screening':'original spline selector with AIS inference','mutual_information':{'n_neighbors':30,'random_state':42}},
        'ais':{'pilot_iters':10,'pilot_samples':320,'max_final_samples':51200,'final_batch_size':800,'min_denominator_ess':80,'min_tail_ess':15,'stop_when_ess_met':True},
        'base_seed':20260908,'design_seed':314159,'shards':32,'iterations_per_condition':200,
        'selection_event':'exact_set','signal_strengths':[.3,.5,.75,1.],
        'provenance':'Selected existing runs; no new simulation or independent validation.'},indent=2))
    previous=analysis.OUT;analysis.OUT=OUT
    p,c=analysis.summarize(data);analysis.OUT=previous
    plotting.SOURCE=OUT;plotting.OUT=OUT;plotting.DATA_FILENAME='unified_tests.csv'
    plotting.REPORT_LABEL='Combined AIS'
    plotting.METHODS['mutual_information']='Mutual information (30 neighbors)'
    plotting.FIGURES.clear();plotting.main()
    # Explicit acceptance note: MI-30 was selected with a known beta=0.5 shortfall.
    note='\n\n## Selected configuration\n\nThis folder contains SHAP, LIME with 40 explanations, spline screening, and MI with 30 neighbors. All use combined AIS; spline screening does not use exact-spline inference. MI-30 was accepted with 77.0% conditional power at beta=0.50, below the original 80% target. At beta=1.00 the four methods reach 97.1%, 92.3%, 96.2% and 99.0%, respectively. No new simulation was run. Do not append this selected dataset to its source datasets.\n\nRegenerate with `.venv/Scripts/python.exe examples/build_selected_four_method_dataset.py`. Full source settings are in source_manifest.json, and configuration/trial audits are included.\n'
    with (OUT/'README.md').open('a',encoding='utf-8') as f:f.write(note)
    page=(OUT/'index.html').read_text(encoding='utf-8')
    page=page.replace('<h1>Combined AIS analysis</h1>','<h1>Selected four methods: combined AIS, MI-30</h1><p>MI-30 is retained with 77.0% conditional power at beta=0.50, below the original 80% target. This dataset reuses existing experiments.</p>')
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(p[p.beta.eq(.5)][['method','conditional_power','signal_targets','detections']].to_string(index=False))


if __name__=='__main__':main()
