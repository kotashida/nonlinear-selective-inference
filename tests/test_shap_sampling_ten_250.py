import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import run_shap_sampling_ten_250 as experiment
from si_shap.inference import _robust_pilot_islands
from si_shap import compare_selection_event_null_calibration, compare_selection_event_power


def test_manifest_allocation_and_mcmc_cost():
    m = experiment.read_manifest()
    counts = [experiment.allocation(m, s) for s in range(32)]
    assert sum(counts) == 250
    assert counts.count(8) == 26 and counts.count(7) == 6
    assert [(a['max_final_samples']+1)*a['mcmc_steps'] for a in m['arms'][7:]] == [10240,51200,51200]
    assert experiment.expected_settings(m,m['arms'][0],0,2)['seed'] == experiment.expected_settings(m,m['arms'][9],0,2)['seed']


def test_tail_stable_order_and_bounds_survive_cdf_saturation():
    z=np.array([12.,10.,13.,11.])
    selected=np.array([True,False,False,True])
    weights=np.zeros(4)
    islands, _ = _robust_pilot_islands(z,selected,weights,3,tail_stable=True)
    assert islands[0]['lower_z'] == 10.5
    assert islands[0]['upper_z'] == 12.5
    permutation=np.array([3,2,0,1])
    again, _ = _robust_pilot_islands(z[permutation],selected[permutation],weights,3,tail_stable=True)
    assert islands == again


@pytest.mark.parametrize('condition',['null','power'])
def test_sampler_salt_preserves_data_and_target(condition):
    kwargs=dict(n_iters=2,n_samples=30,n_features=5,k_select=2,
                selection_method='spline_screening',selection_events=('exact_set',),seed=817,
                inference_method='mcmc_rank',max_final_samples=9,mcmc_steps=2)
    fn=compare_selection_event_null_calibration if condition=='null' else compare_selection_event_power
    if condition=='null': kwargs.update(alpha_levels=(.05,),design_seed=314159)
    else: kwargs.update(alpha=.05,signal_strength=.75)
    first=fn(**kwargs,sampler_seed_salt=1)
    second=fn(**kwargs,sampler_seed_salt=2)
    name='p_value_results' if condition=='null' else 'target_results'
    for col in ('target_feature','auxiliary_u','target_seed','observed_selected_features'):
        assert first[name][col].equals(second[name][col])
    assert (first[name].ais_seed != second[name].ais_seed).all()
    if condition=='null': np.testing.assert_array_equal(first['fixed_design'],second['fixed_design'])
    else: np.testing.assert_array_equal(first['feature_results'].t_obs,second['feature_results'].t_obs)


def test_real_shap_bundles_resume_and_detect_corruption(tmp_path):
    m=copy.deepcopy(experiment.read_manifest())
    m['output_root']=str(tmp_path)
    m['iterations_per_condition']=32
    m['design'].update(n_samples=30,n_features=5)
    m['rf_params'].update(n_estimators=3,max_depth=2)
    m['shared'].update(pilot_samples=8,final_batch_size=10,min_denominator_ess=2.,min_tail_ess=1.)
    for index in (6,9):
        a=copy.deepcopy(m['arms'][index])
        a.update(pilot_iters=2,max_final_samples=40 if index==6 else 9,mcmc_steps=2)
        experiment.worker(m,a,0)
        receipts=[]
        for c in range(3):
            frame=experiment.verify_bundle(m,a,0,c)
            assert len(frame)==1
            receipts.append((experiment.folder(m,a,0,c)/'bundle_receipt.json').read_bytes())
        experiment.worker(m,a,0)
        assert receipts == [(experiment.folder(m,a,0,c)/'bundle_receipt.json').read_bytes() for c in range(3)]
    csv=experiment.folder(m,a,0,0)/'p_value_results.csv'
    csv.write_text(csv.read_text()+'\n')
    with pytest.raises(ValueError,match='Corrupt bundle'):
        experiment.verify_bundle(m,a,0,0)


def test_pooled_verifier_and_failure_accounting(tmp_path, monkeypatch):
    m=copy.deepcopy(experiment.read_manifest())
    m['output_root']=str(tmp_path)
    def bundle(m,a,s,c):
        n=experiment.allocation(m,s)
        f=pd.DataFrame(dict(iteration=range(1,n+1),target_feature=0,auxiliary_u=.2,
            target_seed=100+s,t_obs=2.,observed_selected_features='(0, 1)',
            ais_seed=a['sampler_seed_salt']*10000+s, p_value=.04,failed=False,
            target_is_signal=c!=0,proposals=800,denominator_ess=100.,status='ok',
            resolution_status='resolved',arm=a['name'],shard=s,
            condition='null' if c==0 else f'beta_{m["betas"][c-1]}'))
        if a['name']=='test_02' and s==0 and c==1:
            f.loc[0,['p_value','failed','status']]=[np.nan,True,'insufficient_ess']
        path=experiment.folder(m,a,s,c)
        path.mkdir(parents=True,exist_ok=True)
        (path/'bundle_receipt.json').write_text(json.dumps({'elapsed_seconds':1.}))
        return f
    monkeypatch.setattr(experiment,'verify_bundle',bundle)
    experiment.verify(m)
    assert len(pd.read_csv(tmp_path/'unified_tests.csv'))==7500
    summary=pd.read_csv(tmp_path/'sampling_summary.csv')
    row=summary[(summary.arm=='test_02') & (summary.condition=='beta_0.75')].iloc[0]
    assert row.n_failed==1 and pd.isna(row.complete_rate)
    assert row.resolved_only_rate==1.
    assert row.missing_outcome_rate_lower==249/250
    assert row.missing_outcome_rate_upper==1.
    # Editing the local checkout must not prevent a read-only downloaded audit,
    # while the same edit must continue to block execution/resume.
    original_hashes=experiment.hashes()
    altered_hashes={**original_hashes, 'examples/new_local_analysis.py': 'a'*64}
    existing_files={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    monkeypatch.setattr(experiment,'hashes',lambda:altered_hashes)
    experiment.verify_downloaded(m)
    assert existing_files=={name:(tmp_path/name).read_bytes() for name in existing_files}
    report=json.loads((tmp_path/'local_verification_report.json').read_text())
    assert report['status']=='passed'
    assert report['additional_local_files']==['examples/new_local_analysis.py']
    with pytest.raises(ValueError,match='source_hashes.json drift'):
        experiment.lock(m)
    # Downloaded verification still rejects corrupted pooled data.
    data=pd.read_csv(tmp_path/'unified_tests.csv')
    data.loc[0,'p_value']=.9
    data.to_csv(tmp_path/'unified_tests.csv',index=False)
    with pytest.raises(AssertionError):
        experiment.verify_downloaded(m)
    assert json.loads((tmp_path/'local_verification_report.json').read_text())['status']=='failed'
