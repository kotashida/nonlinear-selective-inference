"""Check sampler constructions, pairing, receipts and portable verification."""
import copy
import itertools
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples import run_shap_sampling_followup as exp
from examples import shap_sampling_followup_samplers as samplers
from si_shap.inference import (_mcmc_transition,_run_mcmc_rank,_prepare_refinement_pilot,
                               _run_ais_refinement)
from si_shap.selection import make_selector


def test_manifest_budget_matching_and_random_design():
    m=exp.read_manifest()
    assert sum(c['iterations'] for c in m['conditions'])*len(m['arms'])==7000
    assert [(a['replicas']+(a['construction']=='parallel'))*a['steps'] for a in m['arms'][4:]]==[10240,51200,51100]
    for c in m['conditions']:
        assert sum(exp.allocation(m,c,s) for s in range(32))==c['iterations']
    assert 1000*(3200+12800+10240+51200+51100)==m['budget_reference']['fresh_fixed_evaluation_subtotal']
    assert 1000*(3200+12800+3*51200+10240+51200+51100)==m['budget_reference']['fresh_max_candidate_evaluations']
    X,y,_,identity=exp.fresh_input(m,0,0,1)
    X2,y2,_,identity2=exp.fresh_input(m,0,0,1)
    np.testing.assert_array_equal(X,X2);np.testing.assert_array_equal(y,y2)
    assert identity==identity2
    assert not np.array_equal(X,exp.fresh_input(m,0,0,2)[0])
    assert m['arms'][0]['sampler_salt']==m['arms'][3]['sampler_salt']
    assert len({a['sampler_salt'] for a in m['arms']})==6


@pytest.mark.parametrize('lower',[0.,4.])
def test_instrumented_transition_and_parallel_rank_match_production(lower):
    oracle=lambda z:z>=lower
    kwargs=dict(rank=3,is_selected=oracle,rng=np.random.default_rng(419),n_steps=15,
                step_scales=(.25,.5,1.,2.),global_jump_probability=.1)
    value,accepted=_mcmc_transition(5.,**kwargs)
    again,counters,path=samplers.transition(5.,3,oracle,np.random.default_rng(419),15,
                                           (.25,.5,1.,2.),.1,trace=True)
    assert again==value and accepted==counters[1]+counters[3] and len(path)==15
    original,_=_run_mcmc_rank(5.,3,oracle,np.random.default_rng(617),n_replicates=9,n_steps=3)
    arm=dict(construction='parallel',replicas=9,steps=3)
    settings=dict(step_scales=[.25,.5,1.,2.],global_jump_probability=.1)
    row,trace=samplers.run_rank(5.,3,oracle,np.random.default_rng(617),arm,settings)
    assert row['p_value']==original and row['proposals']==30
    assert row['denominator_ess'] is None and len(trace['endpoints'])==9


def test_serial_rank_exact_superuniformity_on_a_reversible_finite_chain(monkeypatch):
    # Enumerate the entire experiment, rather than relying on a noisy empirical
    # rejection assertion. pi*K satisfies detailed balance; both directions must
    # use the same K, and the observation position must be uniform.
    pi=np.array([.25,.75]);K=np.array([[.7,.3],[.1,.9]])
    masses={.25:0.,.5:0.,.75:0.}
    arm=dict(construction='serial',replicas=3,steps=1)
    settings=dict(step_scales=[1.],global_jump_probability=0.)
    for position in range(4):
        for chain in itertools.product(range(2),repeat=4):
            transitions=[chain[i] for i in range(position-1,-1,-1)]
            transitions.extend(chain[i] for i in range(position+1,4))
            forced=iter(transitions)
            monkeypatch.setattr(samplers,'transition',lambda initial,**kw:
                                (float(next(forced)),np.zeros(4,dtype=int),[]))
            rng=Mock();rng.integers.return_value=position
            row,_=samplers.run_rank(float(chain[position]),3,lambda z:True,rng,arm,settings)
            assert row['mcmc_observation_position']==position
            assert row['p_value']==sum(x>=chain[position] for x in chain)/4
            rng.integers.assert_called_once_with(4)
            weight=pi[chain[position]]/4
            for i in range(position-1,-1,-1):weight*=K[chain[i+1],chain[i]]
            for i in range(position+1,4):weight*=K[chain[i-1],chain[i]]
            for alpha in masses:
                if row['p_value']<=alpha:masses[alpha]+=weight
    assert all(mass<=alpha+1e-12 for alpha,mass in masses.items())


def test_preserved_tail_regions_and_stable_interval_mass():
    pilot=dict(z_values=np.array([8.,8.1,8.2,8.3,8.4]),
               selected=np.array([True,False,False,False,True]),
               log_weights=np.array([0.,np.nan,np.nan,np.nan,0.]))
    islands,before=samplers.preserved_islands(pilot,3)
    assert before==2 and len(islands)==2
    assert islands[0]['upper_z']<islands[1]['lower_z']
    from scipy import stats
    expected=stats.chi.sf(10.,3)-stats.chi.sf(10.1,3)
    assert expected>0
    assert stats.chi.cdf(10.1,3)-stats.chi.cdf(10.,3)==0
    assert np.exp(samplers.log_interval_mass(10.,10.1,3))==pytest.approx(expected,rel=1e-12)


def test_reference_ais_matches_production_and_continuation_prefix():
    rng=np.random.default_rng(112)
    oracle=lambda z:1.2<=z<=5.
    pilot=_prepare_refinement_pilot(2.,3,oracle,rng,2,24,pilot15=False)
    state=copy.deepcopy(rng.bit_generator.state)
    settings=dict(allocation_cap=160,draw_floor=128,batch_size=16,
                  min_denominator_ess=5.,min_tail_ess=2.)
    original,diag=_run_ais_refinement(2.,3,oracle,np.random.default_rng(112),
        pilot_iters=2,pilot_samples=24,final_batch_size=16,max_final_samples=160,
        min_denominator_ess=5.,min_tail_ess=2.,stop_when_ess_met=True,
        ais_strategy='a_robust_draw_floor')
    rng.bit_generator.state=copy.deepcopy(state)
    arm=dict(variant='reference',early_stop=True,final_cap=160)
    row,trace=samplers.run_ais(2.,3,oracle,rng,pilot,arm,settings)
    assert row['p_value']==original and row['proposals']==diag['proposals']
    rng.bit_generator.state=copy.deepcopy(state)
    fixed,continued=samplers.run_ais(2.,3,oracle,rng,pilot,
                                   dict(variant='reference',early_stop=False,final_cap=80),settings)
    n=min(len(trace['checkpoints']),len(continued['checkpoints']))
    np.testing.assert_array_equal(trace['checkpoints'][:n],continued['checkpoints'][:n])
    assert fixed['proposals']==80


def test_region_map_budget_and_uncertified_intervals():
    oracle=lambda z:(1.<=z<=2.) or (4.<=z<=4.2)
    row,trace=samplers.map_regions(oracle,4.1,3,budget=512)
    assert row['grid_evaluations']<=512 and not row['regions_certified']
    assert row['interpretation']=='finite_grid_diagnostic_only'
    assert len(row['intervals'])==2
    assert len(trace['grid_z'])==row['grid_evaluations']


def test_real_shap_resume_corruption_and_downloaded_aggregate_audit(tmp_path,monkeypatch):
    m=copy.deepcopy(exp.read_manifest())
    m['output_root']=str(tmp_path)
    m['workers']=m['shards']=1
    m['conditions']=[dict(name='null',iterations=1,beta=0.)]
    m['design'].update(n_samples=30,n_features=5)
    m['rf_params'].update(n_estimators=3,max_depth=2)
    m['ais'].update(pilot_iters=1,pilot_samples=12,batch_size=16,allocation_cap=128,
                    draw_floor=1,min_denominator_ess=2.,min_tail_ess=1.)
    for a in m['arms']:
        if a['method']=='ais':a['final_cap']=64 if a['early_stop'] else 32
        else:a.update(replicas=9,steps=2)
    data=exp.fresh_input(m,0,0,1)
    probe=exp.Ray(data[0],data[1],.2,m)
    case=dict(case_id='smoke_case',file='unused.npz',auxiliary_u=.2,beta=0.,
              input_hash=data[3]['input_hash'],target_feature=probe.target,
              observed_selected_features=probe.selected,t_obs=probe.t_obs)
    fixtures=dict(cases=[case],replay_cases=['smoke_case'])
    m['diagnostics'].update(ais_final_cap=32,map_budget=128)
    monkeypatch.setattr(exp,'read_fixtures',lambda m:fixtures)
    monkeypatch.setattr(exp,'fixture_arrays',lambda m,c:data[:3])
    exp.lock(m)
    exp.execute_trial(m,*data)
    for repeat in range(2):
        exp.execute_trial(m,*exp.diagnostic_input(m,case,repeat),expected_case=case)
    exp.execute_map(m,case)
    files={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    exp.execute_trial(m,*data)
    assert all(p.read_bytes()==value for p,value in files.items())
    exp.verify(m)
    assert len(pd.read_csv(tmp_path/'unified_tests.csv'))==19
    manifest_before=(tmp_path/'source_hashes.json').read_bytes()
    current=exp.hashes(m)
    monkeypatch.setattr(exp,'hashes',lambda m:{**current,'unrelated.py':'a'*64})
    exp.verify_downloaded(m)
    assert (tmp_path/'source_hashes.json').read_bytes()==manifest_before
    original_input=exp.fresh_input
    def regenerated_input(*args):
        X,y,mean,identity=original_input(*args)
        return X,y,mean,dict(identity,input_hash='b'*64)
    with monkeypatch.context() as portable:
        portable.setattr(exp,'fresh_input',regenerated_input)
        exp.verify_downloaded(m)
    regeneration=json.loads((tmp_path/'local_input_regeneration_report.json').read_text())
    assert regeneration['byte_identical_fresh_inputs']==0
    assert len(regeneration['mismatches'])==1
    assert (tmp_path/'source_hashes.json').read_bytes()==manifest_before
    original_errors=(tmp_path/'verification_errors.csv').read_bytes()
    def wrong_seed(*args):
        X,y,mean,identity=original_input(*args)
        return X,y,mean,dict(identity,pilot_seed=identity['pilot_seed']+1)
    with monkeypatch.context() as portable:
        portable.setattr(exp,'fresh_input',wrong_seed)
        with pytest.raises(ValueError,match='local_verification_errors.csv'):
            exp.verify_downloaded(m)
    assert 'Settings drift' in (tmp_path/'local_verification_errors.csv').read_text()
    assert (tmp_path/'verification_errors.csv').read_bytes()==original_errors
    with pytest.raises(ValueError,match='source_hashes.json drift'):exp.lock(m)
    frame=pd.read_csv(tmp_path/'unified_tests.csv')
    frame.loc[0,'p_value']=.999
    frame.to_csv(tmp_path/'unified_tests.csv',index=False)
    with pytest.raises(AssertionError):exp.verify_downloaded(m)
    path=exp.trial_path(m,data[3])/m['arms'][0]['name']/'trace.npz'
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(ValueError,match='Corrupt bundle'):exp.execute_trial(m,*data)


def test_archive_rejects_links_and_windows_paths(tmp_path):
    m=exp.read_manifest()
    for name,link in [('other/file',False),(m['experiment']+'/../bad',False),
                      (m['experiment']+'/C:bad',False),(m['experiment']+'/link',True)]:
        archive=tmp_path/'bad.tar'
        with tarfile.open(archive,'w') as handle:
            member=tarfile.TarInfo(name)
            if link:member.type=tarfile.SYMTYPE;member.linkname='../bad'
            handle.addfile(member)
        with pytest.raises(ValueError,match='Unsafe archive'):exp.validate_archive(archive,m)


def test_all_packaged_inputs_reproduce_observed_selection_and_statistic():
    m=exp.read_manifest()
    for case in exp.read_fixtures(m)['cases']:
        X,y,_=exp.fixture_arrays(m,case)
        ray=exp.Ray(X,y,case['auxiliary_u'],m)
        assert ray.target==case['target_feature']
        assert ray.selected==case['observed_selected_features']
        assert ray.t_obs==pytest.approx(case['t_obs'],rel=1e-12,abs=1e-12)
