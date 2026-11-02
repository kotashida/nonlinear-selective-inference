import copy
import io
import json
import tarfile
import numpy as np
import pandas as pd
import pytest
from examples import run_presentation_four_selectors_mcmc20_250_20261008 as e


def small(tmp_path):
    m=copy.deepcopy(e.read_manifest());m['output_root']=str(tmp_path)
    m.update(iterations=1,betas=[0,.5],workers=1,shards=1)
    m['design'].update(n_samples=30,n_features=5)
    m['rf_params'].update(n_estimators=3,max_depth=2)
    m['mcmc'].update(replicas=7,steps=2)
    return m


def test_production_settings_and_shared_inputs():
    m=e.read_manifest()
    assert len(m['methods'])*len(m['betas'])*m['iterations']==5000
    assert m['betas']==[0,.25,.5,.75,1]
    assert m['mcmc']['replicas']==511 and m['mcmc']['steps']==20
    assert 5000*512*20==51200000
    assert m['lime']['num_explanations']==40 and m['lime']['num_samples']==64
    assert m['mutual_information']['estimator']=='kde' and m['mutual_information']['bandwidth']==2.5
    assert m['methods']==['shap','lime','spline_screening','mutual_information']
    assert 'linear_shapley' not in m
    for c in range(5):
        a=e.generated(m,c,1);b=e.generated(m,c,1)
        for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
        assert a[3]==b[3]
    assert not np.array_equal(e.generated(m,0,1)[0],e.generated(m,0,2)[0])
    trials=[i for s in range(32) for i in range(s+1,251,32)]
    assert sorted(trials)==list(range(1,251))


def test_all_selectors_receipts_resume_portable_audit_and_corruption(tmp_path):
    m=small(tmp_path);e.lock(m)
    for c in range(2):
        e.save_input(m,c,1)
        for method in m['methods']:e.execute(m,method,c,1)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    for c in range(2):
        e.save_input(m,c,1)
        for method in m['methods']:e.execute(m,method,c,1)
    assert all(p.read_bytes()==v for p,v in before.items())
    e.verify(m);e.verify_downloaded(m)
    frame=pd.read_csv(tmp_path/'unified_tests.csv')
    assert len(frame)==8 and frame.proposals.eq(16).all()
    assert frame.denominator_ess.isna().all() and frame.tail_ess.isna().all()
    assert (frame.groupby(['condition','trial']).input_sha256.nunique()==1).all()
    assert len(pd.read_csv(tmp_path/'power_summary.csv'))==4
    original=(tmp_path/'source_hashes.json').read_bytes()
    frame.loc[0,'p_value']=.123;frame.to_csv(tmp_path/'unified_tests.csv',index=False)
    with pytest.raises(AssertionError):e.verify_downloaded(m)
    assert (tmp_path/'source_hashes.json').read_bytes()==original
    e.verify(m)
    path=e.result_path(m,'shap',0,1)/'test_basis.npz'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='Corrupt result'):e.execute(m,'shap',0,1)
    with pytest.raises(ValueError,match='local_verification_errors.csv'):e.verify_downloaded(m)
    assert 'Corrupt result' in (tmp_path/'local_verification_errors.csv').read_text()



def test_input_corruption_and_locked_settings(tmp_path):
    m=small(tmp_path);e.lock(m);e.save_input(m,0,1)
    changed=copy.deepcopy(m);changed['mcmc']['steps']=3
    with pytest.raises(ValueError,match='requested_manifest.json drift'):e.lock(changed)
    p=e.input_path(m,0,1)/'input.npz';p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='Corrupt input'):e.save_input(m,0,1)


def test_archive_paths_and_links(tmp_path):
    m=e.read_manifest()
    for name,link in [('other/file',False),(m['experiment']+'/../bad',False),(m['experiment']+'/C:bad',False),(m['experiment']+'/link',True)]:
        p=tmp_path/'bad.tar'
        with tarfile.open(p,'w') as archive:
            entry=tarfile.TarInfo(name)
            if link:entry.type=tarfile.SYMTYPE;entry.linkname='../bad'
            archive.addfile(entry)
        with pytest.raises(ValueError,match='Unsafe archive'):e.validate_archive(p,m)
