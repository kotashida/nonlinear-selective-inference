"""Signal equivalence, analytic numerics, matching and lifecycle regressions."""
from pathlib import Path
import copy
import io
import sys
import tarfile

import numpy as np
import pytest
from scipy import integrate, stats

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import run_rbf_shapley_analytic_power as study


def smoke():
    return study.read_config(ROOT/'benchmarks'/f'{study.SMOKE}.json')


def test_signal_matches_presentation_function_and_is_shared_across_strengths(monkeypatch):
    from si_shap.power import _nonlinear_effect
    m=smoke();saved=study.make_input(m,0)
    x=np.asarray(saved['X'])[:,0]
    np.testing.assert_array_equal(study.standardized_effect(x),_nonlinear_effect(x))
    f=study.standardized_effect(x)
    assert f.mean()==pytest.approx(0.,abs=1e-15)
    assert np.sqrt(np.mean(f*f))==pytest.approx(1.)
    # No MCMC path is allowed, even though the numerical core also exposes it.
    def forbidden(*args,**kwargs): raise AssertionError('MCMC must never run')
    monkeypatch.setattr(study.core,'mcmc_rank_test',forbidden)
    rows=study.trial_rows(m,saved)
    assert len(rows)==10 and {r['method'] for r in rows}=={'analytic'}
    assert len({r['x_sha256'] for r in rows})==1
    assert study.make_input(m,0)==saved
    assert study.make_input(m,1)['X']!=saved['X']
    for ci,beta in enumerate(m['betas']):
        a,b=rows[2*ci:2*ci+2]
        for key in ['selected','target','response_sha256','auxiliary_u']:
            assert a[key]==b[key]
        assert a['beta']==beta
        if beta==0: assert a['null_true'] and b['null_true']


def test_analytic_tail_matches_direct_quadrature_and_interval_selection():
    m=smoke();saved=study.make_input(m,2)
    X=np.asarray(saved['X']);noise=np.asarray(saved['noise'])
    net=study.AdditiveRBFNetwork().fit_basis(X)
    g=study.core.Geometry(net.transform_blocks(X))
    y=g.Q.T@(.5*study.standardized_effect(X[:,0])+noise)
    selected=g.select(y);mask=sum(1<<j for j in selected)
    for arm in m['targets']:
        _,B=g.target(arm,mask,selected[0]);p=B@(B.T@y)
        radius=np.linalg.norm(p);u=p/radius;z=y-p
        intervals=study.core.selection_intervals(g,z,u,selected)
        for r in np.geomspace(.001,30,101):
            assert any(lo<r<hi for lo,hi in intervals)==(g.select(z+r*u)==selected)
        den=sum(integrate.quad(lambda r:stats.chi.pdf(r,B.shape[1]),lo,hi)[0] for lo,hi in intervals)
        num=sum(integrate.quad(lambda r:stats.chi.pdf(r,B.shape[1]),max(lo,radius),hi)[0] for lo,hi in intervals if hi>radius)
        result=study.core.radial_test(g,y,B,selected)
        assert result['p_selective']==pytest.approx(num/den,abs=1e-8)


def test_receipts_resume_full_verification_and_truth_summaries(tmp_path):
    m=smoke();study.setup_output(m,tmp_path)
    for job in study.jobs(m): study.run_job(m,job,tmp_path)
    original={p.name:p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}
    study.setup_output(m,tmp_path)
    report=study.verify(m,tmp_path,export=True)
    assert report['total_tests']==40 and report['trials']==20
    assert study.verify(m,tmp_path)['status']=='verified'
    assert {p.name:p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}==original
    receipt=study.core.read(tmp_path/'jobs'/study.job_name(study.jobs(m)[0]))
    broken=copy.deepcopy(receipt);broken['rows'][0]['p_value']=.123
    with pytest.raises(ValueError,match='checksum'): study.validate_receipt(m,study.jobs(m)[0],broken)
    broken=copy.deepcopy(receipt);broken['inputs'][0]['X'][0][0]+=1
    with pytest.raises(ValueError,match='checksum'): study.validate_receipt(m,study.jobs(m)[0],broken)
    broken=copy.deepcopy(receipt);broken['rows'][0]['null_true']=False
    broken['rows_sha256']=study.core.sha(broken['rows'])
    with pytest.raises(ValueError,match='null_true|truth'): study.validate_receipt(m,study.jobs(m)[0],broken,recompute=True)
    with (tmp_path/'results.csv').open('a') as f: f.write('\ncorrupt\n')
    with pytest.raises(ValueError,match='checksum'): study.verify(m,tmp_path)


def test_frozen_settings_and_cpu_budget(tmp_path):
    production=study.read_config()
    assert production['workers']==32 and production['threads_per_worker']==1
    assert production['iterations']==1000 and len(production['betas'])==5
    assert len(study.jobs(production))==100
    m=smoke();m['workers']=33
    path=tmp_path/'changed.json';study.core.atomic(path,m)
    with pytest.raises(ValueError,match='Manifest changed'): study.read_config(path)


def test_unsafe_archive_rejected(tmp_path):
    path=tmp_path/'bad.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        entry=tarfile.TarInfo('../escape');entry.size=1
        archive.addfile(entry,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe archive'): study.core.validate_archive(smoke(),path)


def test_launcher_has_lifecycle_and_exact_group_guards():
    source=(ROOT/'examples/rbf_shapley_analytic_power_cpu32.ps1').read_text()
    for token in ['Preflight','Launch','Status','Resume','Stop','Verify','Download',
                  'launcher.startticks','kill -TERM -- "-$PID"','flock -n 8',
                  'Refusing to overwrite','validate-archive','plot_rbf_shapley_analytic_power.py',
                  '[r]un_rbf_shapley_analytic_power.py run']:
        assert token in source
