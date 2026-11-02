"""Independent geometry, numerical, matching and resumable benchmark checks."""
from pathlib import Path
import copy
import io
import re
import sys
import tarfile
import numpy as np
import pytest
from scipy import integrate,stats

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import run_rbf_shapley_realdata_10000 as study


def smoke(): return study.read_config(ROOT/'benchmarks'/f'{study.SMOKE}.json')


def test_geometry_matches_original_and_independent_coalition_fits():
    rng=np.random.default_rng(13);X=rng.normal(size=(50,4));X[:,1]+=.5*X[:,0]
    net=study.AdditiveRBFNetwork().fit_basis(X);blocks=net.transform_blocks(X)
    g=study.Geometry(blocks);old=study.core.Geometry(blocks)
    np.testing.assert_allclose(g.selector,old.selector,atol=1e-11)
    y=rng.normal(size=len(X));yr=g.Q.T@y
    assert g.select(yr)==old.select(old.Q.T@y)
    def rss(mask):
        a=net.design_matrix(X,[j for j in range(4) if mask&(1<<j)])
        e=y-a@np.linalg.lstsq(a,y,rcond=None)[0];return e@e
    for arm,M in [('selected_shapley',3),('full_shapley',15)]:
        A,B=g.target(arm,M,0);C,D=old.target(arm,M,0)
        np.testing.assert_allclose(A,C,atol=1e-11)
        np.testing.assert_allclose(B@B.T,D@D.T,atol=1e-10)
        size=M.bit_count()
        score=sum((rss(s)-rss(s|1))/(size*study.comb(size-1,s.bit_count()))
                  for s in range(16) if not s&~M and not s&1)
        assert yr@A@yr==pytest.approx(score,abs=1e-8)


def test_analytic_tail_and_selection_intervals_by_quadrature():
    rng=np.random.default_rng(9);X=rng.normal(size=(50,4))
    g=study.Geometry(study.AdditiveRBFNetwork().fit_basis(X).transform_blocks(X));y=rng.normal(size=g.r)
    selected=g.select(y);M=sum(1<<j for j in selected)
    for arm in study.LABELS:
        _,B=g.target(arm,M,selected[0]);p=B@B.T@y;radius=np.linalg.norm(p);u=p/radius;z=y-p
        intervals=study.core.selection_intervals(g,z,u,selected)
        for r in np.geomspace(.001,30,100):
            assert any(lo<r<hi for lo,hi in intervals)==(g.select(z+r*u)==selected)
        den=sum(integrate.quad(lambda r:stats.chi.pdf(r,B.shape[1]),lo,hi)[0] for lo,hi in intervals)
        num=sum(integrate.quad(lambda r:stats.chi.pdf(r,B.shape[1]),max(lo,radius),hi)[0] for lo,hi in intervals if hi>radius)
        assert study.core.radial_test(g,y,B,selected)['p_selective']==pytest.approx(num/den,abs=1e-8)


def test_nested_uniform_sampling_and_effect_pairing():
    m=smoke();inputs=study.sample_inputs(m,'housing',0)
    assert set(inputs[0]['indices'])<set(inputs[1]['indices'])<set(inputs[2]['indices'])
    assert len({v['auxiliary_u'] for v in inputs})==1
    assert study.sample_inputs(m,'housing',0)==inputs
    assert study.sample_inputs(m,'housing',1)!=inputs
    rows=study.evaluate(m,inputs[0]);assert all(r['status']=='ok' for r in rows)
    for key in ['selected','target','input_sha256','rows_sha256']:
        assert rows[0][key]==rows[1][key]
    assert all(r['null_true'] is None for r in rows)


def test_receipt_verification_corruption_and_export_counts(tmp_path):
    m=smoke();study.setup(m,tmp_path)
    for job in study.jobs(m): study.run_job(m,job,tmp_path)
    before={p.name:p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}
    report=study.verify(m,tmp_path,export=True)
    assert report['total_tests']==28 and report['failures']==0
    study.setup(m,tmp_path)
    assert study.verify(m,tmp_path)['status']=='verified'
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}
    job=study.jobs(m)[0];r=study.core.read(tmp_path/'jobs'/study.job_name(job))
    broken=copy.deepcopy(r);broken['rows'][0]['p_value']=.123
    with pytest.raises(ValueError,match='checksum'): study.validate_receipt(m,job,broken)
    broken=copy.deepcopy(r);broken['inputs'][0]['indices'][0]+=1
    with pytest.raises(ValueError,match='checksum'): study.validate_receipt(m,job,broken)
    broken=copy.deepcopy(r);broken['rows'][0]['target']=-1;broken['rows_sha256']=study.core.sha(broken['rows'])
    with pytest.raises(ValueError,match='Target|Unpaired'): study.validate_receipt(m,job,broken)
    with (tmp_path/'summary.csv').open('a') as f: f.write('corrupt\n')
    with pytest.raises(ValueError,match='checksum'): study.verify(m,tmp_path)


def test_failures_remain_in_denominators(monkeypatch,tmp_path):
    m=smoke()
    def fail(*args,**kwargs): raise RuntimeError('intentional failure')
    monkeypatch.setattr(study,'Geometry',fail)
    inp=study.sample_inputs(m,'housing',0)[0];rows=study.evaluate(m,inp)
    assert len(rows)==2 and all(r['status']=='error' and r['p_value'] is None for r in rows)
    assert all('intentional failure' in r['error'] for r in rows)
    # An all-failure cell remains part of the intended accounting.
    small=dict(m,datasets=['housing'],sample_sizes=[100],repetitions=1)
    study.summarize(small,rows,tmp_path)
    import csv
    with (tmp_path/'summary.csv').open(newline='') as f: summary=list(csv.DictReader(f))
    assert all(r['attempted']=='1' and r['failures']=='1' and r['resolved']=='0' for r in summary)


def test_frozen_cpu64_configuration_and_launcher(tmp_path):
    m=study.read_config()
    assert m['workers']==64 and m['threads_per_worker']==1 and m['sample_sizes']==[100,150,200]
    assert len(study.jobs(m))==18000 and study.expected_tests(m)==500000
    assert m['repetitions']==10000 and m['synthetic_null']['repetitions']==10000
    changed=dict(m,workers=65);path=tmp_path/'changed.json';study.core.atomic(path,changed)
    with pytest.raises(ValueError,match='Manifest changed'): study.read_config(path)
    s=(ROOT/'examples/rbf_shapley_realdata_10000_cpu64.ps1').read_text()
    for token in ['Preflight','Launch','Status','Resume','Stop','Verify','Download','launcher.startticks',
                  'kill -TERM -- "-$PID"','flock -n 8','flock -n 7','Refusing to overwrite',
                  'validate-archive','plot_rbf_shapley_realdata_10000.py','[r]un_rbf_shapley_realdata_10000.py run']:
        assert token in s
    listed=set(re.findall("'([^']+)'",s.split('$files = @(',1)[1].split(')\n',1)[0]))
    assert listed==set(study.PACKAGED)


def test_unsafe_result_archive(tmp_path):
    path=tmp_path/'bad.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        info=tarfile.TarInfo('../escape');info.size=1;archive.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe archive'): study.core.validate_archive(smoke(),path)


def test_synthetic_null_generation_truth_and_pairing():
    m=smoke();inp=study.sample_inputs(m,study.NULL_DATASET,0)[0]
    cfg=m['synthetic_null']
    expected_seed=int(np.random.SeedSequence([cfg['seed'],1,0]).generate_state(1)[0])
    assert inp['subset_seed']==expected_seed and inp['n']==100 and inp['indices']==[]
    rng=np.random.default_rng(expected_seed)
    X=rng.standard_normal((100,4));y=rng.standard_normal(100)
    import hashlib
    rows=study.evaluate(m,inp)
    assert all(r['status']=='ok' and r['null_true'] is True for r in rows)
    assert all(r['input_sha256']==hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest() for r in rows)
    assert rows[0]['selected']==rows[1]['selected'] and rows[0]['target']==rows[1]['target']
    assert study.sample_inputs(m,study.NULL_DATASET,1)!=inp
