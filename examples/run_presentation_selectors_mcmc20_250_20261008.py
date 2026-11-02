"""Matched presentation selectors with parallel MCMC rank inference.

250 null and 250 power inputs at each beta, shared across five selectors.
Inputs are saved once so downloaded verification does not rely on byte-exact
cross-platform random-number transforms. No server launch from Python.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from si_shap import selective_inference
from si_shap.selection import ShapSelector,LimeSelector,SplineScreeningSelector,MutualInformationSelector,LinearShapleySelector,target_from_selected_set
from si_shap.inference import _spline_effect_basis
from si_shap.power import _generate_power_dataset
from si_shap.simulation import _generate_null_dataset
NAME='presentation_selectors_mcmc20_250_20261008'
MANIFEST=ROOT/'examples'/f'{NAME}.json'
MANIFEST_HASH='cc17d082990eff6a5798f26f75527edc8ddf5d51dc4059324f5f07e42eec27c6'

def require(value,message):
    if not value:raise ValueError(message)

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

def read_manifest():
    require(digest(MANIFEST)==MANIFEST_HASH,'Manifest changed; use a new experiment')
    return read(MANIFEST)

def hashes():
    files=[*sorted((ROOT/'src').rglob('*.py')),MANIFEST,ROOT/'pyproject.toml',Path(__file__).resolve(),ROOT/'examples'/f'{NAME}_32cores.ps1',ROOT/'examples'/f'{NAME}_reference.json']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}

def lock(m):
    out=ROOT/m['output_root']
    for name,value in [('requested_manifest.json',m),('source_hashes.json',hashes())]:
        p=out/name
        if p.exists():require(read(p)==value,f'{name} drift; use a new output directory')
        else:atomic(p,value)
    return out

def condition(m,c):return 'null' if m['betas'][c]==0 else f"beta_{m['betas'][c]:g}"

def generated(m,c,trial):
    children=np.random.SeedSequence([m['base_seed'],c,trial]).spawn(3)
    data,target,sampler=[int(s.generate_state(1)[0]) for s in children]
    d=m['design'];rng=np.random.default_rng(data);beta=m['betas'][c]
    if beta==0:
        X,y=_generate_null_dataset(rng,d['n_samples'],d['n_features'],d['feature_correlation']);mean=np.zeros(len(y))
    else:
        X,y,mean=_generate_power_dataset(rng,d['n_samples'],d['n_features'],(0,),beta,feature_correlation=d['feature_correlation'],return_mean=True)
    identity=dict(condition=condition(m,c),beta=beta,trial=trial,data_seed=data,target_seed=target,sampler_seed=sampler,auxiliary_u=float(np.random.default_rng(target).random()))
    return X,y,mean,identity

def input_path(m,c,trial):return ROOT/m['output_root']/'inputs'/condition(m,c)/f'trial_{trial}'
def result_path(m,method,c,trial):return ROOT/m['output_root']/'runs'/method/condition(m,c)/f'trial_{trial}'

def save_input(m,c,trial):
    X,y,mean,identity=generated(m,c,trial);p=input_path(m,c,trial)
    if (p/'receipt.json').exists():
        receipt=read(p/'receipt.json');require(receipt['identity']==identity,'Input seed drift');require(digest(p/'input.npz')==receipt['sha256'],'Corrupt input')
    else:
        p.mkdir(parents=True,exist_ok=True);temp=p/'input.tmp.npz'
        np.savez_compressed(temp,X=X,y=y,mean=mean);temp.replace(p/'input.npz')
        atomic(p/'receipt.json',dict(identity=identity,sha256=digest(p/'input.npz')))

def load_input(m,c,trial):
    p=input_path(m,c,trial);receipt=read(p/'receipt.json')
    require(digest(p/'input.npz')==receipt['sha256'],f'Corrupt input: {p}')
    with np.load(p/'input.npz',allow_pickle=False) as z:X,y,mean=(z[k].copy() for k in ['X','y','mean'])
    return X,y,mean,receipt['identity']

def selector_for(m,method):
    if method=='shap':return ShapSelector(RandomForestRegressor(**m['rf_params']),selection_decimals=m['selection_decimals'])
    if method=='lime':return LimeSelector(RandomForestRegressor(**m['rf_params']),selection_decimals=m['selection_decimals'],**m['lime'])
    if method=='spline_screening':return SplineScreeningSelector()
    if method=='mutual_information':return MutualInformationSelector(**m['mutual_information'])
    if method=='linear_shapley':return LinearShapleySelector(**m['linear_shapley'])
    raise ValueError('Unknown selector')

def requested(m,identity,method):return dict(identity=identity,method=method,manifest=m)

def bundle(p,request):
    receipt=read(p/'receipt.json');require(receipt['requested']==request,f'Receipt settings drift: {p}')
    require(set(receipt['files'])=={'result.json','settings.json','test_basis.npz'},'Missing receipt files')
    for name,checksum in receipt['files'].items():require(digest(p/name)==checksum,f'Corrupt result: {p/name}')
    return read(p/'result.json'),read(p/'settings.json')

def execute(m,method,c,trial):
    X,y,mean,identity=load_input(m,c,trial);p=result_path(m,method,c,trial);request=requested(m,identity,method)
    if (p/'receipt.json').exists():bundle(p,request);return
    selector=selector_for(m,method);mc=m['mcmc'];started=time.perf_counter();cpu=time.process_time()
    result=selective_inference(X,y,k_select=m['design']['k_select'],sigma=m['sigma'],selector=selector,
        test_statistic='linear_shapley' if method=='linear_shapley' else 'spline',
        selection_event=m['selection_event'],target_rule=m['target_rule'],auxiliary_u=identity['auxiliary_u'],target_seed=identity['target_seed'],
        multiplicity=m['multiplicity'],inference_method='mcmc_rank',selection_decimals=m['selection_decimals'],ais_seed=identity['sampler_seed'],
        max_final_samples=mc['replicas'],mcmc_steps=mc['steps'],mcmc_step_scales=mc['step_scales'],mcmc_global_jump_probability=mc['global_jump_probability'])
    row=result['feature_results'].iloc[0].to_dict();feature=int(row['feature'])
    basis=selector.statistic(X).support_basis(feature) if method=='linear_shapley' else _spline_effect_basis(X[:,feature])
    projection=float(np.linalg.norm(basis.T@mean))
    # The core exposes legacy replica-count ESS/SE aliases. This experiment
    # labels those quantities explicitly and never interprets them as ESS.
    row['mc_se_conditional_hub']=row.pop('mc_se',None)
    row['hub_tail_ci_lower']=row.pop('mc_ci_95_lower',None);row['hub_tail_ci_upper']=row.pop('mc_ci_95_upper',None)
    row.update(denominator_ess=None,tail_ess=None,**identity,method=method,
        observed_selected_features=sorted(map(int,result['observed_selected_features'])),
        target_feature=feature,target_is_signal=bool(identity['beta']>0 and feature==0),
        projection_mean_norm=projection,fixed_design_null=bool(projection<=np.sqrt(np.finfo(float).eps)*max(1.,np.linalg.norm(mean))),
        hypothesis=result['hypothesis'],input_sha256=read(input_path(m,c,trial)/'receipt.json')['sha256'],
        failed=not np.isfinite(row['p_value']),rejected=bool(row['p_value']<m['alpha']),
        elapsed_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu)
    atomic(p/'result.json',row);atomic(p/'settings.json',result['settings'])
    temporary=p/'test_basis.tmp.npz';np.savez_compressed(temporary,basis=basis);temporary.replace(p/'test_basis.npz')
    atomic(p/'receipt.json',dict(requested=request,files={n:digest(p/n) for n in ['result.json','settings.json','test_basis.npz']}))
    print(f'{method}/{identity["condition"]}/{trial}: p={row["p_value"]}',flush=True)

def worker(m,method,c,shard):
    require(method in m['methods'] and 0<=c<len(m['betas']) and 0<=shard<m['shards'],'Invalid worker arguments')
    for trial in range(shard+1,m['iterations']+1,m['shards']):execute(m,method,c,trial)

def run(m):
    out=lock(m)
    # Single controller prepares inputs before dispatch: no shared-input races.
    for c in range(len(m['betas'])):
        for trial in range(1,m['iterations']+1):save_input(m,c,trial)
    def launch(job):
        method,c,shard=job;p=out/'logs'/f'{method}_{c}_{shard}.log';p.parent.mkdir(exist_ok=True)
        with p.open('a',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--method',method,'--condition',str(c),'--shard',str(shard)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    jobs=[(method,c,s) for method in m['methods'] for c in range(len(m['betas'])) for s in range(m['shards'])]
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=m['workers']) as pool:
        for future in concurrent.futures.as_completed([pool.submit(launch,j) for j in jobs]):
            try:future.result()
            except Exception as e:errors.append(str(e))
    require(not errors,f'{len(errors)} workers failed; inspect logs and Resume')
    verify(m)

def ci(k,n):
    if not n:return np.nan,np.nan
    return (float(stats.beta.ppf(.025,k,n-k+1)) if k else 0.,float(stats.beta.ppf(.975,k+1,n-k)) if k<n else 1.)

def summaries(f,m):
    rows=[];calibration=[]
    for (method,cond),g in f.groupby(['method','condition']):
        signal=g if g.beta.iloc[0]==0 else g[g.target_is_signal]
        n=len(signal);k=int(signal.rejected.sum());failed=int(signal.failed.sum());lo,hi=ci(k,n)
        rows.append(dict(method=method,condition=cond,beta=g.beta.iloc[0],tests=len(g),signal_targets=n if cond!='null' else None,detections=k,unresolved=failed,
            operational_fraction=k/n if n else np.nan,ci_lower=lo,ci_upper=hi,missing_lower=k/n if n else np.nan,missing_upper=(k+failed)/n if n else np.nan,
            targeting_probability=n/len(g) if cond!='null' else None,marginal_signal_detection=k/len(g) if cond!='null' else None,
            interpretation='global_projection_null' if method=='linear_shapley' else 'feature_spline_projection_null'))
        if cond=='null':
            for alpha in [.01,.05,.1]:
                k=int(g.p_value.lt(alpha).sum());lo,hi=ci(k,len(g))
                calibration.append(dict(method=method,alpha=alpha,n=len(g),rejections=k,unresolved=int(g.failed.sum()),fraction=k/len(g),ci_lower=lo,ci_upper=hi))
    return pd.DataFrame(rows),pd.DataFrame(calibration)

def audit_settings(settings,m,method):
    expected=dict(n_samples=m['design']['n_samples'],n_features=m['design']['n_features'],k_select=m['design']['k_select'],sigma=m['sigma'],selection_event='exact_set',target_rule='uniform_from_selected',multiplicity='none',inference_method='mcmc_rank',max_final_samples=m['mcmc']['replicas'],mcmc_steps=m['mcmc']['steps'],mcmc_step_scales=m['mcmc']['step_scales'],mcmc_global_jump_probability=m['mcmc']['global_jump_probability'],selection_method=method,test_statistic='linear_shapley' if method=='linear_shapley' else 'spline')
    for k,v in expected.items():require(settings[k]==v,f'Inference setting drift: {k}')
    reference=clean(dict(selector_for(m,method).get_settings()))
    for k,v in reference.items():
        if k!='resolved_mode':require(settings[k]==v,f'Selector setting drift: {method}/{k}')
    if method=='linear_shapley':require(settings['resolved_mode']==('monte_carlo' if m['design']['n_features']>m['linear_shapley']['exact_max_features'] else 'exact'),'Linear Shapley mode drift')

def verify(m,downloaded=False):
    out=ROOT/m['output_root'] if downloaded else lock(m)
    require(read(out/'requested_manifest.json')==m,'Recorded manifest drift')
    rows=[];errors=[]
    for c in range(len(m['betas'])):
        for trial in range(1,m['iterations']+1):
            try:
                X,y,mean,identity=load_input(m,c,trial)
                expected=generated(m,c,trial);require(identity==expected[3],'Seed/target identity drift')
                for actual,regenerated in zip((X,y,mean),expected[:3]):np.testing.assert_allclose(actual,regenerated,rtol=1e-13,atol=1e-13)
                input_hash=read(input_path(m,c,trial)/'receipt.json')['sha256']
                for method in m['methods']:
                    row,settings=bundle(result_path(m,method,c,trial),requested(m,identity,method));audit_settings(settings,m,method)
                    for k,v in identity.items():require(row[k]==v,f'Result identity drift: {k}')
                    require(row['input_sha256']==input_hash and row['method']==method,'Unmatched input or method')
                    selected=row['observed_selected_features'];require(selected==sorted(set(selected)) and len(selected)==m['design']['k_select'] and all(0<=x<m['design']['n_features'] for x in selected),'Invalid selected set')
                    require(row['target_feature']==target_from_selected_set(selected,identity['auxiliary_u']),'Target randomization drift')
                    require(row['target_feature']==settings['observed_target_feature'] and settings['auxiliary_u']==identity['auxiliary_u'],'Target settings drift')
                    require(settings['ais_seed']==identity['sampler_seed'] and settings['target_seed']==identity['target_seed'],'Sampler seed drift')
                    p=row['p_value'];mc=m['mcmc'];require(p is not None and np.isfinite(p) and 0<p<=1,'Unresolved or invalid rank p-value')
                    require(row['proposals']==(mc['replicas']+1)*mc['steps'] and row['mcmc_replicates']==mc['replicas'] and row['mcmc_steps']==mc['steps'],'Incomplete MCMC construction')
                    require(p==(row['tail_samples']+1)/(mc['replicas']+1) and row['minimum_attainable_p_value']==1/(mc['replicas']+1),'Rank construction drift')
                    require(row['denominator_ess'] is None and row['tail_ess'] is None,'Replica counts mislabeled ESS')
                    require(row['finite_sample_valid'] and row['p_value_method']=='parallel_reversible_mcmc_rank','Wrong inference construction')
                    require(row['failed']==False and row['rejected']==(p<m['alpha']),'Decision mismatch')
                    require(row['target_is_signal']==bool(identity['beta']>0 and row['target_feature']==0),'Signal-target drift')
                    require(row['test_statistic']==settings['test_statistic'] and 1<=row['test_rank']<=m['design']['n_features'],'Invalid test statistic/rank')
                    with np.load(result_path(m,method,c,trial)/'test_basis.npz',allow_pickle=False) as z:basis=z['basis']
                    require(basis.shape==(len(y),row['test_rank']) and np.isfinite(basis).all(),'Invalid saved test basis')
                    np.testing.assert_allclose(basis.T@basis,np.eye(row['test_rank']),rtol=1e-10,atol=1e-10)
                    require(np.isclose(np.linalg.norm(basis.T@y)/m['sigma'],row['radial_t_obs'],rtol=1e-11,atol=1e-11),'Radial statistic mismatch')
                    require(np.isclose(np.linalg.norm(basis.T@mean),row['projection_mean_norm'],rtol=1e-11,atol=1e-11),'Projection mean mismatch')
                    if method!='linear_shapley':
                        local_basis=_spline_effect_basis(X[:,row['target_feature']])
                        np.testing.assert_allclose(basis@basis.T,local_basis@local_basis.T,rtol=1e-10,atol=1e-10)
                    rows.append(row)
            except Exception as e:errors.append(dict(condition=condition(m,c),trial=trial,reason=str(e)))
    errorfile='local_verification_errors.csv' if downloaded else 'verification_errors.csv'
    pd.DataFrame(errors,columns=['condition','trial','reason']).to_csv(out/errorfile,index=False)
    require(not errors,f'{len(errors)} invalid inputs/results; inspect {errorfile}')
    f=pd.DataFrame(rows);expected_count=len(m['methods'])*len(m['betas'])*m['iterations']
    require(len(f)==expected_count and not f.duplicated(['method','condition','trial']).any(),'Result count/duplicate drift')
    summary,calibration=summaries(f,m)
    outputs={'unified_tests.csv':f,'power_summary.csv':summary[summary.beta.gt(0)],'null_summary.csv':summary[summary.beta.eq(0)],'calibration_summary.csv':calibration,
        'runtime_summary.csv':f.groupby(['method','condition']).agg(tests=('method','size'),transitions=('proposals','sum'),worker_seconds=('elapsed_seconds','sum'),cpu_seconds=('cpu_seconds','sum')).reset_index()}
    for filename,data in outputs.items():
        if downloaded:
            old=pd.read_csv(out/filename,keep_default_na=False,na_values=['']);new=pd.read_csv(io.StringIO(data.to_csv(index=False)),keep_default_na=False,na_values=[''])
            pd.testing.assert_frame_equal(old,new,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        else:data.to_csv(out/filename,index=False)
    if not downloaded:
        atomic(out/'verification.json',dict(tests=expected_count,null_tests=m['iterations']*len(m['methods']),power_tests=m['iterations']*(len(m['betas'])-1)*len(m['methods']),transitions=int(f.proposals.sum()),shared_inputs=len(m['betas'])*m['iterations'],workers=m['workers']))
        (out/'COMPLETED').write_text(f'Verified {expected_count} tests with parallel MCMC ({m["mcmc"]["replicas"]} replicas x {m["mcmc"]["steps"]} steps plus hub).\n',encoding='utf-8')
    require((out/'COMPLETED').exists(),'Missing completion marker')
    print((out/'COMPLETED').read_text(encoding='utf-8'))

def verify_downloaded(m):
    out=ROOT/m['output_root'];saved=read(out/'source_hashes.json');current=hashes()
    require(saved and all(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in saved.values()),'Invalid source provenance')
    report=dict(original_provenance_preserved=True,changed_recorded_files=sorted(k for k,v in saved.items() if current.get(k)!=v))
    try:verify(m,True);report['status']='passed'
    except Exception as e:report.update(status='failed',reason=str(e));raise
    finally:atomic(out/'local_verification_report.json',report)

def validate_archive(path,m):
    with tarfile.open(path) as archive:
        for entry in archive.getmembers():
            p=PurePosixPath(entry.name)
            require(p.parts and not p.is_absolute() and '..' not in p.parts and p.parts[0]==m['experiment'] and '\\' not in entry.name and ':' not in entry.name and (entry.isfile() or entry.isdir()),f'Unsafe archive entry: {entry.name}')

def status(m):
    out=ROOT/m['output_root']
    for method in m['methods']:
        count=len(list((out/'runs'/method).glob('*/trial_*/receipt.json')))
        print(f'{method}: {count}/{m["iterations"]*len(m["betas"])} receipts')
    if (out/'COMPLETED').exists():print((out/'COMPLETED').read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','run','worker','status','verify','verify-downloaded','validate-archive']);p.add_argument('--method');p.add_argument('--condition',type=int);p.add_argument('--shard',type=int);p.add_argument('--archive');a=p.parse_args();m=read_manifest()
    if a.action=='check':print(json.dumps(dict(experiment=NAME,methods=m['methods'],betas=m['betas'],iterations=m['iterations'],tests=6250,workers=32,transitions_per_test=10240,total_transitions=64000000),indent=2))
    elif a.action=='run':run(m)
    elif a.action=='worker':worker(m,a.method,a.condition,a.shard)
    elif a.action=='status':status(m)
    elif a.action=='verify':verify(m)
    elif a.action=='verify-downloaded':verify_downloaded(m)
    else:validate_archive(a.archive,m)
if __name__=='__main__':main()
