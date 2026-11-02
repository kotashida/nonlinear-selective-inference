"""Locked, resumable SHAP sampling follow-up. No automatic remote launch.

Fresh random-design calibration/power and selected mechanism cases are separate
cohorts. Receipts checkpoint common pilots and every arm on every input. A
downloaded audit checks saved aggregates without rewriting original provenance.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import numpy as np
import pandas as pd
from scipy import stats
from si_shap.inference import _chi_statistic, _spline_effect_basis, _prepare_refinement_pilot
from si_shap.power import _generate_power_dataset
from si_shap.simulation import _generate_null_dataset
from si_shap.selection import make_selector, target_from_selected_set
from examples.build_shap_sampling_followup_inputs import arrays_hash
from examples.shap_sampling_followup_samplers import run_ais, run_rank, map_regions

MANIFEST = ROOT/'examples/shap_sampling_followup_20261005.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonable(value):
    if isinstance(value, dict):
        return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value,np.generic):
        return jsonable(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest():
    m = json.loads(MANIFEST.read_text())
    require(m['workers']==m['shards']==32,'Production requires 32 serial workers')
    require(m['rf_params']==dict(n_estimators=50,max_depth=5,random_state=42,n_jobs=1), 'Forest drift')
    require(m['alpha']==.05 and m['sigma']==1 and m['selection_event']=='exact_set'
            and m['target_rule']=='uniform_from_selected' and m['multiplicity']=='none','Inference drift')
    require([(c['iterations'],c['beta']) for c in m['conditions']]==[(500,0.),(250,.35),(250,.6)],'Condition drift')
    require(len(m['arms'])==7 and len({a['name'] for a in m['arms']})==7,'Arm drift')
    require(m['ais']==dict(pilot_iters=10,pilot_samples=320,batch_size=800,
        allocation_cap=51200,draw_floor=128,min_denominator_ess=80.,min_tail_ess=15.),'AIS settings drift')
    require(m['mcmc']==dict(step_scales=[.25,.5,1.,2.],global_jump_probability=.1),'MCMC settings drift')
    expected = [
        ('ais_reference','ais','reference',True,51200,1),
        ('ais_stable','ais','stable',True,51200,2),
        ('ais_preserve','ais','preserve',True,51200,3),
        ('ais_reference_fixed','ais','reference',False,12800,1)]
    require([(a['name'],a['method'],a['variant'],a['early_stop'],a['final_cap'],a['sampler_salt'])
             for a in m['arms'][:4]]==expected,'AIS arm drift')
    require([(a['name'],a['construction'],a['replicas'],a['steps'],a['sampler_salt'])
             for a in m['arms'][4:]]==[('mcmc_parallel_20','parallel',511,20,5),
             ('mcmc_parallel_100','parallel',511,100,6),('mcmc_serial_100','serial',511,100,7)],'Rank arm drift')
    fixtures = read_fixtures(m)
    require(len(fixtures['cases'])==28 and len(fixtures['replay_cases'])==8,'Fixture count drift')
    for case in fixtures['cases']:
        X,y,_ = fixture_arrays(m,case)
        require(arrays_hash(X,y)==case['input_hash'],f"Fixture corrupted: {case['case_id']}")
    return m


def read_fixtures(m):
    return json.loads((ROOT/m['diagnostics']['fixtures']).read_text())


def fixture_arrays(m, case):
    base = (ROOT/m['diagnostics']['fixtures']).parent
    path = base/case['file']
    require(path.resolve().is_relative_to(base.resolve()),'Fixture path escapes directory')
    with np.load(path,allow_pickle=False) as saved:
        return saved['X'],saved['y'],saved['mean']


def hashes(m):
    files = [*sorted((ROOT/'src').rglob('*.py')), MANIFEST, ROOT/'pyproject.toml',
             *[ROOT/'examples'/p for p in ('run_shap_sampling_followup.py',
               'shap_sampling_followup_samplers.py','build_shap_sampling_followup_inputs.py',
               'shap_sampling_followup_32cores.ps1')],
             *sorted((ROOT/m['diagnostics']['fixtures']).parent.glob('*'))]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files if p.is_file()}


def lock(m):
    out = ROOT/m['output_root']
    for name,value in [('requested_manifest.json',m),('source_hashes.json',hashes(m))]:
        path = out/name
        if path.exists():
            require(json.loads(path.read_text())==value,f'{name} drift; use a new output folder')
        else:
            atomic_json(path,value)
    return out


def allocation(m, condition, shard):
    q,r = divmod(condition['iterations'],m['shards'])
    return q+int(shard<r)


def fresh_input(m, condition_index, shard, iteration):
    seed = m['base_seed']+condition_index*10000+shard
    child = np.random.SeedSequence(seed).spawn(iteration)[-1]
    data_seed,target_seed,pilot_seed = child.spawn(3)
    c,d = m['conditions'][condition_index],m['design']
    rng = np.random.default_rng(data_seed)
    if c['beta']==0:
        X,y = _generate_null_dataset(rng,d['n_samples'],d['n_features'],d['feature_correlation'])
        mean = np.zeros(d['n_samples'])
    else:
        X,y,mean = _generate_power_dataset(rng,d['n_samples'],d['n_features'],(0,),c['beta'],
            feature_correlation=d['feature_correlation'],return_mean=True)
    target_seed_int = int(target_seed.generate_state(1)[0])
    u = float(np.random.default_rng(target_seed_int).random())
    identity = dict(cohort='fresh',condition=c['name'],shard=shard,iteration=iteration,
                    case_id=f"{c['name']}_s{shard}_i{iteration}",repeat=0,
                    root_seed=seed,target_seed=target_seed_int,auxiliary_u=u,
                    beta=c['beta'],input_hash=arrays_hash(X,y),
                    pilot_seed=int(pilot_seed.generate_state(1)[0]))
    return X,y,mean,identity


def diagnostic_input(m, case, repeat):
    X,y,mean = fixture_arrays(m,case)
    position = [c['case_id'] for c in read_fixtures(m)['cases']].index(case['case_id'])
    child = np.random.SeedSequence([m['diagnostic_seed'],position,repeat])
    identity = dict(cohort='diagnostic',condition='selected_cases',shard=position,
                    iteration=1,case_id=case['case_id'],repeat=repeat,
                    root_seed=m['diagnostic_seed'],target_seed=None,
                    auxiliary_u=case['auxiliary_u'],beta=case['beta'],
                    input_hash=case['input_hash'],pilot_seed=int(child.generate_state(1)[0]))
    return X,y,mean,identity


class Ray:
    def __init__(self,X,y,u,m,selector=None):
        self.X,self.y = X,y
        self.selector = selector or make_selector(selection_method='shap',rf_params=m['rf_params'],
                                                   selection_decimals=m['selection_decimals'])
        observed = self.selector.select(X,y,m['design']['k_select'])
        self.selected = sorted(map(int,observed.selected_features))
        self.target = target_from_selected_set(self.selected,u)
        basis = _spline_effect_basis(X[:,self.target])
        self.rank = basis.shape[1]
        self.t_obs,projected = _chi_statistic(y,basis,sigma=m['sigma'])
        norm = float(np.linalg.norm(projected))
        require(norm>np.sqrt(np.finfo(float).eps)*max(1.,np.linalg.norm(y)), 'Undefined radial direction')
        self.basis,self.orthogonal,self.direction = basis,y-projected,projected/norm
        self.k,self.sigma = m['design']['k_select'],m['sigma']
        self.cache = {}
        self.calls,self.fits = 0,0
        self.record = False
        self.query_z,self.query_selected,self.query_margin,self.query_structure = [],[],[],[]
        require(self(self.t_obs),'Observed statistic did not reproduce selection')
        again = self.selector.select(X,y,self.k)
        require(np.array_equal(observed.importance,again.importance),'Nondeterministic observed selector')

    def __call__(self,z):
        z = float(z)
        self.calls += 1
        if z not in self.cache:
            candidate = self.orthogonal+self.sigma*self.direction*z
            result = self.selector.select(self.X,candidate,self.k)
            state = sorted(map(int,result.selected_features))==self.selected
            scores = result.importance
            other = np.setdiff1d(np.arange(len(scores)),self.selected)
            margin = float(min(scores[self.selected])-max(scores[other])) if len(other) else np.inf
            # The selector interface exposes scores, not forest structure. Save
            # the full score-vector hash as a reproducible change diagnostic.
            fingerprint = hashlib.sha256(scores.tobytes()).hexdigest()[:16]
            self.cache[z] = state,margin,fingerprint
            self.fits += 1
        state,margin,fingerprint = self.cache[z]
        if self.record:
            self.query_z.append(z); self.query_selected.append(state)
            self.query_margin.append(margin); self.query_structure.append(fingerprint)
        return state

    def begin(self,record=False):
        self.record=record
        self.query_z,self.query_selected,self.query_margin,self.query_structure = [],[],[],[]
        return self.calls,self.fits

    def queries(self):
        return dict(query_z=np.asarray(self.query_z),query_selected=np.asarray(self.query_selected),
                    query_margin=np.asarray(self.query_margin),score_fingerprint=np.asarray(self.query_structure))


def trial_path(m,identity):
    if identity['cohort']=='fresh':
        return ROOT/m['output_root']/'runs'/'fresh'/identity['condition']/f"shard_{identity['shard']}"/f"trial_{identity['iteration']}"
    return ROOT/m['output_root']/'runs'/'diagnostic'/identity['case_id']/f"repeat_{identity['repeat']}"


def receipt_request(m,identity,arm):
    return jsonable(dict(identity=identity,arm=arm,ais=m['ais'],mcmc=m['mcmc'],
                         alpha=m['alpha'],selection_event=m['selection_event'],sigma=m['sigma']))


def save_bundle(path,requested,row,trace):
    path.mkdir(parents=True,exist_ok=True)
    atomic_json(path/'result.json',row)
    temporary = path/'trace.tmp.npz'
    np.savez_compressed(temporary,**trace)
    temporary.replace(path/'trace.npz')
    atomic_json(path/'bundle_receipt.json',dict(requested=requested,
        files={name:digest(path/name) for name in ('result.json','trace.npz')}))


def read_bundle(path,requested):
    receipt = json.loads((path/'bundle_receipt.json').read_text())
    require(receipt['requested']==jsonable(requested),f'Settings drift: {path}')
    require(set(receipt['files'])=={'result.json','trace.npz'},f'Incomplete receipt: {path}')
    for name,checksum in receipt['files'].items():
        require(digest(path/name)==checksum,f'Corrupt bundle: {path/name}')
    return json.loads((path/'result.json').read_text())


def pilot_for(m,ray,identity,path):
    requested = dict(identity=identity,ais=m['ais'])
    common = path/'common'
    if (common/'bundle_receipt.json').exists():
        row = read_bundle(common,requested)
        with np.load(common/'trace.npz',allow_pickle=False) as data:
            pilot = {k:data[k] for k in ('z_values','selected','log_weights','fit_mask','validation_mask')}
        pilot.update(mean=row['mean'],sd=row['sd'],sampling_scheme=row['sampling_scheme'])
        return pilot
    before = ray.begin(identity['cohort']=='diagnostic')
    started = time.perf_counter()
    a = m['ais']
    pilot = _prepare_refinement_pilot(ray.t_obs,ray.rank,ray,np.random.default_rng(identity['pilot_seed']),
                                      a['pilot_iters'],a['pilot_samples'],pilot15=False)
    row = dict(mean=pilot['mean'],sd=pilot['sd'],sampling_scheme=pilot['sampling_scheme'],
               pilot_evaluations=a['pilot_iters']*a['pilot_samples'],
               selector_calls=ray.calls-before[0],uncached_selector_fits=ray.fits-before[1],
               elapsed_seconds=time.perf_counter()-started)
    trace = {k:pilot[k] for k in ('z_values','selected','log_weights','fit_mask','validation_mask')}
    trace.update(ray.queries())
    save_bundle(common,requested,row,trace)
    return pilot


def effective_arms(m,cohort):
    arms = copy.deepcopy(m['arms'])
    if cohort=='diagnostic':
        arms = [a for a in arms if a['name']!='ais_reference_fixed']
        for a in arms:
            if a['method']=='ais':
                a.update(final_cap=m['diagnostics']['ais_final_cap'],early_stop=False)
    return arms


def execute_trial(m,X,y,mean,identity,expected_case=None,selector=None):
    path = trial_path(m,identity)
    arms = effective_arms(m,identity['cohort'])
    if all((path/a['name']/'bundle_receipt.json').exists() for a in arms):
        read_bundle(path/'common',dict(identity=identity,ais=m['ais']))
        for a in arms:
            read_bundle(path/a['name'],receipt_request(m,identity,a))
        return
    ray = Ray(X,y,identity['auxiliary_u'],m,selector)
    if expected_case:
        require(ray.selected==expected_case['observed_selected_features'] and ray.target==expected_case['target_feature']
                and np.isclose(ray.t_obs,expected_case['t_obs'],rtol=1e-12,atol=1e-12),'Reconstructed input mismatch')
    pilot = pilot_for(m,ray,identity,path)
    projection = float(np.linalg.norm(ray.basis.T@mean)) if np.isfinite(mean).all() else None
    for a in arms:
        arm_path = path/a['name']
        requested = receipt_request(m,identity,a)
        if (arm_path/'bundle_receipt.json').exists():
            read_bundle(arm_path,requested)
            continue
        rng_seed = np.random.SeedSequence([identity['pilot_seed'],a['sampler_salt'],773])
        rng = np.random.default_rng(rng_seed)
        before = ray.begin(identity['cohort']=='diagnostic')
        wall,cpu = time.perf_counter(),time.process_time()
        if a['method']=='ais':
            row,trace = run_ais(ray.t_obs,ray.rank,ray,rng,pilot,a,m['ais'])
        else:
            row,trace = run_rank(ray.t_obs,ray.rank,ray,rng,a,m['mcmc'],trace=identity['cohort']=='diagnostic')
        row.update(identity,arm=a['name'],method=a['method'],settings_json=json.dumps(a,sort_keys=True),
            target_feature=ray.target,observed_selected_features=json.dumps(ray.selected),
            t_obs=float(ray.t_obs),test_rank=ray.rank,
            target_is_signal=(ray.target==0) if identity['beta'] and projection is not None else False,
            null_projection_norm=projection,
            fixed_design_null=(projection<=np.sqrt(np.finfo(float).eps)*max(1.,np.linalg.norm(mean))) if projection is not None else None,
            failed=not np.isfinite(row['p_value']),rejected=bool(row['p_value']<m['alpha']),
            selector_calls=ray.calls-before[0],uncached_selector_fits=ray.fits-before[1],
            elapsed_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
            sampler_seed=int(rng_seed.generate_state(1)[0]),pilot_evaluations_shared=0)
        trace.update(ray.queries())
        save_bundle(arm_path,requested,row,trace)
        print(f"{identity['case_id']}/{identity['repeat']}/{a['name']}: {row['status']}",flush=True)


def execute_map(m,case):
    identity = dict(cohort='geometry',case_id=case['case_id'],input_hash=case['input_hash'])
    path = ROOT/m['output_root']/'geometry'/case['case_id']
    requested = dict(identity=identity,map_budget=m['diagnostics']['map_budget'])
    if (path/'bundle_receipt.json').exists():
        read_bundle(path,requested)
        return
    X,y,_ = fixture_arrays(m,case)
    ray = Ray(X,y,case['auxiliary_u'],m)
    require(ray.selected==case['observed_selected_features'] and ray.target==case['target_feature']
            and np.isclose(ray.t_obs,case['t_obs'],rtol=1e-12,atol=1e-12),'Geometry input mismatch')
    anchors = []
    if case['case_id'] in read_fixtures(m)['replay_cases']:
        for repeat in range(m['diagnostics']['replay_seeds']):
            _,_,_,replay_id = diagnostic_input(m,case,repeat)
            pilot_path = trial_path(m,replay_id)/'common'
            read_bundle(pilot_path,dict(identity=replay_id,ais=m['ais']))
            with np.load(pilot_path/'trace.npz',allow_pickle=False) as data:
                anchors.extend(data['z_values'])
    before = ray.begin(True)
    started = time.perf_counter()
    row,trace = map_regions(ray,ray.t_obs,ray.rank,m['diagnostics']['map_budget'],anchors)
    row.update(identity,target_feature=ray.target,t_obs=float(ray.t_obs),
               selector_calls=ray.calls-before[0],uncached_selector_fits=ray.fits-before[1],
               elapsed_seconds=time.perf_counter()-started)
    trace.update(ray.queries())
    save_bundle(path,requested,row,trace)


def jobs(m):
    for c in range(len(m['conditions'])):
        for s in range(m['shards']):
            yield ('fresh',c,s)
    for index,case_id in enumerate(read_fixtures(m)['replay_cases']):
        for repeat in range(m['diagnostics']['replay_seeds']):
            yield ('diagnostic',index,repeat)


def worker(m,cohort,index,shard):
    if cohort=='fresh':
        require(0<=index<len(m['conditions']) and 0<=shard<m['shards'],'Invalid worker index')
        for iteration in range(1,allocation(m,m['conditions'][index],shard)+1):
            execute_trial(m,*fresh_input(m,index,shard,iteration))
    elif cohort=='diagnostic':
        fixtures = read_fixtures(m)
        require(0<=index<len(fixtures['replay_cases']) and 0<=shard<m['diagnostics']['replay_seeds'],'Invalid replay index')
        case_id = fixtures['replay_cases'][index]
        case = next(c for c in fixtures['cases'] if c['case_id']==case_id)
        execute_trial(m,*diagnostic_input(m,case,shard),expected_case=case)
    elif cohort=='geometry':
        execute_map(m,read_fixtures(m)['cases'][index])
    else:
        raise ValueError('Unknown worker cohort')


def dispatch(m,assignments):
    out = ROOT/m['output_root']
    def launch(job):
        cohort,index,shard = job
        log = out/'logs'/f'{cohort}_{index}_{shard}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker',
                            '--cohort',cohort,'--index',str(index),'--shard',str(shard)],
                           cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=m['workers']) as pool:
        futures = [pool.submit(launch,job) for job in assignments]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as exc:
                errors.append(str(exc))
    require(not errors,f'{len(errors)} workers failed; inspect logs and Resume')


def binomial_interval(k,n):
    return (float(stats.beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(stats.beta.ppf(.975,k+1,n-k)) if k<n else 1.) if n else (np.nan,np.nan)


def summary(frame,m):
    rows = []
    fresh = frame[frame.cohort.eq('fresh')]
    for (arm,condition),f in fresh.groupby(['arm','condition'],sort=True):
        signal = pd.Series(True,index=f.index) if condition=='null' else f.target_is_signal.astype(bool)
        g = f[signal]
        failed = int(g.failed.sum()); k = int(g.rejected.sum()); n = len(g)
        lower,upper = binomial_interval(k,n)  # operational fraction: failure is non-detection
        resolved = len(g)-failed
        rlower,rupper = binomial_interval(k,resolved)
        rows.append(dict(arm=arm,condition=condition,alpha=m['alpha'],n_tests=len(f),
            n_failed=int(f.failed.sum()),n_signal_targets=n if condition!='null' else None,
            denominator=n,detections_or_rejections=k,unresolved_targets=failed,
            operational_fraction=k/n if n else np.nan,operational_ci_lower=lower,operational_ci_upper=upper,
            missing_outcome_lower=k/n if n else np.nan,
            missing_outcome_upper=(k+failed)/n if n else np.nan,
            resolved_only_fraction=k/resolved if resolved else np.nan,
            resolved_only_ci_lower=rlower,resolved_only_ci_upper=rupper,
            marginal_signal_detection=k/len(f) if condition!='null' else None,
            metric='global_null_rejection' if condition=='null' else 'operational_signal_detection'))
    return pd.DataFrame(rows)


def paired_comparisons(frame):
    keys = ['cohort','condition','case_id','repeat']
    base = frame[frame.arm.eq('ais_reference')]
    rows = []
    for arm,g in frame.groupby('arm'):
        if arm=='ais_reference':
            continue
        paired = base.merge(g,on=keys,suffixes=('_base','_arm'),validate='one_to_one')
        for (cohort,condition),p in paired.groupby(['cohort','condition']):
            for target_group in ('all','signal'):
                q = p if target_group=='all' else p[p.target_is_signal_base.eq(True)]
                if not len(q):
                    continue
                complete = q[~q.failed_base & ~q.failed_arm]
                rows.append(dict(cohort=cohort,condition=condition,arm=arm,target_group=target_group,n_pairs=len(q),
                    complete_pairs=len(complete),baseline_failures=int(q.failed_base.sum()),arm_failures=int(q.failed_arm.sum()),
                    gains=int((~complete.rejected_base & complete.rejected_arm).sum()),
                    losses=int((complete.rejected_base & ~complete.rejected_arm).sum()),
                    recovered_baseline_failures=int((q.failed_base & ~q.failed_arm).sum()),
                    recovered_and_rejected=int((q.failed_base & q.rejected_arm).sum()),
                    median_abs_p_difference=float(abs(complete.p_value_base-complete.p_value_arm).median()) if len(complete) else np.nan))
    return pd.DataFrame(rows)


def expected_trials(m):
    for index,c in enumerate(m['conditions']):
        for shard in range(m['shards']):
            for iteration in range(1,allocation(m,c,shard)+1):
                X,y,mean,identity = fresh_input(m,index,shard,iteration)
                yield X,y,mean,identity
    fixtures = read_fixtures(m)
    for case_id in fixtures['replay_cases']:
        case = next(c for c in fixtures['cases'] if c['case_id']==case_id)
        for repeat in range(m['diagnostics']['replay_seeds']):
            yield diagnostic_input(m,case,repeat)


def audit_result(row,identity,arm,m):
    for key,value in jsonable(identity).items():
        require(row[key]==value,f'Input/seed mismatch: {identity["case_id"]}/{key}')
    require(row['arm']==arm['name'] and json.loads(row['settings_json'])==arm,'Arm settings mismatch')
    p = row['p_value']
    require(p is None or (np.isfinite(p) and 0<=p<=1),'Invalid p-value')
    require(row['failed']==(p is None) and row['rejected']==(p is not None and p<m['alpha']),'Decision mismatch')
    if arm['method']=='mcmc':
        expected = (arm['replicas']+int(arm['construction']=='parallel'))*arm['steps']
        require(row['proposals']==expected and p is not None,'Incomplete MCMC construction')
        require(abs(p*(arm['replicas']+1)-round(p*(arm['replicas']+1)))<1e-10,'Invalid rank grid')
        require(row['denominator_ess'] is None and row['tail_ess'] is None,'Replica count incorrectly labelled ESS')
    else:
        require(0<row['proposals']<=arm['final_cap'],'AIS cap mismatch')
        met = row['denominator_ess']>=m['ais']['min_denominator_ess'] and row['tail_ess']>=m['ais']['min_tail_ess']
        require((p is not None)==met,'AIS threshold mismatch')
        require(arm['early_stop'] or row['proposals']==arm['final_cap'],'Incomplete fixed AIS draws')


def verify(m,downloaded=False):
    out = ROOT/m['output_root'] if downloaded else lock(m)
    require(json.loads((out/'requested_manifest.json').read_text())==m,'Recorded manifest drift')
    rows,common,errors,geometry = [],[],[],[]
    regeneration_mismatches = []
    for X,y,mean,identity in expected_trials(m):
        path = trial_path(m,identity)
        try:
            if downloaded and identity['cohort']=='fresh':
                # Receipts describe the original server input. Floating-point
                # RNG transforms can differ across platforms; regeneration is
                # supplementary, not the checksum of the transferred bundle.
                recorded = json.loads((path/'common'/'bundle_receipt.json').read_text())['requested']['identity']
                recorded_hash = recorded['input_hash']
                require(isinstance(recorded_hash,str) and len(recorded_hash)==64
                        and all(c in '0123456789abcdef' for c in recorded_hash),'Invalid recorded input hash')
                if recorded_hash!=identity['input_hash']:
                    regeneration_mismatches.append(dict(case_id=identity['case_id'],
                        recorded_input_hash=recorded_hash,regenerated_input_hash=identity['input_hash']))
                    identity = dict(identity,input_hash=recorded_hash)
            pilot_row = read_bundle(path/'common',dict(identity=identity,ais=m['ais']))
            require(pilot_row['pilot_evaluations']==m['ais']['pilot_iters']*m['ais']['pilot_samples'],'Pilot count drift')
            common.append(dict(identity,**pilot_row))
            with np.load(path/'common'/'trace.npz',allow_pickle=False) as pilot:
                require(len(pilot['z_values'])==pilot_row['pilot_evaluations'],'Incomplete shared pilot')
            paired = []
            for arm in effective_arms(m,identity['cohort']):
                row = read_bundle(path/arm['name'],receipt_request(m,identity,arm))
                audit_result(row,identity,arm,m)
                selected = json.loads(row['observed_selected_features'])
                require(selected==sorted(set(selected)) and len(selected)==m['design']['k_select'],'Invalid selected set')
                require(row['target_feature']==target_from_selected_set(selected,identity['auxiliary_u']),'Target randomization mismatch')
                t,_ = _chi_statistic(y,_spline_effect_basis(X[:,row['target_feature']]),sigma=m['sigma'])
                require(np.isclose(row['t_obs'],t,rtol=1e-12,atol=1e-12),'Observed statistic mismatch')
                paired.append(tuple(row[k] for k in ('target_feature','t_obs','observed_selected_features','test_rank')))
                rows.append(row)
            require(len(set(paired))==1,'Unmatched arms on observed inputs')
            # Reference continuation shares its exact final proposal prefix.
            if identity['cohort']=='fresh':
                a,b = path/'ais_reference'/'trace.npz',path/'ais_reference_fixed'/'trace.npz'
                with np.load(a,allow_pickle=False) as early,np.load(b,allow_pickle=False) as fixed:
                    n = min(len(early['checkpoints']),len(fixed['checkpoints']))
                    np.testing.assert_allclose(early['checkpoints'][:n],fixed['checkpoints'][:n],rtol=1e-12,atol=1e-12,equal_nan=True)
        except Exception as exc:
            errors.append(dict(case_id=identity['case_id'],repeat=identity['repeat'],reason=str(exc)))
    for case in read_fixtures(m)['cases']:
        try:
            identity = dict(cohort='geometry',case_id=case['case_id'],input_hash=case['input_hash'])
            row = read_bundle(out/'geometry'/case['case_id'],dict(identity=identity,map_budget=m['diagnostics']['map_budget']))
            require(row['grid_evaluations']<=m['diagnostics']['map_budget'] and not row['regions_certified'],'Invalid geometry certification/budget')
            geometry.append(row)
        except Exception as exc:
            errors.append(dict(case_id=case['case_id'],repeat=-1,reason=str(exc)))
    error_name = 'local_verification_errors.csv' if downloaded else 'verification_errors.csv'
    pd.DataFrame(errors,columns=['case_id','repeat','reason']).to_csv(out/error_name,index=False)
    if downloaded:
        atomic_json(out/'local_input_regeneration_report.json',dict(
            scope='Local regeneration comparison; original bundle checksums and seed/arm identities remain strict.',
            byte_identical_fresh_inputs= sum(c['iterations'] for c in m['conditions'])-len(regeneration_mismatches),
            mismatches=regeneration_mismatches))
    require(not errors,f'{len(errors)} missing/invalid bundles; inspect {error_name}')
    frame = pd.DataFrame(rows)
    require(not frame.duplicated(['cohort','case_id','repeat','arm']).any(),'Duplicate results')
    fresh_expected = sum(c['iterations'] for c in m['conditions'])*len(m['arms'])
    diagnostic_expected = len(read_fixtures(m)['replay_cases'])*m['diagnostics']['replay_seeds']*len(effective_arms(m,'diagnostic'))
    require(len(frame)==fresh_expected+diagnostic_expected,'Missing result rows')
    common_frame = pd.DataFrame(common)
    outputs = {'unified_tests.csv':frame,'sampling_summary.csv':summary(frame,m),
               'paired_comparisons.csv':paired_comparisons(frame),
               'shared_pilot_costs.csv':common_frame,'geometry_summary.csv':pd.DataFrame(geometry) if geometry else pd.DataFrame(columns=['case_id','diagnostic_p','regions_certified']),
               'status_counts.csv':frame.groupby(['cohort','arm','condition','status'],dropna=False).size().rename('count').reset_index(),
               'runtime_summary.csv':frame.groupby(['cohort','arm']).agg(
                   tests=('arm','size'),candidate_evaluations=('selector_calls','sum'),
                   uncached_selector_fits=('uncached_selector_fits','sum'),
                   worker_seconds=('elapsed_seconds','sum'),cpu_seconds=('cpu_seconds','sum')).reset_index()}
    for name,data in outputs.items():
        if downloaded:
            saved = pd.read_csv(out/name,keep_default_na=False,na_values=[''])
            # Round trip normalizes JSON null columns, CSV object strings and
            # booleans exactly as when saved originally.
            import io
            expected = pd.read_csv(io.StringIO(data.to_csv(index=False)),keep_default_na=False,na_values=[''])
            pd.testing.assert_frame_equal(saved,expected,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        else:
            data.to_csv(out/name,index=False)
    if not downloaded:
        atomic_json(out/'budget_actual.json',dict(
            candidate_evaluations=int(frame.selector_calls.sum()+common_frame.selector_calls.sum()+sum(g['selector_calls'] for g in geometry)),
            uncached_selector_fits=int(frame.uncached_selector_fits.sum()+common_frame.uncached_selector_fits.sum()+sum(g['uncached_selector_fits'] for g in geometry)),
            reference_candidate_evaluations=m['budget_reference']['recorded_candidate_evaluations'],
            shared_pilots_counted_once=True,concurrent_worker_seconds_are_not_elapsed_time=True))
        (out/'COMPLETED').write_text(f'Verified {fresh_expected} fresh tests, {diagnostic_expected} selected-case replays, and {len(geometry)} diagnostic maps.\nUnresolved AIS outcomes are retained. Completion does not establish calibration.\n')
    require((out/'COMPLETED').exists(),'Missing completion marker')
    print((out/'COMPLETED').read_text())
    if downloaded and regeneration_mismatches:
        print(f'{len(regeneration_mismatches)} fresh inputs regenerated with different byte hashes; '
              'see local_input_regeneration_report.json. Original server identities retained; '
              'bundle checksums, seeds, observed statistics and paired arms verified.')


def verify_downloaded(m):
    out = ROOT/m['output_root']
    saved = json.loads((out/'source_hashes.json').read_text())
    require(isinstance(saved,dict) and bool(saved) and all(
        isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)
        for h in saved.values()),'Invalid recorded source hashes')
    current = hashes(m)
    report = dict(original_provenance_preserved=True,
                  changed_recorded_files=sorted(p for p in saved if saved[p]!=current.get(p)),
                  additional_local_files=sorted(set(current)-set(saved)))
    try:
        verify(m,downloaded=True)
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',reason=str(exc))
        raise
    finally:
        atomic_json(out/'local_verification_report.json',report)


def validate_archive(path,m):
    with tarfile.open(path) as archive:
        for entry in archive.getmembers():
            name=PurePosixPath(entry.name)
            require(bool(name.parts) and not name.is_absolute() and '..' not in name.parts
                    and name.parts[0]==m['experiment'] and '\\' not in entry.name and ':' not in entry.name
                    and (entry.isfile() or entry.isdir()),f'Unsafe archive entry: {entry.name}')


def status(m):
    out = ROOT/m['output_root']
    require(out.exists(),'Experiment has not been launched')
    for cohort in ('fresh','diagnostic'):
        root = out/'runs'/cohort
        receipts = list(root.rglob('bundle_receipt.json')) if root.exists() else []
        for a in effective_arms(m,cohort):
            count = sum(p.parent.name==a['name'] for p in receipts)
            expected = (sum(c['iterations'] for c in m['conditions']) if cohort=='fresh'
                        else len(read_fixtures(m)['replay_cases'])*m['diagnostics']['replay_seeds'])
            print(f"{cohort}/{a['name']}: {count}/{expected} arm receipts")
    maps = list((out/'geometry').rglob('bundle_receipt.json')) if (out/'geometry').exists() else []
    print(f"geometry: {len(maps)}/{len(read_fixtures(m)['cases'])} maps")
    if (out/'COMPLETED').exists():
        print((out/'COMPLETED').read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['check','status','run','worker','verify','verify-downloaded','validate-archive'])
    parser.add_argument('--cohort',choices=['fresh','diagnostic','geometry'])
    parser.add_argument('--index',type=int)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--archive')
    args=parser.parse_args()
    m=read_manifest()
    if args.action=='check':
        print(json.dumps(dict(experiment=m['experiment'],fresh_inputs=1000,fresh_tests=7000,
            diagnostic_maps=28,diagnostic_replays=96,max_workers=32,budget=m['budget_reference']),indent=2))
    elif args.action=='status':
        status(m)
    elif args.action=='run':
        lock(m)
        dispatch(m,list(jobs(m)))
        dispatch(m,[('geometry',i,0) for i in range(len(read_fixtures(m)['cases']))])
        verify(m)
    elif args.action=='worker':
        require(args.cohort is not None and args.index is not None,'Worker requires cohort/index')
        worker(m,args.cohort,args.index,args.shard)
    elif args.action=='verify':
        verify(m)
    elif args.action=='verify-downloaded':
        verify_downloaded(m)
    else:
        require(args.archive is not None,'Missing archive path')
        validate_archive(args.archive,m)


if __name__=='__main__':
    main()
