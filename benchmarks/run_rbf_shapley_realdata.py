"""Analytic RBF real-data benchmark: paired n=100,150,200; at most 64 CPUs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import importlib.metadata as metadata
import json
from math import comb
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import run_rbf_shapley as core  # Sets numerical thread caps before NumPy import.
import numpy as np
from rbf_network import AdditiveRBFNetwork
from threadpoolctl import threadpool_info, threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
NAME='rbf_shapley_realdata_n100_150_200_cpu64_1000_20261024'
SMOKE=NAME.replace('_1000_','_smoke_')
DEFAULT=ROOT/'benchmarks'/f'{NAME}.json'
FROZEN={NAME:'20b92b00f5adf85ba876666ef571e537a053723e444101d2c8a3603792aee57e',
        SMOKE:'95553cf0d2ffe40c098cfc607bbb04d629beb0d014e0004337d53db84cd40d9b'}
DATASETS=('heating_load','cooling_load','gas_turbine','red_wine','white_wine','abalone','concrete','housing')
LABELS={'selected_shapley':'Selected-Features Shapley Effect','full_shapley':'All-Features Shapley Effect'}
PACKAGED=('benchmarks/run_rbf_shapley_realdata.py','benchmarks/run_rbf_shapley.py','benchmarks/rbf_network.py',
          'benchmarks/table_dataset.pkl','benchmarks/process.ipynb',f'benchmarks/{NAME}.json',f'benchmarks/{SMOKE}.json',
          'examples/rbf_shapley_realdata_cpu64.ps1','examples/plot_rbf_shapley_realdata.py',
          'docs/rbf_shapley_realdata_study.md','docs/commands.md','tests/test_rbf_shapley_realdata.py')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path=DEFAULT):
    m=core.read(path)
    core.require(core.sha(m)==FROZEN.get(m['experiment']),'Manifest changed; use a new frozen experiment')
    core.require(1<=m['workers']<=64 and m['threads_per_worker']==1,'CPU budget exceeded')
    core.require(m['output_root']=='outputs/benchmarks/'+m['experiment'],'Unsafe output root')
    return m


def source_hashes(): return {name:digest(ROOT/name) for name in PACKAGED}


def provenance(m):
    for p,v in m['required_versions'].items(): core.require(metadata.version(p)==v,f'Dependency mismatch: {p}')
    core.require(digest(ROOT/m['input_path'])==m['input_sha256'],'Dataset checksum mismatch')
    return dict(manifest_sha256=core.sha(m),sources=source_hashes(),versions={p:metadata.version(p) for p in m['required_versions']},
                python=sys.version,platform=platform.platform())


@lru_cache(maxsize=1)
def load_data(path,expected_hash):
    # Only the repository's explicitly pinned prepared pickle is accepted.
    core.require(digest(path)==expected_hash,'Dataset checksum mismatch')
    with Path(path).open('rb') as f: data=pickle.load(f)
    result={}
    for name in DATASETS:
        X,y=(np.asarray(v,dtype=float) for v in data[name])
        core.require(X.ndim==2 and y.shape==(len(X),) and len(X)>=200,'Invalid dataset shape')
        core.require(7<=X.shape[1]<=11 and np.isfinite(X).all() and np.isfinite(y).all(),'Invalid dataset values')
        result[name]=(X,y)
    return result


class Geometry:
    """Exact coalition geometry without caching every feature's residual basis.

    Keeps one projector per coalition and builds support only for the tested
    feature. Unlike the older pilot Geometry, caches are local to each sample.
    """
    def __init__(self,blocks,tol=1e-10):
        self.tol,self.p=tol,len(blocks)
        self.Q=core.orthobasis(np.column_stack(blocks),tol);self.r=self.Q.shape[1]
        self.blocks=[self.Q.T@b for b in blocks]
        self.projectors=[]
        for mask in range(1<<self.p):
            cols=[b for j,b in enumerate(self.blocks) if mask&(1<<j)]
            B=core.orthobasis(np.column_stack(cols),tol) if cols else np.empty((self.r,0))
            self.projectors.append(B@B.T)
        self.selector=np.zeros((self.p,self.r,self.r))
        for j in range(self.p):
            for mask in range(1<<self.p):
                if not mask&(1<<j):
                    self.selector[j]+=(self.projectors[mask|(1<<j)]-self.projectors[mask])/(self.p*comb(self.p-1,mask.bit_count()))
        core.require(np.allclose(self.selector.sum(axis=0),np.eye(self.r),atol=1e-8),'Shapley efficiency failure')
        self.cache={}

    def select(self,y,k=2):
        scores=np.einsum('i,jik,k->j',y,self.selector,y)
        order=np.lexsort((np.arange(self.p),-scores))
        return tuple(sorted(int(j) for j in order[:k]))

    def target(self,arm,M,j):
        key=(arm,M,j)
        if key in self.cache: return self.cache[key]
        U=M if arm=='selected_shapley' else (1<<self.p)-1
        size=U.bit_count();directions=[];A=np.zeros((self.r,self.r))
        for mask in range(1<<self.p):
            if mask&~U or mask&(1<<j): continue
            D=self.projectors[mask|(1<<j)]-self.projectors[mask]
            A+=D/(size*comb(size-1,mask.bit_count()))
            directions.append(core.orthobasis((np.eye(self.r)-self.projectors[mask])@self.blocks[j],self.tol))
        B=core.orthobasis(np.column_stack(directions),self.tol)
        core.require(np.allclose(B@B.T@A,A,atol=1e-8),'Target support failure')
        self.cache[key]=(A,B)
        return A,B


def seeds(m,name,repetition):
    di=DATASETS.index(name)
    return {key:int(np.random.SeedSequence([m['seed'],di,repetition,stream]).generate_state(1)[0])
            for stream,key in [(1,'subset_seed'),(2,'target_seed')]}


def sample_inputs(m,name,repetition):
    X,y=load_data(str(ROOT/m['input_path']),m['input_sha256'])[name]
    s=seeds(m,name,repetition)
    order=np.random.default_rng(s['subset_seed']).permutation(len(y))
    u=float(np.random.default_rng(s['target_seed']).uniform())
    return [dict(dataset=name,repetition=repetition,n=n,indices=sorted(int(v) for v in order[:n]),
                 auxiliary_u=u,**s) for n in m['sample_sizes']]


def evaluate(m,inp):
    Xall,yall=load_data(str(ROOT/m['input_path']),m['input_sha256'])[inp['dataset']]
    idx=inp['indices'];X,y=Xall[idx],yall[idx]
    identity=dict(dataset=inp['dataset'],repetition=inp['repetition'],n=inp['n'],p=X.shape[1],
                  subset_seed=inp['subset_seed'],target_seed=inp['target_seed'],auxiliary_u=inp['auxiliary_u'],
                  rows_sha256=core.sha(idx),input_sha256=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),
                  method='analytic',null_true=None)
    start=time.perf_counter();common={};rows=[]
    try:
        net=AdditiveRBFNetwork(m['center_quantiles'],m['width_multiplier']).fit_basis(X)
        g=Geometry(net.transform_blocks(X),m['rank_tolerance']);yr=g.Q.T@y
        selected=g.select(yr,m['k']);target=selected[int(inp['auxiliary_u']*m['k'])]
        mask=sum(1<<j for j in selected)
        common=dict(selected=list(selected),target=target,full_rank=g.r)
    except Exception as e:
        return [dict(**identity,arm=arm,status='error',p_value=None,rejected=None,error=f'{type(e).__name__}: {e}',
                     elapsed_seconds=time.perf_counter()-start) for arm in m['targets']]
    setup_seconds=time.perf_counter()-start
    for arm in m['targets']:
        arm_start=time.perf_counter()
        try:
            A,B=g.target(arm,mask,target)
            result=core.radial_test(g,yr,B,selected,m['sigma'])
            p=result.pop('p_selective');untruncated=result.pop('p_naive')
            core.require(np.isfinite(p) and 0<=p<=1,'Invalid analytic p-value')
            row=dict(**identity,**common,arm=arm,status='ok',error='',p_value=p,rejected=p<m['primary_alpha'],
                     rank=B.shape[1],observed_shapley_score=float(yr@A@yr),untruncated_reference_p=untruncated,**result)
        except Exception as e:
            row=dict(**identity,**common,arm=arm,status='error',p_value=None,rejected=None,error=f'{type(e).__name__}: {e}')
        row.update(setup_seconds=setup_seconds,inference_seconds=time.perf_counter()-arm_start,
                   elapsed_seconds=time.perf_counter()-start)
        rows.append(row)
    return rows


def jobs(m):
    return [(name,start,min(start+m['batch_size'],m['repetitions'])) for name in m['datasets']
            for start in range(0,m['repetitions'],m['batch_size'])]


def job_name(job): return f'{job[0]}_{job[1]:05}_{job[2]:05}.json'


def run_job(m,job,out):
    inputs,rows=[],[]
    with threadpool_limits(limits=1):
        for repetition in range(job[1],job[2]):
            for inp in sample_inputs(m,job[0],repetition):
                rows.extend(evaluate(m,inp));inputs.append(inp)
        core.require(all(v['num_threads']<=1 for v in threadpool_info()),'Worker numerical thread cap failed')
    receipt=dict(manifest_sha256=core.sha(m),job=list(job),inputs=inputs,rows=rows,
                 inputs_sha256=core.sha(inputs),rows_sha256=core.sha(rows))
    core.atomic(Path(out)/'jobs'/job_name(job),receipt)
    return job_name(job)


def validate_receipt(m,job,receipt,recompute=False):
    core.require(receipt['manifest_sha256']==core.sha(m) and receipt['job']==list(job),'Receipt identity mismatch')
    rows,inputs=receipt['rows'],receipt['inputs']
    core.require(core.sha(rows)==receipt['rows_sha256'] and core.sha(inputs)==receipt['inputs_sha256'],'Receipt checksum mismatch')
    expected={(rep,n,arm) for rep in range(job[1],job[2]) for n in m['sample_sizes'] for arm in m['targets']}
    core.require(len(rows)==len(expected) and {(r['repetition'],r['n'],r['arm']) for r in rows}==expected,'Missing/duplicate tests')
    core.require(len(inputs)==(job[2]-job[1])*len(m['sample_sizes']),'Input count mismatch')
    saved={(r['repetition'],r['n'],r['arm']):r for r in rows}
    input_map={(v['repetition'],v['n']):v for v in inputs}
    for rep in range(job[1],job[2]):
        for inp in sample_inputs(m,job[0],rep):
            core.require(input_map[rep,inp['n']]==inp,'Input indices/seed mismatch')
            pair=[saved[rep,inp['n'],a] for a in m['targets']]
            for r in pair:
                for key in ('dataset','repetition','n','subset_seed','target_seed','auxiliary_u'):
                    core.require(r[key]==inp[key],f'Input field mismatch: {key}')
                core.require(r['rows_sha256']==core.sha(inp['indices']),'Row index hash mismatch')
                core.require(r['method']=='analytic' and r['null_true'] is None,'Method/null-truth mismatch')
                core.require(r['status'] in ('ok','error'),'Invalid status')
                if r['status']=='ok':
                    core.require(np.isfinite(r['p_value']) and 0<=r['p_value']<=1,'Invalid p-value')
                    core.require(r['rejected']==(r['p_value']<m['primary_alpha']),'Decision mismatch')
                    core.require(r['selected']==sorted(set(r['selected'])) and len(r['selected'])==m['k'],'Bad selection')
                    core.require(r['target']==r['selected'][int(inp['auxiliary_u']*m['k'])],'Target mismatch')
                    core.require(0<r['rank']<=r['full_rank']<=3*r['p'],'Invalid rank')
                else: core.require(bool(r.get('error')),'Missing failure diagnostic')
            for key in ('input_sha256','rows_sha256','selected','target','full_rank'):
                core.require(pair[0].get(key)==pair[1].get(key),f'Unpaired {key}')
            if recompute and rep in m['verification_repetitions']:
                for a in evaluate(m,inp):
                    b=saved[rep,inp['n'],a['arm']]
                    for key in ('status','selected','target','rank','full_rank','input_sha256'):
                        core.require(a.get(key)==b.get(key),f'Recomputed {key} mismatch')
                    if a['status']=='ok':
                        for key in ('p_value','observed_shapley_score','radius','log_selection_mass'):
                            core.require(np.isclose(a[key],b[key],rtol=1e-6,atol=1e-8),f'Recomputed {key} mismatch')
    return rows


def summarize(m,rows,out):
    summary=[]
    for name in m['datasets']:
        for n in m['sample_sizes']:
            for arm in m['targets']:
                group=[r for r in rows if (r['dataset'],r['n'],r['arm'])==(name,n,arm)]
                ok=[r for r in group if r['status']=='ok']
                for alpha in m['alpha_levels']:
                    k=sum(r['p_value']<alpha for r in ok)
                    conditional=core.binomial(k,len(ok));attempted=core.binomial(k,len(group))
                    summary.append(dict(dataset=name,n=n,arm=arm,alpha=alpha,attempted=len(group),resolved=len(ok),
                        failures=len(group)-len(ok),rejections=k,resolved_rejection_rate=conditional['rate'],
                        resolved_ci_lower=conditional['ci_lower'],resolved_ci_upper=conditional['ci_upper'],
                        detection_fraction=attempted['rate'],detection_ci_lower=attempted['ci_lower'],detection_ci_upper=attempted['ci_upper'],
                        unresolved_upper_fraction=(k+len(group)-len(ok))/len(group)))
    core.write_csv(out/'results.csv',rows);core.write_csv(out/'summary.csv',summary)
    targeting=[]
    for name in m['datasets']:
        for n in m['sample_sizes']:
            g=[r for r in rows if r['dataset']==name and r['n']==n and r['arm']==m['targets'][0] and 'target' in r]
            for j in range(load_data(str(ROOT/m['input_path']),m['input_sha256'])[name][0].shape[1]):
                targeting.append(dict(dataset=name,n=n,target=j,target_count=sum(r['target']==j for r in g),
                                      selection_resolved=len(g),attempted=m['repetitions']))
    core.write_csv(out/'targeting_summary.csv',targeting)


def setup(m,out):
    info=provenance(m)
    if (out/'requested_manifest.json').exists():
        core.require(core.read(out/'requested_manifest.json')==m,'Settings drift on resume')
        old=core.read(out/'provenance.json')
        core.require(old['sources']==info['sources'] and old['versions']==info['versions'],'Source/dependency drift on resume')
    else:
        core.require(not any((out/'jobs').glob('*.json')),'Orphan receipts')
        core.atomic(out/'requested_manifest.json',m);core.atomic(out/'provenance.json',info)
    (out/'jobs').mkdir(exist_ok=True)


def verify(m,out,export=False):
    core.require(core.read(out/'requested_manifest.json')==m,'Manifest mismatch')
    old=core.read(out/'provenance.json')
    core.require(old['sources']==source_hashes() and old['manifest_sha256']==core.sha(m),'Source provenance mismatch')
    core.require({p.name for p in (out/'jobs').glob('*.json')}=={job_name(j) for j in jobs(m)},'Incomplete/extra batches')
    rows=[];hashes={}
    with threadpool_limits(limits=1):
        for job in jobs(m):
            rec=core.read(out/'jobs'/job_name(job));rows.extend(validate_receipt(m,job,rec,recompute=True))
            hashes[job_name(job)]=(rec['inputs_sha256'],rec['rows_sha256'])
    expected=len(m['datasets'])*len(m['sample_sizes'])*m['repetitions']*2
    core.require(len(rows)==expected,'Wrong total count')
    failures=sum(r['status']!='ok' for r in rows)
    report=dict(status='verified',total_tests=len(rows),subdatasets=len(rows)//2,failures=failures,
                manifest_sha256=core.sha(m),receipts_sha256=core.sha(hashes),
                checks='All coverage, row indices, seeds, pairing, decisions, checksums; numerical reconstruction at first/last repetition per dataset and sample size',
                interpretation='Observed real-data selective rejection rates; null truth unknown; sigma=1 is a working scale')
    if export:
        summarize(m,rows,out);core.atomic(out/'verification.json',report)
        core.atomic(out/'COMPLETED',dict(**report,export_sha256={p.name:digest(p) for p in out.glob('*.csv')}))
    else:
        done=core.read(out/'COMPLETED');core.require(done['receipts_sha256']==report['receipts_sha256'],'Completion mismatch')
        for name,sha in done['export_sha256'].items():
            core.require(Path(name).name==name and digest(out/name)==sha,'Export checksum mismatch')
        # Recreate summaries from the checksummed receipts, rather than trust CSV hashes alone.
        import tempfile
        import csv
        with tempfile.TemporaryDirectory() as temp:
            summarize(m,rows,Path(temp))
            for path in Path(temp).glob('*.csv'):
                with path.open(newline='') as f,(out/path.name).open(newline='') as g:
                    left,right=list(csv.DictReader(f)),list(csv.DictReader(g))
                core.require(len(left)==len(right),'Export length mismatch')
                for a,b in zip(left,right):
                    core.require(a.keys()==b.keys(),'Export schema mismatch')
                    for key in a:
                        if a[key]!=b[key]:
                            try: equal=np.isclose(float(a[key]),float(b[key]),rtol=1e-6,atol=1e-8)
                            except ValueError: equal=False
                            core.require(equal,f'Recomputed export mismatch: {key}')
    return report


def worker_count(m):
    allowed=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    # ProcessPoolExecutor on Windows supports at most 61 processes.
    count=min(m['workers'],len(allowed),61 if os.name=='nt' else 64)
    if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,allowed[:count])
    return count


def run(m,out):
    with core.exclusive(out):
        setup(m,out)
        if (out/'COMPLETED').exists(): print(json.dumps(verify(m,out)),flush=True);return
        workers=worker_count(m)
        core.atomic(out/'runtime.json',dict(pid=os.getpid(),workers=workers,cpu_budget=m['workers'],threads_per_worker=1,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            thread_limits={k:os.environ[k] for k in core.THREAD_KEYS},started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        pending=[]
        for job in jobs(m):
            path=out/'jobs'/job_name(job)
            if path.exists(): validate_receipt(m,job,core.read(path))
            else: pending.append(job)
        print(f'Pending batches: {len(pending)}/{len(jobs(m))}; workers={workers}',flush=True)
        with threadpool_limits(limits=1),ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_job,m,j,str(out)) for j in pending]
            for i,f in enumerate(as_completed(futures),1): print(f'Committed {i}/{len(pending)}: {f.result()}',flush=True)
        print(json.dumps(verify(m,out,export=True)),flush=True)


def profile(m):
    timings=[]
    with threadpool_limits(limits=1):
        for name in m['datasets']:
            for inp in sample_inputs(m,name,0):
                start=time.perf_counter();rows=evaluate(m,inp)
                entry=dict(dataset=name,n=inp['n'],seconds=time.perf_counter()-start,
                           statuses=[r['status'] for r in rows],errors=[r.get('error') for r in rows if r['status']!='ok'])
                print(json.dumps(entry),flush=True);timings.append(entry)
    core.require(all(all(s=='ok' for s in e['statuses']) for e in timings),'Representative cost profile has failed tests')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','profile','run','verify','status','validate-archive'])
    parser.add_argument('--config',type=Path,default=DEFAULT);parser.add_argument('--output',type=Path);parser.add_argument('--archive',type=Path)
    args=parser.parse_args();m=read_config(args.config);out=args.output or ROOT/m['output_root']
    if args.action=='check':
        provenance(m);data=load_data(str(ROOT/m['input_path']),m['input_sha256'])
        print(json.dumps(dict(status='preflight_ok',cpu_budget=m['workers'],sample_sizes=m['sample_sizes'],
            total_tests=len(m['datasets'])*len(m['sample_sizes'])*m['repetitions']*2,
            datasets={n:list(data[n][0].shape) for n in m['datasets']}),indent=2))
    elif args.action=='profile': provenance(m);profile(m)
    elif args.action=='run': run(m,out)
    elif args.action=='verify':
        with core.exclusive(out): print(json.dumps(verify(m,out),indent=2))
    elif args.action=='status':
        print(json.dumps(dict(committed_batches=len(list((out/'jobs').glob('*.json'))),expected_batches=len(jobs(m)),complete=(out/'COMPLETED').exists())))
    else: core.validate_archive(m,args.archive);print('Archive paths and member types validated')


if __name__=='__main__': main()
