"""Frozen, resumable fresh-design RBF Shapley analytic power study."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

# This import caps numerical threads before NumPy/SciPy are imported.
import run_rbf_shapley as core
import numpy as np
from rbf_network import AdditiveRBFNetwork
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rbf_shapley_analytic_power_cpu32_1000_20261024'
SMOKE = NAME.replace('_1000_', '_smoke_')
DEFAULT = ROOT/'benchmarks'/f'{NAME}.json'
FROZEN = {NAME: 'dd6bb9b23381f3f772f25bddcb725a1408bb4bf95a33a94db82abe650a1a2b87',
          SMOKE: '04cf7c2249ba8e8bec3c56949e575787de86df8e17e9f4effe2658936849de0a'}
PACKAGED = ('benchmarks/run_rbf_shapley_analytic_power.py',
            'benchmarks/run_rbf_shapley.py', 'benchmarks/rbf_network.py',
            f'benchmarks/{NAME}.json', f'benchmarks/{SMOKE}.json',
            'examples/rbf_shapley_analytic_power_cpu32.ps1',
            'examples/plot_rbf_shapley_analytic_power.py',
            'docs/rbf_shapley_analytic_power_study.md', 'docs/commands.md',
            'tests/test_rbf_shapley_analytic_power.py')
LABELS = {'selected_shapley': 'Selected-Features Shapley Effect',
          'full_shapley': 'All-Features Shapley Effect'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path=DEFAULT):
    m = core.read(path)
    core.require(FROZEN.get(m['experiment']) == core.sha(m), 'Manifest changed; create a newly named frozen experiment')
    core.require(1 <= m['workers'] <= 32 and m['threads_per_worker'] == 1, 'CPU budget exceeded')
    core.require(m['output_root'] == 'outputs/benchmarks/'+m['experiment'], 'Unsafe output root')
    return m


def source_hashes():
    return {name: digest(ROOT/name) for name in PACKAGED}


def provenance(m):
    for name, version in m['required_versions'].items():
        core.require(metadata.version(name) == version, f'Dependency mismatch: {name}')
    return dict(manifest_sha256=core.sha(m), sources=source_hashes(),
                versions={name: metadata.version(name) for name in m['required_versions']},
                python=sys.version, platform=platform.platform())


def input_seed(m, stream, iteration):
    return int(np.random.SeedSequence([m['seed'], stream, iteration]).generate_state(1)[0])


def standardized_effect(x):
    """Frozen copy of si_shap.power._nonlinear_effect used by the reference."""
    raw = x + .5*(x*x-1.)
    centered = raw-raw.mean()
    scale = float(centered.std())
    core.require(np.isfinite(scale) and scale > np.finfo(float).eps, 'Degenerate signal')
    return centered/scale


def make_input(m, iteration):
    data_seed, target_seed = input_seed(m, 1, iteration), input_seed(m, 2, iteration)
    rng = np.random.default_rng(data_seed)
    X = rng.standard_normal((m['n'], m['p']))
    noise = rng.standard_normal(m['n'])
    u = float(np.random.default_rng(target_seed).uniform())
    return dict(iteration=iteration, data_seed=data_seed, target_seed=target_seed,
                X=X.tolist(), noise=noise.tolist(), auxiliary_u=u)


def trial_rows(m, saved_input):
    X, noise = np.asarray(saved_input['X']), np.asarray(saved_input['noise'])
    core.require(X.shape == (m['n'], m['p']) and noise.shape == (m['n'],), 'Invalid input shapes')
    core.require(np.isfinite(X).all() and np.isfinite(noise).all(), 'Nonfinite input')
    net = AdditiveRBFNetwork(m['center_quantiles'], m['width_multiplier']).fit_basis(X)
    g = core.Geometry(net.transform_blocks(X), m['rank_tolerance'])
    effect = standardized_effect(X[:, m['signal_feature']])
    u = saved_input['auxiliary_u']
    xhash = hashlib.sha256(X.tobytes()).hexdigest()
    rows = []
    for ci, beta in enumerate(m['betas']):
        mu = beta*effect
        y = mu+m['sigma']*noise
        yr, mur = g.Q.T@y, g.Q.T@mu
        selected = g.select(yr, m['k'])
        target = selected[int(u*len(selected))]
        mask = sum(1 << j for j in selected)
        yhash = hashlib.sha256(y.tobytes()).hexdigest()
        for arm in m['targets']:
            A, B = g.target(arm, mask, target)
            result = core.radial_test(g, yr, B, selected, m['sigma'])
            # There is only one inferential method: the analytic selected tail.
            p = result.pop('p_selective')
            untruncated = result.pop('p_naive')
            signal_norm = float(np.linalg.norm(B.T@mur))
            rows.append(dict(iteration=saved_input['iteration'], condition_index=ci, beta=beta,
                condition='global_null' if beta == 0 else f'beta_{beta:g}',
                arm=arm, method='analytic', design=m['design'], basis=m['basis'],
                data_seed=saved_input['data_seed'], target_seed=saved_input['target_seed'],
                auxiliary_u=u, x_sha256=xhash, response_sha256=yhash,
                selected=list(selected), selected_mask=mask, target=target,
                target_is_x0=target == 0, rank=B.shape[1], full_rank=g.r,
                p_value=p, rejected=p < m['primary_alpha'],
                untruncated_reference_p=untruncated, mean_contribution=max(0., float(mur@A@mur)),
                projected_mean_norm=signal_norm,
                null_true=bool(signal_norm <= m['null_tolerance']*max(1., np.linalg.norm(mu))), **result))
    return rows


def jobs(m):
    return [(start, min(start+m['batch_size'], m['iterations']))
            for start in range(0, m['iterations'], m['batch_size'])]


def job_name(job):
    return f'batch_{job[0]:05}_{job[1]:05}.json'


def run_job(m, job, out):
    inputs, rows = [], []
    with threadpool_limits(limits=1):
        for i in range(*job):
            saved = make_input(m, i)
            rows.extend(trial_rows(m, saved)); inputs.append(saved)
        core.require(all(t['num_threads'] <= 1 for t in threadpool_info()), 'Numerical thread cap failed')
    receipt = dict(manifest_sha256=core.sha(m), job=list(job), inputs=inputs, rows=rows,
                   inputs_sha256=core.sha(inputs), rows_sha256=core.sha(rows))
    core.atomic(Path(out)/'jobs'/job_name(job), receipt)
    return job_name(job)


def validate_receipt(m, job, receipt, recompute=False):
    core.require(receipt['manifest_sha256'] == core.sha(m) and receipt['job'] == list(job), 'Receipt identity mismatch')
    rows, inputs = receipt['rows'], receipt['inputs']
    core.require(core.sha(rows) == receipt['rows_sha256'] and core.sha(inputs) == receipt['inputs_sha256'], 'Receipt checksum mismatch')
    expected = {(i, ci, a) for i in range(*job) for ci in range(len(m['betas'])) for a in m['targets']}
    core.require(len(rows) == len(expected) and {(r['iteration'],r['condition_index'],r['arm']) for r in rows} == expected,
                 'Missing or duplicate trial arms')
    core.require([v['iteration'] for v in inputs] == list(range(*job)), 'Missing inputs')
    indexed = {(r['iteration'],r['condition_index'],r['arm']): r for r in rows}
    for saved in inputs:
        i = saved['iteration']
        core.require(saved['data_seed'] == input_seed(m,1,i) and saved['target_seed'] == input_seed(m,2,i), 'Input seed mismatch')
        core.require(saved['auxiliary_u'] == float(np.random.default_rng(saved['target_seed']).uniform()), 'Target randomization mismatch')
        # Saved raw arrays are authoritative for cross-platform verification;
        # seed reconstruction is checked with numerical tolerances.
        regenerated = make_input(m, i)
        core.require(np.allclose(saved['X'], regenerated['X'], rtol=1e-13, atol=1e-13) and
                     np.allclose(saved['noise'], regenerated['noise'], rtol=1e-13, atol=1e-13), 'Input seed reconstruction mismatch')
        if recompute:
            recreated = trial_rows(m, saved)
            for a in recreated:
                b = indexed[i,a['condition_index'],a['arm']]
                for field in ('iteration','condition_index','beta','condition','arm','method','design','basis',
                              'data_seed','target_seed','auxiliary_u','x_sha256','selected',
                              'selected_mask','target','target_is_x0','rank','full_rank','null_true'):
                    core.require(a[field] == b[field], f'Recomputed {field} mismatch')
                for field in ('p_value','untruncated_reference_p','mean_contribution','radius','projected_mean_norm','log_selection_mass'):
                    core.require(np.isclose(a[field],b[field],rtol=1e-7,atol=1e-9), f'Recomputed {field} mismatch')
        for ci,beta in enumerate(m['betas']):
            pair=[indexed[i,ci,a] for a in m['targets']]
            for r in pair:
                core.require(r['beta'] == beta and r['method'] == 'analytic', 'Method/condition drift')
                core.require(np.isfinite(r['p_value']) and 0 <= r['p_value'] <= 1, 'Invalid p-value')
                core.require(r['rejected'] == (r['p_value'] < m['primary_alpha']), 'Decision mismatch')
                core.require(r['selected'] == sorted(set(r['selected'])) and len(r['selected']) == m['k'], 'Invalid selected set')
                core.require(r['target'] == r['selected'][int(saved['auxiliary_u']*m['k'])], 'Wrong target')
                core.require(r['data_seed'] == saved['data_seed'] and r['target_seed'] == saved['target_seed']
                             and r['auxiliary_u'] == saved['auxiliary_u'], 'Saved input identity mismatch')
                core.require(0 < r['rank'] <= r['full_rank'] <= m['p']*m['rbf_units'], 'Invalid rank')
                core.require(beta != 0 or r['null_true'], 'Global-null truth error')
            for field in ('selected','target','response_sha256','x_sha256'):
                core.require(pair[0][field] == pair[1][field], f'Unpaired {field}')
        core.require(len({indexed[i,ci,a]['x_sha256'] for ci in range(len(m['betas'])) for a in m['targets']}) == 1,
                     'Design not shared across strengths')
    return rows


def export_summaries(m, rows, out):
    power, calibration, truth_rows, paired = [], [], [], []
    for beta in m['betas']:
        groups = {}
        for arm in m['targets']:
            group = [r for r in rows if r['beta'] == beta and r['arm'] == arm]
            groups[arm] = group
            core.require(len(group) == m['iterations'], 'Wrong per-condition count')
            for alpha in m['alpha_levels']:
                if beta == 0:
                    calibration.append(dict(beta=beta, arm=arm, alpha=alpha,
                        **core.binomial(sum(r['p_value'] < alpha for r in group),len(group))))
                for truth in (True,False):
                    subset = [r for r in group if r['null_true'] == truth]
                    truth_rows.append(dict(beta=beta, arm=arm, alpha=alpha, null_true=truth,
                        **core.binomial(sum(r['p_value'] < alpha for r in subset),len(subset))))
            if beta > 0:
                target = [r for r in group if r['target'] == 0]
                other = [r for r in group if r['target'] != 0]
                detected = sum(r['rejected'] for r in target)
                counts = [('targeting_probability',len(target),len(group)),
                          ('conditional_signal_detection',detected,len(target)),
                          ('marginal_signal_detection',detected,len(group)),
                          ('non_x0_rejection',sum(r['rejected'] for r in other),len(other)),
                          ('all_signal_rejection',sum(r['rejected'] for r in group),len(group))]
                for metric,k,n in counts:
                    power.append(dict(beta=beta,arm=arm,metric=metric,**core.binomial(k,n)))
        a,b=([r for r in groups[arm]] for arm in m['targets'])
        # Rows are exported in iteration order for each arm.
        paired.append(dict(beta=beta,n=len(a),decisions_disagree=sum(x['rejected'] != y['rejected'] for x,y in zip(a,b)),
                           null_truth_disagrees=sum(x['null_true'] != y['null_true'] for x,y in zip(a,b))))
    for name, data in [('results.csv',rows),('power_summary.csv',power),('calibration_summary.csv',calibration),
                       ('null_truth_summary.csv',truth_rows),('paired_target_comparisons.csv',paired)]:
        core.write_csv(out/name,data)


def setup_output(m,out):
    info = provenance(m)
    if (out/'requested_manifest.json').exists():
        core.require(core.read(out/'requested_manifest.json') == m, 'Saved settings changed')
        prior=core.read(out/'provenance.json')
        core.require(prior['sources'] == info['sources'] and prior['versions'] == info['versions'], 'Source/dependency drift on resume')
    else:
        core.require(not any((out/'jobs').glob('*.json')), 'Orphan receipts')
        core.atomic(out/'requested_manifest.json',m);core.atomic(out/'provenance.json',info)
    (out/'jobs').mkdir(exist_ok=True)


def verify(m,out,export=False):
    core.require(core.read(out/'requested_manifest.json') == m, 'Saved manifest mismatch')
    prov=core.read(out/'provenance.json')
    core.require(prov['manifest_sha256'] == core.sha(m) and prov['sources'] == source_hashes(), 'Source provenance mismatch')
    expected={job_name(job) for job in jobs(m)}
    core.require({p.name for p in (out/'jobs').glob('*.json')} == expected, 'Incomplete/extra receipts')
    rows, hashes=[],{}
    for job in jobs(m):
        receipt=core.read(out/'jobs'/job_name(job))
        rows.extend(validate_receipt(m,job,receipt,recompute=True))
        hashes[job_name(job)]=dict(inputs=receipt['inputs_sha256'],rows=receipt['rows_sha256'])
    count=m['iterations']*len(m['betas'])*len(m['targets'])
    core.require(len(rows) == count,'Wrong total count')
    report=dict(status='verified',total_tests=count,trials=count//2,independent_designs=m['iterations'],
                manifest_sha256=core.sha(m),receipts_sha256=core.sha(hashes),
                checks='Every test reconstructed from saved inputs; full pairing, seed, truth, rank and tail checks',
                statistical_calibration='Report pointwise binomial intervals; no stochastic pass/fail cutoff')
    if export:
        export_summaries(m,rows,out);core.atomic(out/'verification.json',report)
        core.atomic(out/'COMPLETED',dict(**report,export_sha256={p.name:digest(p) for p in out.glob('*.csv')}))
    else:
        done=core.read(out/'COMPLETED')
        core.require(done['receipts_sha256'] == report['receipts_sha256'],'Completion receipt mismatch')
        # Compare derived exports with recomputed summaries, as well as hashes.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            export_summaries(m,rows,Path(temp))
            for path in Path(temp).glob('*.csv'):
                core.require(digest(out/path.name) == done['export_sha256'][path.name], 'Export checksum mismatch')
                # Float formatting can differ across supported platforms; compare values.
                import csv
                with path.open(newline='') as f, (out/path.name).open(newline='') as g:
                    left,right=list(csv.DictReader(f)),list(csv.DictReader(g))
                core.require(len(left)==len(right),'Export row mismatch')
                for a,b in zip(left,right):
                    core.require(a.keys()==b.keys(),'Export schema mismatch')
                    for key in a:
                        if a[key] != b[key]:
                            try: equal=np.isclose(float(a[key]),float(b[key]),rtol=1e-7,atol=1e-9)
                            except ValueError: equal=False
                            core.require(equal,f'Recomputed export mismatch: {path.name}:{key}')
    return report


def run(m,out):
    with core.exclusive(out):
        setup_output(m,out)
        if (out/'COMPLETED').exists():
            print(json.dumps(verify(m,out)),flush=True);return
        if hasattr(os,'sched_getaffinity'):
            allowed=sorted(os.sched_getaffinity(0))
            core.require(len(allowed)>=m['workers'],'Insufficient allowed CPUs')
            os.sched_setaffinity(0,allowed[:m['workers']])
        core.atomic(out/'runtime.json',dict(pid=os.getpid(),workers=m['workers'],threads_per_worker=1,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            thread_limits={k:os.environ[k] for k in core.THREAD_KEYS},
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        pending=[]
        for job in jobs(m):
            path=out/'jobs'/job_name(job)
            if path.exists(): validate_receipt(m,job,core.read(path))
            else: pending.append(job)
        print(f'Pending batches: {len(pending)}/{len(jobs(m))}; workers={m["workers"]}',flush=True)
        with threadpool_limits(limits=1),ProcessPoolExecutor(max_workers=m['workers']) as pool:
            futures=[pool.submit(run_job,m,job,str(out)) for job in pending]
            for i,future in enumerate(as_completed(futures),1):
                print(f'Committed {i}/{len(pending)}: {future.result()}',flush=True)
        print(json.dumps(verify(m,out,export=True)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','run','status','verify','validate-archive'])
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args();m=read_config(args.config);out=args.output or ROOT/m['output_root']
    if args.action=='check':
        info=provenance(m)
        with threadpool_limits(limits=1):
            rows=trial_rows(m,make_input(m,0))
        core.require(len(rows)==len(m['betas'])*2,'Preflight count mismatch')
        print(json.dumps(dict(status='preflight_ok',workers=m['workers'],total_tests=m['iterations']*len(m['betas'])*2,
                              manifest_sha256=core.sha(m),sources=info['sources']),indent=2))
    elif args.action=='run': run(m,out)
    elif args.action=='status':
        print(json.dumps(dict(committed_batches=len(list((out/'jobs').glob('*.json'))),
                              expected_batches=len(jobs(m)),complete=(out/'COMPLETED').exists())))
    elif args.action=='verify':
        with core.exclusive(out): print(json.dumps(verify(m,out),indent=2))
    else:
        core.validate_archive(m,args.archive);print('Archive paths and member types validated')


if __name__=='__main__':
    main()
