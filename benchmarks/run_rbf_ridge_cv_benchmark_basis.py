"""Frozen benchmark RBF ridge/CV basis sensitivity with observed responses."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import run_rbf_shapley as core
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import rbf_ridge_cv as model

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rbf_ridge_cv_benchmark_basis_cpu32_20261031'
SMOKE = NAME.replace('_cpu32_', '_smoke_cpu32_')
DEFAULT = ROOT/'benchmarks'/f'{NAME}.json'
FROZEN = {'rbf_ridge_cv_benchmark_basis_cpu32_20261031': '8744860726610b70f414cdbd22c58cd150ac130f8a6f70efe52128f68f783ef1', 'rbf_ridge_cv_benchmark_basis_smoke_cpu32_20261031': 'd99efc57121855e127894a6cd7e652c3458349868863a459904d9b5358133649'}
DATASETS = ('heating_load', 'cooling_load', 'gas_turbine', 'red_wine', 'white_wine', 'abalone', 'concrete', 'housing')
PACKAGED = ('benchmarks/run_rbf_ridge_cv_benchmark_basis.py', 'benchmarks/rbf_ridge_cv.py', 'benchmarks/run_rbf_shapley.py', 'benchmarks/rbf_network.py', 'benchmarks/table_dataset.pkl', 'benchmarks/rbf_ridge_cv_benchmark_basis_cpu32_20261031.json', 'benchmarks/rbf_ridge_cv_benchmark_basis_smoke_cpu32_20261031.json', 'examples/rbf_ridge_cv_benchmark_basis_cpu32.ps1', 'examples/plot_rbf_ridge_cv_benchmark_basis.py', 'docs/rbf_ridge_cv_benchmark_basis_study.md', 'docs/commands.md', 'tests/test_rbf_ridge_cv_benchmark_basis.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path=DEFAULT):
    m = core.read(path)
    core.require(FROZEN.get(m['experiment']) == core.sha(m), 'Manifest changed; create a new named experiment')
    core.require(1 <= m['workers'] <= 32 and m['threads_per_worker'] == 1, 'CPU budget exceeded')
    core.require(m['output_root'] == 'outputs/benchmarks/'+m['experiment'], 'Unsafe output root')
    return m


def source_hashes():
    return {name: digest(ROOT/name) for name in PACKAGED}


def provenance(m):
    for name, version in m['required_versions'].items():
        core.require(metadata.version(name) == version, f'Dependency mismatch: {name}')
    core.require(digest(ROOT/m['input_path']) == m['input_sha256'], 'Dataset checksum mismatch')
    return dict(manifest_sha256=core.sha(m), sources=source_hashes(),
                versions={name: metadata.version(name) for name in m['required_versions']},
                python=sys.version, platform=platform.platform())


@lru_cache(maxsize=1)
def load_data(path, checksum):
    core.require(digest(path) == checksum, 'Dataset checksum mismatch')
    with Path(path).open('rb') as stream:
        data = pickle.load(stream)  # Only the SHA256-pinned repository fixture.
    result = {}
    for name in DATASETS:
        X, y = (np.asarray(v, dtype=float) for v in data[name])
        core.require(X.ndim == 2 and y.shape == (len(X),) and len(X) >= 75, 'Dataset shape')
        core.require(np.isfinite(X).all() and np.isfinite(y).all(), 'Dataset values')
        result[name] = X, y
    return result


def seed(m, dataset, iteration, stream):
    di = DATASETS.index(dataset)+1
    return int(np.random.SeedSequence([m['seed'], di, iteration, stream]).generate_state(1)[0])


def inputs(m, dataset, iteration, n):
    """Subsample observed benchmark rows; preserve reference seeds and nested n."""
    seeds = {name: seed(m, dataset, iteration, stream) for stream, name in
             enumerate(('data_seed', 'target_seed', 'basis_seed', 'fold_seed'), 1)}
    rng = np.random.default_rng(seeds['data_seed'])
    allX, ally = load_data(str(ROOT/m['input_path']), m['input_sha256'])[dataset]
    indices = sorted(int(i) for i in rng.permutation(len(ally))[:n])
    X, y = allX[indices], ally[indices]
    u = float(np.random.default_rng(seeds['target_seed']).uniform())
    return dict(dataset=dataset, iteration=iteration, n=n, indices=indices, auxiliary_u=u,
                X=X.tolist(), y_base=y.tolist(), **seeds)


def evaluate(m, saved, fractions, betas):
    core.require(betas == [0.], 'Observed responses do not support a beta sweep')
    X, base = np.asarray(saved['X']), np.asarray(saved['y_base'])
    n, p = X.shape
    rows = []
    for fraction in fractions:
        started = time.perf_counter()
        fitted = model.basis(X, fraction, saved['basis_seed'])
        blocks = model.transform(X, fitted)
        C, folds, fold_centers = model.cv_matrices(X, fraction, m['lambdas'], m['cv_folds'],
                                                  saved['basis_seed'], saved['fold_seed'])
        geometries = {}
        for beta in betas:
            y = base
            losses = [float(y@c@y) for c in C]
            chosen = int(np.argmin(losses))  # Ascending lambda tie break.
            if chosen not in geometries:
                geometries[chosen] = model.Geometry(blocks, m['lambdas'][chosen], m['rank_tolerance'])
            g = geometries[chosen]
            yr = g.Q.T@y
            selected = g.select(yr, m['k'])
            target = selected[int(saved['auxiliary_u']*len(selected))]
            mask = sum(1 << j for j in selected)
            identity = dict(dataset=saved['dataset'], iteration=saved['iteration'], n=n, p=p,
                            fraction=fraction, beta=beta, centers_per_feature=len(fitted['indices']),
                            basis_centers=fitted['indices'].tolist(), widths=fitted['widths'].tolist(),
                            folds=folds, fold_centers=fold_centers, cv_losses=losses,
                            chosen_lambda=m['lambdas'][chosen], chosen_index=chosen,
                            auxiliary_u=saved['auxiliary_u'], selected=list(selected), target=target,
                            input_sha256=digest_arrays(X, y), full_rank=g.r)
            for arm in m['targets']:
                A, B = g.target(arm, mask, target)
                result = model.radial_test(g, y, B, selected, m['sigma'], C, chosen)
                rows.append(dict(**identity, arm=arm, method='analytic_joint_cv_topk',
                                 **result, rejected=result['p_value'] < m['primary_alpha'],
                                 null_true=None,
                                 projected_mean_norm=None, observed_score=float(yr@A@yr),
                                 elapsed_seconds=time.perf_counter()-started))
    return rows


def digest_arrays(X, y):
    return hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()


def jobs(m):
    result = []
    for section, dataset, count in [('real', d, m['real_iterations']) for d in m['datasets']]:
        for start in range(0, count, m['batch_size']):
            result.append((section, dataset, start, min(start+m['batch_size'], count)))
    return result


def job_name(job):
    return f'{job[0]}_{job[1]}_{job[2]:05}_{job[3]:05}.json'


def trial_inputs(m, job):
    return [inputs(m, job[1], iteration, n) for iteration in range(job[2], job[3]) for n in m['sample_sizes']]


def settings(m, job):
    return m['basis_fractions'], [0.]


def expected_tests(m):
    return len(m['targets'])*len(m['sample_sizes'])*len(m['basis_fractions'])*len(m['datasets'])*m['real_iterations']



def run_job(m, job, out):
    saved = trial_inputs(m, job)
    fractions, betas = settings(m, job)
    rows = []
    with threadpool_limits(limits=1):
        for inp in saved:
            rows.extend(evaluate(m, inp, fractions, betas))
        core.require(all(v['num_threads'] <= 1 for v in threadpool_info()), 'Numerical thread cap')
    receipt = dict(manifest_sha256=core.sha(m), job=list(job), inputs=saved, rows=rows,
                   inputs_sha256=core.sha(saved), rows_sha256=core.sha(rows))
    core.atomic(Path(out)/'jobs'/job_name(job), receipt)
    return job_name(job)


def compare_rows(left, right):
    core.require(len(left) == len(right), 'Recomputed row count')
    numeric = {'p_value', 'radius', 'log_selection_mass', 'untruncated_reference_p',
               'projected_mean_norm', 'observed_score', 'cv_losses', 'widths'}
    for a, b in zip(left, right):
        core.require(a.keys() == b.keys(), 'Recomputed row schema')
        for key in a:
            if key == 'elapsed_seconds':
                continue
            if key in numeric and a[key] is not None:
                core.require(np.allclose(a[key], b[key], rtol=1e-6, atol=1e-8), f'Recomputed {key}')
            else:
                core.require(a[key] == b[key], f'Recomputed {key}')


def validate_receipt(m, job, receipt, recompute=False):
    core.require(receipt['manifest_sha256'] == core.sha(m) and receipt['job'] == list(job), 'Receipt identity')
    saved, rows = receipt['inputs'], receipt['rows']
    core.require(core.sha(saved) == receipt['inputs_sha256'] and core.sha(rows) == receipt['rows_sha256'], 'Receipt checksum')
    regenerated = trial_inputs(m, job)
    core.require(len(saved) == len(regenerated), 'Input count')
    fractions, betas = settings(m, job)
    expected = {(i, n, f, b, a) for i in range(job[2], job[3]) for n in m['sample_sizes']
                for f in fractions for b in betas for a in m['targets']}
    actual = {(r['iteration'], r['n'], r['fraction'], r['beta'], r['arm']) for r in rows}
    core.require(len(rows) == len(expected) and actual == expected, 'Missing/duplicate trial coverage')
    for a, b in zip(saved, regenerated):
        for key in a:
            if key in ('X', 'y_base'):
                core.require(np.allclose(a[key], b[key], rtol=1e-13, atol=1e-13), 'Input regeneration')
            else:
                core.require(a[key] == b[key], 'Input seed/index mismatch')
        selected_rows = [r for r in rows if r['iteration'] == a['iteration'] and r['n'] == a['n']]
        for fraction in fractions:
            for beta in betas:
                pair = [r for r in selected_rows if r['fraction'] == fraction and r['beta'] == beta]
                core.require(len(pair) == 2, 'Unpaired targets')
                for key in ('selected', 'target', 'input_sha256', 'chosen_lambda', 'cv_losses', 'basis_centers', 'folds'):
                    core.require(pair[0][key] == pair[1][key], f'Unpaired {key}')
                for r in pair:
                    core.require(r['dataset'] == job[1] and r['method'] == 'analytic_joint_cv_topk', 'Method/dataset drift')
                    core.require(np.isfinite(r['p_value']) and 0 <= r['p_value'] <= 1, 'Invalid p-value')
                    core.require(r['rejected'] == (r['p_value'] < m['primary_alpha']), 'Wrong decision')
                    core.require(r['selected'] == sorted(set(r['selected'])) and len(r['selected']) == m['k'], 'Selection shape')
                    core.require(r['target'] == r['selected'][int(a['auxiliary_u']*m['k'])], 'Wrong target')
                    core.require(r['auxiliary_u'] == a['auxiliary_u'], 'Auxiliary randomization drift')
                    X, base = np.asarray(a['X']), np.asarray(a['y_base'])
                    y = base
                    core.require(r['input_sha256'] == digest_arrays(X, y), 'Unmatched response/design')
                    core.require(r['chosen_index'] == int(np.argmin(r['cv_losses'])) and
                                 r['chosen_lambda'] == m['lambdas'][r['chosen_index']], 'Wrong CV choice')
                    core.require(r['centers_per_feature'] == max(1, int(np.ceil(fraction*r['n']))), 'Basis count')
                    core.require(0 <= r['rank'] <= r['full_rank'] <= r['n']-1, 'Rank bounds')
                    if job[0] == 'real': core.require(r['null_true'] is None, 'Unknown real-data truth')
        if recompute and a['iteration'] in (0, m[f'{job[0]}_iterations']-1):
            with threadpool_limits(limits=1):
                compare_rows(evaluate(m, a, fractions, betas), selected_rows)
    return rows


def summaries(m, rows, out):
    grouped = {}
    for r in rows:
        key = (r['dataset'], r['n'], r['fraction'], r['beta'], r['arm'])
        grouped.setdefault(key, []).append(r)
    summary, tuning, targets, ranks = [], [], [], []
    for key, group in sorted(grouped.items()):
        dataset, n, fraction, beta, arm = key
        identity = dict(dataset=dataset, n=n, fraction=fraction, beta=beta, arm=arm)
        for alpha in m['alpha_levels']:
            metrics = [('rejection_rate', sum(r['p_value'] < alpha for r in group), len(group))]
            for metric, k, total in metrics:
                counts = core.binomial(k, total)
                counts['trials'] = counts.pop('n')
                summary.append(dict(**identity, alpha=alpha, metric=metric, **counts))
        for lam in m['lambdas']:
            tuning.append(dict(**identity, chosen_lambda=lam, count=sum(r['chosen_lambda'] == lam for r in group), total=len(group)))
        for j in range(group[0]['p']):
            targets.append(dict(**identity, target=j, count=sum(r['target'] == j for r in group), total=len(group)))
        for rank in sorted({r['rank'] for r in group}):
            ranks.append(dict(**identity, rank=rank, count=sum(r['rank'] == rank for r in group), total=len(group)))
    core.write_csv(out/'results.csv', rows)
    core.write_csv(out/'summary.csv', summary)
    core.write_csv(out/'lambda_summary.csv', tuning)
    core.write_csv(out/'target_summary.csv', targets)
    core.write_csv(out/'rank_summary.csv', ranks)


def setup_output(m, out):
    info = provenance(m)
    if (out/'requested_manifest.json').exists():
        core.require(core.read(out/'requested_manifest.json') == m, 'Saved settings changed')
        prior = core.read(out/'provenance.json')
        core.require(prior['sources'] == info['sources'] and prior['versions'] == info['versions'], 'Source/dependency drift on resume')
    else:
        core.require(not any(out.iterdir()) or set(p.name for p in out.iterdir()) <= {
            'launcher.log', 'launcher.pid', 'launcher.startticks', 'launch.lock', 'controller.lock'}, 'Unrecognized output directory')
        core.atomic(out/'requested_manifest.json', m)
        core.atomic(out/'provenance.json', info)
    (out/'jobs').mkdir(exist_ok=True)


def verify(m, out, export=False):
    core.require(core.read(out/'requested_manifest.json') == m, 'Saved manifest mismatch')
    prov = core.read(out/'provenance.json')
    core.require(prov['sources'] == source_hashes() and prov['manifest_sha256'] == core.sha(m), 'Source provenance mismatch')
    core.require({p.name for p in (out/'jobs').glob('*.json')} == {job_name(j) for j in jobs(m)}, 'Incomplete/extra batches')
    rows, hashes = [], {}
    for job in jobs(m):
        receipt = core.read(out/'jobs'/job_name(job))
        rows.extend(validate_receipt(m, job, receipt, recompute=True))
        hashes[job_name(job)] = receipt['rows_sha256']
    core.require(len(rows) == expected_tests(m), 'Wrong total test count')
    report = dict(status='verified', total_tests=len(rows), manifest_sha256=core.sha(m),
                  receipts_sha256=core.sha(hashes), errors=0,
                  numerical_replay='First and last repetition of each experiment section/dataset/n/fraction',
                  statistical_calibration='Observed responses: descriptive rejection rates with exact binomial intervals; truth unknown')
    if export:
        summaries(m, rows, out)
        core.atomic(out/'verification.json', report)
        core.atomic(out/'COMPLETED', dict(**report, export_sha256={p.name: digest(p) for p in out.glob('*.csv')}))
    else:
        done = core.read(out/'COMPLETED')
        core.require(done['receipts_sha256'] == report['receipts_sha256'], 'Completion receipt mismatch')
        for name, checksum in done['export_sha256'].items():
            core.require(digest(out/name) == checksum, 'Export checksum mismatch')
        # Independently reconstruct derived exports from checked receipts.
        import tempfile
        import csv
        with tempfile.TemporaryDirectory() as temp:
            summaries(m, rows, Path(temp))
            for path in Path(temp).glob('*.csv'):
                with path.open(newline='') as a, (out/path.name).open(newline='') as b:
                    core.require(list(csv.DictReader(a)) == list(csv.DictReader(b)), 'Derived export mismatch')
    return report


def run(m, out):
    with core.exclusive(out):
        setup_output(m, out)
        if (out/'COMPLETED').exists():
            print(json.dumps(verify(m, out)), flush=True)
            return
        if hasattr(os, 'sched_getaffinity'):
            allowed = sorted(os.sched_getaffinity(0))
            core.require(len(allowed) >= m['workers'], 'Insufficient allowed CPUs')
            os.sched_setaffinity(0, allowed[:m['workers']])
        core.atomic(out/'runtime.json', dict(pid=os.getpid(), workers=m['workers'], threads_per_worker=1,
                    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    thread_limits={k: os.environ[k] for k in core.THREAD_KEYS},
                    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        pending = []
        for job in jobs(m):
            path = out/'jobs'/job_name(job)
            if path.exists(): validate_receipt(m, job, core.read(path))
            else: pending.append(job)
        print(f'Pending batches: {len(pending)}/{len(jobs(m))}; workers={m["workers"]}', flush=True)
        with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=m['workers']) as pool:
            futures = [pool.submit(run_job, m, job, str(out)) for job in pending]
            for i, future in enumerate(as_completed(futures), 1):
                print(f'Committed {i}/{len(pending)}: {future.result()}', flush=True)
        print(json.dumps(verify(m, out, export=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run', 'status', 'verify', 'validate-archive'])
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    m = read_config(args.config)
    out = args.output or ROOT/m['output_root']
    if args.action == 'check':
        provenance(m)
        with threadpool_limits(limits=1):
            for dataset in m['datasets']:
                for n in m['sample_sizes']:
                    evaluate(m, inputs(m, dataset, 0, n), m['basis_fractions'], [0.])
        print(json.dumps(dict(status='preflight_ok', workers=m['workers'], total_tests=expected_tests(m))))
    elif args.action == 'run': run(m, out)
    elif args.action == 'status':
        print(json.dumps(dict(committed_batches=len(list((out/'jobs').glob('*.json'))),
                              expected_batches=len(jobs(m)), complete=(out/'COMPLETED').exists())))
    elif args.action == 'verify':
        with core.exclusive(out): print(json.dumps(verify(m, out)))
    else:
        core.validate_archive(m, args.archive)
        print('Archive paths and member types validated')


if __name__ == '__main__':
    main()
