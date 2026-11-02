"""Synthetic global-null calibration of the frozen September 24 pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings
import importlib.metadata
import tomllib

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[key] = '1'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:32])
ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'benchmarks/fpr_frozen'
sys.path.insert(0, str(FROZEN))
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
import si_shap

spec = importlib.util.spec_from_file_location('frozen_benchmark', FROZEN / 'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
MANIFEST = ROOT / 'benchmarks/selected_five_null_fpr_10000.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def settings(repetitions=10000):
    m = read(MANIFEST)
    require(1 <= m['n_shards'] <= m['cpu_budget'] <= 32, 'CPU cap violated')
    require(m['sample_size'] == 100 and m['n_features'] == 20, 'Null design drift')
    require(m['methods'] == list(b.METHOD_ORDER), 'Method drift')
    require(m['repetitions'] == 10000 and m['alpha_levels'] == [.01, .05, .1], 'Manifest drift')
    require(Path(si_shap.__file__).resolve().parent == FROZEN / 'si_shap', 'Wrong source imported')
    for relative, expected in read(FROZEN / 'source_hashes.json').items():
        require(b._sha256(FROZEN / relative) == expected, f'Frozen source drift: {relative}')
    archive = read(FROZEN / 'benchmark_configuration.json')
    for relative, expected in archive['archived_core_source_hashes'].items():
        require(b._sha256(FROZEN / Path(relative).relative_to('src')) == expected,
                f'Benchmark source mismatch: {relative}')
    require(archive['common_arguments'] == b.COMMON and archive['rf_params'] == b.RF_PARAMS,
            'Benchmark argument drift')
    versions = {}
    for requirement in tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['dependencies']:
        name, expected = requirement.split('==')
        versions[name] = importlib.metadata.version(name)
        require(versions[name] == expected, f'Dependency drift: {name}={versions[name]}, expected {expected}')
    sources = {p.relative_to(ROOT).as_posix(): b._sha256(p) for p in
               [Path(__file__).resolve(), MANIFEST, ROOT / 'benchmarks/run_selected_five_null_fpr_32shards.sh',
                ROOT / 'examples/selected_five_null_fpr_10000_cpu32.ps1',
                ROOT / 'pyproject.toml', *sorted(FROZEN.rglob('*.py')),
                FROZEN / 'source_hashes.json', FROZEN / 'benchmark_configuration.json']}
    return {**m, 'repetitions': repetitions, 'repetition_start': 0,
            'expected_tests': repetitions * len(b.METHOD_ORDER),
            'datasets': ['iid_gaussian_null'], 'common_arguments': b.COMMON,
            'rf_params': b.RF_PARAMS, 'method_specific': archive['method_specific'],
            'package_versions': versions,
            'archived_core_source_hashes': archive['archived_core_source_hashes'],
            'source_hashes': sources, 'pilot': repetitions != m['repetitions']}


def trial(repetition, config):
    seeds = np.random.SeedSequence([config['master_seed'], 100, 20, repetition]).spawn(4)
    names = ('design_seed', 'response_seed', 'target_seed', 'ais_seed')
    seeds = {name: int(seed.generate_state(1, dtype=np.uint32)[0])
             for name, seed in zip(names, seeds)}
    X = np.random.default_rng(seeds['design_seed']).normal(size=(100, 20))
    y = np.random.default_rng(seeds['response_seed']).normal(size=100)
    return X, y, seeds


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def interval(k, n):
    return (0.0 if k == 0 else float(stats.beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(stats.beta.ppf(.975, k+1, n-k)))


def summarize(frame, config):
    rows = []
    for method in config['methods']:
        group = frame[frame.method.eq(method)]
        if group.empty:
            continue
        def numeric(name):
            return pd.to_numeric(group.get(name, pd.Series(np.nan, index=group.index)), errors='coerce')
        for column, label in [('p_value', 'selective'), ('naive_p_value', 'naive_same_target')]:
            p = pd.to_numeric(group[column], errors='coerce')
            valid = np.isfinite(p) & p.between(0, 1)
            n, v = len(group), int(valid.sum())
            for alpha in config['alpha_levels']:
                k = int((valid & (p < alpha)).sum())
                low, high = interval(k, n)
                vlo, vhi = interval(k, v) if v else (np.nan, np.nan)
                rows.append(dict(method=method, p_value_kind=label, alpha=alpha,
                    n_attempted=n, n_valid=v, n_failed=n-v, n_rejected=k,
                    fpr_all_attempts=k/n, fpr_valid_only=k/v if v else np.nan,
                    unresolved_fpr_lower=k/n, unresolved_fpr_upper=(k+n-v)/n,
                    ci95_all_lower=low, ci95_all_upper=high,
                    ci95_valid_lower=vlo, ci95_valid_upper=vhi,
                    null_mc_se=float(np.sqrt(alpha*(1-alpha)/n)),
                    inflation_test_p=float(stats.binomtest(k, n, alpha, alternative='greater').pvalue),
                    complete_finite_evidence=v == n,
                    median_proposals=numeric('proposals').median(),
                    median_denominator_ess=numeric('denominator_ess').median(),
                    median_tail_ess=numeric('tail_ess').median()))
    return pd.DataFrame(rows)


def verify(frame, config):
    keys = ['dataset', 'sample_size', 'repetition', 'method']
    require(not frame.duplicated(keys).any(), 'Duplicate result keys')
    expected = {(i, m) for i in range(config['repetitions']) for m in config['methods']}
    require(set(zip(frame.repetition, frame.method)) == expected, 'Missing/unexpected trials')
    require(frame.dataset.eq('iid_gaussian_null').all() and frame.sample_size.eq(100).all(),
            'Design label mismatch')
    audits = []
    for i, group in frame.groupby('repetition', sort=False):
        X, y, seeds = trial(int(i), config)
        for name, value in {**seeds, 'X_sha256': array_hash(X), 'y_sha256': array_hash(y)}.items():
            require(group[name].eq(value).all(), f'Trial {i}: {name} mismatch')
        for row in group.itertuples():
            if pd.notna(getattr(row, 'observed_target_feature', np.nan)):
                selected = json.loads(row.observed_selected_features)
                require(len(selected) == 2 and len(set(selected)) == 2, 'Wrong selected set')
                u = float(np.random.default_rng(seeds['target_seed']).random())
                from si_shap.selection import target_from_selected_set
                require(int(row.observed_target_feature) == target_from_selected_set(selected, u),
                        'Wrong randomized target')
        audits.append({'repetition': i, 'data_and_seeds_match': True})
    spline = b.audit_spline_pairing(frame)
    # Failures are retained; only pairs with two finite statistics can be compared.
    pair_valid = []
    for i, g in frame[frame.method.isin(['spline_screening_ais', 'spline_screening_exact'])].groupby('repetition'):
        finite = pd.to_numeric(g.get('t_obs', pd.Series(np.nan, index=g.index)), errors='coerce').notna().all()
        pair_valid.append((i, finite))
    spline['comparable'] = [dict(pair_valid)[int(i)] for i in spline.repetition]
    require(spline.loc[spline.comparable, 'selection_target_statistic_match'].all(), 'Spline pairing mismatch')
    for col in ['p_value', 'naive_p_value']:
        p = pd.to_numeric(frame[col], errors='coerce')
        require((p.isna() | (np.isfinite(p) & p.between(0, 1))).all(), f'Invalid {col}')
    return pd.DataFrame(audits), spline


def lock(output, config):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'configuration.json'
    if path.exists():
        require(read(path) == config, 'Configuration/source changed; use a new output directory')
    else:
        b._atomic_json(path, config)


def run(args):
    config = settings(args.repetitions)
    lock(args.output, config)
    execution = {'n_shards': args.n_shards, 'shard_index': args.shard_index}
    ep = args.output / 'execution.json'
    require(not ep.exists() or read(ep) == execution, 'Shard allocation changed')
    b._atomic_json(ep, execution)
    runtime = {
        'python': sys.version, 'package_location': str(si_shap.__file__),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scipy', 'scikit-learn', 'shap', 'patsy']},
        'numerical_threads': 1, 'rf_jobs': 1,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    runtime_path = args.output / 'runtime.json'
    if runtime_path.exists():
        prior = read(runtime_path)
        require(prior['python'] == runtime['python'] and prior['packages'] == runtime['packages'],
                'Python/dependency runtime changed on resume')
    else:
        b._atomic_json(runtime_path, runtime)
    path = args.output / 'results.csv'
    records = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {(int(r['repetition']), r['method']) for r in records}
    require(len(done) == len(records), 'Duplicate checkpoint rows')
    for i in b.assigned_repetitions(0, args.repetitions, args.n_shards, args.shard_index):
        X, y, seeds = trial(i, config)
        for method in config['methods']:
            if (i, method) in done:
                continue
            start = time.perf_counter()
            fields = dict(dataset='iid_gaussian_null', sample_size=100, repetition=i,
                          method=method, seeds=seeds, rows=np.arange(100), elapsed=0)
            try:
                with warnings.catch_warnings(), threadpool_limits(limits=1):
                    warnings.filterwarnings('ignore', message='Exact-set conditioning can make the event rare')
                    result = b.selective_inference(X, y, **b.COMMON,
                            target_seed=seeds['target_seed'], ais_seed=seeds['ais_seed'],
                            **b.method_arguments(method))
                require(len(result['feature_results']) == 1, 'Expected one tested target')
                row = b._result_row(result, **fields)
                row['naive_p_value'] = row['unadjusted_p_value']
                row['auxiliary_u'] = result['auxiliary_u']
                row['selection_event'] = result['selection_event']
            except Exception as error:
                row = b._error_row(error, **fields)
                row['naive_p_value'] = np.nan
            row.update(X_sha256=array_hash(X), y_sha256=array_hash(y),
                       runtime_seconds=time.perf_counter()-start)
            records.append(row)
            done.add((i, method))
            b._atomic_csv(path, pd.DataFrame(records))
            print(f'trial={i} {method} p={row.get("p_value")} status={row.get("status")}', flush=True)


def finalize(output, frame, config):
    audit, spline = verify(frame, config)
    lock(output, config)
    b._atomic_csv(output / 'results.csv', frame)
    summary = summarize(frame, config)
    b._atomic_csv(output / 'summary.csv', summary)
    b._atomic_csv(output / 'trial_audit.csv', audit)
    b._atomic_csv(output / 'spline_pairing_audit.csv', spline)
    b._atomic_csv(output / 'status_counts.csv', frame.groupby(['method', 'status'], dropna=False).size().rename('count').reset_index())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, alpha in zip(axes, config['alpha_levels']):
        s = summary[(summary.alpha == alpha) & (summary.p_value_kind == 'selective')]
        x = np.arange(len(s))
        ax.errorbar(x, s.fpr_all_attempts, yerr=[s.fpr_all_attempts-s.ci95_all_lower,
                    s.ci95_all_upper-s.fpr_all_attempts], fmt='o', capsize=3)
        ax.axhline(alpha, color='red', linestyle='--')
        ax.set_xticks(x, ['SHAP', 'LIME', 'Spline AIS', 'Spline exact', 'MI'], rotation=45)
        ax.set_title(f'alpha={alpha}; missing values reported separately')
        ax.set_ylabel('Rejections / attempted trials (95% exact CI)')
    fig.savefig(output / 'fpr.png', dpi=160)
    plt.close(fig)
    b._atomic_json(output / 'verification.json', {'expected_rows': config['repetitions']*5,
                  'observed_rows': len(frame), 'matching_audits_passed': True,
                  'failed_selective_tests': int(pd.to_numeric(frame.p_value, errors='coerce').isna().sum()),
                  'statistical_control_asserted': False})
    (output / 'COMPLETED').write_text(f'Verified {len(frame)} null test attempts; inspect FPR and failures.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run', 'merge', 'verify'])
    parser.add_argument('--output', type=Path, default=ROOT / read(MANIFEST)['output_root'])
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--repetitions', type=int, default=10000)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    require(1 <= args.n_shards <= 32 and 0 <= args.shard_index < args.n_shards, 'Invalid shard/CPU allocation')
    require(args.repetitions >= 1, 'Replications must be positive')
    if args.action == 'check':
        print(json.dumps(settings(args.repetitions), indent=2))
    elif args.action == 'run':
        run(args)
    elif args.action == 'merge':
        require(args.input_root is not None, '--input-root required')
        paths = [args.input_root / f'shard_{i}' for i in range(args.n_shards)]
        configs = [read(p / 'configuration.json') for p in paths]
        require(all(c == configs[0] for c in configs), 'Shard configurations differ')
        require(configs[0] == settings(configs[0]['repetitions']), 'Source/configuration drift')
        for i, p in enumerate(paths):
            require(read(p / 'execution.json') == {'n_shards': args.n_shards, 'shard_index': i}, 'Shard mismatch')
        frame = pd.concat([pd.read_csv(p / 'results.csv') for p in paths], ignore_index=True)
        finalize(args.output, frame.sort_values(['repetition', 'method']), configs[0])
    else:
        config = read(args.output / 'configuration.json')
        require(config == settings(config['repetitions']), 'Source/configuration drift')
        frame = pd.read_csv(args.output / 'results.csv')
        verify(frame, config)
        pd.testing.assert_frame_equal(pd.read_csv(args.output / 'summary.csv'),
                                      summarize(frame, config), check_dtype=False, rtol=1e-10)
        print(f'Verified {len(frame)} null test attempts; completion is not a claim of FPR control.')


if __name__ == '__main__':
    main()
