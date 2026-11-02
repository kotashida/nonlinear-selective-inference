"""Fresh paired spline calibration, preserving the historical fixed design.

Default: 10,000 null responses, 100 resumable batches, <=32 one-thread workers.
Primary test: exact spline FPR at alpha=.05 versus .05 (two-sided binomial).
Other thresholds and AIS comparisons are secondary; Holm correction is reported.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from scipy import stats
from si_shap.null_calibration import compare_selection_event_null_calibration as calibrate

REFERENCE = ROOT / 'examples/spline_fresh_null_reference.json'
DEFAULT_OUTPUT = ROOT / 'outputs/selection_method_validation/exact_set/spline_fresh_null_10000'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run_batch(task):
    index, count, seed, settings, output = task
    folder = Path(output) / 'batches' / f'batch_{index:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    allowed = set(inspect.signature(calibrate).parameters)
    base = {k: v for k, v in settings.items() if k in allowed}
    base.update(n_iters=count, seed=seed, iteration_start=0)
    for method in ('ais', 'exact_spline'):
        path = folder / f'{method}.csv'
        if path.exists():
            frame = pd.read_csv(path)
            if len(frame) != count:
                raise ValueError(f'Incomplete checkpoint: {path}')
        else:
            params = {**base, 'inference_method': method,
                      'stop_when_ess_met': method == 'ais',
                      # Strategy controls AIS proposal construction only. Exact
                      # inference has no AIS strategy and requires this sentinel.
                      'ais_strategy': settings['ais_strategy'] if method == 'ais' else 'baseline'}
            result = calibrate(**params)
            # Audit returned experimental settings, including selector metadata.
            actual = json.loads(json.dumps(result['settings']))
            for key, value in settings.items():
                if key not in {'runtime_metadata', 'n_iters', 'seed', 'iteration_start',
                               'inference_method', 'stop_when_ess_met', 'ais_strategy'}:
                    if actual.get(key) != value:
                        raise ValueError(f'Setting drift: {key}: {actual.get(key)} != {value}')
            expected_strategy = settings['ais_strategy'] if method == 'ais' else 'baseline'
            if actual['ais_strategy'] != expected_strategy:
                raise ValueError(f'Wrong {method} strategy: {actual["ais_strategy"]}')
            frame = result['p_value_results']
            frame.to_csv(folder / f'{method}.tmp', index=False)
            (folder / f'{method}.tmp').replace(path)
            write_json(folder / f'{method}_settings.json', actual)
        frame = frame.copy()
        frame['batch'] = index
        frame['inference_method'] = method
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize(data, output, expected):
    data.to_csv(output / 'unified_tests.csv', index=False)
    summaries = []
    for method, frame in data.groupby('inference_method'):
        if len(frame) != expected:
            raise ValueError('Wrong trial count')
        p = frame.p_value.to_numpy(float)
        valid = np.isfinite(p)
        for alpha in (.01, .05, .1):
            k = int(np.sum(valid & (p <= alpha)))
            test = stats.binomtest(k, len(p), alpha)
            ci = test.proportion_ci(confidence_level=.95, method='exact')
            summaries.append(dict(method=method, alpha=alpha, n=len(p),
                failures=int((~valid).sum()), rejections=k, fpr=k/len(p),
                ci_lower=ci.low, ci_upper=ci.high,
                binomial_two_sided_p=test.pvalue,
                conservative_one_sided_p=stats.binom.cdf(k, len(p), alpha),
                binomial_test_usable=bool(valid.all())))
    summary = pd.DataFrame(summaries)
    order = np.argsort(summary.binomial_two_sided_p.to_numpy())
    adjusted = np.maximum.accumulate(summary.binomial_two_sided_p.to_numpy()[order]
                                    * np.arange(len(order), 0, -1))
    summary['holm_p_six_tests'] = 1.0
    summary.loc[order, 'holm_p_six_tests'] = np.minimum(1, adjusted)
    summary.to_csv(output / 'calibration_summary.csv', index=False)
    a = data[data.inference_method.eq('ais')]
    e = data[data.inference_method.eq('exact_spline')]
    paired = a.merge(e, on=['batch', 'iteration', 'selection_event'],
                     suffixes=('_ais', '_exact'), validate='one_to_one')
    if len(paired) != expected:
        raise ValueError('Missing pairs')
    for key in ('target_feature', 'target_seed', 'ais_seed'):
        np.testing.assert_array_equal(paired[key+'_ais'], paired[key+'_exact'])
    np.testing.assert_array_equal(
        paired.observed_selected_features_ais.map(str),
        paired.observed_selected_features_exact.map(str))
    np.testing.assert_allclose(paired.auxiliary_u_ais, paired.auxiliary_u_exact, rtol=0, atol=1e-14)
    complete = np.isfinite(paired.p_value_ais) & np.isfinite(paired.p_value_exact)
    paired['p_difference'] = paired.p_value_ais - paired.p_value_exact
    paired.to_csv(output / 'paired_tests.csv', index=False)
    comparisons = []
    for alpha in (.01, .05, .1):
        ar = paired.p_value_ais <= alpha
        er = paired.p_value_exact <= alpha
        only_a = int((complete & ar & ~er).sum())
        only_e = int((complete & er & ~ar).sum())
        comparisons.append(dict(alpha=alpha, complete_pairs=int(complete.sum()),
            ais_only_rejections=only_a, exact_only_rejections=only_e,
            mcnemar_exact_p=stats.binomtest(only_a, only_a+only_e, .5).pvalue
                if only_a+only_e else 1.,
            mean_p_difference=paired.loc[complete, 'p_difference'].mean(),
            mean_absolute_p_difference=paired.loc[complete, 'p_difference'].abs().mean()))
    pd.DataFrame(comparisons).to_csv(output / 'paired_summary.csv', index=False)
    ks = []
    for method, frame in data.groupby('inference_method'):
        p = frame.p_value.to_numpy(float)
        p = p[np.isfinite(p)]
        if len(p):
            test = stats.kstest(p, 'uniform')
            ks.append(dict(method=method, n_finite=len(p), ks_statistic=test.statistic,
                           ks_p=test.pvalue, mean_p=p.mean()))
    pd.DataFrame(ks).to_csv(output / 'uniformity_summary.csv', index=False)
    (output / 'README.md').write_text(
        'Fresh paired global-null calibration on the historical fixed design.\n\n'
        'Primary analysis: exact spline two-sided binomial test at alpha=0.05. '
        'Secondary: other thresholds, AIS, paired McNemar, and uniformity KS tests. '
        'Holm p-values adjust the six threshold/method tests. '
        'Failures count as non-rejections in operational FPR; binomial calibration '
        'interpretation requires zero failures. No settings are tuned using this run.\n\n'
        + summary.to_string(index=False) + '\n', encoding='utf-8')
    (output / 'COMPLETED').write_text('All fresh pairs audited.\n', encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-trials', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.n_trials < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error('Require workers 1..32, positive trial/batch counts, nonnegative seed')
    settings = json.loads(REFERENCE.read_text(encoding='utf-8'))
    expected_versions = settings['runtime_metadata']['package_versions']
    versions = {name: metadata.version(name) for name in expected_versions}
    if versions != expected_versions:
        raise ValueError(f'Package versions differ: {versions}; expected {expected_versions}')
    output = args.output.resolve()
    counts = [min(args.batch_size, args.n_trials-i)
              for i in range(0, args.n_trials, args.batch_size)]
    children = np.random.SeedSequence(args.seed).spawn(len(counts))
    seeds = [int(s.generate_state(1)[0]) for s in children]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate batch seeds')
    # Seeds are derived from a new root, rather than any historical shard root.
    historical = {int(np.random.SeedSequence([root, 0, 0]).generate_state(1)[0])
                  for root in range(20260908, 20260940)}
    if historical.intersection(seeds):
        raise ValueError('Historical seed collision')
    config = dict(reference_settings=settings, n_trials=args.n_trials,
                  batch_size=args.batch_size, root_seed=args.seed, batch_seeds=seeds,
                  package_versions=versions, primary_alpha=.05,
                  primary_method='exact_spline', primary_alternative='two-sided')
    print(f'{args.n_trials} fresh paired trials; {len(counts)} batches; '
          f'{args.workers} workers, one thread each; fixed design seed {settings["design_seed"]}', flush=True)
    if args.plan:
        print(json.dumps(config, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'experiment_config.json'
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise ValueError('Existing output has a different experiment configuration')
    write_json(lock, config)
    frames = []
    tasks = [(i, count, seeds[i], settings, str(output)) for i, count in enumerate(counts)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_batch, task) for task in tasks]
        for future in as_completed(futures):
            frames.append(future.result())
            print(f'Completed batches: {len(frames)}/{len(tasks)}', flush=True)
    data = pd.concat(frames, ignore_index=True).sort_values(['batch', 'inference_method', 'iteration'])
    summarize(data, output, args.n_trials)


if __name__ == '__main__':
    main()
