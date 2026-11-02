"""Matched SHAP sampler experiment; reuse the repository's simulation and writers.

run/verify operate on the locked manifest. No benchmark pipeline is rerun.
Each worker resumes only complete, verified condition bundles.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
MANIFEST = ROOT / 'examples/shap_sampling_ten_250.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_manifest():
    m = json.loads(MANIFEST.read_text())
    require(m['workers'] == m['shards'] == 32, 'Expected 32 single-threaded workers')
    require(m['iterations_per_condition'] == 250 and m['alpha'] == .05,
            'Iteration or alpha drift')
    require(m['betas'] == [.75, 1.] and len(m['arms']) == 10, 'Condition/arm drift')
    require(m['rf_params']['n_jobs'] == 1, 'Nested forest parallelism prohibited')
    require(len({a['name'] for a in m['arms']}) == 10, 'Duplicate arms')
    return m


def hashes():
    files = [*sorted((ROOT / 'src').rglob('*.py')),
             *sorted((ROOT / 'examples').glob('*.py')), MANIFEST,
             ROOT / 'examples/shap_sampling_ten_250_32cores.ps1', ROOT / 'pyproject.toml']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files}


def lock(m):
    out = ROOT / m['output_root']
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('requested_manifest.json', m), ('source_hashes.json', hashes())]:
        path = out / name
        if path.exists():
            require(json.loads(path.read_text()) == value, f'{name} drift; use a new experiment')
        else:
            path.write_text(json.dumps(value, indent=2) + '\n')
    return out


def allocation(m, shard):
    q, r = divmod(m['iterations_per_condition'], m['shards'])
    return q + (shard < r)


def expected_settings(m, arm, shard, condition):
    # Distinct conditions; identical condition seeds across all sampling arms.
    seed = m['base_seed'] + shard + condition * 10000
    kw = dict(n_iters=allocation(m, shard), **m['design'], selection_method='shap',
              rf_params=m['rf_params'], selection_events=('exact_set',), seed=seed,
              sampler_seed_salt=arm['sampler_seed_salt'], **m['shared'],
              **{k: arm[k] for k in ('inference_method', 'ais_strategy', 'pilot_iters',
                 'max_final_samples', 'mcmc_steps', 'mcmc_global_jump_probability')},
              stop_when_ess_met=arm['inference_method'] == 'ais')
    if condition == 0:
        kw.update(alpha_levels=(m['alpha'],), design_seed=m['design_seed'],
                  fixed_auxiliary_u=None)
    else:
        kw.update(alpha=m['alpha'], signal_features=(0,),
                  signal_strength=m['betas'][condition-1])
    return kw


def folder(m, arm, shard, condition):
    name = 'null_fresh' if condition == 0 else 'power_beta_' + str(m['betas'][condition-1]).replace('.', 'p')
    return ROOT / m['output_root'] / 'runs' / arm['name'] / f'shard_{shard}' / name


def verify_bundle(m, arm, shard, condition):
    import numpy as np
    import pandas as pd
    path = folder(m, arm, shard, condition)
    kw = expected_settings(m, arm, shard, condition)
    receipt = json.loads((path / 'bundle_receipt.json').read_text())
    require(receipt['requested'] == json.loads(json.dumps(kw)), f'Settings drift: {path}')
    for name, checksum in receipt['files'].items():
        require(hashlib.sha256((path / name).read_bytes()).hexdigest() == checksum,
                f'Corrupt bundle: {path / name}')
    settings = json.loads((path / 'settings.json').read_text())
    for key in ('n_iters', 'seed', 'sampler_seed_salt', 'inference_method', 'ais_strategy',
                'pilot_iters', 'pilot_samples', 'max_final_samples', 'mcmc_steps',
                'mcmc_global_jump_probability', 'rf_params', 'selection_events',
                'stop_when_ess_met', 'min_denominator_ess', 'min_tail_ess'):
        require(settings[key] == json.loads(json.dumps(kw[key])), f'Actual setting drift: {key}: {path}')
    require(settings['target_rule'] == 'uniform_from_selected' and settings['sigma'] == 1.,
            f'Target/noise drift: {path}')
    require(settings.get('alpha_levels', [settings.get('alpha')]) == [.05], f'Alpha drift: {path}')
    if condition == 0:
        require(settings['design_seed'] == m['design_seed'] and settings['fixed_auxiliary_u'] is None,
                f'Null design/auxiliary drift: {path}')
    else:
        require(settings['signal_features'] == [0] and settings['signal_strength'] == m['betas'][condition-1],
                f'Signal drift: {path}')
    name = 'p_value_results.csv' if condition == 0 else 'target_results.csv'
    frame = pd.read_csv(path / name)
    n = allocation(m, shard)
    require(len(frame) == n and sorted(frame.iteration.tolist()) == list(range(1, n+1)),
            f'Missing or duplicate iterations: {path}')
    require(frame.selection_event.eq('exact_set').all(), f'Wrong event: {path}')
    if condition:
        feature = pd.read_csv(path / 'feature_results.csv')
        require(len(feature) == n and feature.iteration.nunique() == n, f'Missing features: {path}')
        extra = [c for c in feature.columns if c not in frame.columns and c != 'selection_event']
        frame = frame.merge(feature[['iteration'] + extra], on='iteration', validate='one_to_one')
    require(frame['failed'].astype(bool).equals(~np.isfinite(frame.p_value)),
            f'Failure/p-value mismatch: {path}')
    require(np.all((frame.p_value.dropna() >= 0) & (frame.p_value.dropna() <= 1)),
            f'Invalid p-values: {path}')
    if condition:
        require(frame.rejected.astype(bool).equals(frame.p_value.lt(.05)), f'Rejection drift: {path}')
    frame['arm'] = arm['name']
    frame['shard'] = shard
    frame['condition'] = 'null' if condition == 0 else f'beta_{m["betas"][condition-1]}'
    return frame


def worker(m, arm, shard):
    from si_shap import compare_selection_event_null_calibration, compare_selection_event_power
    from examples.compare_selection_event_null_calibration import _write_results
    from examples.compare_selection_event_power import _write_results_to_directory
    for condition in range(3):
        path = folder(m, arm, shard, condition)
        if (path / 'bundle_receipt.json').exists():
            verify_bundle(m, arm, shard, condition)
            continue
        kw = expected_settings(m, arm, shard, condition)
        started = time.perf_counter()
        result = (compare_selection_event_null_calibration if condition == 0
                  else compare_selection_event_power)(**kw)
        elapsed = time.perf_counter() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        (_write_results if condition == 0 else _write_results_to_directory)(result, path)
        names = [p.name for p in path.iterdir() if p.suffix in ('.csv', '.json', '.npy')]
        receipt = dict(requested=kw, elapsed_seconds=elapsed,
                       files={name: hashlib.sha256((path/name).read_bytes()).hexdigest() for name in names})
        temporary = path / 'bundle_receipt.tmp'
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(path / 'bundle_receipt.json')
        verify_bundle(m, arm, shard, condition)
    print(f'{arm["name"]}/shard_{shard}: complete', flush=True)


def verify_downloaded(m):
    """Audit saved results without executing inference or replacing provenance."""
    out = ROOT / m['output_root']
    saved = json.loads((out / 'source_hashes.json').read_text())
    require(isinstance(saved, dict) and bool(saved), 'Missing recorded source provenance')
    require(all(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value)
                for value in saved.values()), 'Invalid recorded source hashes')
    current = hashes()
    report = dict(mode='downloaded_result_audit',
                  source_identity_with_current_checkout=saved == current,
                  changed_recorded_files=sorted(p for p in saved if p in current and saved[p] != current[p]),
                  missing_recorded_files=sorted(set(saved) - set(current)),
                  additional_local_files=sorted(set(current) - set(saved)),
                  original_provenance_preserved=True,
                  source_identity_note='Recorded source hashes are retained; original source bytes are not in the result archive.')
    try:
        verify(m, downloaded=True)
    except Exception as exc:
        report.update(status='failed', reason=str(exc))
        raise
    else:
        report.update(status='passed', verified_bundles=960, verified_target_tests=7500)
    finally:
        (out / 'local_verification_report.json').write_text(json.dumps(report, indent=2) + '\n')


def _audit_or_write_csv(out, name, frame, downloaded):
    if downloaded:
        import pandas as pd
        # Compare saved aggregates against audited raw bundles, without replacing them.
        # "null" is an experimental condition label, not a missing-value token.
        saved = pd.read_csv(out / name, keep_default_na=False, na_values=[''])
        pd.testing.assert_frame_equal(saved, frame.reset_index(drop=True),
                                      check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
    else:
        frame.to_csv(out / name, index=False)


def verify(m, *, downloaded=False):
    import numpy as np
    import pandas as pd
    from examples.validate_selection_methods import _clopper_pearson
    if downloaded:
        out = ROOT / m['output_root']
        require(json.loads((out / 'requested_manifest.json').read_text()) == m,
                'Recorded settings differ from the requested experiment')
    else:
        out = lock(m)
    frames, errors, timings = [], [], []
    for arm in m['arms']:
        for shard in range(m['shards']):
            for condition in range(3):
                try:
                    frames.append(verify_bundle(m, arm, shard, condition))
                    receipt = json.loads((folder(m, arm, shard, condition)/'bundle_receipt.json').read_text())
                    timings.append(dict(arm=arm['name'], shard=shard, condition=condition,
                                        elapsed_seconds=receipt['elapsed_seconds']))
                except Exception as exc:
                    errors.append(dict(arm=arm['name'], shard=shard, condition=condition, reason=str(exc)))
    if not downloaded:
        pd.DataFrame(errors, columns=['arm','shard','condition','reason']).to_csv(out/'verification_errors.csv', index=False)
    require(not errors, f'{len(errors)} missing/invalid bundles; see verification_errors.csv')
    all_rows = pd.concat(frames, ignore_index=True)
    require(len(all_rows) == 7500, 'Expected 7500 target tests')
    keys = ['condition','shard','iteration']
    baseline = all_rows[all_rows.arm == 'test_01'].set_index(keys).sort_index()
    for name, frame in all_rows.groupby('arm'):
        other = frame.set_index(keys).sort_index()
        for col in ['target_feature','auxiliary_u','target_seed','t_obs','observed_selected_features']:
            require(col in other and np.array_equal(other[col].values, baseline[col].values),
                    f'Paired input mismatch: {name}/{col}')
        if name != 'test_01':
            require((other.ais_seed != baseline.ais_seed).all(), f'Sampling streams not separated: {name}')
    _audit_or_write_csv(out, 'unified_tests.csv', all_rows, downloaded)
    summaries=[]
    for (arm, condition), f in all_rows.groupby(['arm','condition']):
        signal = f.target_is_signal.fillna(False).astype(bool) if condition != 'null' else pd.Series(True,index=f.index)
        valid = np.isfinite(f.p_value)
        successes = int((f.p_value.lt(.05) & signal).sum())
        denominator = int((signal & valid).sum())
        failures = int((signal & ~valid).sum())
        lower, upper = _clopper_pearson(successes, denominator)
        marginal_lower, marginal_upper = _clopper_pearson(successes, len(f))
        summaries.append(dict(arm=arm, condition=condition, alpha=.05, n_tests=len(f),
            n_failed=int((~valid).sum()), n_target_tests=int(signal.sum()), n_failed_target_tests=failures,
            rejections=successes, metric='type_i_error' if condition=='null' else 'conditional_power',
            complete_rate=successes/denominator if denominator and not failures else np.nan,
            complete_ci95_lower=lower if not failures else np.nan,
            complete_ci95_upper=upper if not failures else np.nan,
            resolved_only_rate=successes/denominator if denominator else np.nan,
            resolved_only_ci95_lower=lower,resolved_only_ci95_upper=upper,
            missing_outcome_rate_lower=successes/int(signal.sum()) if signal.sum() else np.nan,
            missing_outcome_rate_upper=(successes+failures)/int(signal.sum()) if signal.sum() else np.nan,
            marginal_detection_rate=successes/len(f) if condition!='null' and not (~valid).any() else np.nan,
            marginal_detection_ci95_lower=marginal_lower if condition!='null' and not (~valid).any() else np.nan,
            marginal_detection_ci95_upper=marginal_upper if condition!='null' and not (~valid).any() else np.nan,
            median_proposals=f.proposals.median(), median_denominator_ess=f.denominator_ess.median()))
    _audit_or_write_csv(out, 'sampling_summary.csv', pd.DataFrame(summaries), downloaded)
    _audit_or_write_csv(out, 'status_counts.csv', all_rows.groupby(['arm','condition','status'],dropna=False).size().rename('count').reset_index(), downloaded)
    _audit_or_write_csv(out, 'resolution_counts.csv', all_rows.groupby(['arm','condition','resolution_status'],dropna=False).size().rename('count').reset_index(), downloaded)
    _audit_or_write_csv(out, 'bundle_timings.csv', pd.DataFrame(timings), downloaded)
    if not downloaded:
        (out/'COMPLETED').write_text('Verified 10 configurations x 3 conditions x 250 = 7500 tests.\nSampling failures remain in unified_tests.csv and status_counts.csv.\n')
    print((out/'COMPLETED').read_text())


def run(m):
    out = lock(m)
    with concurrent.futures.ThreadPoolExecutor(max_workers=m['workers']) as pool:
        futures=[]
        for arm in m['arms']:
            for shard in range(m['shards']):
                log = out/'logs'/f'{arm["name"]}_shard_{shard}.log'
                log.parent.mkdir(exist_ok=True)
                def launch(a=arm, s=shard, p=log):
                    with p.open('a',encoding='utf-8') as stream:
                        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker',
                                        '--arm',a['name'],'--shard',str(s)],cwd=ROOT,
                                       stdout=stream,stderr=subprocess.STDOUT,check=True)
                futures.append(pool.submit(launch))
        errors=[]
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as exc: errors.append(str(exc))
    require(not errors, f'{len(errors)} workers failed; inspect logs, then Resume')
    verify(m)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=['check','run','worker','verify','verify-downloaded','validate-archive'])
    parser.add_argument('--archive')
    parser.add_argument('--arm')
    parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    m=read_manifest()
    if args.action=='validate-archive':
        import tarfile
        from pathlib import PurePosixPath
        with tarfile.open(args.archive) as archive:
            for entry in archive.getmembers():
                p=PurePosixPath(entry.name)
                require(not p.is_absolute() and '..' not in p.parts and
                        p.parts[0]==m['experiment'] and (entry.isfile() or entry.isdir()),
                        f'Unsafe archive member: {entry.name}')
        print('Archive paths and member types verified')
    elif args.action=='check': print(json.dumps(m,indent=2))
    elif args.action=='run': run(m)
    elif args.action=='verify': verify(m)
    elif args.action=='verify-downloaded': verify_downloaded(m)
    else:
        require(args.shard is not None and 0 <= args.shard < m['shards'], 'Invalid shard')
        arm=next(a for a in m['arms'] if a['name']==args.arm)
        worker(m,arm,args.shard)


if __name__=='__main__': main()
