"""Audit an already downloaded frozen run, report replay drift, and make plots.

Does not change original sources, receipts, completion markers, or p-values.
Integrity verification and cross-platform numerical agreement are reported as
separate results. A numerical mismatch never receives a replay-pass label.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types

import run_rbf_ridge_cv_study as study
from threadpoolctl import threadpool_limits


NUMERIC = {'p_value', 'radius', 'log_selection_mass', 'untruncated_reference_p',
           'projected_mean_norm', 'observed_score', 'cv_losses', 'widths'}
IDENTITY = ('dataset', 'iteration', 'n', 'fraction', 'beta', 'arm')


def replay_auditor(alpha_levels, differences, counts):
    def compare(left, right):
        study.core.require(len(left) == len(right), 'Recomputed row count')
        for local, saved in zip(left, right):
            study.core.require(local.keys() == saved.keys(), 'Recomputed row schema')
            # Trial identity must remain exact, even during a failed replay.
            study.core.require(all(local[k] == saved[k] for k in IDENTITY), 'Replay identity drift')
            changes = {}
            for key in local:
                if key == 'elapsed_seconds':
                    continue
                if key in NUMERIC and local[key] is not None:
                    equal = study.np.allclose(local[key], saved[key], rtol=1e-6, atol=1e-8)
                else:
                    equal = local[key] == saved[key]
                if not equal:
                    changes[key] = dict(local=local[key], saved=saved[key])
            counts['replayed_tests'] += 1
            counts['max_absolute_p_difference'] = max(counts['max_absolute_p_difference'],
                abs(local['p_value']-saved['p_value']))
            changed_decisions = [alpha for alpha in alpha_levels
                                 if (local['p_value'] < alpha) != (saved['p_value'] < alpha)]
            if changed_decisions:
                counts['tests_with_changed_alpha_decisions'] += 1
            if changes:
                differences.append(dict(**{k: saved[k] for k in IDENTITY}, changes=changes,
                                        changed_alpha_decisions=changed_decisions))
    return compare


def clone(function, namespace):
    copied = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def validate_job(m, job, receipt):
    differences = []
    counts = dict(replayed_tests=0, tests_with_changed_alpha_decisions=0, max_absolute_p_difference=0.)
    namespace = dict(vars(study))
    namespace['compare_rows'] = replay_auditor(m['alpha_levels'], differences, counts)
    validate = clone(study.validate_receipt, namespace)
    with threadpool_limits(limits=1):
        validate(m, job, receipt, recompute=True)
    return differences, counts


def csv_proxy(rounding):
    class ComparedRow(dict):
        def __eq__(self, other):
            if self.keys() != other.keys():
                return False
            for key in self:
                if self[key] == other[key]:
                    continue
                # Only the derived binomial confidence limits may differ in
                # final binary rounding. Raw p-values and all counts remain
                # exact string comparisons. Do not use this for replay tails.
                if key not in ('ci_lower', 'ci_upper'):
                    return False
                try:
                    delta = abs(float(self[key])-float(other[key]))
                except (ValueError, TypeError):
                    return False
                if not study.np.isfinite(delta) or delta > 1e-14:
                    return False
                rounding['fields'] += 1
                rounding['max_absolute_difference'] = max(rounding['max_absolute_difference'], delta)
            return True

    def reader(*args, **kwargs):
        return (ComparedRow(row) for row in csv.DictReader(*args, **kwargs))
    return types.SimpleNamespace(DictReader=reader)


def integrity_and_replay(m, output):
    differences = []
    counts = dict(replayed_tests=0, tests_with_changed_alpha_decisions=0, max_absolute_p_difference=0.)
    # Retain the complete frozen verifier, including source/manifest, every
    # receipt, every coverage/pairing/seed check, and derived CSV comparisons.
    # Substitute only its numerical comparison callback in an isolated copy.
    namespace = dict(vars(study))
    rounding = dict(fields=0, max_absolute_difference=0., absolute_tolerance=1e-14)
    proxy = csv_proxy(rounding)
    # verify imports csv inside its body; intercept just that import in the
    # isolated globals. The actual csv module and frozen source stay intact.
    import builtins
    real_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        return proxy if name == 'csv' else real_import(name, *args, **kwargs)
    namespace['__builtins__'] = dict(vars(builtins), __import__=isolated_import)
    workers = min(32, os.cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[:32])
        workers = min(workers, len(allowed))

    def merge(future):
        changed, checked = future.result()
        differences.extend(changed)
        counts['replayed_tests'] += checked['replayed_tests']
        counts['tests_with_changed_alpha_decisions'] += checked['tests_with_changed_alpha_decisions']
        counts['max_absolute_p_difference'] = max(counts['max_absolute_p_difference'], checked['max_absolute_p_difference'])

    with threadpool_limits(limits=1), study.core.exclusive(output), ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        checked_batches = 0

        def validate(m, job, receipt, recompute=False):
            nonlocal checked_batches
            pending.append(pool.submit(validate_job, m, job, receipt))
            if len(pending) >= 2*workers:
                merge(pending.pop(0))
                checked_batches += 1
                if checked_batches % 1000 == 0:
                    print(f'Checked {checked_batches}/{len(study.jobs(m))} batches', flush=True)
            return receipt['rows']
        namespace['validate_receipt'] = validate
        verifier = clone(study.verify, namespace)
        verified = verifier(m, output)
        for future in pending:
            merge(future)
    expected_replays = 2*len(m['sample_sizes'])*len(m['targets'])*(
        len(m['synthetic_fractions'])*(1+len(m['betas']))+len(m['datasets']))
    # Smoke runs have both repetitions replayed; production has first/last.
    study.core.require(counts['replayed_tests'] == expected_replays, 'Numerical replay coverage')
    return dict(integrity_status='passed', total_tests=verified['total_tests'],
                manifest_sha256=verified['manifest_sha256'], receipts_sha256=verified['receipts_sha256'],
                numerical_replay_status='passed' if not differences else 'mismatches_detected',
                original_tolerances=dict(rtol=1e-6, atol=1e-8), **counts,
                differing_replayed_tests=len(differences), replay_differences=differences,
                derived_confidence_interval_rounding=rounding,
                interpretation='Saved server results are unchanged. Integrity passing does not certify cross-platform numerical stability.')


def stream_digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        h.update(chunk)
    return h.hexdigest()


def audit_archive(m, archive, output):
    study.core.validate_archive(m, archive)
    checksum = archive.with_name(archive.name+'.sha256')
    study.core.require(checksum.is_file(), 'Missing downloaded archive checksum sidecar')
    with archive.open('rb') as stream:
        actual = stream_digest(stream)
    study.core.require(actual == checksum.read_text().split()[0].lower(), 'Downloaded archive checksum mismatch')
    checked = 0
    ignored = []
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            if member.isdir():
                continue
            relative = Path(*Path(member.name).parts[1:])
            if relative.as_posix() == 'controller.lock':
                # Windows' frozen lock helper writes a byte; Linux flock does
                # not. This synchronization file is not a result artifact.
                ignored.append(member.name)
                continue
            target = (output/relative).resolve()
            study.core.require(target.is_relative_to(output.resolve()), 'Archive path escape')
            study.core.require(target.is_file() and target.stat().st_size == member.size, f'Extracted size mismatch: {relative}')
            with bundle.extractfile(member) as stored, target.open('rb') as local:
                study.core.require(stream_digest(stored) == stream_digest(local), f'Extracted bytes mismatch: {relative}')
            checked += 1
    return dict(status='passed', archive=str(archive), sha256=actual, files_compared=checked,
                excluded_mutable_lock_files=ignored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=study.DEFAULT)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--report', type=Path, default=study.ROOT/'output/audits/rbf_ridge_cv_download_20261030/report.json')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--skip-archive', action='store_true', help='Check receipts/exports without repeating an archive comparison')
    args = parser.parse_args()
    m = study.read_config(args.config)
    output = (args.output or study.ROOT/m['output_root']).resolve()
    study.core.require((output/'COMPLETED').is_file(), 'No completed extracted run; this command never redownloads results')
    archive = None if args.skip_archive else args.archive
    if archive is None and not args.skip_archive:
        candidates = list(Path(tempfile.gettempdir()).glob(m['experiment']+'_*.tar.gz'))
        if candidates:
            archive = max(candidates, key=lambda p: p.stat().st_mtime)
    print('Checking archive and extracted files...' if archive else 'No cached archive found; checking completed receipts and exports...', flush=True)
    archive_report = audit_archive(m, archive, output) if archive else dict(
        status='skipped_by_request' if args.skip_archive else 'not_available')
    print('Checking every receipt, trial, and export; replaying first/last trials...', flush=True)
    report = integrity_and_replay(m, output)
    report['archive_integrity'] = archive_report
    report['output'] = str(output)
    report['recovery_source_sha256'] = study.digest(__file__)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    study.core.atomic(args.report, report)
    if not args.no_plots:
        sys.path.insert(0, str(study.ROOT/'examples'))
        from plot_rbf_ridge_cv_study import plot
        plot(output)
        # Put the portability status beside the original gallery, without
        # changing its source or implying that numeric replay succeeded.
        notice = ('Numerical replay: '+report['numerical_replay_status']+
                  '. See the separate recovery report; plots use unchanged server p-values.')
        (output/'analysis/PORTABILITY.txt').write_text(notice+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'replay_differences'}, indent=2))
    print('Audit report: '+str(args.report))
    if not args.no_plots: print('Plots: '+str(output/'analysis/index.html'))
    if report['numerical_replay_status'] != 'passed':
        print('Integrity passed; numerical replay has mismatches. This is not a numerical-verification pass.')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
