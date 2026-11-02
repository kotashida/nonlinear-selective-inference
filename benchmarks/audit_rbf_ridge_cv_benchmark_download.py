"""Stream-check saved server results and separately report local replay drift.

The frozen regression, p-values, tolerances, and completion record are unchanged.
An integrity pass never substitutes for a numerical replay pass.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import run_rbf_ridge_cv_benchmark_basis as study
from recover_rbf_ridge_cv_download import clone, replay_auditor, stream_digest
from threadpoolctl import threadpool_limits


def validate_job(manifest, job, output):
    receipt = study.core.read(Path(output)/'jobs'/study.job_name(job))
    differences = []
    counts = dict(replayed_tests=0, tests_with_changed_alpha_decisions=0,
                  max_absolute_p_difference=0.)
    namespace = dict(vars(study))
    namespace['compare_rows'] = replay_auditor(manifest['alpha_levels'], differences, counts)
    validate = clone(study.validate_receipt, namespace)
    with threadpool_limits(limits=1):
        rows = validate(manifest, job, receipt, recompute=True)
    return rows, receipt['rows_sha256'], differences, counts


def archive_integrity(manifest, archive, output):
    """Compare every archived file with staging in one decompression pass."""
    output = Path(output).resolve()
    seen, checked, excluded = set(), 0, []
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            relative = PurePosixPath(member.name)
            study.core.require(not relative.is_absolute() and '..' not in relative.parts
                               and '\\' not in member.name and relative.parts
                               and relative.parts[0] == manifest['experiment'], 'Unsafe archive path')
            study.core.require(member.isfile() or member.isdir(), 'Unsafe archive member type')
            key = str(relative).casefold()
            study.core.require(key not in seen and not any(':' in p or p.endswith((' ', '.'))
                               for p in relative.parts), 'Duplicate or unsafe archive destination')
            seen.add(key)
            if member.isdir():
                continue
            parts = relative.parts[1:]
            if parts == ('controller.lock',):
                excluded.append(member.name)
                continue
            target = output.joinpath(*parts).resolve()
            study.core.require(target.is_relative_to(output), 'Archive destination escape')
            study.core.require(target.is_file() and target.stat().st_size == member.size,
                               f'Extracted size mismatch: {relative}')
            with stream.extractfile(member) as stored, target.open('rb') as local:
                study.core.require(stream_digest(stored) == stream_digest(local),
                                   f'Extracted bytes mismatch: {relative}')
            checked += 1
            if checked % 2000 == 0:
                print(f'Archive byte comparison: {checked:,} files', flush=True)
    for required in ('COMPLETED', 'requested_manifest.json'):
        study.core.require((manifest['experiment']+'/'+required).casefold() in seen,
                           'Incomplete result archive')
    return dict(status='passed', files_compared=checked, excluded_mutable_lock_files=excluded)


def csv_value(value):
    return '' if value is None else json.dumps(value) if isinstance(value, list) else str(value)


def compare_csv_row(saved, derived, rounding=None):
    expected = {k: csv_value(v) for k, v in derived.items()}
    study.core.require(saved.keys() == expected.keys(), 'Derived export schema mismatch')
    for key, value in expected.items():
        if saved[key] == value:
            continue
        if rounding is not None and key in ('ci_lower', 'ci_upper'):
            delta = abs(float(saved[key])-float(value))
            if study.np.isfinite(delta) and delta <= 1e-14:
                rounding['fields'] += 1
                rounding['max_absolute_difference'] = max(rounding['max_absolute_difference'], delta)
                continue
        raise ValueError(f'Derived export mismatch: {key}')


def accumulate(groups, manifest, rows):
    for row in rows:
        key = tuple(row[k] for k in ('dataset', 'n', 'fraction', 'beta', 'arm'))
        group = groups.setdefault(key, dict(total=0, p=row['p'], rejects=Counter(),
                                           lambdas=Counter(), targets=Counter(), ranks=Counter()))
        study.core.require(group['p'] == row['p'], 'Feature-count drift')
        group['total'] += 1
        for alpha in manifest['alpha_levels']:
            group['rejects'][alpha] += int(row['p_value'] < alpha)
        group['lambdas'][row['chosen_lambda']] += 1
        group['targets'][row['target']] += 1
        group['ranks'][row['rank']] += 1


def derived_exports(manifest, groups):
    exports = {name: [] for name in ('summary.csv', 'lambda_summary.csv', 'target_summary.csv', 'rank_summary.csv')}
    for key, group in sorted(groups.items()):
        identity = dict(zip(('dataset', 'n', 'fraction', 'beta', 'arm'), key))
        total = group['total']
        for alpha in manifest['alpha_levels']:
            counts = study.core.binomial(group['rejects'][alpha], total)
            counts['trials'] = counts.pop('n')
            exports['summary.csv'].append(dict(**identity, alpha=alpha, metric='rejection_rate', **counts))
        for lam in manifest['lambdas']:
            exports['lambda_summary.csv'].append(dict(**identity, chosen_lambda=lam,
                                                      count=group['lambdas'][lam], total=total))
        for target in range(group['p']):
            exports['target_summary.csv'].append(dict(**identity, target=target,
                                                      count=group['targets'][target], total=total))
        for rank, count in sorted(group['ranks'].items()):
            exports['rank_summary.csv'].append(dict(**identity, rank=rank, count=count, total=total))
    return exports


def audit(manifest, output, archive, workers=None):
    output = Path(output)
    workers = workers or min(8, max(1, (os.cpu_count() or 1)-1))
    study.core.require(1 <= workers <= 31, 'Audit CPU budget exceeded')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[:32])
        workers = min(workers, max(1, len(allowed)-1))
    study.core.require(study.core.read(output/'requested_manifest.json') == manifest, 'Saved manifest mismatch')
    provenance = study.core.read(output/'provenance.json')
    local = study.provenance(manifest)
    study.core.require(provenance['sources'] == local['sources'] and provenance['versions'] == local['versions']
                       and provenance['manifest_sha256'] == study.core.sha(manifest), 'Source/dependency provenance mismatch')
    done = study.core.read(output/'COMPLETED')
    study.core.require(done['status'] == 'verified' and done['errors'] == 0
                       and done['total_tests'] == study.expected_tests(manifest)
                       and done['manifest_sha256'] == study.core.sha(manifest), 'Completion identity mismatch')
    names = {'results.csv', 'summary.csv', 'lambda_summary.csv', 'target_summary.csv', 'rank_summary.csv'}
    study.core.require(set(done['export_sha256']) == names, 'Export coverage mismatch')
    for name, checksum in done['export_sha256'].items():
        with (output/name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        study.core.require(actual == checksum, f'Export checksum mismatch: {name}')
    jobs = study.jobs(manifest)
    study.core.require({p.name for p in (output/'jobs').glob('*.json')} == {study.job_name(j) for j in jobs},
                       'Incomplete/extra batches')
    hashes, groups, differences = {}, {}, []
    counts = dict(replayed_tests=0, tests_with_changed_alpha_decisions=0, max_absolute_p_difference=0.)
    rounding = dict(fields=0, max_absolute_difference=0., absolute_tolerance=1e-14)
    total = 0
    print(f'Checking {len(jobs):,} receipts and all CSV rows with {workers} audit workers.', flush=True)
    with threadpool_limits(limits=1), study.core.exclusive(output), \
            ProcessPoolExecutor(max_workers=workers) as pool, (output/'results.csv').open(newline='', encoding='utf-8') as stream:
        csv_rows = csv.DictReader(stream)
        iterator = iter(jobs)
        pending = deque()

        def submit_next():
            job = next(iterator, None)
            if job is not None:
                pending.append((job, pool.submit(validate_job, manifest, job, str(output))))

        for _ in range(2*workers):
            submit_next()
        checked = 0
        while pending:
            job, future = pending.popleft()
            rows, checksum, changed, replay_counts = future.result()
            hashes[study.job_name(job)] = checksum
            differences.extend(changed)
            for key in ('replayed_tests', 'tests_with_changed_alpha_decisions'):
                counts[key] += replay_counts[key]
            counts['max_absolute_p_difference'] = max(counts['max_absolute_p_difference'],
                                                       replay_counts['max_absolute_p_difference'])
            for row in rows:
                saved = next(csv_rows, None)
                study.core.require(saved is not None, 'Missing results.csv row')
                compare_csv_row(saved, row)
            accumulate(groups, manifest, rows)
            total += len(rows)
            checked += 1
            if checked % 1000 == 0:
                print(f'Checked {checked:,}/{len(jobs):,} batches; {total:,} saved tests', flush=True)
            submit_next()
        study.core.require(next(csv_rows, None) is None, 'Extra results.csv rows')
    study.core.require(total == study.expected_tests(manifest), 'Wrong total test count')
    study.core.require(study.core.sha(hashes) == done['receipts_sha256'], 'Completion receipt mismatch')
    expected_replays = min(2, manifest['real_iterations'])*len(manifest['datasets'])*len(manifest['sample_sizes'])*len(manifest['basis_fractions'])*len(manifest['targets'])
    study.core.require(counts['replayed_tests'] == expected_replays, 'Numerical replay coverage mismatch')
    for name, expected in derived_exports(manifest, groups).items():
        with (output/name).open(newline='', encoding='utf-8') as stream:
            saved = csv.DictReader(stream)
            for row in expected:
                actual = next(saved, None)
                study.core.require(actual is not None, f'Missing derived export row: {name}')
                compare_csv_row(actual, row, rounding=rounding)
            study.core.require(next(saved, None) is None, f'Extra derived export row: {name}')
    print('Comparing staged files with the checksum-matched downloaded archive.', flush=True)
    archived = archive_integrity(manifest, archive, output)
    return dict(integrity_status='passed', total_tests=total, manifest_sha256=study.core.sha(manifest),
                receipts_sha256=study.core.sha(hashes), archive_integrity=archived,
                numerical_replay_status='mismatches_detected' if differences else 'passed',
                differing_replayed_tests=len(differences), replay_differences=differences,
                original_tolerances=dict(rtol=1e-6, atol=1e-8), **counts,
                derived_confidence_interval_rounding=rounding, audit_workers=workers,
                source_sha256=study.digest(__file__), saved_server_results_unchanged=True,
                interpretation='Integrity and numerical replay are separate. Plots use unchanged server p-values. '
                               'An integrity pass does not establish cross-platform numerical stability.')
