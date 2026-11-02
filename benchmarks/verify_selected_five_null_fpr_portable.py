"""Verify downloaded null results across platforms without changing run sources.

Normal tail draws can differ by one float64 step between math-library builds.
Reconcile only a single adjacent tail value, and only when the entire array's
SHA-256 then exactly matches the recorded originating-platform hash. All other
source, seed, target, pairing, p-value, and summary checks remain mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import tarfile
import types
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# The original runner sets thread limits and loads the frozen inference package.
RUNNER = runpy.run_path(str(ROOT / 'benchmarks/run_selected_five_null_fpr.py'))
np = RUNNER['np']
pd = RUNNER['pd']
require = RUNNER['require']
array_hash = RUNNER['array_hash']
NORMAL_TAIL_BOUNDARY = 3.6541528853610088


def reconcile_tail_rounding(value, expected_hash):
    """Return an exact hash match or fail; no approximate hash acceptance."""
    if array_hash(value) == expected_hash:
        return value, None
    for index in np.flatnonzero(np.abs(value.ravel()) >= NORMAL_TAIL_BOUNDARY):
        for direction in (-np.inf, np.inf):
            candidate = value.copy()
            candidate.ravel()[index] = np.nextafter(candidate.ravel()[index], direction)
            if array_hash(candidate) == expected_hash:
                return candidate, {
                    'flat_index': int(index), 'ulp_steps': 1,
                    'local_value': float(value.ravel()[index]),
                    'origin_value': float(candidate.ravel()[index]),
                    'origin_sha256': expected_hash,
                }
    raise RuntimeError('Array hash mismatch is not reconciled by one adjacent normal tail value')


def digest_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive, output):
    """Compare every archived file with its downloaded, extracted bytes."""
    count = 0
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            require(member.name.startswith(output.name + '/')
                    or (member.isdir() and member.name == output.name), 'Unexpected archive root')
            target = (output.parent / member.name).resolve()
            require(target.is_relative_to(output.resolve()), 'Archive path escapes result directory')
            if member.isdir():
                continue
            require(member.isfile(), 'Archive contains a link or special entry')
            require(target.is_file() and target.stat().st_size == member.size,
                    f'Missing/wrong-size downloaded file: {member.name}')
            with bundle.extractfile(member) as stored, target.open('rb') as local:
                require(digest_stream(stored) == digest_stream(local),
                        f'Downloaded bytes differ from archive: {member.name}')
            count += 1
    require(count > 0, 'Empty result archive')
    with archive.open('rb') as stream:
        return {'path': str(archive), 'sha256': digest_stream(stream),
                'files_compared': count, 'all_extracted_file_bytes_match': True}


def verify_output(output, archive=None):
    config = RUNNER['read'](output / 'configuration.json')
    require(config == RUNNER['settings'](config['repetitions']), 'Source/configuration drift')
    archive_audit = verify_archive(archive, output) if archive is not None else None
    frame = pd.read_csv(output / 'results.csv')
    hashes = {}
    for repetition, group in frame.groupby('repetition', sort=False):
        require(group.X_sha256.nunique(dropna=False) == group.y_sha256.nunique(dropna=False) == 1,
                f'Trial {repetition}: method data hashes differ')
        hashes[int(repetition)] = (group.X_sha256.iloc[0], group.y_sha256.iloc[0])
    reconciled = []

    def portable_trial(repetition, trial_config):
        X, y, seeds = RUNNER['trial'](repetition, trial_config)
        values = []
        for name, value, expected in zip(('X', 'y'), (X, y), hashes[repetition]):
            recovered, detail = reconcile_tail_rounding(value, expected)
            values.append(recovered)
            if detail is not None:
                reconciled.append({'repetition': repetition, 'array': name, **detail})
        return *values, seeds

    # Keep the complete original verification function and its exact hash checks.
    # Only its trial generator is supplied with hash-confirmed adjacent floats.
    original = RUNNER['verify']
    verify = types.FunctionType(original.__code__,
        {**original.__globals__, 'trial': portable_trial}, original.__name__, original.__defaults__)
    trial_audit, spline_audit = verify(frame, config)
    pd.testing.assert_frame_equal(pd.read_csv(output / 'summary.csv'),
        RUNNER['summarize'](frame, config), check_dtype=False, rtol=1e-10)
    pd.testing.assert_frame_equal(pd.read_csv(output / 'trial_audit.csv'),
        trial_audit, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(output / 'spline_pairing_audit.csv'),
        spline_audit, check_dtype=False, rtol=1e-10)
    status = frame.groupby(['method', 'status'], dropna=False).size().rename('count').reset_index()
    pd.testing.assert_frame_equal(pd.read_csv(output / 'status_counts.csv'), status, check_dtype=False)
    origin = RUNNER['read'](output / 'verification.json')
    require(origin['expected_rows'] == origin['observed_rows'] == len(frame)
            and origin['matching_audits_passed'], 'Origin verification record mismatch')
    failed = int(frame.p_value.isna().sum())
    require(origin['failed_selective_tests'] == failed, 'Origin failure count mismatch')
    report = {
        'verified_rows': len(frame), 'source_and_configuration_match': True,
        'all_regenerated_array_hashes_match_origin': True,
        'trial_target_pairing_and_summary_audits_passed': True,
        'failed_selective_tests': failed,
        'tail_rounding_reconciled_arrays': len(reconciled),
        'maximum_allowed_adjustment': 'one float64 ULP in one normal tail value per array',
        'rounding_reconciliations': reconciled,
        'archive_audit': archive_audit,
        'verifier_sha256': RUNNER['b']._sha256(Path(__file__)),
        'statistical_control_asserted': False,
    }
    RUNNER['b']._atomic_json(output / 'portable_verification.json', report)
    print(f'Verified {len(frame)} downloaded null test attempts; all origin array hashes match.')
    print(f'Reconciled {len(reconciled)} single-step normal tail rounding differences; '
          f'{failed} selective-inference failures retained.')
    if archive_audit:
        print(f"Downloaded bytes match all {archive_audit['files_compared']} archived files.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / RUNNER['read'](RUNNER['MANIFEST'])['output_root'])
    parser.add_argument('--archive', type=Path, help='Optional original downloaded tar.gz for byte integrity checks')
    args = parser.parse_args()
    if args.archive is None:
        candidates = list(Path(tempfile.gettempdir()).glob(args.output.name + '_*.tar.gz'))
        if candidates:
            args.archive = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    verify_output(args.output, args.archive)


if __name__ == '__main__':
    main()
