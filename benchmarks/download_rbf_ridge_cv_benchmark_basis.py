"""Finalize a cached benchmark download without changing the frozen experiment.

Validate the transport checksum and archive, extract into private staging,
run the original strict verifier and plotter, and publish only on success.
Failed verification leaves the archive and staged files available for diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile

import run_rbf_ridge_cv_benchmark_basis as study
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
from plot_rbf_ridge_cv_benchmark_basis import plot


def expected_checksum(path):
    parts = Path(path).read_text(encoding='utf-8').split()
    if not parts or not re.fullmatch(r'[0-9a-fA-F]{64}', parts[0]):
        raise ValueError('Malformed archive checksum file')
    return parts[0].lower()


def extract_archive(manifest, archive, destination):
    study.core.validate_archive(manifest, archive)
    # A fresh staging directory and exclusive file creation prevent overwrites.
    # Inspect normalized names too: two spellings must not address one file.
    seen = set()
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            relative = PurePosixPath(member.name)
            key = str(relative).casefold()
            if key in seen or any(':' in part or part.endswith((' ', '.'))
                                  for part in relative.parts):
                raise ValueError('Duplicate or unsafe archive destination')
            seen.add(key)
            path = destination.joinpath(*relative.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target, length=1024*1024)


def finalize(archive, checksum, repo=ROOT, audit_replay=False):
    archive, repo = Path(archive), Path(repo)
    manifest = study.read_config()
    target = repo/manifest['output_root']
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {target}')
    expected = expected_checksum(checksum)
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Archive checksum mismatch; cached files have been preserved')
    staging_root = repo/'tmp'/'downloads'/manifest['experiment']
    staging_root.mkdir(parents=True, exist_ok=True)
    state_path = staging_root/'state.json'
    state = study.core.read(state_path) if state_path.exists() else {}
    staging = None
    if state.get('archive_sha256') == expected:
        name = state.get('directory', '')
        if name and Path(name).name == name and name not in ('.', '..'):
            candidate = staging_root/name
            if (candidate/manifest['experiment']/'COMPLETED').is_file():
                staging = candidate
    if staging is None:
        staging = Path(tempfile.mkdtemp(prefix=expected[:12]+'_', dir=staging_root))
        print(f'Extracting into {staging}', flush=True)
        extract_archive(manifest, archive, staging)
        study.core.atomic(state_path, dict(archive_sha256=expected, directory=staging.name))
    output = staging/manifest['experiment']
    try:
        if audit_replay:
            from audit_rbf_ridge_cv_benchmark_download import audit
            report = audit(manifest, output, archive)
            report['archive_sha256'] = expected
            report_path = repo/'output'/'audits'/manifest['experiment']/'report.json'
            report_path.parent.mkdir(parents=True, exist_ok=True)
            study.core.atomic(report_path, report)
            print({k: v for k, v in report.items() if k != 'replay_differences'}, flush=True)
            print(f'Audit report: {report_path}', flush=True)
        else:
            print('Running the original strict local verifier (including numerical replay).', flush=True)
            with threadpool_limits(limits=1), study.core.exclusive(output):
                report = study.verify(manifest, output)
            print(report, flush=True)
        plot(output)
        if audit_replay:
            analysis = output/'analysis'
            study.core.atomic(analysis/'portability_audit.json', report)
            notice = ('Saved server results: integrity passed. Local numerical replay: '
                      +report['numerical_replay_status']+'. '
                      'Plots use unchanged server p-values. See portability_audit.json for details.')
            (analysis/'PORTABILITY.txt').write_text(notice+'\n', encoding='utf-8')
            gallery = analysis/'index.html'
            content = gallery.read_text(encoding='utf-8')
            import html
            content = content.replace('<h1>', '<p><strong>'+html.escape(notice)+'</strong></p><h1>', 1)
            gallery.write_text(content, encoding='utf-8')
    except Exception as error:
        raise RuntimeError(f'Local validation or plotting failed: {error}. '
                           f'Cached archive and staged results are preserved at {output}; '
                           'no final results directory was created.') from error
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {target}')
    output.rename(target)
    if audit_replay:
        print(f'Integrity-audited server results and plots saved to {target}', flush=True)
        if report['numerical_replay_status'] != 'passed':
            print('Local numerical replay has mismatches; this is not a numerical-verification pass.', flush=True)
    else:
        print(f'Verified results and plots saved to {target}', flush=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--checksum', type=Path, required=True)
    parser.add_argument('--audit-replay', action='store_true',
                        help='Keep integrity checks strict and report local replay differences separately')
    args = parser.parse_args()
    try:
        target = finalize(args.archive, args.checksum, audit_replay=args.audit_replay)
        if args.audit_replay:
            report = study.core.read(target/'analysis'/'portability_audit.json')
            if report['numerical_replay_status'] != 'passed':
                raise SystemExit(2)
    except Exception as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
