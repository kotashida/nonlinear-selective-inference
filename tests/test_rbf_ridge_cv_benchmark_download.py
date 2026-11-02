"""Retry transport failures and keep failed downloads out of final results."""
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
import download_rbf_ridge_cv_benchmark_basis as download


def fixture_archive(tmp_path):
    manifest = download.study.read_config()
    archive = tmp_path/'results.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name, data in [('COMPLETED', b'{}'), ('requested_manifest.json', b'{}'),
                           ('results.csv', b'p_value\n0.02\n')]:
            member = tarfile.TarInfo(f'{manifest["experiment"]}/{name}')
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    checksum = tmp_path/'results.tar.gz.sha256'
    checksum.write_text(download.study.digest(archive)+'  results.tar.gz\n')
    return manifest, archive, checksum


def test_failed_validation_is_staged_and_retry_uses_cached_extraction(tmp_path, monkeypatch):
    manifest, archive, checksum = fixture_archive(tmp_path)
    repo = tmp_path/'repo'
    seen = []

    def fail(m, output):
        seen.append(output)
        raise ValueError('Numerical replay mismatch')

    monkeypatch.setattr(download.study, 'verify', fail)
    with pytest.raises(RuntimeError, match='Numerical replay mismatch'):
        download.finalize(archive, checksum, repo)
    assert not (repo/manifest['output_root']).exists()
    assert (seen[0]/'results.csv').read_text() == 'p_value\n0.02\n'
    assert archive.exists() and checksum.exists()

    def verify(m, output):
        assert output == seen[0]
        assert m == manifest
        return dict(status='verified')

    def plot(output):
        (output/'analysis').mkdir()
        (output/'analysis'/'index.html').write_text('Verified plot gallery')

    monkeypatch.setattr(download.study, 'verify', verify)
    monkeypatch.setattr(download, 'plot', plot)
    target = download.finalize(archive, checksum, repo)
    assert target == repo/manifest['output_root']
    assert (target/'analysis'/'index.html').is_file()
    assert not seen[0].exists()
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        download.finalize(archive, checksum, repo)


def test_checksum_failure_stops_before_extraction(tmp_path):
    manifest, archive, checksum = fixture_archive(tmp_path)
    repo = tmp_path/'repo'
    checksum.write_text('not a sha256\n')
    with pytest.raises(ValueError, match='Malformed'):
        download.finalize(archive, checksum, repo)
    checksum.write_text('0'*64+'  results.tar.gz\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        download.finalize(archive, checksum, repo)
    assert not repo.exists()


@pytest.mark.parametrize('unsafe', ['../outside', 'RUN:stream', './COMPLETED'])
def test_archive_rejects_escaping_windows_streams_and_duplicate_names(tmp_path, unsafe):
    manifest = download.study.read_config()
    archive = tmp_path/'unsafe.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name in ('COMPLETED', 'requested_manifest.json', unsafe):
            member = tarfile.TarInfo(f'{manifest["experiment"]}/{name}')
            member.size = 1
            stream.addfile(member, io.BytesIO(b'x'))
    destination = tmp_path/'staging'
    destination.mkdir()
    with pytest.raises(ValueError, match='Unsafe archive|unsafe archive|Duplicate'):
        download.extract_archive(manifest, archive, destination)
    assert not (tmp_path/'outside').exists()


def powershell(script, tmp_path):
    executable = shutil.which('powershell.exe')
    if executable is None:
        pytest.skip('Windows PowerShell is unavailable')
    path = tmp_path/'test.ps1'
    path.write_text(script, encoding='utf-8')
    result = subprocess.run([executable, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-File', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    return result.stdout


def test_powershell_retries_only_transport_failures_and_limits_attempts(tmp_path):
    source = (ROOT/'examples/download_rbf_ridge_cv_benchmark_basis_cpu32.ps1').read_text()
    retry = source[source.index('function Invoke-Transport {'):source.index('function Invoke-RemoteBash {')]
    script = r'''
$ErrorActionPreference = 'Stop'
$Attempts = 3
$RetryDelaySeconds = 0
'''+retry+r'''
$global:count = 0
Invoke-Transport 'test transient failure' {
    $global:count++
    if ($global:count -lt 3) { $global:LASTEXITCODE = 255 } else { $global:LASTEXITCODE = 0 }
}
if ($global:count -ne 3) { throw 'Did not retry transient failure' }
$global:count = 0
$caught = $false
try {
    Invoke-Transport 'test action failure' { $global:count++; $global:LASTEXITCODE = 2 }
} catch { $caught = $true }
if (-not $caught -or $global:count -ne 1) { throw 'Action failure was retried' }
$global:count = 0
$caught = $false
try {
    Invoke-Transport 'test persistent transport failure' { $global:count++; $global:LASTEXITCODE = 255 }
} catch { $caught = $true }
if (-not $caught -or $global:count -ne 3) { throw 'Retry limit was not enforced' }
Write-Output 'Transport retry checks passed'
'''
    assert 'Transport retry checks passed' in powershell(script, tmp_path)


def test_powershell_syntax_and_remote_archive_atomic_commit(tmp_path):
    path = ROOT/'examples/download_rbf_ridge_cv_benchmark_basis_cpu32.ps1'
    source = path.read_text()
    script = r'''
$tokens = $null
$parseErrors = $null
[Management.Automation.Language.Parser]::ParseFile('''+"'"+str(path).replace("'", "''")+"'"+r''', [ref]$tokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
Write-Output 'Syntax OK'
'''
    assert 'Syntax OK' in powershell(script, tmp_path)
    assert source.index('verify --config "$CONFIG"') < source.index('tar -czf "$ARCHIVE_PART"')
    assert source.index('mv -T -- "$ARCHIVE_PART"') < source.index('mv -T -- "$CHECKSUM_PART"')
    assert 'ServerAliveInterval=15' in source
    assert 'Reusing the previously verified remote archive' in source
    assert 'Reusing the checksum-matched local archive' in source
    # The completed experiment's source list remains unchanged by this repair.
    assert 'examples/download_rbf_ridge_cv_benchmark_basis_cpu32.ps1' not in download.study.PACKAGED
    assert 'benchmarks/download_rbf_ridge_cv_benchmark_basis.py' not in download.study.PACKAGED
