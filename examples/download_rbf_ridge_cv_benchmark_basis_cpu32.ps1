param(
    [string] $Server = 'shida.kota@cpu-a10.',
    [ValidateRange(1,5)] [int] $Attempts = 3,
    [ValidateRange(0,30)] [int] $RetryDelaySeconds = 3,
    [switch] $LocalOnly,
    [switch] $AuditReplay
)
$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$pythonLocal = Join-Path $localRepo '.venv/Scripts/python.exe'
$experiment = 'rbf_ridge_cv_benchmark_basis_cpu32_20261031'
$archiveName = "${experiment}_verified_download.tar.gz"
$cache = Join-Path $localRepo "tmp/downloads/$experiment"
$archive = Join-Path $cache $archiveName
$checksum = "$archive.sha256"
$target = Join-Path $localRepo "outputs/benchmarks/$experiment"
$sshOptions = @('-o','ConnectTimeout=20','-o','ConnectionAttempts=1',
                '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4')
Set-Location $localRepo
if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing results: $target" }
New-Item -ItemType Directory -Path $cache -Force | Out-Null

function Invoke-Transport {
    param([string] $Description, [scriptblock] $Command)
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        Write-Host "$Description (attempt $attempt/$Attempts)"
        & $Command
        $code = $LASTEXITCODE
        if ($code -eq 0) { return }
        if ($code -ne 255 -or $attempt -eq $Attempts) {
            throw "$Description failed (exit $code). Cached files are preserved; rerun this downloader to retry."
        }
        Write-Host 'SSH transport failed; retrying the same download step.'
        if ($RetryDelaySeconds -gt 0) { Start-Sleep -Seconds $RetryDelaySeconds }
    }
}

function Invoke-RemoteBash {
    param([string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
    Invoke-Transport 'Prepare verified remote archive' {
        & ssh @sshOptions $Server "printf '%s' '$encoded' | base64 --decode | bash"
    }
}

if (-not $LocalOnly) {
    foreach ($command in 'ssh','scp') {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing command: $command" }
    }
    $prepare = @'
set -Eeuo pipefail
trap 'rc=$?; printf "Remote Bash failed (exit %s), line %s: %s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR
REPO="$HOME/nonlinear-selective-inference_rbf_ridge_cv_benchmark_basis_cpu32_20261031"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
CONFIG="benchmarks/rbf_ridge_cv_benchmark_basis_cpu32_20261031.json"
OUT="$REPO/outputs/benchmarks/rbf_ridge_cv_benchmark_basis_cpu32_20261031"
ARCHIVE="$HOME/rbf_ridge_cv_benchmark_basis_cpu32_20261031_verified_download.tar.gz"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 BLIS_NUM_THREADS=1
cd "$REPO"
test -f "$OUT/COMPLETED"
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Experiment active; use Status.' >&2; exit 2; }
exec 8>"$ARCHIVE.lock"
flock -n 8 || { echo 'Another recovery downloader is packaging this archive; retry later.' >&2; exit 2; }
IDENTITY="$(sha256sum "$OUT/COMPLETED" "$OUT/requested_manifest.json" "$OUT/provenance.json")"
if [[ -f "$ARCHIVE" && -f "$ARCHIVE.sha256" && -f "$ARCHIVE.identity" && -f "$ARCHIVE.verification.json" ]]; then
  [[ "$(cat "$ARCHIVE.identity")" == "$IDENTITY" ]] || { echo 'Cached archive belongs to different completion metadata; refusing to replace it.' >&2; exit 2; }
  sha256sum -c "$ARCHIVE.sha256"
  echo 'Reusing the previously verified remote archive.'
  cat "$ARCHIVE.verification.json"
else
  REPORT_PART="$(mktemp "$ARCHIVE.report.part.XXXXXX")"
  ARCHIVE_PART="$(mktemp "$ARCHIVE.part.XXXXXX")"
  CHECKSUM_PART="$(mktemp "$ARCHIVE.checksum.part.XXXXXX")"
  IDENTITY_PART="$(mktemp "$ARCHIVE.identity.part.XXXXXX")"
  echo 'Verifying completed results, then packaging them in this SSH session.'
  "$PYTHON" benchmarks/run_rbf_ridge_cv_benchmark_basis.py verify --config "$CONFIG" > "$REPORT_PART"
  cat "$REPORT_PART"
  tar -czf "$ARCHIVE_PART" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  mv -T -- "$ARCHIVE_PART" "$ARCHIVE"
  sha256sum "$ARCHIVE" > "$CHECKSUM_PART"
  printf '%s\n' "$IDENTITY" > "$IDENTITY_PART"
  mv -T -- "$REPORT_PART" "$ARCHIVE.verification.json"
  mv -T -- "$IDENTITY_PART" "$ARCHIVE.identity"
  mv -T -- "$CHECKSUM_PART" "$ARCHIVE.sha256"
fi
'@
    Invoke-RemoteBash $prepare
    Invoke-Transport 'Download archive checksum' {
        & scp @sshOptions "${Server}:~/$archiveName.sha256" "$checksum.part"
    }
    Move-Item -LiteralPath "$checksum.part" -Destination $checksum -Force
    $expected = ((Get-Content -LiteralPath $checksum -Raw).Trim() -split '\s+')[0]
    if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw 'Malformed archive checksum.' }
    $cached = (Test-Path -LiteralPath $archive) -and
              ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ieq $expected)
    if ($cached) {
        Write-Host 'Reusing the checksum-matched local archive.'
    } else {
        Invoke-Transport 'Download result archive' {
            & scp @sshOptions "${Server}:~/$archiveName" "$archive.part"
        }
        if ((Get-FileHash -LiteralPath "$archive.part" -Algorithm SHA256).Hash -ine $expected) {
            throw 'Archive checksum mismatch; partial download is preserved.'
        }
        Move-Item -LiteralPath "$archive.part" -Destination $archive -Force
    }
} elseif (-not (Test-Path -LiteralPath $archive) -or -not (Test-Path -LiteralPath $checksum)) {
    throw 'No cached archive/checksum pair is available. Run once without -LocalOnly.'
}

$finalizeArgs = @('benchmarks/download_rbf_ridge_cv_benchmark_basis.py','--archive',$archive,'--checksum',$checksum)
if ($AuditReplay) { $finalizeArgs += '--audit-replay' }
& $pythonLocal @finalizeArgs
if ($LASTEXITCODE -eq 2 -and $AuditReplay) {
    Write-Warning 'Server results and plots were saved after passing integrity checks. Local numerical replay has differences; inspect analysis/portability_audit.json. This is not a numerical-verification pass.'
    exit 2
}
if ($LASTEXITCODE -ne 0) { throw 'Local finalization failed. Archive and staging are preserved; see the diagnostic above.' }
