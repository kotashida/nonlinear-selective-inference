param(
    [ValidateSet('Preflight','Launch','Status','Resume','Stop','Verify','Download','VerifyLocal')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)
$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'publication_five_mcmc20_20261009'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$pythonLocal = Join-Path $localRepo '.venv/Scripts/python.exe'
Set-Location $localRepo

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
    & ssh -o ConnectTimeout=15 $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}

if ($Action -eq 'VerifyLocal') {
    & $pythonLocal benchmarks/run_publication_mcmc.py verify-downloaded
    if ($LASTEXITCODE -ne 0) { throw 'Local result verification failed.' }
    return
}
foreach ($command in 'ssh','scp','tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing command: $command" }
}

$header = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_publication_five_mcmc20_20261009"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUT="$REPO/outputs/benchmarks/publication_five_mcmc20_20261009"
export PYTHONPATH="$REPO/src:$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1
'@
$idle = @'
if pgrep -u "$(id -u)" -af '[v]alidate_selection_methods.py|[r]un_.*shards|[r]un_.*validation|[r]un_shap_sampling|[r]un_presentation.*selectors|[r]un_exact_pipeline_benchmark|[r]un_linear_shapley|[r]un_mi_cpu_sweep|[r]un_selected_five|[r]un_publication_mcmc.py run'; then
  echo 'A competing experiment is active; nothing was stopped.' >&2
  exit 2
fi
'@
$dependency = @'
cd "$REPO"
"$PYTHON" - <<'PY'
from benchmarks.run_publication_mcmc import read_config, provenance
provenance(read_config())
print('Validated sources, prepared data, Python 3.11 and pinned dependencies verified.')
PY
'@

if ($Action -in 'Preflight','Launch') {
    # This check only reads already prepared inputs. Preparation is an explicit
    # local step in the study guide and never mutates a prior data snapshot.
    & $pythonLocal benchmarks/run_publication_mcmc.py check
    if ($LASTEXITCODE -ne 0) { throw 'Local read-only preflight failed.' }
    $preflight = @'
hostname
for tool in bash tar base64 nohup setsid pgrep ps awk mktemp getconf readlink tr flock sha256sum id; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32
test -x "$PYTHON"
"$PYTHON" -c 'import sys,numpy,pandas,scipy,sklearn,shap,patsy,threadpoolctl; assert sys.version_info[:2]==(3,11); print("CPU-server imports available")'
test ! -e "$REPO" || { echo 'Checkout already exists; use Status or Resume.' >&2; exit 2; }
awk '/MemAvailable/{print "Available memory (GiB):", $2/1048576}' /proc/meminfo
df -h "$HOME"
'@
    Invoke-RemoteBash ($header + "`n" + $preflight + "`n" + $idle)
}

if ($Action -eq 'Launch') {
    $sourceName = "publication_mcmc20_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $sourceName
    $files = @('src','pyproject.toml','README.md',
        'benchmarks/run_publication_mcmc.py','benchmarks/prepare_publication_mcmc_data.py','benchmarks/plot_publication_mcmc.py',
        'benchmarks/publication_mcmc_reference.json','benchmarks/publication_mcmc20_20261009.json',
        'benchmarks/publication_mcmc20_smoke.json','benchmarks/publication_mcmc_requirements.txt',
        'benchmarks/publication_mcmc_data','benchmarks/dataset',
        'docs/publication_mcmc_benchmark_study.md','docs/commands.md',
        'examples/publication_five_mcmc20_cpu32.ps1','tests/test_publication_mcmc_benchmark.py')
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache -C $localRepo @files
    if ($LASTEXITCODE -ne 0) { throw 'Source packaging failed.' }
    $checksum = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    & scp -o ConnectTimeout=15 $archive "${Server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Source upload failed.' }
    $deploy = @'
echo '__CHECKSUM__  '"$HOME/__SOURCE__" | sha256sum -c -
STAGING="$(mktemp -d "$HOME/publication_mcmc20_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/benchmarks/run_publication_mcmc.py"
test ! -e "$REPO"
mv -T -- "$STAGING" "$REPO"
'@
    Invoke-RemoteBash ($header + "`n" + $deploy.Replace('__SOURCE__',$sourceName).Replace('__CHECKSUM__',$checksum) + "`n" + $dependency)
}

if ($Action -eq 'Resume') {
    $resume = @'
test -f "$OUT/requested_manifest.json"
test -f "$OUT/provenance.json"
'@
    Invoke-RemoteBash ($header + "`n" + $resume + "`n" + $dependency)
}

if ($Action -in 'Launch','Resume') {
    $launch = @'
cd "$REPO"
test ! -f "$OUT/COMPLETED" || { cat "$OUT/COMPLETED"; exit 0; }
# A shared account-level budget lock also prevents this new launcher from
# overlapping another launcher using the same CPU budget convention.
exec 8>"$HOME/.nonlinear-selective-inference-cpu32.lock"
flock -n 8 || { echo 'The shared CPU budget is in use.' >&2; exit 2; }
mkdir -p "$OUT"
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'This experiment is active.' >&2; exit 2; }
"$PYTHON" -c 'from benchmarks.run_publication_mcmc import read_config,lock; lock(read_config())'
STAMP="$(date +%Y%m%d_%H%M%S)"
for name in launcher.pid launcher.log; do
  if [[ -f "$OUT/$name" ]]; then cp "$OUT/$name" "$OUT/$name.Backup_$STAMP"; fi
done
nohup setsid "$PYTHON" -u benchmarks/run_publication_mcmc.py run > "$OUT/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
sleep 3
kill -0 "$PID" || { tail -n 80 "$OUT/launcher.log"; exit 1; }
ps -p "$PID" -o pid,pgid,etime,args
tail -n 10 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $idle + "`n" + $launch)
}

if ($Action -eq 'Status') {
    $status = @'
test -d "$OUT"
if [[ -f "$OUT/launcher.pid" ]]; then
  PID="$(cat "$OUT/launcher.pid")"
  ps -p "$PID" -o pid,pgid,etime,args || true
  echo 'Processes and CPU usage in the recorded group:'
  ps -e -o pid,pgid,pcpu,rss,etime,args | awk -v group="$PID" 'NR==1 || $2==group'
fi
cd "$REPO"
"$PYTHON" benchmarks/run_publication_mcmc.py status
tail -n 30 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}

if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUT/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
[[ "$(ps -p "$PID" -o args=)" == *"benchmarks/run_publication_mcmc.py run"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
kill -TERM -- "-$PID"
sleep 2
pgrep -a -g "$PID" || echo 'Stopped; all committed arm receipts retained.'
'@
    Invoke-RemoteBash ($header + "`n" + $stop)
}

if ($Action -in 'Verify','Download') {
    $verify = @'
cd "$REPO"
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Controller is still active; use Status.' >&2; exit 2; }
"$PYTHON" benchmarks/run_publication_mcmc.py verify
"$PYTHON" benchmarks/plot_publication_mcmc.py
test -f "$OUT/COMPLETED"
cat "$OUT/verification.json"
'@
    Invoke-RemoteBash ($header + "`n" + $verify)
}

if ($Action -eq 'Download') {
    $archiveName = "${experiment}_${stamp}.tar.gz"
    $pack = @'
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Controller active; cannot archive changing outputs.' >&2; exit 2; }
tar -czf "$HOME/__ARCHIVE__" -C "$(dirname "$OUT")" "$(basename "$OUT")"
sha256sum "$HOME/__ARCHIVE__" > "$HOME/__ARCHIVE__.sha256"
'@
    Invoke-RemoteBash ($header + "`n" + $pack.Replace('__ARCHIVE__',$archiveName))
    $archive = Join-Path $env:TEMP $archiveName
    & scp -o ConnectTimeout=15 "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Result download failed.' }
    & scp -o ConnectTimeout=15 "${Server}:~/$archiveName.sha256" "$archive.sha256"
    if ($LASTEXITCODE -ne 0) { throw 'Checksum download failed.' }
    $expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split '\s+')[0]
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Result archive checksum mismatch.' }
    & $pythonLocal benchmarks/run_publication_mcmc.py validate-archive --archive $archive
    if ($LASTEXITCODE -ne 0) { throw 'Unsafe result archive.' }
    $parent = Join-Path $localRepo 'outputs/benchmarks'
    $target = [IO.Path]::GetFullPath((Join-Path $parent $experiment))
    $prefix = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
    if (-not $target.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output target escapes benchmarks.' }
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing results: $target" }
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Result extraction failed.' }
    & $pythonLocal benchmarks/run_publication_mcmc.py verify-downloaded
    if ($LASTEXITCODE -ne 0) { throw 'Local verification failed; inspect downloaded receipts.' }
    Write-Host "Verified results saved to $target"
}
