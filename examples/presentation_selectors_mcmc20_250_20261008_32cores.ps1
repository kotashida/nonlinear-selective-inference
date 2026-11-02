param(
    [ValidateSet('Preflight','Launch','Status','Stop','Resume','Verify','VerifyLocal','Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'presentation_selectors_mcmc20_250_20261008'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Set-Location $localRepo
$pythonLocal = Join-Path $localRepo '.venv/Scripts/python.exe'

if ($Action -eq 'VerifyLocal') {
    & $pythonLocal examples/run_presentation_selectors_mcmc20_250_20261008.py verify-downloaded
    if ($LASTEXITCODE -ne 0) { throw 'Local verification failed.' }
    return
}

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
    & ssh -o ConnectTimeout=15 $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}

foreach ($command in 'ssh','scp','tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing command: $command" }
}

$header = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_presentation_selectors_mcmc20_250_20261008"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUT="$REPO/outputs/selection_method_validation/exact_set/presentation_selectors_mcmc20_250_20261008"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1
export PYTHONPATH="$REPO/src:$REPO"
'@

$idleCheck = @'
if pgrep -af '[v]alidate_selection_methods.py|[r]un_.*shards|[r]un_shap_sampling_ten_250.py|[r]un_shap_sampling_followup.py|[r]un_presentation_selectors_mcmc20_250_20261008.py|[r]un_exact_pipeline_benchmark.py|[r]un_linear_shapley.*|[r]un_mi_cpu_sweep.py|[r]un_selected_five_ais_exact_mi30.py'; then
  echo 'A competing validation experiment is active. No process was stopped.' >&2
  exit 2
fi
'@

$dependencyCheck = @'
cd "$REPO"
"$PYTHON" - <<'PY'
import importlib.metadata as md
from pathlib import Path
import tomllib
import si_shap
assert Path(si_shap.__file__).resolve().is_relative_to(Path.cwd()/'src')
for requirement in tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies']:
    name,version=requirement.split('==')
    assert md.version(name)==version,(name,md.version(name),version)
print('Source location and pinned package versions verified.')
PY
'@

if ($Action -in 'Preflight','Launch') {
    & $pythonLocal examples/run_presentation_selectors_mcmc20_250_20261008.py check
    if ($LASTEXITCODE -ne 0) { throw 'Local manifest/fixture check failed.' }
    $preflight = @'
hostname
for tool in bash tar base64 nohup setsid pgrep ps awk mktemp getconf readlink tr flock; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32 || { echo 'Server needs at least 32 online CPU cores.' >&2; exit 2; }
test -x "$PYTHON"
"$PYTHON" -c 'import numpy,pandas,scipy,sklearn,shap,matplotlib,patsy; print("Required imports available")'
test ! -e "$REPO" || { echo 'Experiment checkout already exists; use Status or Resume.' >&2; exit 2; }
'@
    Invoke-RemoteBash ($header + "`n" + $preflight + "`n" + $idleCheck)
}

if ($Action -eq 'Launch') {
    $sourceName = "presentation_mcmc250_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $sourceName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache -C $localRepo src examples tests docs pyproject.toml README.md
    if ($LASTEXITCODE -ne 0) { throw 'Source archive creation failed.' }
    & scp -o ConnectTimeout=15 $archive "${Server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Source upload failed.' }
    $deploy = @'
STAGING="$(mktemp -d "$HOME/presentation_mcmc250_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/examples/run_presentation_selectors_mcmc20_250_20261008.py"
test ! -e "$REPO"
mv -T -- "$STAGING" "$REPO"
'@
    Invoke-RemoteBash ($header + "`n" + $deploy.Replace('__SOURCE__',$sourceName) + "`n" + $dependencyCheck)
}

if ($Action -eq 'Resume') {
    $resumeCheck = @'
test -f "$OUT/requested_manifest.json"
'@
    Invoke-RemoteBash ($header + "`n" + $resumeCheck + "`n" + $dependencyCheck)
}

if ($Action -in 'Launch','Resume') {
    $launch = @'
cd "$REPO"
"$PYTHON" -c 'from examples.run_presentation_selectors_mcmc20_250_20261008 import read_manifest,lock; lock(read_manifest())'
test ! -f "$OUT/COMPLETED" || { cat "$OUT/COMPLETED"; exit 0; }
# The lock remains held by the launcher and its workers. It closes simultaneous
# Launch/Resume races and is released automatically when the process group ends.
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'This experiment is already running.' >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
for name in launcher.log launcher.pid; do
  if [[ -f "$OUT/$name" ]]; then cp "$OUT/$name" "$OUT/$name.Backup_$STAMP"; fi
done
nohup setsid "$PYTHON" -u examples/run_presentation_selectors_mcmc20_250_20261008.py run > "$OUT/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
sleep 5
kill -0 "$PID" || { tail -n 80 "$OUT/launcher.log"; exit 1; }
echo "Launcher PID: $PID"
ps -p "$PID" -o pid,pgid,etime,args
tail -n 20 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $idleCheck + "`n" + $launch)
}

if ($Action -eq 'Status') {
    $status = @'
test -d "$OUT"
if [[ -f "$OUT/launcher.pid" ]]; then ps -p "$(cat "$OUT/launcher.pid")" -o pid,pgid,etime,args || true; fi
cd "$REPO"
"$PYTHON" examples/run_presentation_selectors_mcmc20_250_20261008.py status
echo 'Active follow-up workers (must be <=32):'
pgrep -fc '[r]un_shap_sampling_followup.py worker' || true
tail -n 30 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}

if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUT/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
[[ "$(ps -p "$PID" -o args=)" == *"examples/run_presentation_selectors_mcmc20_250_20261008.py run"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
kill -TERM -- "-$PID"
sleep 2
pgrep -a -g "$PID" || echo 'Experiment stopped; completed arm receipts were retained.'
'@
    Invoke-RemoteBash ($header + "`n" + $stop)
}

if ($Action -in 'Verify','Download') {
    $verify = @'
cd "$REPO"
# Verification must not race with an active controller.
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Experiment is still running; use Status.' >&2; exit 2; }
"$PYTHON" examples/run_presentation_selectors_mcmc20_250_20261008.py verify
test -f "$OUT/unified_tests.csv"
test -f "$OUT/power_summary.csv"
test -f "$OUT/null_summary.csv"
test "$(awk 'END{print NR-1}' "$OUT/unified_tests.csv")" -eq 6250
cat "$OUT/COMPLETED"
'@
    Invoke-RemoteBash ($header + "`n" + $verify)
}

if ($Action -eq 'Download') {
    $archiveName = "${experiment}_${stamp}.tar.gz"
    $pack = @'
tar -czf "$HOME/__ARCHIVE__" -C "$(dirname "$OUT")" "$(basename "$OUT")"
'@
    Invoke-RemoteBash ($header + "`n" + $pack.Replace('__ARCHIVE__',$archiveName))
    $archive = Join-Path $env:TEMP $archiveName
    & scp -o ConnectTimeout=15 "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Result download failed.' }
    & $pythonLocal examples/run_presentation_selectors_mcmc20_250_20261008.py validate-archive --archive $archive
    if ($LASTEXITCODE -ne 0) { throw 'Unsafe result archive.' }
    $parent = Join-Path $localRepo 'outputs/selection_method_validation/exact_set'
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $target = [IO.Path]::GetFullPath((Join-Path $parent $experiment))
    $prefix = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
    if (-not $target.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes output directory.' }
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing output: $target" }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Result extraction failed.' }
    & $pythonLocal examples/run_presentation_selectors_mcmc20_250_20261008.py verify-downloaded
    if ($LASTEXITCODE -ne 0) { throw 'Local verification failed.' }
    Write-Host "Verified results saved to $target"
}
