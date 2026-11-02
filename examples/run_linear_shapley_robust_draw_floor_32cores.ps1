param(
    [ValidateSet('Launch', 'Status', 'Resume', 'Stop', 'Verify', 'Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'linear_shapley_selected_five_matched_300_20260928'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Set-Location $localRepo

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
    & ssh -o ConnectTimeout=15 $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}

foreach ($command in 'ssh', 'scp', 'tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing command: $command" }
}

$header = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_linear_shapley_matched_300"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUT="$REPO/outputs/selection_method_validation/exact_set/linear_shapley_selected_five_matched_300_20260928"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="$REPO/src:$REPO"
'@

$idleCheck = @'
if pgrep -af '[v]alidate_selection_methods.py|[r]un_.*selection_method.*shards|[r]un_linear_shapley_selected_five_matched_300.py'; then
  echo 'A competing validation experiment is active. No process was stopped.' >&2
  exit 2
fi
'@

if ($Action -eq 'Launch') {
    $preflight = @'
hostname
for tool in bash tar base64 nohup setsid pgrep ps xargs awk seq getconf readlink tr; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32 || { echo 'Server needs at least 32 online CPU cores.' >&2; exit 2; }
test -x "$PYTHON"
"$PYTHON" -c 'import numpy,pandas,scipy,sklearn,shap,matplotlib,patsy; print("Required imports available")'
test ! -e "$REPO" || { echo 'Experiment checkout already exists; use Status or Resume.' >&2; exit 2; }
'@
    Invoke-RemoteBash ($header + "`n" + $preflight + "`n" + $idleCheck)

    $sourceName = "linear_shapley_matched_300_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $sourceName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache -C $localRepo src examples tests docs pyproject.toml README.md
    if ($LASTEXITCODE -ne 0) { throw 'Source archive creation failed.' }
    & scp -o ConnectTimeout=15 $archive "${Server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Source upload failed.' }
    $deploy = @'
STAGING="$(mktemp -d "$HOME/linear_shapley_matched_300_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/examples/run_linear_shapley_selected_five_matched_300.py"
"$PYTHON" - "$STAGING" <<'PY'
from pathlib import Path
import sys
for name in ('run_selection_method_validation_shards.sh', 'run_linear_shapley_robust_draw_floor_32cores.sh'):
    path = Path(sys.argv[1]) / 'examples' / name
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
PY
bash -n "$STAGING/examples/run_selection_method_validation_shards.sh"
bash -n "$STAGING/examples/run_linear_shapley_robust_draw_floor_32cores.sh"
mv -- "$STAGING" "$REPO"
cd "$REPO"
"$PYTHON" - <<'PY'
import importlib.metadata as md
from pathlib import Path
import tomllib
import si_shap
root = Path.cwd()
assert Path(si_shap.__file__).resolve().is_relative_to(root / 'src')
for requirement in tomllib.loads((root / 'pyproject.toml').read_text())['project']['dependencies']:
    name, version = requirement.split('==')
    assert md.version(name) == version, (name, md.version(name), version)
print('Source location and pinned package versions verified.')
PY
"$PYTHON" examples/run_linear_shapley_selected_five_matched_300.py check
mkdir -p "$OUT"
'@
    Invoke-RemoteBash ($header + "`n" + $deploy.Replace('__SOURCE__', $sourceName))
}

if ($Action -in 'Launch', 'Resume') {
    $launch = @'
cd "$REPO"
test ! -f "$OUT/COMPLETED" || { cat "$OUT/COMPLETED"; exit 0; }
"$PYTHON" -c 'from examples.run_linear_shapley_selected_five_matched_300 import read_manifest,lock; lock(read_manifest())'
STAMP="$(date +%Y%m%d_%H%M%S)"
for name in launcher.log launcher.pid; do
  if [[ -f "$OUT/$name" ]]; then cp "$OUT/$name" "$OUT/$name.Backup_$STAMP"; fi
done
nohup setsid "$PYTHON" -u examples/run_linear_shapley_selected_five_matched_300.py run > "$OUT/launcher.log" 2>&1 < /dev/null &
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
if [[ -f "$OUT/COMPLETED" ]]; then cat "$OUT/COMPLETED"; fi
if [[ -f "$OUT/launcher.pid" ]]; then ps -p "$(cat "$OUT/launcher.pid")" -o pid,pgid,etime,args || true; fi
count=$(find "$OUT/runs/linear_shapley" -mindepth 2 -maxdepth 2 -type f -name validation_settings.json -path '*/shard_*/validation_settings.json' 2>/dev/null | wc -l)
echo "linear_shapley: $count/32 completed shards"
echo 'Active validation workers (must be <=32):'
pgrep -fc '[v]alidate_selection_methods.py' || true
if [[ -f "$OUT/launcher.log" ]]; then tail -n 30 "$OUT/launcher.log"; fi
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}

if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUT/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
[[ "$(ps -p "$PID" -o args=)" == *"examples/run_linear_shapley_selected_five_matched_300.py run"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
kill -TERM -- "-$PID"
sleep 2
pgrep -a -g "$PID" || echo 'Experiment stopped; completed shards were retained.'
'@
    Invoke-RemoteBash ($header + "`n" + $stop)
}

if ($Action -in 'Verify', 'Download') {
    $verify = @'
cd "$REPO"
test -f "$OUT/COMPLETED"
"$PYTHON" examples/run_linear_shapley_selected_five_matched_300.py verify
test -f "$OUT/verification.json"
test -f "$OUT/configuration_audit.csv"
test -f "$OUT/runs/linear_shapley/pooled_summary/calibration_decisions.csv"
test -f "$OUT/runs/linear_shapley/pooled_summary/power_decisions.csv"
cat "$OUT/COMPLETED"
'@
    Invoke-RemoteBash ($header + "`n" + $verify)
}

if ($Action -eq 'Download') {
    $archiveName = "${experiment}_${stamp}.tar.gz"
    $pack = @'
tar -czf "$HOME/__ARCHIVE__" -C "$(dirname "$OUT")" "$(basename "$OUT")"
'@
    Invoke-RemoteBash ($header + "`n" + $pack.Replace('__ARCHIVE__', $archiveName))
    $archive = Join-Path $env:TEMP $archiveName
    & scp -o ConnectTimeout=15 "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Result download failed.' }
    $members = & tar -tzf $archive
    if ($LASTEXITCODE -ne 0) { throw 'Downloaded archive is invalid.' }
    foreach ($member in $members) {
        if (-not $member.StartsWith("$experiment/") -or ($member -split '/') -contains '..') {
            throw "Unsafe archive entry: $member"
        }
    }
    $parent = Join-Path $localRepo 'outputs/selection_method_validation/exact_set'
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $target = [IO.Path]::GetFullPath((Join-Path $parent $experiment))
    $prefix = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
    if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes output directory.' }
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing output: $target" }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Result extraction failed.' }
    & (Join-Path $localRepo '.venv/Scripts/python.exe') examples/run_linear_shapley_selected_five_matched_300.py verify
    if ($LASTEXITCODE -ne 0) { throw 'Local verification failed.' }
    Write-Host "Verified results saved to $target"
}
