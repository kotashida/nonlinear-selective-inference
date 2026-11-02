param(
    [ValidateSet('Preflight','Launch','Status','Stop','Resume','Verify','Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)
$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'mi_cpu_sweep_1000_20260920'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Set-Location $localRepo

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
    & ssh -o ConnectTimeout=15 $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}
foreach ($command in 'ssh','scp','tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing $command" }
}
$header = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_mi_cpu_sweep_1000_20260920"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUT="$REPO/outputs/selection_method_validation/exact_set/mi_parameter_tuning/mi_cpu_sweep_1000_20260920"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1
export PYTHONPATH="$REPO/src:$REPO"
'@
$idleCheck = @'
if pgrep -af '[v]alidate_selection_methods.py|[r]un_.*shards|[c]ompare_selection_event|[r]un_mi_cpu_sweep.py|[r]un_selected_five|[r]un_spline_fresh_null|[r]un_linear_shapley'; then
  echo 'Competing experiment found. No processes were stopped. Wait before launch/resume.' >&2
  exit 2
fi
'@
if ($Action -in 'Preflight','Launch') {
    $preflight = @'
hostname
for tool in bash tar base64 nohup setsid pgrep ps xargs awk seq cmp mktemp getconf readlink tr; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32
test -x "$PYTHON"
"$PYTHON" -c 'import numpy,pandas,scipy,sklearn,shap,matplotlib,patsy; print("Required imports available")'
test ! -e "$REPO" || { echo 'Checkout already exists; use Status/Resume.' >&2; exit 2; }
'@
    Invoke-RemoteBash ($header + "`n" + $preflight + "`n" + $idleCheck)
}
if ($Action -eq 'Launch') {
    $sourceName = "mi_cpu_sweep_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $sourceName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache -C $localRepo src examples tests docs pyproject.toml README.md
    if ($LASTEXITCODE -ne 0) { throw 'Source archive failed.' }
    & scp -o ConnectTimeout=15 $archive "${Server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Upload failed.' }
    $deploy = @'
test ! -e "$REPO"
STAGING="$(mktemp -d "$HOME/mi_cpu_sweep_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/examples/run_mi_cpu_sweep.py"
"$PYTHON" - "$STAGING" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])/'examples/run_selection_method_validation_shards.sh'
p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
PY
bash -n "$STAGING/examples/run_selection_method_validation_shards.sh"
mv -- "$STAGING" "$REPO"
cd "$REPO"
# Use this isolated source through PYTHONPATH; do not alter the shared venv's
# editable registration or affect other experiment checkouts.
"$PYTHON" - <<'PY'
import importlib.metadata as md
from pathlib import Path
import tomllib
import si_shap
root=Path.cwd()
assert Path(si_shap.__file__).resolve().is_relative_to(root/'src')
for requirement in tomllib.loads((root/'pyproject.toml').read_text())['project']['dependencies']:
    name,version=requirement.split('==')
    assert md.version(name)==version,(name,md.version(name),version)
print('Source location and all pinned package versions verified.')
PY
"$PYTHON" examples/run_mi_cpu_sweep.py check
mkdir -p "$OUT"
'@
    Invoke-RemoteBash ($header + "`n" + $deploy.Replace('__SOURCE__',$sourceName))
}
if ($Action -in 'Launch','Resume') {
    $launch = @'
cd "$REPO"
test ! -f "$OUT/COMPLETED" || { echo 'Already completed; use Verify/Download.'; exit 0; }
"$PYTHON" -c 'from examples.run_mi_cpu_sweep import read_manifest,lock; lock(read_manifest())'
STAMP="$(date +%Y%m%d_%H%M%S)"
for name in launcher.log launcher.pid; do
  if [[ -f "$OUT/$name" ]]; then cp "$OUT/$name" "$OUT/$name.Backup_$STAMP"; fi
done
nohup setsid "$PYTHON" -u examples/run_mi_cpu_sweep.py run > "$OUT/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
sleep 5
kill -0 "$PID" || { tail -n 50 "$OUT/launcher.log"; exit 1; }
test -f "$OUT/knn_30/run_config.txt"
echo "Launcher PID: $PID"
cat "$OUT/knn_30/run_config.txt"
ps -p "$PID" -o pid,pgid,etime,args
tail -n 15 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $idleCheck + "`n" + $launch)
}
if ($Action -eq 'Status') {
    $status = @'
test -d "$OUT"
if [[ -f "$OUT/COMPLETED" ]]; then cat "$OUT/COMPLETED"; fi
if [[ -f "$OUT/launcher.pid" ]]; then ps -p "$(cat "$OUT/launcher.pid")" -o pid,pgid,etime,args || true; fi
"$PYTHON" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);m=json.loads((out/'requested_manifest.json').read_text())
for c in m['configurations']:
    root=out/c['name']
    count=len(list(root.glob('shard_*/validation_settings.json')))
    print(c['name'],str(count)+'/32 completed shards','diagnostics='+str((root/'observed_diagnostics.csv').exists()))
PY
echo 'Active validation workers (must be <=32):'
pgrep -fc '[v]alidate_selection_methods.py' || true
tail -n 20 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}
if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUT/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
[[ "$(ps -p "$PID" -o args=)" == *"examples/run_mi_cpu_sweep.py run"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
kill -TERM -- "-$PID"
sleep 2
pgrep -a -g "$PID" || echo 'Experiment stopped; partial results retained.'
'@
    Invoke-RemoteBash ($header + "`n" + $stop)
}
if ($Action -in 'Verify','Download') {
    $verify = @'
cd "$REPO"
test -f "$OUT/COMPLETED"
"$PYTHON" examples/run_mi_cpu_sweep.py verify
test -f "$OUT/power_selection_summary.csv"
test -f "$OUT/calibration_summary.csv"
cat "$OUT/power_selection_summary.csv"
cat "$OUT/calibration_summary.csv"
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
    if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }
    $members = & tar -tzf $archive
    if ($LASTEXITCODE -ne 0) { throw 'Invalid archive.' }
    foreach ($member in $members) {
        if (-not $member.StartsWith("$experiment/") -or ($member -split '/') -contains '..') { throw "Unsafe archive entry: $member" }
    }
    $parent = Join-Path $localRepo 'outputs/selection_method_validation/exact_set/mi_parameter_tuning'
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $target = [IO.Path]::GetFullPath((Join-Path $parent $experiment))
    $prefix = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
    if (-not $target.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes output directory.' }
    if (Test-Path -LiteralPath $target) { Move-Item -LiteralPath $target -Destination "${target}_Backup_${stamp}" }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Extraction failed; previous results remain backed up.' }
    $pythonLocal = Join-Path $localRepo '.venv/Scripts/python.exe'
    & $pythonLocal -c "import json; from pathlib import Path; from examples.run_mi_cpu_sweep import read_manifest,verify_variant; m=read_manifest(); assert json.loads((Path(m['output_root'])/'requested_manifest.json').read_text())==m; assert sum(verify_variant(m,c) for c in m['configurations'])==50000; print('Verified all downloaded settings and 50,000 raw tests.')"
    if ($LASTEXITCODE -ne 0) { throw 'Local result verification failed.' }
    Write-Host "Verified results saved to $target"
}
