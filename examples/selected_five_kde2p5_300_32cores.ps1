param(
    [ValidateSet('Preflight','Launch','Status','Diagnose','Stop','Resume','Verify','Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'selected_five_kde2p5_300_20260923'
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
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing command: $command" }
}

$header = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_selected_five_kde2p5_300"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUT="$REPO/outputs/selection_method_validation/exact_set/selected_five_kde2p5_300_20260923"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1
export PYTHONPATH="$REPO/src:$REPO"
'@

$idleCheck = @'
if pgrep -af '[v]alidate_selection_methods.py|[r]un_.*shards|[r]un_selected_five_kde2p5_300.py|[r]un_mi_cpu_sweep.py|[r]un_selected_five_ais_exact_mi30.py'; then
  echo 'A competing validation experiment is active. No process was stopped.' >&2
  exit 2
fi
'@

if ($Action -in 'Preflight','Launch') {
    $preflight = @'
hostname
for tool in bash tar base64 nohup setsid pgrep ps xargs awk seq cmp mktemp getconf readlink tr; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32 || { echo 'Server needs at least 32 online CPU cores.' >&2; exit 2; }
test -x "$PYTHON"
"$PYTHON" -c 'import numpy,pandas,scipy,sklearn,shap,matplotlib,patsy; print("Required imports available")'
test ! -e "$REPO" || { echo 'Experiment checkout already exists; use Status or Resume.' >&2; exit 2; }
'@
    Invoke-RemoteBash ($header + "`n" + $preflight + "`n" + $idleCheck)
}

if ($Action -eq 'Launch') {
    $sourceName = "selected_five_kde2p5_300_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $sourceName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache -C $localRepo src examples tests docs pyproject.toml README.md
    if ($LASTEXITCODE -ne 0) { throw 'Source archive creation failed.' }
    & scp -o ConnectTimeout=15 $archive "${Server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Source upload failed.' }
    $deploy = @'
STAGING="$(mktemp -d "$HOME/selected_five_kde2p5_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/examples/run_selected_five_kde2p5_300.py"
"$PYTHON" - "$STAGING" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])/'examples/run_selection_method_validation_shards.sh'
p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
PY
bash -n "$STAGING/examples/run_selection_method_validation_shards.sh"
mv -- "$STAGING" "$REPO"
cd "$REPO"
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
print('Source location and pinned package versions verified.')
PY
"$PYTHON" examples/run_selected_five_kde2p5_300.py check
mkdir -p "$OUT"
'@
    Invoke-RemoteBash ($header + "`n" + $deploy.Replace('__SOURCE__',$sourceName))
}

if ($Action -eq 'Resume') {
    # Repair the original launcher bug in an existing isolated checkout. The
    # repair is allowed only before any exact-spline result has been written.
    $runner = Join-Path $localRepo 'examples/run_selected_five_kde2p5_300.py'
    $repairName = "run_selected_five_kde2p5_300_repair_${stamp}.py"
    & scp -o ConnectTimeout=15 $runner "${Server}:~/$repairName"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading the exact-spline repair failed.' }
    $repair = @'
cd "$REPO"
TARGET="$REPO/examples/run_selected_five_kde2p5_300.py"
NEW="$HOME/__REPAIR__"
test -f "$TARGET" && test -f "$NEW"
if cmp -s "$TARGET" "$NEW"; then
  rm -f -- "$NEW"
  echo 'Exact-spline repair is already installed.'
else
  test -f "$OUT/runs/ais/pooled_summary/validation_settings.json"
  exact_results=$(find "$OUT/runs/exact_spline" -type f \( -name p_value_results.csv -o -name target_results.csv \) 2>/dev/null | wc -l)
  test "$exact_results" -eq 0 || { echo 'Refusing source repair after exact-spline results exist.' >&2; exit 2; }
  REPAIR_STAMP="$(date +%Y%m%d_%H%M%S)"
  cp -- "$TARGET" "$TARGET.BeforeExactRepair_$REPAIR_STAMP"
  mv -- "$NEW" "$TARGET"
  if [[ -f "$OUT/runs/exact_spline/run_config.txt" ]]; then
    mv -- "$OUT/runs/exact_spline/run_config.txt" "$OUT/runs/exact_spline/run_config.txt.BeforeExactRepair_$REPAIR_STAMP"
  fi
  "$PYTHON" -m py_compile "$TARGET"
  "$PYTHON" examples/run_selected_five_kde2p5_300.py check >/dev/null
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
from examples.run_selected_five_kde2p5_300 import read_manifest, source_hashes
m=read_manifest()
out=Path(m['output_root'])
(out/'source_hashes.json').write_text(json.dumps(source_hashes(),indent=2)+'\n')
(out/'EXACT_INFERENCE_REPAIR').write_text('Exact arm disables AIS stopping and uses baseline AIS strategy; AIS outputs preserved.\n')
PY
  echo 'Installed exact-spline compatibility repair; AIS results were preserved.'
fi
'@
    Invoke-RemoteBash ($header + "`n" + $repair.Replace('__REPAIR__',$repairName))
}

if ($Action -in 'Launch','Resume') {
    $launch = @'
cd "$REPO"
test ! -f "$OUT/COMPLETED" || { cat "$OUT/COMPLETED"; exit 0; }
"$PYTHON" -c 'from examples.run_selected_five_kde2p5_300 import read_manifest,lock; lock(read_manifest())'
STAMP="$(date +%Y%m%d_%H%M%S)"
for name in launcher.log launcher.pid; do
  if [[ -f "$OUT/$name" ]]; then cp "$OUT/$name" "$OUT/$name.Backup_$STAMP"; fi
done
nohup setsid "$PYTHON" -u examples/run_selected_five_kde2p5_300.py run > "$OUT/launcher.log" 2>&1 < /dev/null &
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
for arm in ais exact_spline; do
  count=$(find "$OUT/runs/$arm" -mindepth 2 -maxdepth 2 -type f -name validation_settings.json -path '*/shard_*/validation_settings.json' 2>/dev/null | wc -l)
  echo "$arm: $count/32 completed shards"
done
echo 'Active validation workers (must be <=32):'
pgrep -fc '[v]alidate_selection_methods.py' || true
tail -n 30 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}

if ($Action -eq 'Diagnose') {
    $diagnose = @'
test -d "$OUT"
"$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
arms = {
    "ais": ("shap", "lime", "spline_screening", "mutual_information"),
    "exact_spline": ("spline_screening",),
}
folders = (
    ("null_fresh", "p_value_results.csv"),
    ("power_feature_0_beta_0p3", "target_results.csv"),
    ("power_feature_0_beta_0p5", "target_results.csv"),
    ("power_feature_0_beta_0p75", "target_results.csv"),
    ("power_feature_0_beta_1p0", "target_results.csv"),
)
for arm, methods in arms.items():
    missing = []
    for shard in range(32):
        for method in methods:
            for folder, filename in folders:
                path = out / "runs" / arm / f"shard_{shard}" / method / "baseline" / folder / filename
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(path.relative_to(out).as_posix())
    print(f"{arm}: {len(missing)} missing result files")
    for path in missing[:100]:
        print("  MISSING", path)
    if len(missing) > 100:
        print(f"  ... and {len(missing)-100} more")
print("Grouped fatal errors:")
groups = {}
for log in sorted(out.glob("runs/*/logs/shard_*.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    if not any(marker in text for marker in ("Traceback", "Error:", "Exception:", "Killed", "No space left")):
        continue
    lines = [line for line in text.splitlines() if line.strip()]
    signature = lines[-1] if lines else "empty log"
    groups.setdefault(signature, []).append((log, lines))
if not groups:
    print("  No standard fatal-error marker found.")
for signature, items in groups.items():
    print(f"  {len(items)} shard(s): {signature}")
    log, lines = items[0]
    print(f"===== representative: {log} =====")
    print("\n".join(lines[-80:]))
PY
echo 'Launcher tail:'
tail -n 80 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $diagnose)
}

if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUT/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
[[ "$(ps -p "$PID" -o args=)" == *"examples/run_selected_five_kde2p5_300.py run"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
kill -TERM -- "-$PID"
sleep 2
pgrep -a -g "$PID" || echo 'Experiment stopped; completed shards were retained.'
'@
    Invoke-RemoteBash ($header + "`n" + $stop)
}

if ($Action -in 'Verify','Download') {
    $verify = @'
cd "$REPO"
test -f "$OUT/COMPLETED"
"$PYTHON" examples/run_selected_five_kde2p5_300.py verify
test -f "$OUT/unified_tests.csv"
test -f "$OUT/power_summary.csv"
test -f "$OUT/calibration_summary.csv"
test "$(awk 'END{print NR-1}' "$OUT/unified_tests.csv")" -eq 7500
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
    if (-not $target.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes output directory.' }
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing output: $target" }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Result extraction failed.' }
    $pythonLocal = Join-Path $localRepo '.venv/Scripts/python.exe'
    & $pythonLocal examples/run_selected_five_kde2p5_300.py verify
    if ($LASTEXITCODE -ne 0) { throw 'Local verification failed.' }
    Write-Host "Verified results saved to $target"
}
