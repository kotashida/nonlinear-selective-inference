param(
    [ValidateSet('Launch', 'Status', 'Resume', 'Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$controlName = 'selected_five_ais_exact_mi30_64cores'
$outputNames = @(
    'selected_five_methods_robust_draw_floor_ais_exact_mi30',
    'selected_five_methods_robust_floor_multimodal_denominator_ais_exact_mi30'
)
Set-Location $localRepo

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
    & ssh $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}

foreach ($command in 'ssh', 'scp', 'tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command missing: $command"
    }
}
$remoteHeader = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_selected_five_ais_exact_mi30_64cores"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
BASE="$REPO/outputs/selection_method_validation/exact_set"
CONTROL="$BASE/selected_five_ais_exact_mi30_64cores"
ARMS=(selected_five_methods_robust_draw_floor_ais_exact_mi30 selected_five_methods_robust_floor_multimodal_denominator_ais_exact_mi30)
'@

if ($Action -eq 'Launch') {
    $preflight = @'
for tool in tar base64 bash nohup setsid pgrep ps xargs awk seq cmp mktemp getconf; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 64 || { echo 'Server needs at least 64 online CPU cores.' >&2; exit 2; }
test -x "$PYTHON"
test ! -e "$REPO" || { echo 'Experiment checkout exists. Use Status or Resume.' >&2; exit 2; }
if pgrep -af '[v]alidate_selection_methods.py|[r]un_selected_five_ais_exact_mi30|[r]un_.*shards.sh'; then
  echo 'A competing experiment is active; wait for it to finish before Launch.' >&2
  exit 2
fi
'@
    Invoke-RemoteBash ($remoteHeader + "`n" + $preflight)

    # Upload only historical configs needed for the 640-condition audit, plus
    # the canonical reference dataset. Historical AIS raw trials are not needed.
    $reference = 'outputs/selection_method_validation/exact_set/selected_five_methods_combined_ais_exact_mi30'
    $rows = Import-Csv -LiteralPath (Join-Path $localRepo "$reference/unified_tests.csv")
    $entries = [Collections.Generic.List[string]]::new()
    foreach ($entry in 'src', 'examples', 'pyproject.toml', 'README.md') { $entries.Add($entry) }
    foreach ($filename in 'unified_tests.csv', 'source_manifest.json', 'selected_configuration.json') {
        $entries.Add("$reference/$filename")
    }
    foreach ($method in 'shap', 'lime', 'mutual_information', 'spline_screening') {
        $source = ($rows | Where-Object selection_method -EQ $method | Select-Object -First 1).source_run
        if (-not $source) { throw "No historical source for $method" }
        foreach ($shard in 0..31) {
            foreach ($condition in 'null_fresh', 'power_feature_0_beta_0p3', 'power_feature_0_beta_0p5', 'power_feature_0_beta_0p75', 'power_feature_0_beta_1p0') {
                $entries.Add("outputs/selection_method_validation/exact_set/$source/shard_$shard/$method/baseline/$condition/settings.json")
            }
        }
    }
    foreach ($entry in $entries) {
        $resolved = [IO.Path]::GetFullPath((Join-Path $localRepo $entry))
        $repoPrefix = [IO.Path]::GetFullPath($localRepo).TrimEnd('\') + '\'
        if (-not $resolved.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Archive path escapes repository: $entry" }
        if (-not (Test-Path -LiteralPath $resolved)) { throw "Missing input: $entry" }
    }
    $archiveName = "selected_five_ais_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $archiveName
    $fileList = Join-Path $env:TEMP "selected_five_ais_source_${stamp}.txt"
    [IO.File]::WriteAllLines($fileList, $entries, [Text.UTF8Encoding]::new($false))
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache -C $localRepo -T $fileList
    if ($LASTEXITCODE -ne 0) { throw 'Creating source archive failed.' }
    & scp $archive "${Server}:~/$archiveName"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading source archive failed.' }
    $deploy = @'
test ! -e "$REPO"
mkdir -p "$REPO"
tar -xzf "$HOME/__SOURCE__" -C "$REPO"
cd "$REPO"
"$PYTHON" - <<'PY'
from pathlib import Path
for name in ('run_selected_five_ais_exact_mi30_64cores.sh', 'run_selection_method_validation_shards.sh'):
    p = Path('examples') / name
    p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
PY
bash -n examples/run_selected_five_ais_exact_mi30_64cores.sh
bash -n examples/run_selection_method_validation_shards.sh
# Use this isolated checkout directly; do not change the shared environment's
# editable-install registration while other checkouts may depend on it.
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
reference = Path('outputs/selection_method_validation/exact_set/selected_five_methods_combined_ais_exact_mi30')
manifest = json.loads((reference / 'source_manifest.json').read_text())
expected = next(iter(manifest.values()))['runtime_metadata']['package_versions']
actual = {name: metadata.version(name) for name in expected}
assert actual == expected, (actual, expected)
print('Historical package versions verified.')
PY
for setting in robust_draw_floor robust_floor_multimodal_denominator; do
  "$PYTHON" examples/run_selected_five_ais_exact_mi30.py "examples/selected_five_ais_${setting}.json" >/dev/null
done
mkdir -p "$CONTROL"
'@
    Invoke-RemoteBash ($remoteHeader + "`n" + $deploy.Replace('__SOURCE__', $archiveName))
}

if ($Action -in 'Launch', 'Resume') {
    $launch = @'
test -d "$REPO"
test "$(getconf _NPROCESSORS_ONLN)" -ge 64
if [[ -f "$CONTROL/COMPLETED" ]]; then echo 'Already completed. Use Download.'; exit 0; fi
if pgrep -af '[v]alidate_selection_methods.py|[r]un_selected_five_ais_exact_mi30|[r]un_.*shards.sh'; then
  echo 'An experiment is active; refusing a duplicate or competing launch.' >&2
  exit 2
fi
cd "$REPO"
mkdir -p "$CONTROL"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$CONTROL/launcher.log" ]]; then cp "$CONTROL/launcher.log" "$CONTROL/launcher.log.$(date +%Y%m%d_%H%M%S)"; fi
nohup setsid env PYTHON_BIN="$PYTHON" bash examples/run_selected_five_ais_exact_mi30_64cores.sh \
  > "$CONTROL/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$CONTROL/launcher.pid"
sleep 3
kill -0 "$PID" || { tail -n 60 "$CONTROL/launcher.log"; exit 1; }
echo "Launched PID $PID: two concurrent arms, 32 one-core shard workers each (64 total)."
'@
    Invoke-RemoteBash ($remoteHeader + "`n" + $launch)
}

if ($Action -eq 'Status') {
    $status = @'
test -d "$CONTROL"
if [[ -f "$CONTROL/COMPLETED" ]]; then echo COMPLETED; else echo RUNNING_OR_INCOMPLETE; fi
if [[ -f "$CONTROL/launcher.pid" ]]; then ps -p "$(cat "$CONTROL/launcher.pid")" -o pid,pgid,etime,args || true; fi
"$PYTHON" - "$REPO" <<'PY'
from pathlib import Path
import collections
import os
import sys
import time
repo = Path(sys.argv[1]).resolve()
def snapshot():
    workers = {}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'examples/validate_selection_methods.py' not in args or (process / 'cwd').resolve() != repo:
                continue
            fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
            workers[process.name] = (int(fields[11]) + int(fields[12]), fields[0], args)
        except (OSError, ValueError):
            pass
    return workers
first = snapshot()
started = time.monotonic()
time.sleep(2)
second = snapshot()
elapsed = time.monotonic() - started
ticks = os.sysconf('SC_CLK_TCK')
cpu = {pid: max(0, values[0] - first[pid][0]) / ticks / elapsed * 100
       for pid, values in second.items() if pid in first}
print(f'Live validation workers: {len(second)}/64 maximum; states: {dict(collections.Counter(v[1] for v in second.values()))}')
print(f'CPU sampled over {elapsed:.1f}s: {sum(cpu.values()):.1f}% total = {sum(cpu.values()) / 100:.1f} cores busy')
for pid in sorted(second, key=lambda pid: cpu.get(pid, 0), reverse=True)[:8]:
    values = second[pid]
    folder = values[2].split('--output-dir ', 1)[-1].split()[0]
    print(f'  PID {pid}: CPU {cpu.get(pid, 0):.1f}%, state {values[1]}, accumulated CPU {values[0]/ticks:.0f}s; {folder}')
if not second:
    print('No validation workers found. Check arm logs below for errors or report generation.')
PY
for arm in "${ARMS[@]}"; do
  echo "$arm"
  if [[ -f "$BASE/$arm/COMPLETED" ]]; then echo '  Report complete and audited'; fi
  for method in shap lime mutual_information spline_screening; do
    count=0
    for file in "$BASE/$arm/runs/$method"/shard_*/validation_settings.json; do
      [[ ! -f "$file" ]] || count=$((count + 1))
    done
    null_count=0
    power_count=0
    for file in "$BASE/$arm/runs/$method"/shard_*/"$method"/baseline/null_fresh/p_value_results.csv; do
      [[ ! -f "$file" ]] || null_count=$((null_count + 1))
    done
    for file in "$BASE/$arm/runs/$method"/shard_*/"$method"/baseline/power_feature_0_beta_*/target_results.csv; do
      [[ ! -f "$file" ]] || power_count=$((power_count + 1))
    done
    echo "  $method: $count/32 complete shards; null conditions $null_count/32; power conditions $power_count/128"
    shard_log="$BASE/$arm/runs/$method/logs/shard_0.log"
    if [[ -f "$shard_log" ]]; then
      echo "  $method shard 0 progress:"
      "$PYTHON" - "$shard_log" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
with p.open('rb') as f:
    f.seek(max(0, p.stat().st_size - 8192))
    lines = f.read().decode(errors='replace').replace('\r', '\n').splitlines()
for line in [line for line in lines if line.strip()][-4:]:
    print('    ' + line)
PY
    fi
  done
done
for setting in robust_draw_floor robust_floor_multimodal_denominator; do
  if [[ -f "$CONTROL/$setting.log" ]]; then echo "--- $setting ---"; tail -n 12 "$CONTROL/$setting.log"; fi
done
if [[ -f "$CONTROL/launcher.log" ]]; then tail -n 12 "$CONTROL/launcher.log"; fi
'@
    Invoke-RemoteBash ($remoteHeader + "`n" + $status)
}

if ($Action -eq 'Download') {
    $parent = Join-Path $localRepo 'outputs/selection_method_validation/exact_set'
    foreach ($name in @($outputNames) + @($controlName)) {
        if (Test-Path -LiteralPath (Join-Path $parent $name)) {
            throw "Local destination exists and will not be overwritten: $(Join-Path $parent $name)"
        }
    }
    $archiveName = "selected_five_ais_results_${stamp}.tar.gz"
    $pack = @'
test -f "$CONTROL/COMPLETED" || { echo 'Both arms must finish before Download. Use Status.' >&2; exit 2; }
for arm in "${ARMS[@]}"; do
  test -f "$BASE/$arm/COMPLETED"
  test -f "$BASE/$arm/trial_audit.csv"
  test -f "$BASE/$arm/unified_tests.csv"
done
tar -czf "$HOME/__ARCHIVE__" -C "$BASE" "${ARMS[@]}" selected_five_ais_exact_mi30_64cores
'@
    Invoke-RemoteBash ($remoteHeader + "`n" + $pack.Replace('__ARCHIVE__', $archiveName))
    $archive = Join-Path $env:TEMP $archiveName
    & scp "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Downloading output archive failed.' }
    & tar -tzf $archive > $null
    if ($LASTEXITCODE -ne 0) { throw 'Downloaded archive is invalid.' }
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Extracting output archive failed.' }
    foreach ($name in $outputNames) {
        $target = Join-Path $parent $name
        $selected = Get-Content -Raw -LiteralPath (Join-Path $target 'selected_configuration.json') | ConvertFrom-Json
        if (-not $selected.exact_reference_reused) { throw "Unexpected exact reference provenance: $target" }
        if (-not (Test-Path -LiteralPath (Join-Path $target 'COMPLETED'))) { throw "Missing completion marker: $target" }
        Write-Host "Downloaded $target"
    }
    Write-Host "Launcher logs: $(Join-Path $parent $controlName)"
}
