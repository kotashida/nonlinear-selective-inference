param(
    [ValidateSet('Launch', 'Status', 'Repair', 'Finalize', 'Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$remoteRepoName = 'nonlinear-selective-inference_ais_refinement_200'
$outputRelative = 'outputs/selection_method_validation/exact_set/ais_refinement_200'
$baselineRelative = 'outputs/selection_method_validation/exact_set/ais_fix_followup_200/all_target_diagnostics.csv'
$mechanismRelative = 'outputs/selection_method_validation/exact_set/ais_mechanism_ablation_200/all_target_diagnostics.csv'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
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

$remoteHeader = @"
set -euo pipefail
REPO="`$HOME/$remoteRepoName"
PYTHON="`$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUTPUT_BASE="`$REPO/$outputRelative"
BASELINE_CSV="`$REPO/$baselineRelative"
MECHANISM_CSV="`$REPO/$mechanismRelative"
"@

if ($Action -eq 'Launch') {
    foreach ($relative in $baselineRelative, $mechanismRelative) {
        $path = Join-Path $localRepo $relative
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing reference data: $path" }
    }
    $preflight = @"
for tool in tar base64 bash nohup setsid pgrep xargs; do command -v "`$tool" >/dev/null; done
test -x "`$PYTHON"
test ! -e "`$REPO" || { echo 'Remote refinement checkout already exists.' >&2; exit 2; }
if pgrep -af '[v]alidate_selection_methods.py|[r]un_ais_refinement_200.sh'; then
  echo 'A competing validation experiment is already running.' >&2
  exit 2
fi
"@
    Invoke-RemoteBash ($remoteHeader + "`n" + $preflight)

    $archiveName = "ais_refinement_200_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $archiveName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache `
        -C $localRepo src examples tests pyproject.toml README.md `
        $baselineRelative $mechanismRelative
    if ($LASTEXITCODE -ne 0) { throw 'Creating source archive failed.' }
    & scp $archive "${Server}:~/$archiveName"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading source archive failed.' }

    $deploy = @"
mkdir -p "`$REPO"
tar -xzf "`$HOME/$archiveName" -C "`$REPO"
cd "`$REPO"
"`$PYTHON" - <<'PY'
from pathlib import Path
for name in ('run_ais_refinement_200.sh', 'run_selection_method_validation_shards.sh'):
    path = Path('examples') / name
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
PY
bash -n examples/run_ais_refinement_200.sh
bash -n examples/run_selection_method_validation_shards.sh
"`$PYTHON" -m pip install -e . --no-deps
"`$PYTHON" -m pytest -q tests/test_inference.py tests/test_ais_fix_strategies.py tests/test_analyze_ais_refinement_200.py
mkdir -p "`$OUTPUT_BASE"
nohup setsid env PYTHON_BIN="`$PYTHON" OUTPUT_BASE="`$OUTPUT_BASE" \
  BASELINE_CSV="`$BASELINE_CSV" MECHANISM_CSV="`$MECHANISM_CSV" \
  bash examples/run_ais_refinement_200.sh > "`$OUTPUT_BASE/launcher.log" 2>&1 < /dev/null &
PID=`$!
echo "`$PID" > "`$OUTPUT_BASE/launcher.pid"
sleep 3
kill -0 "`$PID" || { tail -n 100 "`$OUTPUT_BASE/launcher.log"; exit 1; }
echo "Launched PID `$PID with 64 aggregate shard workers"
tail -n 30 "`$OUTPUT_BASE/launcher.log"
"@
    Invoke-RemoteBash ($remoteHeader + "`n" + $deploy)
}

if ($Action -eq 'Status') {
    $status = @'
test -d "$OUTPUT_BASE"
if [[ -f "$OUTPUT_BASE/COMPLETED" ]]; then echo COMPLETED; else echo RUNNING_OR_INCOMPLETE; fi
if [[ -f "$OUTPUT_BASE/launcher.pid" ]]; then
  ps -p "$(cat "$OUTPUT_BASE/launcher.pid")" -o pid,pgid,etime,args || true
fi
for condition in \
  a_robust_consolidation a_robust_draw_floor a_robust_draw_floor_pilot15 \
  a_cdf_interval a_robust_floor_multimodal_denominator a_full_stack_optimized; do
  count=0
  for shard in "$OUTPUT_BASE/$condition"/shard_*; do
    [[ -f "$shard/validation_settings.json" ]] || continue
    [[ -f "$shard/lime/baseline/null_fresh/p_value_results.csv" ]] || continue
    [[ -f "$shard/lime/baseline/power_feature_0_beta_1p0/feature_results.csv" ]] || continue
    count=$((count + 1))
  done
  echo "$condition: $count/48 completed shards"
done
if [[ -f "$OUTPUT_BASE/repair.log" ]]; then
  tail -n 60 "$OUTPUT_BASE/repair.log"
else
  tail -n 60 "$OUTPUT_BASE/launcher.log"
fi
'@
    Invoke-RemoteBash ($remoteHeader + "`n" + $status)
}

if ($Action -eq 'Repair') {
    $localInference = Join-Path $localRepo 'src\si_shap\inference.py'
    $localTests = Join-Path $localRepo 'tests\test_ais_fix_strategies.py'
    & scp $localInference "${Server}:~/ais_refinement_inference.py"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading corrected inference.py failed.' }
    & scp $localTests "${Server}:~/test_ais_fix_strategies.py"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading corrected regression test failed.' }
    $repair = @"
test -d "`$REPO"
test -d "`$OUTPUT_BASE"
if pgrep -af '[v]alidate_selection_methods.py|[r]un_ais_refinement_200.sh'; then
  echo 'The original run still has active workers; wait for them to exit before Repair.' >&2
  exit 2
fi
cp "`$HOME/ais_refinement_inference.py" "`$REPO/src/si_shap/inference.py"
cp "`$HOME/test_ais_fix_strategies.py" "`$REPO/tests/test_ais_fix_strategies.py"
cd "`$REPO"
"`$PYTHON" -m py_compile src/si_shap/inference.py
"`$PYTHON" -m pytest -q tests/test_ais_fix_strategies.py
if [[ -d "`$OUTPUT_BASE/a_cdf_interval" ]]; then
  ARCHIVED_INTERVAL="`$OUTPUT_BASE/a_cdf_interval_pre_repair_$stamp"
  test ! -e "`$ARCHIVED_INTERVAL"
  mv -- "`$OUTPUT_BASE/a_cdf_interval" "`$ARCHIVED_INTERVAL"
  echo "Archived prior interval-arm results at `$ARCHIVED_INTERVAL; rerunning all interval shards consistently."
fi
nohup setsid env PYTHON_BIN="`$PYTHON" OUTPUT_BASE="`$OUTPUT_BASE" \
  BASELINE_CSV="`$BASELINE_CSV" MECHANISM_CSV="`$MECHANISM_CSV" \
  bash examples/run_ais_refinement_200.sh > "`$OUTPUT_BASE/repair.log" 2>&1 < /dev/null &
PID=`$!
echo "`$PID" > "`$OUTPUT_BASE/launcher.pid"
sleep 3
kill -0 "`$PID" || { tail -n 100 "`$OUTPUT_BASE/repair.log"; exit 1; }
echo "Repair deployed; resumed PID `$PID, rerunning interval arm and caching the other five arms."
tail -n 30 "`$OUTPUT_BASE/repair.log"
"@
    Invoke-RemoteBash ($remoteHeader + "`n" + $repair)
}

if ($Action -eq 'Finalize') {
    $analyzer = Join-Path $localRepo 'examples\analyze_ais_refinement_200.py'
    & scp $analyzer "${Server}:~/analyze_ais_refinement_200.py"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading analyzer failed.' }
    $finalize = @"
test -d "`$OUTPUT_BASE"
test -f "`$BASELINE_CSV"
test -f "`$MECHANISM_CSV"
cp "`$HOME/analyze_ais_refinement_200.py" "`$REPO/examples/analyze_ais_refinement_200.py"
cd "`$REPO"
"`$PYTHON" examples/analyze_ais_refinement_200.py "`$OUTPUT_BASE" \
  --baseline-csv "`$BASELINE_CSV" --mechanism-csv "`$MECHANISM_CSV"
test -f "`$OUTPUT_BASE/COMPLETED"
cat "`$OUTPUT_BASE/README.md"
"@
    Invoke-RemoteBash ($remoteHeader + "`n" + $finalize)
}

if ($Action -eq 'Download') {
    $archiveName = "ais_refinement_200_results_${stamp}.tar.gz"
    $pack = @"
test -f "`$OUTPUT_BASE/COMPLETED" || { echo 'Run is not complete.' >&2; exit 2; }
test -f "`$OUTPUT_BASE/arm_summary.csv"
test -f "`$OUTPUT_BASE/paired_failure_comparisons.csv"
test -f "`$OUTPUT_BASE/paired_cost_comparisons.csv"
tar -czf "`$HOME/$archiveName" -C "`$(dirname "`$OUTPUT_BASE")" "`$(basename "`$OUTPUT_BASE")"
"@
    Invoke-RemoteBash ($remoteHeader + "`n" + $pack)
    $archive = Join-Path $env:TEMP $archiveName
    & scp "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Downloading result archive failed.' }
    & tar -tzf $archive > $null
    if ($LASTEXITCODE -ne 0) { throw 'Downloaded archive is invalid.' }
    $parent = Join-Path $localRepo 'outputs\selection_method_validation\exact_set'
    New-Item -ItemType Directory -Force $parent | Out-Null
    $target = Join-Path $parent 'ais_refinement_200'
    if (Test-Path -LiteralPath $target) {
        throw "Local output already exists and was not overwritten: $target"
    }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Extracting results failed.' }
    Get-Content (Join-Path $target 'README.md')
    Write-Host "Results saved to $target"
}
