param(
    [ValidateSet('Launch', 'Status', 'Finalize', 'Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$remoteRepoName = 'nonlinear-selective-inference_ais_mechanism_ablation_200'
$outputRelative = 'outputs/selection_method_validation/exact_set/ais_mechanism_ablation_200'
$referenceRelative = 'outputs/selection_method_validation/exact_set/ais_fix_followup_200/all_target_diagnostics.csv'
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
REFERENCE_CSV="`$REPO/$referenceRelative"
"@

if ($Action -eq 'Launch') {
    $referencePath = Join-Path $localRepo $referenceRelative
    if (-not (Test-Path -LiteralPath $referencePath)) {
        throw "Missing carried ais_fix_followup_200 diagnostics: $referencePath"
    }
    $preflight = @"
for tool in tar base64 bash nohup setsid pgrep xargs; do command -v "`$tool" >/dev/null; done
test -x "`$PYTHON"
test ! -e "`$REPO" || { echo 'Remote AIS-ablation checkout already exists.' >&2; exit 2; }
if pgrep -af '[v]alidate_selection_methods.py|[r]un_ais_mechanism_ablation_200.sh'; then
  echo 'A competing validation experiment is already running.' >&2
  exit 2
fi
"@
    Invoke-RemoteBash ($remoteHeader + "`n" + $preflight)

    $archiveName = "ais_mechanism_ablation_200_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $archiveName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache `
        -C $localRepo src examples tests pyproject.toml README.md $referenceRelative
    if ($LASTEXITCODE -ne 0) { throw 'Creating source archive failed.' }
    & scp $archive "${Server}:~/$archiveName"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading source archive failed.' }

    $deploy = @"
mkdir -p "`$REPO"
tar -xzf "`$HOME/$archiveName" -C "`$REPO"
cd "`$REPO"
"`$PYTHON" - <<'PY'
from pathlib import Path
for name in (
    'run_ais_mechanism_ablation_200.sh',
    'run_selection_method_validation_shards.sh',
):
    path = Path('examples') / name
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
PY
bash -n examples/run_ais_mechanism_ablation_200.sh
bash -n examples/run_selection_method_validation_shards.sh
"`$PYTHON" -m pip install -e . --no-deps
"`$PYTHON" -m pytest -q tests/test_inference.py tests/test_ais_fix_strategies.py
mkdir -p "`$OUTPUT_BASE"
nohup setsid env PYTHON_BIN="`$PYTHON" OUTPUT_BASE="`$OUTPUT_BASE" REFERENCE_CSV="`$REFERENCE_CSV" \
  bash examples/run_ais_mechanism_ablation_200.sh > "`$OUTPUT_BASE/launcher.log" 2>&1 < /dev/null &
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
  a_multimodal_retention b_denominator_targeting c_stratified_exploration \
  ab_multimodal_denominator ac_multimodal_stratified \
  bc_denominator_stratified abc_all_mechanisms; do
  count=0
  for file in "$OUTPUT_BASE/$condition"/shard_*/validation_settings.json; do
    [[ ! -f "$file" ]] || count=$((count + 1))
  done
  echo "$condition: $count/48 completed shards"
done
tail -n 50 "$OUTPUT_BASE/launcher.log"
'@
    Invoke-RemoteBash ($remoteHeader + "`n" + $status)
}

if ($Action -eq 'Finalize') {
    $localAnalyzer = Join-Path $localRepo 'examples\analyze_ais_mechanism_ablation_200.py'
    if (-not (Test-Path -LiteralPath $localAnalyzer)) {
        throw "Missing local analyzer: $localAnalyzer"
    }
    & scp $localAnalyzer "${Server}:~/analyze_ais_mechanism_ablation_200.py"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading the corrected analyzer failed.' }
    $finalize = @"
test -d "`$OUTPUT_BASE"
test -f "`$REFERENCE_CSV"
cp "`$HOME/analyze_ais_mechanism_ablation_200.py" \
  "`$REPO/examples/analyze_ais_mechanism_ablation_200.py"
cd "`$REPO"
"`$PYTHON" examples/analyze_ais_mechanism_ablation_200.py \
  "`$OUTPUT_BASE" --reference-csv "`$REFERENCE_CSV"
test -f "`$OUTPUT_BASE/COMPLETED"
echo 'Final pooling completed successfully.'
cat "`$OUTPUT_BASE/README.md"
"@
    Invoke-RemoteBash ($remoteHeader + "`n" + $finalize)
}

if ($Action -eq 'Download') {
    $archiveName = "ais_mechanism_ablation_200_results_${stamp}.tar.gz"
    $pack = @"
test -f "`$OUTPUT_BASE/COMPLETED" || { echo 'Run is not complete.' >&2; exit 2; }
test -f "`$OUTPUT_BASE/failure_summary.csv"
test -f "`$OUTPUT_BASE/paired_failure_comparisons.csv"
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
    $target = Join-Path $parent 'ais_mechanism_ablation_200'
    if (Test-Path -LiteralPath $target) {
        throw "Local output already exists and was not overwritten: $target"
    }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Extracting results failed.' }
    Get-Content (Join-Path $target 'README.md')
    Write-Host "Results saved to $target"
}
