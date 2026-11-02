param(
    [ValidateSet('Launch', 'Status', 'Download', 'Stop', 'LaunchK2', 'StatusK2', 'DownloadK2')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'lime_selection_regions_pilot10'
$remoteRepo = 'nonlinear-selective-inference_lime_selection_regions_pilot10'
$remoteOutput = "`$HOME/$remoteRepo/outputs/selection_regions/sweep/lime_ais_pilot_iters10"
$remoteK2Output = "`$HOME/$remoteRepo/outputs/selection_regions/sweep/lime_k2_ais_pilot_iters10"
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Set-Location $localRepo

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String(
        [System.Text.Encoding]::UTF8.GetBytes($normalized)
    )
    & ssh $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote Bash failed: $LASTEXITCODE" }
}

foreach ($command in 'ssh', 'scp', 'tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required local command missing: $command"
    }
}

if ($Action -eq 'Launch') {
    $preflight = @"
set -euo pipefail
for tool in tar base64 bash nohup pgrep; do command -v "`$tool" >/dev/null; done
test -x "`$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
test ! -e "`$HOME/$remoteRepo" || {
  echo 'Remote sweep directory already exists; use Status.' >&2
  exit 2
}
if pgrep -af '[s]weep_selection_region_settings.py'; then
  echo 'A selection-region sweep is already running.' >&2
  exit 2
fi
"@
    Invoke-RemoteBash -Script $preflight

    $archiveName = "${experiment}_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $archiveName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache `
        -C $localRepo src examples tests pyproject.toml README.md
    if ($LASTEXITCODE -ne 0) { throw 'Creating source archive failed.' }
    & scp $archive "${Server}:~/$archiveName"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading source archive failed.' }

    $launch = @"
set -euo pipefail
REPO="`$HOME/$remoteRepo"
PYTHON="`$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
mkdir -p "`$REPO"
tar -xzf "`$HOME/$archiveName" -C "`$REPO"
cd "`$REPO"
"`$PYTHON" -m pip install -e . --no-deps
mkdir -p "$remoteOutput"
nohup env PYTHONPATH="`$REPO/src" "`$PYTHON" \
  examples/sweep_selection_region_settings.py 1 2 3 4 5 6 \
  --preset recommended \
  --selection-method lime \
  --proposal-pilot-iters 10 \
  --proposal-pilot-samples 320 \
  --lime-num-explanations 40 \
  --lime-num-samples 64 \
  --lime-ridge-alpha 1 \
  --grid-size 1001 \
  --grid-refinements 1 \
  --n-jobs 32 \
  --output-dir "$remoteOutput" \
  > "$remoteOutput/launcher.log" 2>&1 < /dev/null &
echo "`$!" > "$remoteOutput/launcher.pid"
sleep 3
kill -0 "`$(cat "$remoteOutput/launcher.pid")"
echo "Launcher PID: `$(cat "$remoteOutput/launcher.pid")"
tail -n 10 "$remoteOutput/launcher.log"
"@
    Invoke-RemoteBash -Script $launch
}

if ($Action -eq 'Status') {
    $status = @"
set -euo pipefail
OUTPUT="$remoteOutput"
test -d "`$OUTPUT"
if [[ -f "`$OUTPUT/launcher.pid" ]]; then
  ps -p "`$(cat "`$OUTPUT/launcher.pid")" -o pid,etime,args || true
fi
find "`$OUTPUT" -name selection_regions.csv -type f | sort
tail -n 20 "`$OUTPUT/launcher.log"
"@
    Invoke-RemoteBash -Script $status
}

if ($Action -eq 'Stop') {
    $stop = @"
set -euo pipefail
OUTPUT="$remoteOutput"
PID="`$(cat "`$OUTPUT/launcher.pid")"
[[ "`$PID" =~ ^[1-9][0-9]*`$ ]]
ARGS="`$(ps -p "`$PID" -o args=)"
[[ "`$ARGS" == *'examples/sweep_selection_region_settings.py'* ]]
[[ "`$ARGS" == *'lime_ais_pilot_iters10'* ]]
kill -TERM "`$PID"
for _ in {1..20}; do
  kill -0 "`$PID" 2>/dev/null || { echo "Stopped PID `$PID"; exit 0; }
  sleep 1
done
echo "PID `$PID is still exiting; inspect with Status before LaunchK2." >&2
exit 2
"@
    Invoke-RemoteBash -Script $stop
}

if ($Action -eq 'LaunchK2') {
    $launchK2 = @"
set -euo pipefail
REPO="`$HOME/$remoteRepo"
PYTHON="`$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUTPUT="$remoteK2Output"
test -d "`$REPO"
test -x "`$PYTHON"
test ! -e "`$OUTPUT" || {
  echo 'The k=2 output directory already exists and was not overwritten.' >&2
  exit 2
}
if pgrep -af '[s]weep_selection_region_settings.py'; then
  echo 'A selection-region sweep is still running.' >&2
  exit 2
fi
cd "`$REPO"
mkdir -p "`$OUTPUT"
nohup env PYTHONPATH="`$REPO/src" "`$PYTHON" \
  examples/sweep_selection_region_settings.py 1 2 3 4 5 6 \
  --selection-method lime \
  --k-select 2 \
  --rf-config baseline \
  --selection-event exact_set \
  --proposal-pilot-iters 10 \
  --proposal-pilot-samples 320 \
  --lime-num-explanations 40 \
  --lime-num-samples 64 \
  --lime-ridge-alpha 1 \
  --grid-size 1001 \
  --grid-refinements 1 \
  --n-jobs 32 \
  --output-dir "`$OUTPUT" \
  > "`$OUTPUT/launcher.log" 2>&1 < /dev/null &
echo "`$!" > "`$OUTPUT/launcher.pid"
sleep 3
kill -0 "`$(cat "`$OUTPUT/launcher.pid")"
echo "Launcher PID: `$(cat "`$OUTPUT/launcher.pid")"
tail -n 10 "`$OUTPUT/launcher.log"
"@
    Invoke-RemoteBash -Script $launchK2
}

if ($Action -eq 'StatusK2') {
    $statusK2 = @"
set -euo pipefail
OUTPUT="$remoteK2Output"
test -d "`$OUTPUT"
if [[ -f "`$OUTPUT/launcher.pid" ]]; then
  ps -p "`$(cat "`$OUTPUT/launcher.pid")" -o pid,etime,args || true
fi
find "`$OUTPUT" -name selection_regions.csv -type f | sort
tail -n 20 "`$OUTPUT/launcher.log"
"@
    Invoke-RemoteBash -Script $statusK2
}

if ($Action -eq 'Download') {
    $archiveName = "${experiment}_${stamp}.tar.gz"
    $pack = @"
set -euo pipefail
OUTPUT="$remoteOutput"
test -f "`$OUTPUT/all_selection_regions.csv" || {
  echo 'Sweep is not complete; use Status.' >&2
  exit 2
}

tar -czf "`$HOME/$archiveName" -C "`$(dirname "`$OUTPUT")" "`$(basename "`$OUTPUT")"
"@
    Invoke-RemoteBash -Script $pack
    $archive = Join-Path $env:TEMP $archiveName
    & scp "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Downloading results failed.' }
    & tar -tzf $archive > $null
    if ($LASTEXITCODE -ne 0) { throw 'Downloaded archive is invalid.' }
    $localParent = Join-Path $localRepo 'outputs\selection_regions\sweep'
    $localTarget = Join-Path $localParent 'lime_ais_pilot_iters10'
    if (Test-Path -LiteralPath $localTarget) {
        throw "Local target already exists and was not overwritten: $localTarget"
    }
    New-Item -ItemType Directory -Force $localParent | Out-Null
    & tar -xzf $archive -C $localParent
    if ($LASTEXITCODE -ne 0) { throw 'Extracting results failed.' }
    Get-ChildItem -LiteralPath $localTarget
}

if ($Action -eq 'DownloadK2') {
    $archiveName = "lime_k2_selection_regions_pilot10_${stamp}.tar.gz"
    $packK2 = @"
set -euo pipefail
OUTPUT="$remoteK2Output"
test -f "`$OUTPUT/all_selection_regions.csv" || {
  echo 'The k=2 sweep is not complete; use StatusK2.' >&2
  exit 2
}
tar -czf "`$HOME/$archiveName" -C "`$(dirname "`$OUTPUT")" "`$(basename "`$OUTPUT")"
"@
    Invoke-RemoteBash -Script $packK2
    $archive = Join-Path $env:TEMP $archiveName
    & scp "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Downloading k=2 results failed.' }
    & tar -tzf $archive > $null
    if ($LASTEXITCODE -ne 0) { throw 'Downloaded archive is invalid.' }
    $localParent = Join-Path $localRepo 'outputs\selection_regions\sweep'
    $localTarget = Join-Path $localParent 'lime_k2_ais_pilot_iters10'
    if (Test-Path -LiteralPath $localTarget) {
        throw "Local target already exists and was not overwritten: $localTarget"
    }
    New-Item -ItemType Directory -Force $localParent | Out-Null
    & tar -xzf $archive -C $localParent
    if ($LASTEXITCODE -ne 0) { throw 'Extracting k=2 results failed.' }
    Get-ChildItem -LiteralPath $localTarget
}
