param(
    [ValidateSet('Launch', 'Status', 'Resume', 'Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$remoteName = 'nonlinear-selective-inference_spline_fresh_null_10000'
$outputName = 'spline_fresh_null_10000'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Set-Location $localRepo

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Script)
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes(($Script -replace "`r`n", "`n")))
    & ssh $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}

foreach ($command in 'ssh', 'scp', 'tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command missing: $command"
    }
}

$header = @'
set -euo pipefail
REPO="$HOME/__REMOTE__"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUT="$REPO/outputs/selection_method_validation/exact_set/__OUTPUT__"
'@.Replace('__REMOTE__', $remoteName).Replace('__OUTPUT__', $outputName)

if ($Action -eq 'Launch') {
    $preflight = @'
for tool in tar base64 bash nohup setsid pgrep getconf; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32 || { echo 'Server needs at least 32 online CPU cores.' >&2; exit 2; }
test -x "$PYTHON"
test ! -e "$REPO" || { echo 'Experiment checkout exists. Use Status or Resume.' >&2; exit 2; }
'@
    Invoke-RemoteBash ($header + "`n" + $preflight)

    $entries = @(
        'src', 'pyproject.toml', 'README.md',
        'examples/run_spline_fresh_null.py',
        'examples/spline_fresh_null_reference.json'
    )
    foreach ($entry in $entries) {
        if (-not (Test-Path -LiteralPath (Join-Path $localRepo $entry))) {
            throw "Missing source input: $entry"
        }
    }
    $archiveName = "spline_fresh_null_source_$stamp.tar.gz"
    $archive = Join-Path $env:TEMP $archiveName
    $list = Join-Path $env:TEMP "spline_fresh_null_source_$stamp.txt"
    [IO.File]::WriteAllLines($list, $entries, [Text.UTF8Encoding]::new($false))
    & tar -czf $archive --exclude=__pycache__ -C $localRepo -T $list
    if ($LASTEXITCODE -ne 0) { throw 'Creating source archive failed.' }
    & scp $archive "${Server}:~/$archiveName"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading source archive failed.' }
    $deploy = @'
mkdir -p "$REPO"
tar -xzf "$HOME/__ARCHIVE__" -C "$REPO"
cd "$REPO"
export PYTHONPATH="$REPO/src"
"$PYTHON" examples/run_spline_fresh_null.py --plan --workers 32 >/dev/null
'@.Replace('__ARCHIVE__', $archiveName)
    Invoke-RemoteBash ($header + "`n" + $deploy)
}

if ($Action -in 'Launch', 'Resume') {
    $launch = @'
test -d "$REPO"
test "$(getconf _NPROCESSORS_ONLN)" -ge 32
if [[ -f "$OUT/COMPLETED" ]]; then echo 'Already completed. Use Download.'; exit 0; fi
if [[ -f "$OUT/launcher.pid" ]] && kill -0 "$(cat "$OUT/launcher.pid")" 2>/dev/null; then
  echo 'Experiment is already running.' >&2; exit 2
fi
cd "$REPO"
mkdir -p "$OUT"
export PYTHONPATH="$REPO/src"
if [[ -f "$OUT/launcher.log" ]]; then cp "$OUT/launcher.log" "$OUT/launcher.log.$(date +%Y%m%d_%H%M%S)"; fi
nohup setsid "$PYTHON" examples/run_spline_fresh_null.py --workers 32 \
  > "$OUT/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
sleep 3
kill -0 "$PID" || { tail -n 80 "$OUT/launcher.log"; exit 1; }
echo "Launched PID $PID with at most 32 one-thread workers."
'@
    Invoke-RemoteBash ($header + "`n" + $launch)
}

if ($Action -eq 'Status') {
    $status = @'
test -d "$REPO"
if [[ -f "$OUT/COMPLETED" ]]; then echo COMPLETED; else echo RUNNING_OR_INCOMPLETE; fi
if [[ -f "$OUT/launcher.pid" ]]; then ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,%cpu,%mem,args || true; fi
completed=0
for file in "$OUT"/batches/batch_*/exact_spline.csv; do [[ ! -f "$file" ]] || completed=$((completed + 1)); done
echo "Exact batches complete: $completed/100"
completed=0
for file in "$OUT"/batches/batch_*/ais.csv; do [[ ! -f "$file" ]] || completed=$((completed + 1)); done
echo "AIS batches complete: $completed/100"
if [[ -f "$OUT/launcher.log" ]]; then tail -n 25 "$OUT/launcher.log"; fi
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}

if ($Action -eq 'Download') {
    $destination = Join-Path $localRepo "outputs/selection_method_validation/exact_set/$outputName"
    if (Test-Path -LiteralPath $destination) {
        throw "Local destination exists and will not be overwritten: $destination"
    }
    $archiveName = "spline_fresh_null_results_$stamp.tar.gz"
    $pack = @'
test -f "$OUT/COMPLETED" || { echo 'Experiment is not complete. Use Status.' >&2; exit 2; }
tar -czf "$HOME/__ARCHIVE__" -C "$(dirname "$OUT")" "$(basename "$OUT")"
'@.Replace('__ARCHIVE__', $archiveName)
    Invoke-RemoteBash ($header + "`n" + $pack)
    $archive = Join-Path $env:TEMP $archiveName
    & scp "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Downloading result archive failed.' }
    & tar -tzf $archive > $null
    if ($LASTEXITCODE -ne 0) { throw 'Downloaded result archive is invalid.' }
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Extracting results failed.' }
    if (-not (Test-Path -LiteralPath (Join-Path $destination 'COMPLETED'))) {
        throw 'Downloaded output lacks its completion marker.'
    }
    Write-Host "Downloaded $destination"
}
