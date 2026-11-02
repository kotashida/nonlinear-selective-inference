param(
    [ValidateSet('Launch', 'Status', 'Stop', 'Resume', 'Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@172.31.1.18'
)
$ErrorActionPreference = 'Stop'
$localRepo = 'C:\Users\shida\OneDrive\selective-inference\nonlinear-selective-inference'
$experiment = 'estimators_histogram_kde_200_trials_20260912'
$remoteDirectory = 'nonlinear-selective-inference_mi_estimators_200_20260912'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Set-Location $localRepo

function Invoke-RemoteBash {
    param([Parameter(Mandatory)] [string] $Server,
          [Parameter(Mandatory)] [string] $Script)
    $normalized = $Script -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($normalized))
    & ssh $Server "printf '%s' '$encoded' | base64 --decode | bash"
    if ($LASTEXITCODE -ne 0) { throw "Remote Bash failed: $LASTEXITCODE" }
}

foreach ($command in 'ssh', 'scp', 'tar') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required local command missing: $command"
    }
}
$remoteHeader = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_mi_estimators_200_20260912"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUTPUT_BASE="$REPO/outputs/selection_method_validation/exact_set/mi_parameter_tuning/estimators_histogram_kde_200_trials_20260912"
'@

if ($Action -eq 'Launch') {
    # Read-only preflight before making archives or changing the server.
    $preflight = @'
set -euo pipefail
hostname
for tool in tar base64 bash nohup setsid pgrep ps xargs awk seq cmp mktemp; do
  command -v "$tool" || { echo "Required command missing: $tool" >&2; exit 2; }
done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32
test -x "$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
"$HOME/.venvs/nonlinear-selective-inference-py311/bin/python" -c 'import numpy, pandas, scipy, sklearn, shap, matplotlib; print("Imports OK")'
if pgrep -af '[v]alidate_selection_methods.py|[r]un_.*shards|[c]ompare_selection_event|[r]un_lime_.*sweep.*sh|[r]un_combined_ais_other_methods.sh|[r]un_mi_.*sh'; then
  echo 'Competing experiments found; launch postponed to preserve the 32-core limit.' >&2
  exit 2
fi
test ! -e "$HOME/nonlinear-selective-inference_mi_estimators_200_20260912" || {
  echo 'Sweep checkout already exists. Use Status or Resume; it will not be overwritten.' >&2
  exit 2
}
'@
    Invoke-RemoteBash -Server $server -Script $preflight
    $sourceName = "mi_sweep_source_${stamp}.tar.gz"
    $sourceArchive = Join-Path $env:TEMP $sourceName
    & tar -czf $sourceArchive --exclude=__pycache__ --exclude=.pytest_cache `
        -C $localRepo src examples tests docs pyproject.toml README.md
    if ($LASTEXITCODE -ne 0) { throw 'Creating source archive failed.' }
    & scp $sourceArchive "${server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Uploading source archive failed.' }
    $deploy = @'
test ! -e "$REPO"
STAGING="$(mktemp -d "$HOME/mi_sweep_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/pyproject.toml"
test -f "$STAGING/examples/run_mi_estimators.sh"
# Windows working-tree files may contain CRLF. Normalize the Bash entry points.
"$PYTHON" - "$STAGING" <<'PY'
from pathlib import Path
import sys
for name in ('run_mi_estimators.sh', 'run_selection_method_validation_shards.sh'):
    p = Path(sys.argv[1]) / 'examples' / name
    p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
PY
bash -n "$STAGING/examples/run_mi_estimators.sh"
bash -n "$STAGING/examples/run_selection_method_validation_shards.sh"
mv -- "$STAGING" "$REPO"
cd "$REPO"
"$PYTHON" -m pip install -e . --no-deps
"$PYTHON" examples/validate_selection_methods.py --help >/dev/null
mkdir -p "$OUTPUT_BASE"
cp examples/run_mi_estimators.sh "$OUTPUT_BASE/requested_manifest.sh"
'@
    Invoke-RemoteBash -Server $server -Script ($remoteHeader + "`n" + $deploy.Replace('__SOURCE__', $sourceName))
}

if ($Action -in 'Launch', 'Resume') {
    $launch = @'
cd "$REPO"
test ! -f "$OUTPUT_BASE/COMPLETED" || { echo 'Already completed. Use Download.'; exit 0; }
if pgrep -af '[v]alidate_selection_methods.py|[r]un_.*shards|[c]ompare_selection_event|[r]un_lime_.*sweep.*sh|[r]un_combined_ais_other_methods.sh|[r]un_mi_.*sh'; then
  echo 'An experiment is active; refusing a duplicate or competing launch.' >&2
  exit 2
fi
cmp examples/run_mi_estimators.sh "$OUTPUT_BASE/requested_manifest.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -f "$OUTPUT_BASE/launcher.log" ]]; then
  cp "$OUTPUT_BASE/launcher.log" "$OUTPUT_BASE/launcher.log.Backup_$STAMP"
fi
if [[ -f "$OUTPUT_BASE/launcher.pid" ]]; then
  cp "$OUTPUT_BASE/launcher.pid" "$OUTPUT_BASE/launcher.pid.Backup_$STAMP"
fi
nohup setsid env PYTHON_BIN="$PYTHON" OUTPUT_BASE="$OUTPUT_BASE" \
  bash examples/run_mi_estimators.sh > "$OUTPUT_BASE/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUTPUT_BASE/launcher.pid"
sleep 3
kill -0 "$PID" || { tail -n 40 "$OUTPUT_BASE/launcher.log"; exit 1; }
test -f "$OUTPUT_BASE/histogram/run_config.txt"
echo "Launcher PID: $PID"
cat "$OUTPUT_BASE/histogram/run_config.txt"
tail -n 12 "$OUTPUT_BASE/launcher.log"
'@
    Invoke-RemoteBash -Server $server -Script ($remoteHeader + "`n" + $launch)
}

if ($Action -eq 'Status') {
    $status = @'
test -d "$OUTPUT_BASE"
if [[ -f "$OUTPUT_BASE/COMPLETED" ]]; then echo 'COMPLETED'; fi
PID="$(cat "$OUTPUT_BASE/launcher.pid")"
ps -p "$PID" -o pid,pgid,etime,args || true
for variant in histogram kde; do
  count=0
  for file in "$OUTPUT_BASE/$variant"/shard_*/validation_settings.json; do
    [[ ! -f "$file" ]] || count=$((count + 1))
  done
  echo "$variant: $count/32 completed shards"
done
tail -n 20 "$OUTPUT_BASE/launcher.log"
'@
    Invoke-RemoteBash -Server $server -Script ($remoteHeader + "`n" + $status)
}

if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUTPUT_BASE/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
# Require the expected command, working directory and process-group leader.
[[ "$(ps -p "$PID" -o args=)" == *"bash examples/run_mi_estimators.sh"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
kill -TERM -- "-$PID"
sleep 2
if pgrep -a -g "$PID"; then
  echo 'Some group members are still exiting; inspect Status before resuming.'
else
  echo 'Experiment process group stopped. Partial results retained.'
fi
'@
    Invoke-RemoteBash -Server $server -Script ($remoteHeader + "`n" + $stop)
}

if ($Action -eq 'Download') {
    $archiveName = "${experiment}_${stamp}.tar.gz"
    $finish = @'
test -f "$OUTPUT_BASE/COMPLETED" || { echo 'Not completed; use Status.' >&2; exit 2; }
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# Verify both MI configurations, counts, seeds and fixed AIS settings.
"$PYTHON" examples/verify_mi_estimators.py "$OUTPUT_BASE"
tar -czf "$HOME/__ARCHIVE__" -C "$(dirname "$OUTPUT_BASE")" "$(basename "$OUTPUT_BASE")"
'@
    Invoke-RemoteBash -Server $server -Script ($remoteHeader + "`n" + $finish.Replace('__ARCHIVE__', $archiveName))
    $archive = Join-Path $env:TEMP $archiveName
    & scp "${server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Downloading results failed.' }
    & tar -tzf $archive > $null
    if ($LASTEXITCODE -ne 0) { throw 'Archive validation failed.' }
    $parent = Join-Path $localRepo 'outputs\selection_method_validation\exact_set\mi_parameter_tuning'
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $target = Join-Path $parent $experiment
    $resolvedParent = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
    $target = [IO.Path]::GetFullPath($target)
    if (-not $target.StartsWith($resolvedParent, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output target escapes expected directory.' }
    if (Test-Path -LiteralPath $target) {
        Move-Item -LiteralPath $target -Destination "${target}_Backup_${stamp}"
    }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Extraction failed; any old results remain backed up.' }
    foreach ($variant in 'histogram','kde') {
        $settings = Get-Content -Raw (Join-Path $target "$variant/pooled_summary/validation_settings.json") | ConvertFrom-Json
        if ($settings.n_shards -ne 32 -or $settings.n_null_iters -ne 200 -or $settings.n_power_iters -ne 200) {
            throw "Unexpected iteration counts: $variant"
        }
        if (($settings.methods -join ' ') -ne 'mutual_information' -or $settings.max_final_samples -ne 51200 -or $settings.pilot_iters -ne 10 -or $settings.pilot_samples -ne 320 -or $settings.final_batch_size -ne 800 -or $settings.design_seed -ne 314159 -or ($settings.shard_root_seeds -join ',') -ne ((20260908..20260939) -join ',')) {
            throw 'Unexpected methods, seeds or AIS parameters.'
        }
        if ($settings.mi_neighbors -ne 30 -or $settings.mi_params.estimator -ne $variant -or $settings.mi_params.n_bins -ne 5 -or $settings.mi_params.bandwidth -ne 1.0) { throw "Wrong MI configuration: $variant" }
        Write-Host "Verified $variant; 200 trials per condition; combined AIS."
    }
    Get-ChildItem -LiteralPath (Join-Path $target 'histogram/pooled_summary')
    Write-Host "Results saved to $target"
}
