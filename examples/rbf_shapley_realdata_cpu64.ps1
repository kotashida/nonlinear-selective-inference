param(
    [ValidateSet('Preflight','Launch','Status','Resume','Stop','Verify','Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)
$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$pythonLocal = Join-Path $localRepo '.venv/Scripts/python.exe'
$experiment = 'rbf_shapley_realdata_n100_150_200_cpu64_1000_20261024'
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
set -Eeuo pipefail
trap 'rc=$?; printf "Remote Bash failed (exit %s), line %s: %s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR
REPO="$HOME/nonlinear-selective-inference_rbf_shapley_realdata_n100_150_200_cpu64_1000_20261024"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
CONFIG="benchmarks/rbf_shapley_realdata_n100_150_200_cpu64_1000_20261024.json"
OUT="$REPO/outputs/benchmarks/rbf_shapley_realdata_n100_150_200_cpu64_1000_20261024"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 BLIS_NUM_THREADS=1
'@
$idle = @'
if pgrep -u "$(id -u)" -af '[v]alidate_selection_methods.py|[r]un_.*shards|[r]un_.*validation|[r]un_shap_sampling|[r]un_presentation.*selectors|[r]un_exact_pipeline_benchmark|[r]un_linear_shapley|[r]un_mi_cpu_sweep|[r]un_selected_five|[r]un_publication_mcmc.py run|[r]un_shapley_targets.py run|[r]un_rbf_shapley.py run|[r]un_rbf_shapley_analytic_power.py run|[r]un_rbf_shapley_realdata.py run'; then
  echo 'A competing experiment is active; no process was stopped.' >&2
  exit 2
fi
'@

if ($Action -in 'Preflight','Launch') {
    & $pythonLocal benchmarks/run_rbf_shapley_realdata.py check
    if ($LASTEXITCODE -ne 0) { throw 'Local read-only preflight failed.' }
    $preflight = @'
hostname
for tool in bash tar base64 nohup setsid pgrep ps awk mktemp getconf readlink tr flock sha256sum id; do command -v "$tool" >/dev/null; done
test -x "$PYTHON"
"$PYTHON" - <<'PY'
import os, sys, importlib.metadata as md
assert sys.version_info[:2] == (3, 11), sys.version
assert len(os.sched_getaffinity(0)) >= 1, 'No allowed CPU cores'
print('Effective worker cap:', min(64, len(os.sched_getaffinity(0))))
for name, version in {'numpy':'1.26.4', 'scipy':'1.17.1', 'threadpoolctl':'3.6.0'}.items():
    assert md.version(name) == version, (name, md.version(name), version)
import threadpoolctl
print('Server Python, numerical dependencies and at-most-64-core affinity allocation verified')
PY
test ! -e "$REPO" || { echo 'Dedicated checkout exists; use Status or Resume.' >&2; exit 2; }
awk '/MemAvailable/{print "Available memory (GiB):", $2/1048576}' /proc/meminfo
df -h "$HOME"
'@
    Invoke-RemoteBash ($header + "`n" + $preflight + "`n" + $idle)
}

if ($Action -eq 'Launch') {
    $sourceName = "rbf_shapley_realdata_cpu64_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $sourceName
    $files = @('benchmarks/run_rbf_shapley_realdata.py',
        'benchmarks/run_rbf_shapley.py',
        'examples/plot_rbf_shapley_realdata.py',
        'benchmarks/rbf_network.py',
        'benchmarks/table_dataset.pkl',
        'benchmarks/process.ipynb',
        'benchmarks/rbf_shapley_realdata_n100_150_200_cpu64_1000_20261024.json',
        'benchmarks/rbf_shapley_realdata_n100_150_200_cpu64_smoke_20261024.json',
        'examples/rbf_shapley_realdata_cpu64.ps1',
        'docs/rbf_shapley_realdata_study.md',
        'tests/test_rbf_shapley_realdata.py',
        'docs/commands.md')
    & tar -czf $archive -C $localRepo @files
    if ($LASTEXITCODE -ne 0) { throw 'Source packaging failed.' }
    $checksum = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    & scp -o ConnectTimeout=15 $archive "${Server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Source upload failed.' }
    $deploy = @'
echo '__CHECKSUM__  '"$HOME/__SOURCE__" | sha256sum -c -
STAGING="$(mktemp -d "$HOME/rbf_shapley_realdata_cpu64_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/benchmarks/run_rbf_shapley_realdata.py"
test ! -e "$REPO"
mv -T -- "$STAGING" "$REPO"
cd "$REPO"
"$PYTHON" benchmarks/run_rbf_shapley_realdata.py check --config "$CONFIG"
'@
    Invoke-RemoteBash ($header + "`n" + $deploy.Replace('__CHECKSUM__',$checksum).Replace('__SOURCE__',$sourceName))
}

if ($Action -in 'Launch','Resume') {
    $launch = @'
cd "$REPO"
"$PYTHON" benchmarks/run_rbf_shapley_realdata.py check --config "$CONFIG"
test ! -f "$OUT/COMPLETED" || { cat "$OUT/COMPLETED"; exit 0; }
exec 8>"$HOME/.nonlinear-selective-inference-cpu32.lock"
flock -n 8 || { echo 'Account experiment CPU budget is in use.' >&2; exit 2; }
exec 7>"$HOME/.nonlinear-selective-inference-cpu64.lock"
flock -n 7 || { echo '64-core account budget is in use.' >&2; exit 2; }
mkdir -p "$OUT"
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'This experiment is active.' >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S_%N)"
for name in launcher.pid launcher.log launcher.startticks; do
  if [[ -f "$OUT/$name" ]]; then cp -- "$OUT/$name" "$OUT/$name.Backup_$STAMP"; fi
done
nohup setsid "$PYTHON" -u benchmarks/run_rbf_shapley_realdata.py run --config "$CONFIG" >"$OUT/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
"$PYTHON" - "$PID" "$OUT/launcher.startticks" <<'PY'
from pathlib import Path
import sys
pid = int(sys.argv[1])
ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
Path(sys.argv[2]).write_text(ticks+'\n')
PY
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  test -f "$OUT/COMPLETED" || { tail -n 80 "$OUT/launcher.log"; exit 1; }
fi
ps -p "$PID" -o pid,pgid,sid,etime,args || true
tail -n 10 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $idle + "`n" + $launch)
}

if ($Action -eq 'Status') {
    $status = @'
if [[ ! -d "$OUT" ]]; then
  printf 'Status unavailable on %s: expected output directory is missing:\n  %s\n' "$(hostname)" "$OUT" >&2
  if [[ -d "$REPO" ]]; then
    echo 'The dedicated checkout exists, but its output directory has not been created.' >&2
  else
    printf 'The dedicated checkout is also missing:\n  %s\n' "$REPO" >&2
  fi
  echo 'Existing related checkouts on this server:' >&2
  found=0
  for candidate in "$HOME"/nonlinear-selective-inference*rbf_shapley*; do
    if [[ -d "$candidate" ]]; then
      printf '  %s\n' "$candidate" >&2
      found=1
    fi
  done
  if [[ "$found" == 0 ]]; then echo '  (none found under the remote home directory)' >&2; fi
  echo 'Each experiment uses a separate checkout. Use the matching script and server; check these paths before Launch or Resume.' >&2
  exit 1
fi
cd "$REPO"
"$PYTHON" benchmarks/run_rbf_shapley_realdata.py status --config "$CONFIG"
if [[ -f "$OUT/runtime.json" ]]; then cat "$OUT/runtime.json"; fi
if [[ -f "$OUT/launcher.pid" ]]; then
  PID="$(cat "$OUT/launcher.pid")"
  ps -e -o pid,pgid,pcpu,rss,etime,args | awk -v group="$PID" 'NR==1 || $2==group'
fi
if [[ -f "$OUT/launcher.log" ]]; then
  tail -n 20 "$OUT/launcher.log"
else
  printf 'Launcher log is not available: %s\n' "$OUT/launcher.log"
fi
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}

if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUT/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
[[ "$(ps -p "$PID" -o args=)" == *"benchmarks/run_rbf_shapley_realdata.py run"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
[[ "$(ps -p "$PID" -o sid= | tr -d ' ')" == "$PID" ]]
"$PYTHON" - "$PID" "$OUT/launcher.startticks" <<'PY'
from pathlib import Path
import sys
actual = Path(f'/proc/{int(sys.argv[1])}/stat').read_text().rsplit(')', 1)[1].split()[19]
assert actual == Path(sys.argv[2]).read_text().strip(), 'PID was reused; refusing to signal'
PY
kill -TERM -- "-$PID"
sleep 2
pgrep -a -g "$PID" || echo 'Stopped this process group; committed batches remain resumable.'
'@
    Invoke-RemoteBash ($header + "`n" + $stop)
}

if ($Action -in 'Verify','Download') {
    $verify = @'
cd "$REPO"
test -f "$OUT/COMPLETED"
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Experiment still active; use Status.' >&2; exit 2; }
"$PYTHON" benchmarks/run_rbf_shapley_realdata.py verify --config "$CONFIG"
'@
    Invoke-RemoteBash ($header + "`n" + $verify)
}

if ($Action -eq 'Download') {
    $archiveName = "${experiment}_${stamp}.tar.gz"
    $pack = @'
exec 9>"$OUT/launch.lock"
flock -n 9 || { echo 'Experiment active; cannot archive changing outputs.' >&2; exit 2; }
tar -czf "$HOME/__ARCHIVE__" -C "$(dirname "$OUT")" "$(basename "$OUT")"
sha256sum "$HOME/__ARCHIVE__" > "$HOME/__ARCHIVE__.sha256"
'@
    Invoke-RemoteBash ($header + "`n" + $pack.Replace('__ARCHIVE__',$archiveName))
    $archive = Join-Path $env:TEMP $archiveName
    & scp -o ConnectTimeout=15 "${Server}:~/$archiveName" $archive
    if ($LASTEXITCODE -ne 0) { throw 'Result download failed.' }
    & scp -o ConnectTimeout=15 "${Server}:~/$archiveName.sha256" "$archive.sha256"
    if ($LASTEXITCODE -ne 0) { throw 'Checksum download failed.' }
    $expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split '\s+')[0]
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Archive checksum mismatch.' }
    & $pythonLocal benchmarks/run_rbf_shapley_realdata.py validate-archive --archive $archive
    if ($LASTEXITCODE -ne 0) { throw 'Unsafe result archive.' }
    $parent = Join-Path $localRepo 'outputs/benchmarks'
    $target = [IO.Path]::GetFullPath((Join-Path $parent $experiment))
    $prefix = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
    if (-not $target.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output target escapes benchmarks.' }
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing results: $target" }
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Result extraction failed.' }
    & $pythonLocal benchmarks/run_rbf_shapley_realdata.py verify --output $target
    if ($LASTEXITCODE -ne 0) { throw 'Local result verification failed.' }
    & $pythonLocal examples/plot_rbf_shapley_realdata.py --input $target
    if ($LASTEXITCODE -ne 0) { throw 'Plot generation failed; verified numerical results are preserved.' }
    Write-Host "Verified results and plots saved to $target"
}
