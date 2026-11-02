param(
    [ValidateSet('Launch','Status','Resume','Stop','Verify','Download')]
    [string] $Action = 'Status',
    [string] $Server = 'shida.kota@cpu-a10.'
)

$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'selected_five_kde2p5_1000_cpu32_20260924'
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
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Missing command: $command"
    }
}

$header = @'
set -euo pipefail
REPO="$HOME/nonlinear-selective-inference_benchmark_kde2p5_cpu32"
PYTHON="$HOME/.venvs/nonlinear-selective-inference-py311/bin/python"
OUT="$REPO/outputs/benchmarks/selected_five_kde2p5_1000_cpu32_20260924"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1
export PYTHONPATH="$REPO/src:$REPO"
'@

$idleCheck = @'
if pgrep -af '[r]un_exact_pipeline_benchmark.py|[r]un_exact_pipeline_benchmark_32shards.sh'; then
  echo 'A competing benchmark process is active. No process was stopped.' >&2
  exit 2
fi
'@

if ($Action -eq 'Launch') {
    $preflight = @'
hostname
for tool in bash tar base64 nohup setsid pgrep ps awk seq find wc readlink tr sed; do command -v "$tool" >/dev/null; done
test "$(getconf _NPROCESSORS_ONLN)" -ge 32 || { echo 'Server needs at least 32 online CPU cores.' >&2; exit 2; }
test -x "$PYTHON"
"$PYTHON" -c 'import numpy,pandas,scipy,sklearn,shap,matplotlib,patsy; print("Required imports available")'
test ! -e "$REPO" || { echo 'Experiment checkout already exists; use Status or Resume.' >&2; exit 2; }
'@
    Invoke-RemoteBash ($header + "`n" + $preflight + "`n" + $idleCheck)

    $sourceName = "exact_pipeline_benchmark_source_${stamp}.tar.gz"
    $archive = Join-Path $env:TEMP $sourceName
    & tar -czf $archive --exclude=__pycache__ --exclude=.pytest_cache -C $localRepo src benchmarks tests docs pyproject.toml README.md
    if ($LASTEXITCODE -ne 0) { throw 'Source archive creation failed.' }
    & scp -o ConnectTimeout=15 $archive "${Server}:~/$sourceName"
    if ($LASTEXITCODE -ne 0) { throw 'Source upload failed.' }

    $deploy = @'
STAGING="$(mktemp -d "$HOME/exact_pipeline_benchmark_import_XXXXXX")"
tar -xzf "$HOME/__SOURCE__" -C "$STAGING"
test -f "$STAGING/benchmarks/run_exact_pipeline_benchmark.py"
test -f "$STAGING/benchmarks/table_dataset.pkl"
"$PYTHON" - "$STAGING" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for relative in ('benchmarks/run_exact_pipeline_benchmark_32shards.sh',):
    path = root / relative
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
PY
bash -n "$STAGING/benchmarks/run_exact_pipeline_benchmark_32shards.sh"
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
"$PYTHON" benchmarks/run_exact_pipeline_benchmark.py check
mkdir -p "$OUT"
'@
    Invoke-RemoteBash ($header + "`n" + $deploy.Replace('__SOURCE__', $sourceName))
}

if ($Action -in 'Launch','Resume') {
    $launch = @'
cd "$REPO"
test -f benchmarks/exact_pipeline_benchmark_1000.json
test ! -f "$OUT/COMPLETED" || { cat "$OUT/COMPLETED"; exit 0; }
'@ + "`n" + $idleCheck + "`n" + @'
STAMP="$(date +%Y%m%d_%H%M%S)"
for name in launcher.log launcher.pid; do
  if [[ -f "$OUT/$name" ]]; then cp -- "$OUT/$name" "$OUT/$name.Backup_$STAMP"; fi
done
nohup setsid env REPO="$REPO" PYTHON="$PYTHON" OUT="$OUT" \
  bash benchmarks/run_exact_pipeline_benchmark_32shards.sh \
  >"$OUT/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
sleep 5
kill -0 "$PID" || { tail -n 100 "$OUT/launcher.log"; exit 1; }
echo "Launcher PID: $PID"
ps -p "$PID" -o pid,pgid,etime,args
echo 'Active shard workers:'
pgrep -fc '[r]un_exact_pipeline_benchmark.py run' || true
tail -n 30 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $launch)
}

if ($Action -eq 'Status') {
    $status = @'
test -d "$OUT"
if [[ -f "$OUT/COMPLETED" ]]; then cat "$OUT/COMPLETED"; fi
if [[ -f "$OUT/launcher.pid" ]]; then ps -p "$(cat "$OUT/launcher.pid")" -o pid,pgid,etime,args || true; fi
workers=$(pgrep -fc '[r]un_exact_pipeline_benchmark.py run' || true)
echo "Active shard workers: $workers (limit 32)"
test "$workers" -le 32
completed=0
rows=0
for shard in $(seq 0 31); do
  result="$OUT/shards/shard_$shard/results.csv"
  if [[ -f "$result" ]]; then
    count=$(awk 'END{print NR-1}' "$result")
    rows=$((rows + count))
    assigned=$((1000 / 32))
    if [[ "$shard" -lt $((1000 % 32)) ]]; then assigned=$((assigned + 1)); fi
    expected=$((assigned * 8 * 5))
    if [[ "$count" -eq "$expected" ]]; then completed=$((completed + 1)); fi
  fi
done
echo "Complete shards: $completed/32"
echo "Checkpointed result rows: $rows/40000"
echo 'Recent shard progress:'
find "$OUT/logs" -type f -name 'shard_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1,5p' | while read -r _ log; do echo "===== $log ====="; tail -n 2 "$log"; done
echo 'Launcher tail:'
tail -n 40 "$OUT/launcher.log"
'@
    Invoke-RemoteBash ($header + "`n" + $status)
}

if ($Action -eq 'Stop') {
    $stop = @'
PID="$(cat "$OUT/launcher.pid")"
[[ "$PID" =~ ^[1-9][0-9]*$ ]]
[[ "$(ps -p "$PID" -o args=)" == *"run_exact_pipeline_benchmark_32shards.sh"* ]]
[[ "$(readlink "/proc/$PID/cwd")" == "$REPO" ]]
[[ "$(ps -p "$PID" -o pgid= | tr -d ' ')" == "$PID" ]]
kill -TERM -- "-$PID"
sleep 2
pgrep -a -g "$PID" || echo 'Benchmark stopped; all checkpointed method rows were retained.'
'@
    Invoke-RemoteBash ($header + "`n" + $stop)
}

if ($Action -in 'Verify','Download') {
    $verify = @'
cd "$REPO"
test -f "$OUT/COMPLETED"
"$PYTHON" benchmarks/run_exact_pipeline_benchmark.py verify --output "$OUT"
test "$(awk 'END{print NR-1}' "$OUT/results.csv")" -eq 40000
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
    $parent = Join-Path $localRepo 'outputs\benchmarks'
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $target = [IO.Path]::GetFullPath((Join-Path $parent $experiment))
    $prefix = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
    if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Target escapes output directory.'
    }
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing output: $target" }
    & tar -xzf $archive -C $parent
    if ($LASTEXITCODE -ne 0) { throw 'Result extraction failed.' }
    & '.\.venv\Scripts\python.exe' benchmarks\run_exact_pipeline_benchmark.py verify --output $target
    if ($LASTEXITCODE -ne 0) { throw 'Local verification failed.' }
    Write-Host "Verified results saved to $target"
}
