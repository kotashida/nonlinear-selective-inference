param(
    [Parameter(Mandatory)]
    [ValidatePattern('^publication_five_mcmc20_focused_20261010_[0-9]{8}_[0-9]{6}_[0-9]{3}\.tar\.gz$')]
    [string] $ArchiveName,
    [string] $Server = 'shida.kota@cpu-a10.'
)
$ErrorActionPreference = 'Stop'
$localRepo = Split-Path -Parent $PSScriptRoot
$experiment = 'publication_five_mcmc20_focused_20261010'
$pythonLocal = Join-Path $localRepo '.venv/Scripts/python.exe'
$archive = Join-Path $env:TEMP $ArchiveName
$checksumFile = "$archive.sha256"
$parent = Join-Path $localRepo 'outputs/benchmarks'
$target = [IO.Path]::GetFullPath((Join-Path $parent $experiment))
$prefix = [IO.Path]::GetFullPath($parent).TrimEnd('\') + '\'
Set-Location $localRepo

if (-not $target.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output target escapes benchmarks.' }
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw "Downloaded archive is missing: $archive" }
if (Test-Path -LiteralPath $target) { throw "Results already exist; use the original launcher with -Action VerifyLocal: $target" }

# Keep the frozen launcher and scientific sources unchanged: their hashes are
# checked against the provenance saved by the completed server experiment.
# Retrieve only the tiny checksum sidecar; never download/repack results again.
if (-not (Test-Path -LiteralPath $checksumFile -PathType Leaf)) {
    $received = $false
    $partial = "$checksumFile.partial"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            & scp -o ConnectTimeout=15 -o ConnectionAttempts=3 "${Server}:~/$ArchiveName.sha256" $partial
            if ($LASTEXITCODE -eq 0) { $received = $true; break }
        } catch {
            Write-Warning $_.Exception.Message
        }
        if ($attempt -lt 3) {
            Write-Warning "Checksum connection failed; retrying ($attempt/3). The results archive is retained."
            Start-Sleep -Seconds 2
        }
    }
    if (-not $received) { throw "Checksum retrieval failed. Archive retained at $archive. Retry this same recovery command when the server is reachable." }
    Move-Item -LiteralPath $partial -Destination $checksumFile
}
$checksumText = (Get-Content -LiteralPath $checksumFile -Raw).Trim()
if ($checksumText -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') { throw 'Malformed checksum sidecar; archive has not been extracted.' }
$expected = $Matches[1].ToLowerInvariant()
$recordedName = ($Matches[2] -split '/')[-1]
if ($recordedName -ne $ArchiveName) { throw 'Checksum sidecar names a different archive.' }
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'Result archive checksum mismatch; archive has not been extracted.' }
Write-Host "Archive checksum passed; reusing $archive"

& $pythonLocal benchmarks/run_publication_mcmc.py validate-archive --config benchmarks/publication_mcmc20_focused_20261010.json --archive $archive
if ($LASTEXITCODE -ne 0) { throw 'Archive validation failed; archive has not been extracted.' }
New-Item -ItemType Directory -Path $parent -Force | Out-Null
& tar -xzf $archive -C $parent
if ($LASTEXITCODE -ne 0) { throw 'Result extraction failed; retained archive can be used for recovery. Inspect partial output before retrying.' }
& $pythonLocal benchmarks/run_publication_mcmc.py verify-downloaded --config benchmarks/publication_mcmc20_focused_20261010.json
if ($LASTEXITCODE -ne 0) { throw 'Local verification failed; downloaded receipts and original archive are retained for inspection.' }
Write-Host "Verified results saved to $target"
