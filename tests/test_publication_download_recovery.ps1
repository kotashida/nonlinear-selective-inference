$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$fixtureName = 'publication_five_mcmc20_focused_20261010_20990101_000000_001.tar.gz'
$fixtureArchive = Join-Path $env:TEMP $fixtureName
$fixtureChecksum = "$fixtureArchive.sha256"
if ((Test-Path -LiteralPath $fixtureArchive) -or (Test-Path -LiteralPath $fixtureChecksum)) { throw 'Recovery test fixture already exists.' }
$recoveryScript = Join-Path $repo 'examples/recover_publication_focused_download.ps1'
function Assert-RecoveryRejection([string] $Expected) {
    try {
        & $recoveryScript -ArchiveName $fixtureName
        throw 'Recovery unexpectedly accepted the invalid fixture.'
    } catch {
        if ($_.Exception.Message -notlike "*$Expected*") { throw }
        Write-Output "Passed rejection: $Expected"
    }
}
try {
    [IO.File]::WriteAllBytes($fixtureArchive, [Text.Encoding]::UTF8.GetBytes('invalid test archive'))
    [IO.File]::WriteAllText($fixtureChecksum, 'malformed')
    Assert-RecoveryRejection 'Malformed checksum'
    [IO.File]::WriteAllText($fixtureChecksum, (('0' * 64) + '  /home/test/wrong.tar.gz'))
    Assert-RecoveryRejection 'different archive'
    [IO.File]::WriteAllText($fixtureChecksum, (('0' * 64) + '  /home/test/' + $fixtureName))
    Assert-RecoveryRejection 'checksum mismatch'
    $fixtureHash = (Get-FileHash -LiteralPath $fixtureArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText($fixtureChecksum, ($fixtureHash + '  /home/test/' + $fixtureName))
    Assert-RecoveryRejection 'Archive validation failed'
    if (Test-Path -LiteralPath (Join-Path $repo 'outputs/benchmarks/publication_five_mcmc20_focused_20261010')) {
        throw 'Invalid fixture created the production output folder.'
    }
} finally {
    Remove-Item -LiteralPath $fixtureArchive,$fixtureChecksum -ErrorAction SilentlyContinue
}
