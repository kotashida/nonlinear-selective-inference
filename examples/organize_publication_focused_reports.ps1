param()
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$root = [IO.Path]::GetFullPath((Join-Path $repo 'outputs/benchmarks/publication_five_mcmc20_focused_20261010'))
$reports = Join-Path $root 'reports'
$rootPrefix = $root.TrimEnd('\') + '\'
if (-not (Test-Path -LiteralPath (Join-Path $root 'verification.json'))) { throw 'Verified benchmark output not found.' }
$mapping = [ordered]@{
    'manuscript_independent_gaussian' = '01_independent_gaussian'
    'manuscript' = '02_other_designs'
    'analysis' = '03_full_diagnostics'
    'figures' = '04_original_exports'
}
function Get-ReportInventory([string] $Directory) {
    $inventory = @{}
    Get-ChildItem -LiteralPath $Directory -Recurse -File | ForEach-Object {
        $relative = $_.FullName.Substring($Directory.Length).TrimStart('\')
        $inventory[$relative] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
    return $inventory
}
$verification = Get-Content -LiteralPath (Join-Path $root 'verification.json') -Raw | ConvertFrom-Json
foreach ($property in $verification.output_sha256.PSObject.Properties) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $root $property.Name) -Algorithm SHA256).Hash
    if ($actual -ne $property.Value) { throw "Scientific output changed: $($property.Name)" }
}
# Validate all absolute source/target paths before moving any directory.
foreach ($item in $mapping.GetEnumerator()) {
    $source = [IO.Path]::GetFullPath((Join-Path $root $item.Key))
    $target = [IO.Path]::GetFullPath((Join-Path $reports $item.Value))
    if (-not $source.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase) -or
        -not $target.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Report move escapes the benchmark directory.' }
    if ((Test-Path -LiteralPath $source) -and (Test-Path -LiteralPath $target)) { throw "Both old and organized reports exist; refusing to merge or overwrite: $source" }
    if (-not (Test-Path -LiteralPath $source) -and -not (Test-Path -LiteralPath $target)) { throw "Report directory missing: $source" }
}
New-Item -ItemType Directory -Path $reports -Force | Out-Null
$moved = @()
foreach ($item in $mapping.GetEnumerator()) {
    $source = [IO.Path]::GetFullPath((Join-Path $root $item.Key))
    $target = [IO.Path]::GetFullPath((Join-Path $reports $item.Value))
    if (Test-Path -LiteralPath $source) {
        $before = Get-ReportInventory $source
        Move-Item -LiteralPath $source -Destination $target
        $after = Get-ReportInventory $target
        if ($before.Count -ne $after.Count) { throw "Report inventory changed during move: $target" }
        foreach ($name in $before.Keys) {
            if ($before[$name] -ne $after[$name]) { throw "Report content changed during move: $name" }
        }
        $moved += $item.Value
        Write-Host "Organized $($item.Key) -> reports/$($item.Value) ($($before.Count) files; all hashes preserved)"
    }
}
$utf8 = New-Object Text.UTF8Encoding($false)
$readme = @'
# Focused five-method benchmark: start here

## Current paper results

Open [the report index](index.html), then
[independent-Gaussian manuscript results](reports/01_independent_gaussian/index.html).
This is the current paper set: 5,000 paired independent-Gaussian null trials,
n=100, p=20, five methods, 25,000 evaluations. It contains the minimal Type I
error figure and the numerical results/computation table. Independent-Gaussian
power was not run; real-predictor power is not part of this set.

## Report folders

| Folder | Purpose |
|---|---|
| `reports/01_independent_gaussian` | Current paper: independent-Gaussian null figure/table, captions, PDF/SVG/PNG, CSV and LaTeX table |
| `reports/02_other_designs` | Earlier compact manuscript set: correlated-null panels and heating/red-wine power; also real-data tables |
| `reports/03_full_diagnostics` | Complete 33-figure gallery, supporting CSV tables, checks and provenance |
| `reports/04_original_exports` | Original server-generated figures, retained for completeness |

The first three folders contain an `index.html` with previews and download links.
Each report's manifests retain their original generation hashes; relocation did
not change report content. There are no duplicate report copies at the root.

## Scientific records: keep these paths unchanged

The root CSV files are the original verified summaries, not additional report
versions: `results.csv` is the complete 70,000-evaluation result table;
`rates.csv`, `diagnostics.csv`, `prediction.csv`, `calibration.csv`,
`selection_frequency.csv` and `paired_comparisons.csv` are analysis inputs.
`null/` and `real/` contain resumable per-trial inputs and receipts.
Configuration, provenance, completion/verification records, locks and launcher
files remain at the root because the original runner/verifier expects them.
Do not relocate or edit them to tidy the display.

## Regenerate reports from the repository root

Current independent-Gaussian paper set:

```powershell
.\.venv\Scripts\python.exe examples\plot_publication_manuscript.py --independent-only
```

Other-design manuscript set:

```powershell
.\.venv\Scripts\python.exe examples\plot_publication_manuscript.py
```

Complete diagnostic gallery:

```powershell
.\.venv\Scripts\python.exe examples\plot_publication_focused_gallery.py
```

The frozen server plotter can recreate the legacy `figures/` folder during
server verification. It was not modified because its source is part of the
experiment provenance. Use the report generators above for the organized local
presentation. No benchmarks or inference procedures were rerun for this cleanup.
'@
[IO.File]::WriteAllText((Join-Path $root 'README.md'),$readme,$utf8)
$page = @'
<!doctype html><html lang="en"><meta charset="utf-8"><title>Focused benchmark reports</title>
<style>body{font:17px system-ui;max-width:950px;margin:48px auto;padding:0 24px;line-height:1.65;color:#20262d}h1,h2{font-weight:500}section{border-top:1px solid #ddd;padding:18px 0}a{color:#185d91}.current{background:#eef6fa;padding:20px;border-radius:6px}small{color:#555}</style>
<h1>Focused benchmark reports</h1>
<section class="current"><h2>1. Current paper: independent Gaussian</h2>
<p>5,000 paired null trials per method; n = 100, p = 20; five methods.</p>
<p><a href="reports/01_independent_gaussian/index.html">Open the current manuscript figure and table</a></p>
<p><a href="reports/01_independent_gaussian/figure_1_type1_error.pdf">Type I error figure (PDF)</a> | <a href="reports/01_independent_gaussian/table_1_null_results.pdf">Results table (PDF)</a></p>
<small>Independent-Gaussian power has not been run. The other-design power curves below do not belong to this condition.</small></section>
<section><h2>2. Other designs</h2><p><a href="reports/02_other_designs/index.html">Open the earlier compact manuscript set</a></p><p>Correlated-Gaussian null, heating/red-wine null and power, real-response rejection and computation tables. Separate from the current independent-Gaussian scope.</p></section>
<section><h2>3. Full diagnostics</h2><p><a href="reports/03_full_diagnostics/index.html">Open the complete 33-figure gallery</a></p><p>Selection, prediction, calibration, runtime, paired spline comparisons, MCMC diagnostics and supporting tables.</p></section>
<section><h2>4. Original exports</h2><p><a href="reports/04_original_exports/">Original server figures</a></p><p>Retained original exports; use the current paper set for manuscript preparation.</p></section>
<p><a href="README.md">Folder guide and regeneration commands</a></p>
<p><small>All verified scientific CSVs, trial receipts and benchmark settings remain unchanged at their original paths.</small></p></html>
'@
[IO.File]::WriteAllText((Join-Path $root 'index.html'),$page,$utf8)
$reportsPage = $page.Replace('href="reports/','href="').Replace('href="README.md"','href="../README.md"')
[IO.File]::WriteAllText((Join-Path $reports 'index.html'),$reportsPage,$utf8)
foreach ($property in $verification.output_sha256.PSObject.Properties) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $root $property.Name) -Algorithm SHA256).Hash
    if ($actual -ne $property.Value) { throw "Scientific output changed: $($property.Name)" }
}
Write-Host 'Finished. Open the benchmark index.html. All seven verified scientific CSV hashes are unchanged.'
