# Matched tests for the two selected AIS settings

The shared runner is `examples/run_selected_five_ais_exact_mi30.py`. Each setting has its own config:

- `examples/selected_five_ais_robust_draw_floor.json`
- `examples/selected_five_ais_robust_floor_multimodal_denominator.json`

## Run on the server using 64 cores

`examples/selected_five_ais_exact_mi30_64cores.ps1` uploads an isolated checkout and the historical files needed for auditing. It runs both settings concurrently with 32 shard workers per setting, one RF/BLAS thread per worker: up to 64 CPU cores total. The 32 statistical shards, their seeds, and all trial counts stay unchanged. Methods run sequentially within each setting. During report generation or when one arm finishes first, usage naturally falls below 64.

From PowerShell in this repository:

```powershell
$aisServer = 'shida.kota@cpu-a10.'
# Upload source/reference configs, audit package versions, and launch both arms.
& .\examples\selected_five_ais_exact_mi30_64cores.ps1 -Action Launch -Server $aisServer
# Check progress, completed shards per method, and recent logs.
& .\examples\selected_five_ais_exact_mi30_64cores.ps1 -Action Status -Server $aisServer
# Download both completed output folders and launcher logs into this repo.
& .\examples\selected_five_ais_exact_mi30_64cores.ps1 -Action Download -Server $aisServer
# If the launcher exited before completion, resume from existing raw results.
& .\examples\selected_five_ais_exact_mi30_64cores.ps1 -Action Resume -Server $aisServer
```

The server defaults match the refinement launcher: SSH `shida.kota@cpu-a10.` and Python `~/.venvs/nonlinear-selective-inference-py311/bin/python`. Replace `$aisServer` if needed. The remote checkout is `~/nonlinear-selective-inference_selected_five_ais_exact_mi30_64cores`. Launch requires at least 64 online CPUs and refuses to overlap another validation experiment. It uses the existing pinned environment without changing its shared editable-install registration.

Download saves both `selected_five_methods_<setting>_ais_exact_mi30` folders, including raw runs and reports, under this repository's `outputs/selection_method_validation/exact_set/`. Launcher/arm logs are saved there in `selected_five_ais_exact_mi30_64cores`. Download requires both settings to complete and refuses to overwrite existing local destination folders. The source upload includes only the canonical reference dataset/manifests and the 640 historical AIS condition settings, avoiding upload of the old raw AIS runs.

Status samples live worker CPU usage over two seconds and shows accumulated CPU time, partial null/power condition outputs, and the latest shard-0 progress log for each started method. A completed shard means all null and four power conditions finished; `0/32` alone does not indicate inactivity. Methods run in order (SHAP, LIME, MI, spline), and condition CSVs are written after all trials in that condition finish. Updated Status runs from the local PowerShell script, so it can inspect an already-running experiment without uploading changes or restarting it.

From the repository root, using the historical pinned package versions and a Python environment with Bash available (Linux is recommended):

```bash
python examples/run_selected_five_ais_exact_mi30.py examples/selected_five_ais_robust_draw_floor.json --run
python examples/run_selected_five_ais_exact_mi30.py examples/selected_five_ais_robust_floor_multimodal_denominator.json --run
```

Omit `--run` to audit inputs and print the full per-method configuration without simulation. `--finalize` audits completed output and regenerates reports. `--workers` controls concurrency only; it does not change the 32-shard partition or seeds. The historical launcher handles resume and rejects changed run configs. No new simulation has been launched as part of this deliverable.

## Parameters matched exactly

The reference is `outputs/selection_method_validation/exact_set/selected_five_methods_combined_ais_exact_mi30`. This is a derived dataset, so the runner follows its `source_manifest.json` and `unified_tests.csv` back to the actual per-method condition configs rather than borrowing the refinement launcher's different experiment.

| Parameter | Matched value |
|---|---|
| Design | Baseline, 100 samples, 20 features, select 2, feature correlation 0, simulation sigma 1 |
| Methods | SHAP; LIME, 40 explanations; MI, 30 neighbors; spline screening with AIS; spline screening with exact inference |
| Trials per method | 200 null; 200 power at each of β = 0.30, 0.50, 0.75, 1.00 |
| Shards | 32; shards 0–7 have 7 trials per condition, shards 8–31 have 6 |
| Root seeds | 20260908 through 20260939 inclusive; original condition-seed derivation and within-shard iteration numbering |
| Design seed | Root 314159; derived null design seed 3405814020, fixed across shards |
| Conditioning and target | Exact selected set; uniform target from selected features; fresh auxiliary draw each iteration; known simulation variance; no multiplicity correction |
| Signal | Feature 0 (`first`), original data-generating process |
| FPR / power evaluation | FPR at α = 0.01, 0.05, 0.10; power at α = 0.05; same Wilson intervals and failed-test handling |
| AIS budget | Pilot 10 × 320; final cap 51,200; final batch 800; denominator ESS 80, tail ESS 15; stop when ESS met |
| Selector controls | Original recorded settings, including LIME 64 neighborhood samples, ridge alpha 1, default kernel width; MI kNN estimator, neighbors 30, random state 42; original RF parameters, n_jobs 1; selection rounding 10 decimals |
| Validation criteria | Minimum calibration iterations 200; minimum signal targets 30; minimum conditional power 0.80; comprehensive preset; MCMC steps 20 |
| Software | Original pinned package versions are required before `--run`; Python/platform provenance is recorded separately |

Every historical AIS condition config (640 per arm) is audited, including current API defaults and selector metadata, before a run. Finalization checks all historical low-level settings, excluding only strategy and runtime provenance. It also checks all 4,000 new AIS rows against the original condition/shard/iteration pairs: target seeds, AIS seeds, targets, auxiliary draws, observed statistics, null selected sets, and power signal-target indicators. A mismatch stops finalization.

## Outputs and assumptions

Each config creates a sibling folder with the original `selected_five_methods_<setting>_ais_exact_mi30` naming pattern. Both produce the historical filenames: `unified_tests.csv`, `source_manifest.json`, `selected_configuration.json`, `configuration_audit.csv`, `trial_audit.csv`, `power_summary.csv`, `calibration_summary.csv`, `sampling_cost_summary.csv`, `signal_outcomes.csv`, the seven numbered PNG/PDF figures, `README.md`, and `index.html`. Historical CSV columns remain first; any new diagnostic columns follow. New raw runs retain `shard_<index>/<method>/baseline/null_fresh` and `power_feature_0_beta_<strength>` names under `runs/<method>`, with the same raw result writers and pooled summary structure.

- **Draw floor / consolidation ambiguity:** chosen setting 1 is `a_robust_draw_floor` (128-draw component floor). `a_robust_consolidation` has a zero draw floor; they are different strategies despite identical refinement failures and costs (4/200 power failures, mean cost 6,640). They must not be assumed interchangeable across all methods and strengths. Setting 2 is exactly `a_robust_floor_multimodal_denominator` (3/200 failures, mean cost 6,808). Both use 10 pilot rounds, with no pilot-15, CDF-interval, or optimization variant.
- **MI terminology:** “MI30” means 30 nearest neighbors, not an additional MI-score threshold. The historical selected configuration explicitly uses `n_neighbors=30`.
- **Exact-spline reference:** its 1,000 existing rows are reused verbatim in both new five-method datasets. It has no AIS setting to replace. This preserves the five-method comparison and avoids rerunning unchanged tests. The four AIS methods each receive the new setting.
- **Historical strategy field:** older AIS configs omit `ais_strategy`; the original implementation's default is treated as `baseline`. “Combined AIS” in the reference denotes its tuned budget, not a strategy enum. The report's `ais_setting=combined` category is retained for format compatibility; the new strategy is explicit in both manifests and the report label.
- **Execution provenance:** the original AIS runs used Linux/Python 3.11, and the exact reference used Windows/Python 3.12. Platform/Python strings cannot all be identical in one new run. Package versions must match; final pairing audits detect numerical or selector differences that affect the comparison. Newly recorded settings metadata is allowed only in addition to historical keys; all historical values must match.

Checks for the deliverable: both configuration plans were audited against the historical files; regression checks cover parameter drift, seed drift, five-method report assembly, exact-reference preservation, and historical output column/file formats. These checks use existing rows for report verification and do not claim new FPR or power results.
