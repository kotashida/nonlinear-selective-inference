# Real-data pipeline benchmark: sample-size sweep

## Required design for the next benchmark

The next benchmark must evaluate **all three sample sizes n=100, 150, 200**,
with **1,000 random row subsamples per dataset at each sample size**. This
matches the sample-size grid and repetition count in Section 6, "Real Data
Experiments", and Table 1 (p. 8) of
[Shiraishi et al. (2025)](../references/papers/2025_Shiraishi_SelectiveInferenceFeatureSelectionPipelines.pdf).
Here n is the **sample/subsample size**, not the population size or the number
of repetitions. The purpose is to evaluate whether detection frequency
increases as more observations are used in each test.

The current RBF benchmark will compare:

- **Selected-Features Shapley Effect**;
- **All-Features Shapley Effect**.

Use the analytic selective tests and the RBF RSS Shapley selector from the
validated synthetic study: k=2, three X-only Gaussian RBF units per feature,
exhaustive coalition fits, top-two full-feature RSS Shapley selection, one
uniformly randomized selected target, exact selected-set conditioning,
known/working sigma=1, alpha=0.05, and no multiplicity adjustment. Retain each
dataset's native features and its existing preprocessed observed response;
**do not add a synthetic signal or sweep beta in this real-data experiment**.
Each subsample defines its own X-only RBF centers, widths and block means.
The two effects differ only in the universe defining the tested contribution.

Use all eight supplied regression tasks described below. The experiment has
8 datasets x 3 sample sizes x 1,000 repetitions = **24,000 subdatasets**, and
24,000 x 2 effect definitions = **48,000 tests**. Repeat the full selection and
inference procedure for each subsample; "1,000 repetitions" does not mean
1,000 tests conditional on choosing any particular feature.

### Matching across sample sizes and effects

For each dataset and repetition, generate a deterministic random ordering of
the dataset's row indices. Take its first 100, 150 and 200 rows to obtain
nested samples, then sort each chosen set of indices before fitting. This
gives a uniform sample without replacement at every n while pairing the
sample-size comparison. Different repetitions use independent random streams;
subsamples may overlap because they come from the same finite source data.
Nested matching is our variance-reduction choice; the paper specifies random
subsamples but does not state that its samples are nested across n.

At each n, both effects receive identical rows, responses, selected sets and
actual tested features. Share the auxiliary target-selection uniform draw
across n within a repetition; actual selected sets and targets can still
change with n. Record the master seed, seed derivation, row indices, input
hashes and target randomization in an immutable, separately named manifest
and result receipts. Do not reuse the old n=100 manifest or append these tests
to earlier outputs.

### Required tables, figures and interpretation

- Produce a table analogous to the paper's Table 1: rows n=100, 150, 200,
  columns datasets, with both effect definitions' rejection fractions,
  numerators, denominators and pointwise 95% binomial intervals.
- Plot **sample size n** on the horizontal axis and selective rejection rate
  on the vertical axis, with one panel per dataset and one curve per effect.
  Every curve must include all three sample sizes. Use the display names above.
- Report failures/unresolved tests separately, retain them in the intended
  1,000-trial accounting, and distinguish successful-test rejection fractions
  from detection frequency over all attempted trials.
- Preserve paired outcomes across n and effects so uncertainty in differences
  can be assessed with paired resampling by repetition. Do not treat the
  nested samples or two effects as independent observations.
- Include tested-space ranks and target-feature frequencies: both the
  selected target and the hypothesis may change as n changes. All-Features
  effects may test signal in the full candidate RBF space, so larger rejection
  rates do not establish better identification of unique feature effects.

The paper calls its real-data metric "power" under the explicit assumption
that selected features are truly relevant. That assumption is not observed
ground truth. In our results, label the primary quantity **selective rejection
rate**; "power under the selected-feature relevance assumption" may be used
only with that assumption stated. Increasing n is expected to improve
detection in many settings, but monotonicity is a scientific question, not a
pass/fail requirement or a property to enforce in the plots. The real-data
benchmark does not itself estimate Type I error; use the separate synthetic
global-null study for that purpose. Sigma=1 remains a working scale rather
than an independently estimated real-data noise SD.

Execution must use at most **64 CPU cores total** (the user's updated limit), with numerical-library
threads fixed at one and no nested worker pools. Before the full run, measure
cost on representative dataset/sample-size cells: 7--11 native features mean
128--2,048 coalitions, rather than the synthetic study's 16. Create dedicated
resumable runner/manifests, verifier, launcher and outputs following
`docs/commands.md` before launching. The legacy commands below do **not**
execute this expanded RBF benchmark.

The expanded experiment is implemented in `run_rbf_shapley_realdata.py`, with
the dedicated launcher `examples/rbf_shapley_realdata_cpu64.ps1` and frozen
manifest `rbf_shapley_realdata_n100_150_200_cpu64_1000_20261024.json`.
See `docs/rbf_shapley_realdata_study.md` for the execution and verification
details. It runs up to 64 workers with one numerical thread each and reduces
the worker count if fewer CPUs are available.

## Legacy selected-five n=100 benchmark

The remaining sections document the original five-arm benchmark for
provenance. They describe its frozen n=100 behavior, not the required design
of the next RBF benchmark. Its runner and production manifest have not been
changed by this specification update.

`run_exact_pipeline_benchmark.py` applies the pipeline archived at
`outputs/selection_method_validation/exact_set/selected_five_kde2p5_300_20260923`
to the datasets derived from `benchmarks/dataset/`.

### What the legacy benchmark holds fixed

- Five settings: SHAP + AIS, LIME (40 explanations) + AIS, spline screening +
  AIS, spline screening + exact inference, and mutual-information KDE
  (bandwidth multiplier 2.5) + AIS.
- `n=100`, `k_select=2`, spline-projection statistic, `sigma=1`, no
  multiplicity adjustment, and exact selected-set conditioning.
- One target drawn uniformly from the two selected features using a stored,
  method-shared target seed. This is the `uniform_from_selected` target rule
  actually recorded by the source run.
- AIS pilot 10 x 320, final batches of 800, at most 51,200 final proposals,
  denominator ESS 80, tail ESS 15, early ESS stopping, and
  `a_robust_draw_floor`.
- Random forest (SHAP and LIME): 50 trees, depth 5, seed 42, one job. LIME uses
  64 neighborhood samples, 40 explanation rows, its default kernel width, and
  ridge penalty 1. KDE MI retains the archived inactive neighbor/bin values
  (30 and 5) as metadata.
- 1,000 paired row subsamples per dataset in the production benchmark, matching
  the real-data repetition count in the comparison paper. This deliberately
  increases only the outer replication count from the source run's 300; it
  does not change the pipeline. All five settings receive the same rows,
  target seed, and inference seed in a repetition.

The runner checks SHA-256 hashes of the three core pipeline source files
against the archived run before executing. It resumes from `results.csv`,
records exceptions instead of silently dropping them, produces an ongoing
summary, and audits that the exact and AIS spline arms have identical observed
selection, target, and statistic.

### Legacy compatibility changes

Only two methodological inputs change: synthetic data are replaced with the
requested benchmark data, and each dataset keeps its native 7--11 feature
columns instead of the synthetic design's 20. The supplied pickle is used
because it is the committed output of `process.ipynb`: full-dataset feature and
response standardization plus the notebook's documented jitter. There is no
within-subsample re-standardization or retuning.

The seven files in `benchmarks/dataset/` define eight regression tasks because
the Energy Efficiency workbook supplies separate heating- and cooling-load
responses. Abalone is included because it is one of the supplied datasets;
Airfoil is not silently substituted or downloaded.

### Repetition count and sample-size coverage

The comparison paper uses 10,000 generated null datasets per synthetic
configuration and a total of 10,000 qualifying tests per synthetic power
condition (paper p. 7). Its real-data experiment instead uses 1,000 random
subdatasets from every original dataset at each sample size (paper p. 8).
The legacy production manifest uses 1,000 subdatasets per dataset but retains
only n=100 from the referenced nonlinear run. It therefore does not implement
the paper's full sample-size comparison. The next benchmark must include
n=100, 150, 200 as specified above, in a separately named experiment. Expanding
the legacy five-arm comparison itself to all three sizes would require
8 x 3 x 1,000 x 5 = 120,000 tests; the two-effect RBF comparison requires
48,000 tests instead.

### Legacy commands (n=100 only)

The production run is launched on the CPU server with the dedicated PowerShell
launcher in `examples/exact_pipeline_benchmark_1000_cpu32.ps1`. It assigns the
1,000 repetitions round-robin to 32 isolated, resumable shards. Each shard uses
one process and all nested numerical and forest thread counts are fixed at one.

For a labelled smoke run without altering any selector or inference setting:

```powershell
python benchmarks/run_exact_pipeline_benchmark.py run `
  --output outputs/benchmarks/selected_five_kde2p5_smoke `
  --repetitions 1
```

Reducing repetitions changes only the amount of benchmark replication. Such an
output is a pilot and must not be merged with the production 1,000-repetition
run.

### Legacy interpretation

Real datasets do not reveal which selected spline-projection null hypotheses
are false. The reported fraction below 0.05 is therefore a selective rejection
rate, not a calibrated estimate of power. AIS values are estimated selective
p-values and must be read together with their ESS, proposal-count, Monte Carlo
error, and failure diagnostics. `sigma=1` is inherited as a working scale from
the standardized-response validation pipeline; it is not an independently
estimated noise standard deviation.
