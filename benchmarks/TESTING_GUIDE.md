# Testing the nonlinear selective-inference pipeline on `benchmarks/`

> **Archived draft:** this guide predates the request to reproduce
> `selected_five_kde2p5_300_20260923` exactly. In particular, its KNN mutual
> information, baseline AIS, all-selected targeting, Airfoil requirement, and
> paper-derived 1,000 x three-sample-size design are not the configuration of
> that run. For the implemented exact-pipeline benchmark, use
> `EXACT_PIPELINE_BENCHMARK.md` and `run_exact_pipeline_benchmark.py`.

## Purpose and scope

This benchmark evaluates this repository's nonlinear selective-inference
pipeline on the real datasets and outer experimental design used in
*Statistical Test for Feature Selection Pipelines by Selective Inference*.
It is **not** an attempt to reproduce the paper's linear feature-selection
pipeline or its truncated-normal selective test.

The paper remains authoritative for the parts of the benchmark that are
compatible with this pipeline:

- the eight real datasets in Appendix D.5 (paper p. 16);
- significance level `alpha = 0.05` (Section 6, paper p. 7);
- sample sizes `n in {100, 150, 200}`;
- 1,000 random subdatasets for every dataset and sample size; and
- the real-data evaluation as a selective rejection-rate or "power" comparison
  under the explicit assumption that selected features are relevant (Section 6,
  paper p. 8 and Table 1).

The following are fixed requirements of this repository and replace the
corresponding paper-specific components:

1. The five settings are **Spline + Exact**, **Spline + AIS**, **SHAP + AIS**,
   **LIME + AIS**, and **Mutual Information + AIS**.
2. Every setting uses the spline-projection test statistic and null hypothesis.
3. Every setting conditions on equality of the complete selected feature set:
   `selection_event="exact_set"`.
4. All selected features are tested with `target_rule="all_selected"`.

Results from this protocol must therefore be described as an **adaptation of
the paper's real-data benchmark for the nonlinear pipeline**, not as a
reproduction of Table 1.

## Deviation register

Every difference from the paper belongs to one of the following categories.

### 1. Necessary for testing this pipeline

| Component | Protocol used here | Why the deviation is necessary |
|---|---|---|
| Feature selection | Spline screening, SHAP, LIME, or mutual information | These are the nonlinear selectors being evaluated. The paper instead uses marginal screening, forward stepwise selection, and Lasso inside two larger pipelines. |
| Inference | Exact truncated-chi inference for Spline + Exact; AIS for the other four settings | These are the pipeline's two inference engines. The paper uses a truncated-normal parametric-programming test. |
| Null and statistic | Zero projection of the fixed-design mean response onto the centered cubic B-spline basis; chi statistic | The paper's coefficient null in equation (3) is not the target of this nonlinear pipeline. |
| Selection event | Exact equality of the selected feature set only | The pipeline has no missing-value-imputation or outlier-detection outputs. The paper conditions on feature and outlier sets in equations (4)-(8), and on the CV-selected pipeline in Appendix F. |
| Compared methods | The five settings listed above | The paper's `proposed` and `w/o-pp` comparison is specific to its inference method and does not represent the five pipeline settings. |
| No paper-style CV pipeline construction | Each of the five settings is pre-specified and run directly | The paper's two-fold CV selects among 32 versions of its own `op1` and `op2` pipelines (Section 6, p. 7; Appendix G, Listing 2, p. 20). Reintroducing those pipelines would replace, rather than evaluate, the selectors of interest here. |
| No MVI or outlier-detection stages | The supplied complete numeric arrays are passed directly to this pipeline | These stages are components of the paper's method, not components of the nonlinear pipeline. |
| Synthetic scenarios | The paper's linear-null and linear-alternative simulations are not part of the primary real-data benchmark | They target the paper's coefficient test. Null calibration and power for this repository should continue to use its separately documented nonlinear simulation experiments. |

### 2. Implementation choices not specified by the paper

The following choices must be frozen before the run and reported with the
results. They must not be presented as settings taken from the paper.

- sampling rows without replacement;
- the master seed and deterministic seed derivation;
- use of the already-preprocessed `table_dataset.pkl`;
- extension of the same preprocessing recipe to Airfoil when it is added;
- `k_select = 2`;
- the selector hyperparameters specified below;
- the AIS proposal budget, ESS targets, and AIS strategy;
- the working value `sigma = 1.0` for the globally standardized responses;
- testing every selected feature without a feature-family multiplicity
  correction;
- confidence intervals, paired method comparisons, failure bounds, runtime
  summaries, and AIS diagnostics.

These choices are based where possible on the repository's existing matched
five-method validation configuration in
`outputs/selection_method_validation/exact_set/selected_five_methods_combined_ais_exact_mi30/selected_configuration.json`.
They are fixed here to prevent tuning on the real-data rejection rates.

### 3. Unnecessary deviations that this guide removes

- Do not substitute Abalone for the paper's Airfoil Self-Noise dataset.
- Do not choose among selectors, `k_select`, forest parameters, LIME settings,
  MI settings, AIS strategies, or AIS budgets after examining these results.
- Do not use `same_target`, `feature_inclusion`, or `exact_ranking` conditioning
  in this benchmark.
- Do not use MCMC or conditional Monte Carlo as one of the five primary
  settings.
- Do not label an ordinary unadjusted p-value as the paper's `w/o-pp` method.
- Do not merge provisional seven-dataset results or runs with different
  configurations into the final benchmark.

## Dataset specification

Appendix D.5 of the paper gives the following authoritative dataset list and
Table 1 order.

| Paper label | Required dataset | Local key | Current status |
|---|---|---|---|
| Data1 | Airfoil Self-Noise | `airfoil` | Missing; must be added before the final run |
| Data2 | Concrete Compressive Strength | `concrete` | Present |
| Data3 | Energy Efficiency, heating load | `heating_load` | Present |
| Data4 | Energy Efficiency, cooling load | `cooling_load` | Present |
| Data5 | Gas Turbine CO and NOx Emission | `gas_turbine` | Present |
| Data6 | Real Estate Valuation | `housing` | Present |
| Data7 | Wine Quality, red wine | `red_wine` | Present |
| Data8 | Wine Quality, white wine | `white_wine` | Present |

The local pickle currently contains Abalone instead of Airfoil. Abalone may be
reported as a separately labeled supplementary dataset, but it must not occupy
Data1 or enter the primary eight-dataset summary.

Use this fixed dataset order in configuration files, seed derivation, tables,
and plots. Do not rely on pickle dictionary order.

### Local preprocessing and provenance

The paper names the datasets but does not document its real-data preprocessing,
response-column choices, or random seeds. The following procedure comes from
`benchmarks/process.ipynb`, not from the paper:

1. Standardize every feature and response over the complete source dataset
   using the population standard deviation (`ddof=0`).
2. For a feature having fewer than `0.3 * N` distinct values in the complete
   dataset, add independent `N(0, 0.1^2)` jitter using NumPy RNG seed 0.
3. Save the resulting arrays as `(X, y)` in `table_dataset.pkl`.

Treat the existing pickle as the fixed benchmark input. Do not re-standardize
or re-jitter within a subdataset. When Airfoil is added, apply the same local
recipe once at the full-dataset level so that it is processed consistently with
the seven supplied datasets. Record the source URL, checksum, response column,
feature columns, and resulting shape.

The final report must also disclose these local dataset-specific choices, which
are not resolved by the paper:

- `gas_turbine` uses only `gt_2015.csv` and uses `TEY` as the response;
- `housing` includes the source file's `No` identifier among its seven feature
  columns; and
- the current Abalone response has NumPy `object` dtype, although Abalone is not
  part of the primary benchmark.

Convert all loaded arrays with `np.asarray(..., dtype=float)` before use and
verify that they are finite.

## Experimental design inherited from the paper

For each of the eight datasets:

1. Draw 1,000 random subdatasets of size 100, 150, and 200.
2. Use the same sampled row indices for all five settings.
3. Run selection and inference on the same subdataset; there is no train/test
   split in the paper's real-data experiment.
4. Use `alpha = 0.05`.
5. Record all selected-feature tests and summarize the selective rejection
   rate for each `(dataset, n, setting)` cell.

The paper says "randomly generated sub-sampled datasets" but does not specify
replacement. This protocol samples without replacement, which is an
implementation choice. The paper also does not report a seed; this protocol
uses a fixed master seed and stores every sampled row set.

Do not add the paper's two-fold pipeline-selection CV here. That CV chooses
among the paper's `op1` and `op2` configurations and is therefore one of the
components being replaced by the five fixed nonlinear settings.

## Frozen nonlinear pipeline specification

### Common target and conditioning

All five settings use:

```python
test_statistic="spline"
selection_event="exact_set"
target_rule="all_selected"
multiplicity="none"
```

For an observed selected set `M_obs`, exact-set conditioning means that a
candidate response is accepted only when its unordered selected set is exactly
`M_obs`. No `same_target`, feature-inclusion, or exact-ranking result belongs in
the primary benchmark.

The null for every selected feature `j` is:

> `H0`: the fixed-design mean response has zero projection onto the centered
> cubic B-spline basis of feature `j`.

This is a marginal nonlinear-association null. It is not a null about a linear
coefficient, SHAP value, LIME coefficient, conditional effect, or causal
effect. The statistic is the norm of the standardized spline coordinates and
has a chi reference distribution before selection adjustment.

`target_rule="all_selected"` is used because the real-data experiment evaluates
the selected features, and `k_select` is fixed. `multiplicity="none"` preserves
the paper's per-hypothesis rejection-rate interpretation. Any familywise claim
must be a separately labeled secondary analysis.

### Five primary settings

| Report label | Selector | Inference method |
|---|---|---|
| Spline + Exact | `spline_screening` | `exact_spline` |
| Spline + AIS | `spline_screening` | `ais` |
| SHAP + AIS | `shap` | `ais` |
| LIME + AIS | `LimeSelector` | `ais` |
| Mutual Information + AIS | `MutualInformationSelector` | `ais` |

Spline + Exact and Spline + AIS must have identical observed spline scores,
rankings, selected sets, target features, and test statistics on every matched
subdataset. This is an important audit: their p-values should differ only
because of the inference engine.

### Frozen selector settings

Use the following settings for the primary run:

- `k_select = 2` for every selector and dataset;
- `selection_decimals = 10`;
- SHAP and LIME forest: `RandomForestRegressor(n_estimators=50,
  max_depth=5, random_state=42, n_jobs=1)`;
- SHAP importance: mean absolute Tree SHAP importance;
- LIME: `random_state=42`, 64 neighborhood samples, 40 explanation rows,
  default kernel width `0.75 * sqrt(n_features)`, and ridge penalty 1.0;
- mutual information: k-nearest-neighbor estimator, 30 neighbors, and
  `random_state=42`;
- spline screening: centered cubic B-spline projection norm with the package's
  three-dimensional spline basis; and
- deterministic decreasing-score/increasing-feature-index tie breaking.

The 40-explanation LIME and 30-neighbor MI choices come from the repository's
existing selected five-method configuration. They are not paper parameters.

### Frozen AIS settings

Use one common AIS configuration for Spline + AIS, SHAP + AIS, LIME + AIS, and
Mutual Information + AIS:

```python
inference_method="ais"
ais_strategy="baseline"
pilot_iters=10
pilot_samples=320
final_batch_size=800
max_final_samples=51_200
min_denominator_ess=80.0
min_tail_ess=15.0
stop_when_ess_met=True
```

These values reproduce the budget of the existing matched five-method
configuration. If a different AIS strategy is later adopted, it must be chosen
from independent simulation validation before viewing this benchmark and run as
a new, separately named configuration. Never select the AIS strategy producing
the largest real-data rejection rate.

AIS produces a self-normalized importance-sampling estimate. The package does
not claim it is an exact finite-sample p-value. Consequently:

- call AIS results **estimated selective p-values** and their summaries
  **estimated selective rejection rates**;
- report ESS, proposal counts, stopping status, Monte Carlo standard error, and
  failure rate with every AIS result; and
- use Spline + Exact as the exact reference for the same spline selector.

### Working noise scale

The API requires a known positive `sigma`. The paper assumes known variance or
variance estimated from an independent dataset (Section 2, equation (1), p. 3)
but does not state the variance used for Table 1. Standardizing the full
response does not prove that regression noise has standard deviation one.

For this benchmark, freeze `sigma = 1.0` as a **working scale convention** for
all datasets, subdatasets, and settings. This makes the five settings
comparable, but it does not by itself justify confirmatory Gaussian-noise
claims. A scientifically justified external noise estimate may replace it only
in a separately named, pre-specified run.

## Reproducible method construction

Construct a fresh selector for each method call. This avoids sharing mutable
selector caches across subdatasets or parallel workers.

```python
from sklearn.ensemble import RandomForestRegressor

from si_shap.selection import LimeSelector, MutualInformationSelector


def make_forest():
    return RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=1,
    )


def method_arguments(label):
    if label == "Spline + Exact":
        return {
            "selection_method": "spline_screening",
            "inference_method": "exact_spline",
        }
    if label == "Spline + AIS":
        return {
            "selection_method": "spline_screening",
            "inference_method": "ais",
        }
    if label == "SHAP + AIS":
        return {
            "selection_method": "shap",
            "estimator": make_forest(),
            "inference_method": "ais",
        }
    if label == "LIME + AIS":
        return {
            "selector": LimeSelector(
                make_forest(),
                selection_decimals=10,
                random_state=42,
                num_samples=64,
                num_explanations=40,
                kernel_width=None,
                ridge_alpha=1.0,
            ),
            "inference_method": "ais",
        }
    if label == "Mutual Information + AIS":
        return {
            "selector": MutualInformationSelector(
                random_state=42,
                n_neighbors=30,
                estimator="knn",
            ),
            "inference_method": "ais",
        }
    raise ValueError(f"Unknown method: {label}")
```

The fixed common arguments are:

```python
COMMON_ARGUMENTS = {
    "k_select": 2,
    "sigma": 1.0,
    "test_statistic": "spline",
    "selection_event": "exact_set",
    "target_rule": "all_selected",
    "multiplicity": "none",
    "selection_decimals": 10,
}

AIS_ARGUMENTS = {
    "ais_strategy": "baseline",
    "pilot_iters": 10,
    "pilot_samples": 320,
    "final_batch_size": 800,
    "max_final_samples": 51_200,
    "min_denominator_ess": 80.0,
    "min_tail_ess": 15.0,
    "stop_when_ess_met": True,
}
```

Do not pass AIS arguments to Spline + Exact.

## Subdataset generation and paired execution

Use this fixed order and fail closed if a primary dataset is absent:

```python
DATASET_ORDER = (
    "airfoil",
    "concrete",
    "heating_load",
    "cooling_load",
    "gas_turbine",
    "housing",
    "red_wine",
    "white_wine",
)

METHODS = (
    "Spline + Exact",
    "Spline + AIS",
    "SHAP + AIS",
    "LIME + AIS",
    "Mutual Information + AIS",
)

ALPHA = 0.05
SAMPLE_SIZES = (100, 150, 200)
N_REPETITIONS = 1_000
MASTER_SEED = 20260914
SUMMARY_SEED = 20260915
```

For each `(dataset_index, n, repetition)`, derive a subdataset seed and one
method-independent inference seed. Use the same row indices and AIS seed for all
applicable settings:

```python
seed_sequence = np.random.SeedSequence(
    [MASTER_SEED, dataset_index, n, repetition]
)
subset_sequence, inference_sequence = seed_sequence.spawn(2)
subset_rng = np.random.default_rng(subset_sequence)
row_indices = np.sort(
    subset_rng.choice(len(y_full), size=n, replace=False)
)
inference_seed = int(inference_sequence.generate_state(1)[0])
```

Before running, require:

```python
missing = [name for name in DATASET_ORDER if name not in datasets]
if missing:
    raise RuntimeError(
        "Primary benchmark is incomplete; missing datasets: "
        + ", ".join(missing)
    )
```

A seven-dataset pilot may be run before Airfoil is available, but its output
directory and tables must contain `provisional_7_dataset` and must never be
merged into the primary result.

Within each subdataset, call:

```python
arguments = method_arguments(method)
if arguments["inference_method"] == "ais":
    arguments.update(AIS_ARGUMENTS)

result = selective_inference(
    X_full[row_indices],
    y_full[row_indices],
    **COMMON_ARGUMENTS,
    **arguments,
    ais_seed=inference_seed,
)
```

Save at least:

- the sorted row indices or a collision-resistant hash of them;
- dataset, sample size, repetition, and method;
- selected set, complete ranking, selector scores, and selector settings;
- target feature and observed statistic for every selected-feature test;
- raw selective p-value and ordinary unadjusted p-value;
- all exact-interval or AIS diagnostic columns returned by the API;
- subdataset and inference seeds;
- elapsed time and exception information; and
- the complete frozen run configuration and package versions.

Write results in restartable shards. A resumed job must validate its complete
configuration before appending.

## Primary evaluation metric

For each selected feature, define rejection using the raw selective p-value:

```python
reject = raw_selective_p_value <= 0.05
```

For every `(dataset, n, method)` cell, the primary metric is the mean rejection
fraction across the 1,000 subdatasets. First compute the fraction of the two
selected features rejected within each subdataset, then average those 1,000
fractions. Because `k_select=2` is fixed, this equals the pooled feature-level
rejection fraction when no tests fail.

Use the following terminology:

- **selective rejection rate** for Spline + Exact;
- **estimated selective rejection rate** for AIS settings; and
- **power under the all-selected-features-relevant assumption** only when
  explicitly invoking the same assumption made by the paper on p. 8.

Real data do not reveal which spline-projection nulls are false. The primary
metric is therefore not ground-truth power.

Table 1 of the paper does not specify its precise aggregation denominator or
confidence intervals. The subdataset-first mean above is an implementation
choice that treats the 1,000 resamples, rather than 2,000 dependent feature
tests, as replication units. Report a two-sided 95% percentile-bootstrap
interval using 10,000 bootstrap resamples of the 1,000 subdataset-level
fractions and `SUMMARY_SEED`. For a difference between two settings, resample
the matched repetition indices and compute the difference inside each
bootstrap resample; do not bootstrap the settings independently.

The primary table should have one row per dataset and sample size and five
clearly labeled method columns. Do not use the paper's `proposed/w/o-pp` slash
notation.

## Failures and AIS resolution

Never silently omit a failed test or subdataset.

With two expected feature tests per subdataset, each complete cell contains
2,000 tests. For every cell, report:

- requested subdatasets and completed subdatasets;
- expected, finite, and failed feature tests;
- strict rejection rate, reported only when there are no failures;
- converged-only rejection rate as a diagnostic;
- lower bound `rejections / expected_tests`; and
- upper bound `(rejections + failed_tests) / expected_tests`.

For each AIS setting, additionally summarize:

- proposals used;
- denominator ESS and tail ESS;
- Monte Carlo standard error;
- whether ESS stopping targets were met;
- fraction reaching the 51,200-draw cap;
- p-value pile-up at 0 or 1; and
- exception and non-finite-result counts.

An unmet ESS target, non-finite estimate, or exception is a computational
failure, not evidence for the null. Do not replace failures with p-values of 1
in the primary point estimate; use the strict bounds above.

For the matched spline settings, audit equality of selections and observed
statistics before comparing exact and AIS p-values. Report their p-value error,
rejection-decision agreement, and AIS coverage of the exact decision as
secondary numerical diagnostics.

## Secondary outputs

The following are useful but are not paper-specified primary metrics:

- the ordinary unadjusted spline p-value and rejection rate, labeled `naive`;
- selector stability and feature-selection frequencies;
- per-feature rejection rates;
- paired rejection-rate differences between the five settings;
- runtime and proposal-cost comparisons; and
- sensitivity to a larger pre-specified AIS budget or an independently chosen
  AIS strategy.

The naive result is not the paper's `w/o-pp` method. The paper defines `w/o-pp`
as an over-conditioned local-interval ablation in Section 6, p. 7; this
repository does not implement that procedure for these nonlinear selectors.

## Execution stages

The paper reports a 96-core CPU and 512 GB RAM (Appendix D.4, p. 16). The full
experiment contains 8 datasets x 3 sample sizes x 1,000 subdatasets x 5
settings, and each subdataset produces two feature tests. AIS selectors may
refit models many times, so run in three predeclared stages:

1. 5 repetitions per cell for API and data validation;
2. 100 repetitions per cell for runtime, failure, and ESS assessment; and
3. 1,000 repetitions per cell for the frozen final analysis.

Stages 1 and 2 are operational checks. Do not use their real-data rejection
rates to alter scientific or computational settings. Use one forest/BLAS thread
inside each worker and parallelize across independent subdatasets or shards.

## Required audit before interpreting results

Confirm all of the following:

1. The primary dataset keys exactly match `DATASET_ORDER`; Abalone is excluded.
2. Every dataset/sample-size cell has 1,000 stored row sets and all five methods
   use the same row set for a given repetition.
3. Every method reports `test_statistic="spline"`,
   `selection_event="exact_set"`, and `target_rule="all_selected"`.
4. Every observed selected set contains exactly two features.
5. Spline + Exact and Spline + AIS have identical selected sets, rankings, and
   observed statistics on matched subdatasets.
6. All selector parameters, AIS parameters, seeds, package versions, and the
   working sigma are identical to the frozen configuration.
7. Failures and low-ESS estimates are retained and reflected in bounds.
8. No results from pilot stages, different AIS strategies, different sigma
   values, or provisional dataset collections are mixed into the final table.

Run the package test suite before launching and after preparing the benchmark
environment:

```powershell
& .\.venv\Scripts\python.exe -m pytest
```

## Final reporting checklist

Report:

- that this is a nonlinear adaptation of the paper's real-data benchmark;
- the complete Data1-Data8 mapping and source provenance;
- the local preprocessing recipe and all undocumented dataset-specific choices;
- `n`, 1,000 subdatasets, sampling without replacement, and the sampling and
  summary seeds;
- the five settings and all selector hyperparameters;
- `k_select=2`, the spline-projection null, and the working `sigma=1.0` caveat;
- exact-set conditioning, all-selected targeting, and no multiplicity
  adjustment;
- the exact or AIS inference engine and fixed computational budget;
- selective rejection rates with subdataset-level uncertainty;
- failure bounds and AIS resolution diagnostics;
- paired exact-versus-AIS checks for spline screening; and
- the assumption required to call a real-data rejection rate "power."

Do not claim that the numerical results reproduce Table 1. The faithful claim
is that the experiment preserves the paper's real-dataset roster, sample-size
grid, replication count, significance level, and real-data rejection-rate
framework while replacing the paper-specific selection and inference machinery
with the five required nonlinear selective-inference settings.
