# SHAP Selective Inference Simulation

This project simulates selective inference after features are selected using SHAP importance from a nonlinear model. It implements Random Forest Tree SHAP selection, chi tests for B-spline effects, finite-sample-valid conditional Monte Carlo rank $p$ values, explicitly exploratory Adaptive Importance Sampling (AIS), and selection-region visualization.

## Real-data API and statistical scope

Use `selective_inference` for externally supplied data. The conservative default
conditions on equality of the **unordered observed top-$k$ set**; weaker target
feature inclusion and ordered exact-ranking events are explicit alternatives.

```python
from sklearn.ensemble import RandomForestRegressor
from si_shap import selective_inference

result = selective_inference(
    X,
    y,
    k_select=2,
    estimator=RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    ),
    sigma=known_sigma,
    selection_event="exact_set",
    multiplicity="holm",
)

print(result["observed_selected_features"])
print(result["importance_table"])
print(result["feature_results"])
```

The tested null is a zero fixed-design projection onto the target feature's
centered cubic B-spline basis: a **marginal nonlinear-association** hypothesis,
not a Random Forest coefficient, a SHAP value, a conditional effect, or a causal
effect. Validity requires fixed `X`, independent Gaussian errors with known,
user-supplied `sigma`, and a deterministic selection pipeline. Unknown-variance
inference is deliberately rejected rather than silently estimating variance from
selected data. The currently supported built-in combination is a cloneable
scikit-learn tree estimator with Tree SHAP; `RandomForestRegressor` is the
officially tested estimator and every exposed `random_state` must be fixed.

`feature_results` contains individual selective p-values, optional Holm or
Bonferroni adjusted values, the event definition, retained conditional-Monte-Carlo
draw counts, and resolution diagnostics. Exact-set conditioning does not itself
correct multiplicity. The default samples a fixed number of null chi radii and
uses the selected-draw rank formula `(tail + 1) / (selected + 1)`. It is
finite-sample valid and becomes conservatively equal to one when no proposal
reproduces selection. `inference_method="ais"` retains the self-normalized AIS
ratio as an explicitly approximate exploratory estimate; it is not an exact
finite-sample p-value, even when ESS diagnostics are satisfactory.

By default, for `k_select=k` the API returns `k` rows. With
`target_rule="uniform_from_selected"`, it instead draws one auxiliary
`u ~ Uniform(0,1)`, maps the canonically sorted selected set to one target, and
returns one row. The same fixed `u` must be reused for every candidate response.
The `same_target` event conditions only on reproducing that complete algorithm's
target; `exact_set` also preserves the full selected set, `feature_inclusion`
only preserves membership of the target, and `exact_ranking` preserves the
complete feature ranking, including features below the top-$k$ boundary.

## Repository Structure

The following tree shows the current layout of source files and retained project artifacts. Local environments, caches, and intermediate build files such as `.venv/` are described later.

```text
nonlinear-selective-inference/
├── .gitignore
├── README.md
├── pyproject.toml
├── src/
│   └── si_shap/
│       ├── __init__.py
│       ├── api.py
│       ├── inference.py
│       ├── null_calibration.py
│       ├── plotting.py
│       ├── power.py
│       ├── selection.py
│       ├── selection_regions.py
│       └── simulation.py
├── examples/
│   ├── compare_selection_event_power.py
│   ├── compare_selection_event_null_calibration.py
│   ├── plot_selection_regions.py
│   ├── run_selective_inference.py
│   └── sweep_selection_region_settings.py
├── notebooks/
│   └── unadjusted_vs_random.ipynb
└── tests/
    └── test_*.py
```

## File Descriptions

### Repository root

- `.gitignore`: Defines paths excluded from Git, including local documentation and presentation materials, Python caches, virtual environments, notebook checkpoints, test caches, build artifacts, and execution results.
- `README.md`: Describes the project's purpose, structure, file roles, setup, and primary workflows.
- `pyproject.toml`: Defines package metadata, the supported Python version, runtime, notebook, and test dependencies, and the setuptools and pytest configuration.

### Python package: `src/si_shap/`

- `src/si_shap/__init__.py`: Defines the package's public API, making the simulation, selection-region calculation, table conversion, and plotting functions importable directly from `si_shap`.
- `src/si_shap/selection.py`: Fits the configurable Random Forest, computes mean absolute Tree SHAP importance, and performs deterministic top-$k$ feature selection with an explicit tie-breaking rule.
- `src/si_shap/inference.py`: Implements an orthonormal basis for centered B-spline effects, the known-variance chi statistic, exact conditional Monte Carlo rank p-values, and exploratory AIS diagnostics.
- `src/si_shap/simulation.py`: Generates data under the global null hypothesis, runs the Random, Unadjusted SHAP, and Selective SHAP methods, and summarizes their $p$ values, false positive rates, failure rates, and Monte Carlo diagnostics.
- `src/si_shap/selection_regions.py`: Numerically scans the SHAP selection event as the response moves along the observed effect direction, then computes selection-interval boundaries, selection probabilities under the chi distribution, and proposal parameters for each dataset.
- `src/si_shap/plotting.py`: Creates $p$-value histograms for the three methods and selection-region plots for one or more datasets, overlaying the chi density, selection-conditional density, and AIS proposal.

### Examples: `examples/`

- `examples/run_selective_inference.py`: Runs the finite-sample-valid workflow by default and optionally runs exploratory AIS.
- `examples/plot_selection_regions.py`: Computes selection regions for random seeds supplied on the command line and generates `outputs/shap_selection_regions/shap_selection_regions.csv` and `outputs/shap_selection_regions/shap_selection_regions.png`.
- `examples/sweep_selection_region_settings.py`: Compares selection regions across several top-$k$ and Random Forest settings.

### Notebooks: `notebooks/`

- `notebooks/unadjusted_vs_random.ipynb`: A self-contained experiment comparing the false positive rates of Random selection and Unadjusted SHAP selection using both an unknown-variance F test and a known-variance chi test.

### Tests: `tests/`

- `tests/test_selection.py`: Tests tie handling in top-$k$ selection, invalid SHAP importance inputs, and Random Forest parameter resolution.
- `tests/test_inference.py`: Tests centering and orthonormality, the chi statistic, exact conditional Monte Carlo rank construction and calibration, and exploratory AIS.
- `tests/test_simulation.py`: Tests simulation-argument validation, aggregation rules that account for AIS failures, and reproducible null-data generation.
- `tests/test_selection_regions.py`: Tests bisection of selection-interval boundaries, anchor points that preserve narrow observed intervals, chi-probability integration, Random Forest parameter forwarding, and proposal display in single-region plots.

## Generated Directories and Files

The following paths are generated during installation, testing, execution, or building. They are not source files and are normally excluded from Git.

- `.venv/`: Local Python virtual environment and installed dependencies.
- `.pytest_cache/`: Cache of information from previous pytest runs.
- `.vscode/`: Local VS Code workspace settings.
- `build/`, `dist/`: Python package build artifacts.
- `src/si_shap.egg-info/`: Metadata generated by editable installation or package builds.
- `outputs/`: CSV, PNG, and other generated results, grouped into one subdirectory per task.
- `tmp/`: Temporary working files.
- `__pycache__/`, `*.pyc`: Python bytecode caches.
- `.ipynb_checkpoints/`: Notebook autosave data generated by Jupyter.

## Setup

From the repository root, install the package together with its notebook and test dependencies:

```powershell
python -m pip install -e ".[notebook,test]"
```

## Configuring the Random Forest

Use the public APIs' `rf_params` argument to customize the Random Forest. Supplied values override the defaults:

```python
{"n_estimators": 50, "max_depth": 5, "random_state": 42}
```

For example:

```python
from si_shap import run_simulation

result = run_simulation(
    n_iters=10,
    n_samples=100,
    n_features=20,
    k_select=1,
    rf_params={"n_estimators": 100, "max_depth": 8},
)
```

`compute_selection_regions` accepts the same `rf_params` argument. Set `random_state` to a fixed integer to preserve reproducibility.

## Usage

To compare all three methods, run `examples/run_selective_inference.py` as
shown below. The retained notebook compares only Random and unadjusted SHAP
inference. Selective inference refits the Random Forest and recomputes SHAP
values for every candidate response, so begin with a small `n_iters` value.

Run finite-sample-valid SHAP selective inference directly from the repository root:

```powershell
python examples/run_selective_inference.py `
    --n-iters 1 `
    --n-samples 100 `
    --n-features 20 `
    --k-select 1 `
    --seed 123
```

The feature-level results are written to
`outputs/selective_inference/selective_inference.csv`. A row records the
finite-sample-valid rank p-value, the number of proposal radii reproducing the
event, the tail count, and the achieved selection-resolution diagnostic. Use
`--inference-method ais` only for explicitly exploratory self-normalized AIS.

The global-null summary reports featurewise FPR, FWER, global-null FDR, failure
rates, confidence intervals, and a p-value uniformity KS statistic. The KS value
is diagnostic only because selected-feature p-values can be dependent within an
iteration.

### Validate selective-p-value super-uniformity

Run the paired fixed-design global-null experiment to compare
`feature_inclusion`, `exact_set`, and `same_target`:

```powershell
python examples/compare_selection_event_null_calibration.py `
    --n-iters 100 `
    --n-samples 100 `
    --n-features 20 `
    --k-select 2
```

The experiment generates one fixed design matrix and independent Gaussian null
responses. In every iteration it chooses one target uniformly from the SHAP
selected set and reuses that target, the realized auxiliary uniform draw, and
the Monte Carlo seed for all three conditioning events. Only raw, unadjusted
selective p-values are assessed. Finite-budget rank p-values are discrete and
super-uniform, so calibration means rejection rates no larger than their nominal
levels rather than exact continuous uniformity.

Outputs under `outputs/selection_event_null_calibration/` include the complete
p-value audit table, strict and converged calibration summaries, paired
rejection-rate comparisons, the fixed design matrix, settings, histograms,
uniform-reference Q-Q plots, empirical-CDF plots, and ECDF-minus-uniform plots
with a one-sided 95% super-uniformity band. If exploratory AIS fails, strict rates are
reported as unavailable and lower/upper rejection-rate bounds retain the failed
replications in the denominator. Use a small run for smoke testing, then at
least 1,000--2,000 iterations with a predeclared fixed Monte Carlo budget for
substantive super-uniformity assessment.

### Compare three target-selection events

Use the paired power experiment to compare `feature_inclusion`, `exact_set`, and
`same_target` under a nonlinear alternative:

```powershell
python examples/compare_selection_event_power.py `
    --preset calibrated
```

The `quick` preset runs 10 iterations with signal strength 0.3 and an AIS
ceiling of 800 for smoke testing. The default `calibrated` preset runs 100
iterations at the same signal strength, 40 samples per proposal pilot, and a
ceiling of 1,600. The `calibrated_plus` preset increases the proposal pilot to
200 samples and the AIS ceiling to 6,400. The explicitly opt-in `stress` preset
uses signal strength 1.0 and a much larger AIS ceiling; it is intended to
diagnose extreme-tail behavior and can be very slow. Any of `--n-iters`,
`--signal-strength`, `--pilot-samples`, or `--max-final-samples` overrides its
preset value.

Every event uses identical data, the same observed selected set, the same single
randomized target, the same fixed auxiliary draw, and the same Monte Carlo seed within
an iteration. The primary `power` estimate is
`P(randomized target is a signal and its selective test rejects)`. The output
also reports `target_signal_rate` and
`conditional_power_given_signal_target`. A strict power estimate is `NaN` if
any signal-target AIS test fails; `converged_power` remains diagnostic.

The plot shows a converged-only fallback only when at least 20 and at least
half of the iterations are complete; otherwise it reports `NA` with the
complete-case count. Result files are staged together and replace the output
directory only after every CSV, JSON, and plot has been generated successfully.
`settings.json` records the selected preset, package versions, Python/platform
details, and Git commit/dirty state.

Use `k_select >= 2` for this comparison. For `k_select=1`, all three events are
identical.

Results are saved under the repository's `outputs/selection_event_power/`
folder by default, even when the command is launched from another working
directory. Use `--output-dir` only when a different location is desired:

- `power_summary.csv`: power, selection, and failure metrics by event.
- `paired_power_comparison.csv`: paired power differences, standard errors, and
  approximate 95% intervals; positive means the comparison event has more power
  than the baseline event.
- `target_results.csv` (also emitted as the compatibility alias
  `signal_results.csv`) and `feature_results.csv`: iteration-level audit data.
- `power_comparison.png`: a bar chart of overall power with simulation-error
  bars.

Increase `--n-iters` for a final comparison. The workflow is computationally
expensive because AIS repeatedly refits the selection model for all events.

Generate the selection-region figure and CSV from the repository root with:

```powershell
python examples/plot_selection_regions.py 101 202 303
```

To compare several `k_select` and Random Forest settings without overwriting
earlier outputs, run the sweep example.  The default quick preset compares
`k_select` values 1 and 2 with the baseline and medium forests (four runs):

```powershell
python examples/sweep_selection_region_settings.py 101 202 303
```

The recommended preset compares `k_select` values 1, 2, 5, and 10 with the
baseline, shallow, medium, and flexible forests under both exact-set and
feature-inclusion conditioning (32 runs per supplied dataset seed):

```powershell
python examples/sweep_selection_region_settings.py 101 202 303 `
    --preset recommended
```

Settings can also be chosen explicitly:

```powershell
python examples/sweep_selection_region_settings.py 101 202 303 `
    --k-select 2 5 `
    --rf-config shallow flexible `
    --selection-event exact_set feature_inclusion
```

Each conditioning/top-k/forest experiment is written below
`outputs/selection_region_sweep/`, and a
combined `all_selection_regions.csv` records the experiment name and complete
forest parameters.  These sweeps are computationally expensive because the
forest is refitted at every selection-region grid point.

Override Random Forest parameters with repeatable `--rf-param NAME=VALUE`
options:

```powershell
python examples/plot_selection_regions.py 101 202 303 `
    --rf-param n_estimators=100 `
    --rf-param max_depth=8 `
    --rf-param min_samples_leaf=2 `
    --rf-param n_jobs=-1
```

Values use JSON syntax when possible. For example, use
`--rf-param max_depth=null` for Python `None` and
`--rf-param bootstrap=false` for Python `False`. Parameters that are not
specified retain the project defaults.

Run the tests with:

```powershell
python -m pytest
```
