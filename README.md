# SHAP Selective Inference Simulation

This project simulates selective inference after features are selected using SHAP importance from a nonlinear model. It implements feature selection based on Random Forest Tree SHAP importance, chi tests for B-spline effects, selective $p$-value estimation using Adaptive Importance Sampling (AIS), and selection-region visualization.

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

`feature_results` contains individual raw selective p-values, optional Holm or
Bonferroni adjusted values, the event definition, denominator and tail ESS,
Monte Carlo standard error, and convergence status. Exact-set conditioning does
not itself correct multiplicity. A non-`ok` row has `NaN` as its selective
p-value and must not be interpreted as a valid estimate. These are
selection-adjusted p-values conditionally valid under the stated assumptions, up
to numerical and Monte Carlo error.

For `k_select=k`, the API returns `k` rows and selection-region visualization
returns `k` plots: every selected feature has a distinct response path
`y_j(z)`. Exact-set conditioning is stronger and can reduce power and effective
sample size; inclusion is weaker, while exact-ranking additionally preserves
the observed order.

## Repository Structure

The following tree shows the current layout of source files and retained project artifacts. Local environments, caches, and intermediate build files such as `.venv/` are described later.

```text
shap-selective-inference/
├── .gitignore
├── README.md
├── pyproject.toml
├── src/
│   └── si_shap/
│       ├── __init__.py
│       ├── inference.py
│       ├── plotting.py
│       ├── selection.py
│       ├── selection_regions.py
│       └── simulation.py
├── examples/
│   └── plot_selection_regions.py
├── notebooks/
│   ├── unadjusted_vs_random.ipynb
│   └── si_vs_unadjusted_vs_random.ipynb
└── tests/
│   ├── test_inference.py
│   ├── test_selection.py
│   ├── test_selection_regions.py
│   └── test_simulation.py
```

## File Descriptions

### Repository root

- `.gitignore`: Defines paths excluded from Git, including local documentation and presentation materials, Python caches, virtual environments, notebook checkpoints, test caches, build artifacts, and execution results.
- `README.md`: Describes the project's purpose, structure, file roles, setup, and primary workflows.
- `pyproject.toml`: Defines package metadata, the supported Python version, runtime, notebook, and test dependencies, and the setuptools and pytest configuration.

### Python package: `src/si_shap/`

- `src/si_shap/__init__.py`: Defines the package's public API, making the simulation, selection-region calculation, table conversion, and plotting functions importable directly from `si_shap`.
- `src/si_shap/selection.py`: Fits the configurable Random Forest, computes mean absolute Tree SHAP importance, and performs deterministic top-$k$ feature selection with an explicit tie-breaking rule.
- `src/si_shap/inference.py`: Implements an orthonormal basis for centered B-spline effects, the known-variance chi statistic, truncated normal distributions, effective sample sizes, and AIS proposal adaptation, sampling, and convergence diagnostics.
- `src/si_shap/simulation.py`: Generates data under the global null hypothesis, runs the Random, Unadjusted SHAP, and Selective SHAP (AIS) methods, and summarizes their $p$ values, false positive rates, failure rates, and Monte Carlo diagnostics.
- `src/si_shap/selection_regions.py`: Numerically scans the SHAP selection event as the response moves along the observed effect direction, then computes selection-interval boundaries, selection probabilities under the chi distribution, and proposal parameters for each dataset.
- `src/si_shap/plotting.py`: Creates $p$-value histograms for the three methods and selection-region plots for one or more datasets, overlaying the chi density, selection-conditional density, and AIS proposal.

### Examples: `examples/`

- `examples/run_selective_inference.py`: Runs the full SHAP selective-inference workflow with AIS and saves feature-level selective p-values and convergence diagnostics.
- `examples/plot_selection_regions.py`: Computes selection regions for random seeds supplied on the command line and generates `outputs/shap_selection_regions/shap_selection_regions.csv` and `outputs/shap_selection_regions/shap_selection_regions.png`.
- `examples/sweep_selection_region_settings.py`: Compares selection regions across several top-$k$ and Random Forest settings.

### Notebooks: `notebooks/`

- `notebooks/unadjusted_vs_random.ipynb`: A self-contained experiment comparing the false positive rates of Random selection and Unadjusted SHAP selection using both an unknown-variance F test and a known-variance chi test.
- `notebooks/si_vs_unadjusted_vs_random.ipynb`: Uses the package's `run_simulation` function to compare Random, Unadjusted SHAP, and Selective SHAP (AIS), including summary tables, AIS diagnostics, and $p$-value histograms.

### Tests: `tests/`

- `tests/test_selection.py`: Tests tie handling in top-$k$ selection, invalid SHAP importance inputs, and Random Forest parameter resolution.
- `tests/test_inference.py`: Tests centering and orthonormality of the B-spline basis, the chi statistic and projection, and effective sample sizes.
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

To compare the three methods, open `notebooks/si_vs_unadjusted_vs_random.ipynb` and run its cells from top to bottom. AIS refits the Random Forest and recomputes SHAP values for every candidate response, so begin with a small `n_iters` value when checking that the workflow runs correctly.

Run SHAP selective inference with AIS directly from the repository root:

```powershell
python examples/run_selective_inference.py `
    --n-iters 1 `
    --n-samples 100 `
    --n-features 20 `
    --k-select 1 `
    --seed 123
```

The feature-level results are written to
`outputs/selective_inference_ais/selective_inference.csv`. A row with
`status=ok` contains the AIS selective p-value in `p_value`, together with the
denominator and tail ESS and the Monte Carlo standard error. A non-`ok` status
has `p_value=NaN`; increase `--max-final-samples` or revise the ESS thresholds
rather than interpreting it as a valid p-value.

### Compare exact-set and feature-inclusion power

Use the paired power experiment to compare the two conditioning events under a
nonlinear alternative:

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

Every event is evaluated on identical generated data and with the same AIS seed
within an iteration. The primary `power` estimate is
`P(signal feature is selected and its selective test rejects)`. The output also
separates `signal_selection_rate` and `conditional_power`, because the former is
identical across conditioning events while the latter can differ. A strict
power estimate is `NaN` if any selected-signal AIS test fails; the explicitly
labeled `converged_power` remains available for diagnosis.

The plot shows a converged-only fallback only when at least 20 and at least
half of the iterations are complete; otherwise it reports `NA` with the
complete-case count. Result files are staged together and replace the output
directory only after every CSV, JSON, and plot has been generated successfully.
`settings.json` records the selected preset, package versions, Python/platform
details, and Git commit/dirty state.

Use `k_select >= 2` for this comparison. When `k_select=1`, preserving the full
selected set and preserving inclusion of its only feature are the same event, so
their exact power is identical.

Results are saved under the repository's `outputs/selection_event_power/`
folder by default, even when the command is launched from another working
directory. Use `--output-dir` only when a different location is desired:

- `power_summary.csv`: power, selection, and failure metrics by event.
- `paired_power_comparison.csv`: paired power differences, standard errors, and
  approximate 95% intervals; positive means the comparison event has more power
  than the baseline event.
- `signal_results.csv` and `feature_results.csv`: iteration-level data for
  auditing the aggregate estimates.
- `power_comparison.png`: a bar chart of overall power with simulation-error
  bars.

Increase `--n-iters` for a final comparison. The workflow is computationally
expensive because AIS repeatedly refits the selection model for both events.

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
