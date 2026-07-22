# SHAP Selective Inference Simulation

This project simulates selective inference after features are selected using SHAP importance from a nonlinear model. It implements feature selection based on Random Forest Tree SHAP importance, chi tests for B-spline effects, selective $p$-value estimation using Adaptive Importance Sampling (AIS), and selection-region visualization.

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
├── docs/
│   ├── unadjusted_vs_random.md
│   └── run_selective_inference.md
├── tests/
│   ├── test_inference.py
│   ├── test_selection.py
│   ├── test_selection_regions.py
│   └── test_simulation.py
└── presentation/
    ├── .gitignore
    ├── presentation.tex
    ├── presentation.pdf
    ├── assets/
    │   ├── unadjusted_vs_random_known.png
    │   ├── unadjusted_vs_random_unknown.png
    │   └── shap_selection_regions.png
    └── examples/
        ├── 250728_shiraishi.pdf
        └── 250908_shiraishi.pdf
```

## File Descriptions

### Repository root

- `.gitignore`: Defines generated files that are excluded from Git, including Python caches, virtual environments, notebook checkpoints, test caches, build artifacts, and execution results.
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

### Mathematical and experimental specifications: `docs/`

- `docs/unadjusted_vs_random.md`: Explains the Random and Unadjusted SHAP comparison, including the global null hypothesis, SHAP selection, B-spline projection, unknown-variance F test, known-variance chi test, and false positive rate definition, with equations linked to the implementation.
- `docs/run_selective_inference.md`: Gives a beginner-oriented, function-by-function derivation of the complete `examples/run_selective_inference.py` workflow: the global-null model, SHAP selection, spline projection and chi test, selective conditioning, AIS estimation and diagnostics, false positive rate aggregation, and output interpretation.

### Tests: `tests/`

- `tests/test_selection.py`: Tests tie handling in top-$k$ selection, invalid SHAP importance inputs, and Random Forest parameter resolution.
- `tests/test_inference.py`: Tests centering and orthonormality of the B-spline basis, the chi statistic and projection, and effective sample sizes.
- `tests/test_simulation.py`: Tests simulation-argument validation, aggregation rules that account for AIS failures, and reproducible null-data generation.
- `tests/test_selection_regions.py`: Tests bisection of selection-interval boundaries, anchor points that preserve narrow observed intervals, chi-probability integration, Random Forest parameter forwarding, and proposal display in single-region plots.

### Presentation: `presentation/`

- `presentation/.gitignore`: Excludes `.aux`, `.log`, `.nav`, `.out`, `.snm`, `.toc`, and SyncTeX files generated by LuaLaTeX and Beamer.
- `presentation/presentation.tex`: Source for the 16:9 Beamer slides covering research progress, the Unadjusted versus Random experiment, selection regions, and AIS challenges. Compile it with LuaLaTeX.
- `presentation/presentation.pdf`: Presentation slides generated from `presentation.tex` for viewing and distribution.
- `presentation/assets/unadjusted_vs_random_known.png`: Slide image comparing Unadjusted SHAP with Random selection under the known-variance chi test.
- `presentation/assets/unadjusted_vs_random_unknown.png`: Slide image comparing Unadjusted SHAP with Random selection under the unknown-variance F test.
- `presentation/assets/shap_selection_regions.png`: Slide image showing SHAP selection regions and densities for multiple datasets.
- `presentation/examples/250728_shiraishi.pdf`: Archived presentation from July 28, 2025, retained as a content and structure reference.
- `presentation/examples/250908_shiraishi.pdf`: Archived presentation from September 8, 2025, retained as a content and structure reference.

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
- `presentation/presentation.aux`, `.log`, `.nav`, `.out`, `.snm`, `.toc`: Intermediate files generated while compiling `presentation.tex` with LuaLaTeX and Beamer.

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

The recommended preset compares `k_select` values 1, 2, and 5 with the
baseline, shallow, medium, and flexible forests (12 runs):

```powershell
python examples/sweep_selection_region_settings.py 101 202 303 `
    --preset recommended
```

Settings can also be chosen explicitly:

```powershell
python examples/sweep_selection_region_settings.py 101 202 303 `
    --k-select 2 5 `
    --rf-config shallow flexible
```

Each experiment is written below `outputs/selection_region_sweep/`, and a
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

Compile the presentation slides with LuaLaTeX from the `presentation/` directory:

```powershell
cd presentation
lualatex presentation.tex
```
