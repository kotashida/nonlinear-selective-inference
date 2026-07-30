import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import nbformat
import pandas as pd
import shap
from nbclient import NotebookClient
from patsy import dmatrix
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm


NOTEBOOK = Path("notebooks/unadjusted_vs_random.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_clean_and_has_no_hidden_cells():
    notebook = _notebook()

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 5
    for cell in notebook["cells"]:
        assert not cell.get("metadata", {}).get("hide_input", False)
        assert not cell.get("metadata", {}).get("collapsed", False)
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_notebook_simulation_computes_both_variance_modes_from_one_run():
    notebook = _notebook()
    namespace = {
        "np": np,
        "pd": pd,
        "shap": shap,
        "stats": stats,
        "RandomForestRegressor": RandomForestRegressor,
        "dmatrix": dmatrix,
        "tqdm": tqdm,
    }
    exec("".join(notebook["cells"][1]["source"]), namespace)

    result = namespace["run_simulation"](
        n_iters=2,
        n_samples=20,
        n_features=4,
        k_select=[1, 2],
        seed=7,
        sigma=1.0,
        rf_params={"n_estimators": 3, "max_depth": 2, "random_state": 42},
        verbose=False,
        show_progress=False,
    )

    assert set(result["summary"]["variance"]) == {"known", "unknown"}
    assert result["pvals_shap"]["known"].shape == (2, 2)
    assert result["pvals_shap"]["unknown"].shape == (2, 2)
    assert result["pvals_random"]["known"].shape == (2, 2)


def test_notebook_executes_all_cells_in_order(monkeypatch, tmp_path):
    output_dir = tmp_path / "notebook-output"
    monkeypatch.setenv("SI_SHAP_NOTEBOOK_N_ITERS", "2")
    monkeypatch.setenv("SI_SHAP_NOTEBOOK_OUTPUT_DIR", str(output_dir))
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    executed = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()

    assert all(
        cell.get("execution_count") is not None
        for cell in executed.cells
        if cell.cell_type == "code"
    )
    assert (output_dir / "unadjusted_vs_random_results.csv").is_file()
    assert (output_dir / "unadjusted_vs_random_rf_sensitivity.png").is_file()
