import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
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
    smf = pytest.importorskip("statsmodels.formula.api")
    notebook = _notebook()
    namespace = {
        "np": np,
        "pd": pd,
        "shap": shap,
        "stats": stats,
        "RandomForestRegressor": RandomForestRegressor,
        "smf": smf,
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
