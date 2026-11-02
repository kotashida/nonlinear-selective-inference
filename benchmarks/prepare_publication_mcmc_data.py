"""Prepare unjittered, disjoint scaling/training/evaluation pools from local files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks/dataset"
OUT = ROOT / "benchmarks/publication_mcmc_data"
SEED = 2026100903


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_tasks():
    energy = pd.read_excel(RAW / "ENB2012_data.xlsx")
    gas = pd.read_csv(RAW / "gt_2015.csv")
    housing = pd.read_excel(RAW / "Real estate valuation data set.xlsx")
    concrete = pd.read_excel(RAW / "Concrete_Data.xls")
    abalone = pd.read_csv(RAW / "abalone.csv", header=None)
    tasks = {
        "heating_load": (energy.iloc[:, :8], energy.iloc[:, 8], "ENB2012_data.xlsx"),
        "cooling_load": (energy.iloc[:, :8], energy.iloc[:, 9], "ENB2012_data.xlsx"),
        "gas_turbine": (gas.drop(columns="TEY"), gas.TEY, "gt_2015.csv"),
        # Sex is deliberately excluded to preserve the previous numeric task.
        "abalone": (abalone.iloc[:, 1:-1], abalone.iloc[:, -1], "abalone.csv"),
        "concrete": (concrete.iloc[:, :-1], concrete.iloc[:, -1], "Concrete_Data.xls"),
        # UCI marks No as an ID, not a predictor.
        "housing": (housing.iloc[:, 1:-1], housing.iloc[:, -1], "Real estate valuation data set.xlsx"),
    }
    for name in ("red_wine", "white_wine"):
        filename = "winequality-" + name.split("_")[0] + ".csv"
        table = pd.read_csv(RAW / filename, sep=";")
        tasks[name] = (table.iloc[:, :-1], table.iloc[:, -1], filename)
    return tasks


def partition(X, name):
    """Separate building shapes, duplicate predictors, or ordered turbine blocks."""
    if name == "gas_turbine":
        count = len(X) // 5
        return np.arange(count), np.arange(count, len(X)-count), np.arange(len(X)-count, len(X))
    grouping = X[:, :5] if name in {"heating_load", "cooling_load"} else X
    _, groups = np.unique(grouping, axis=0, return_inverse=True)
    order = np.random.default_rng(SEED).permutation(int(groups.max())+1)
    count = max(1, len(order) // 5)
    calibration = np.flatnonzero(np.isin(groups, order[:count]))
    evaluation = np.flatnonzero(np.isin(groups, order[count:2*count]))
    training = np.flatnonzero(np.isin(groups, order[2*count:]))
    return calibration, training, evaluation


def prepare(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (features, response, filename) in raw_tasks().items():
        X, y = np.asarray(features, dtype=float), np.asarray(response, dtype=float)
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError(f"Missing/nonfinite raw data in {name}; no silent imputation")
        cal, train, evaluation = partition(X, name)
        assert not (set(cal) & set(train) or set(cal) & set(evaluation) or set(train) & set(evaluation))
        scale = X[cal].std(axis=0)
        scale[scale == 0] = 1  # Preserve genuinely constant columns; no invented jitter.
        y_scale = float(y[cal].std())
        if y_scale <= 0 or len(train) < 200:
            raise ValueError(f"Inadequate calibration or training pool: {name}")
        values = dict(X=(X-X[cal].mean(axis=0))/scale, y=(y-y[cal].mean())/y_scale,
                      calibration_rows=cal, training_rows=train, evaluation_rows=evaluation)
        path = output / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                for key, value in values.items():
                    np.testing.assert_array_equal(saved[key], value, err_msg=f"Prepared data drift: {name}/{key}")
        else:
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **values)
            tmp.replace(path)
        records[name] = dict(file=path.name, sha256=digest(path), raw_file=filename,
            raw_sha256=digest(RAW/filename), n=len(X), p=X.shape[1], feature_names=list(map(str, features.columns)),
            calibration_n=len(cal), training_n=len(train), evaluation_n=len(evaluation),
            response_scale=y_scale, response_center=float(y[cal].mean()),
            split="ordered_blocks" if name == "gas_turbine" else "building_shape_groups" if name in {"heating_load", "cooling_load"} else "identical_predictor_groups")
    metadata = dict(schema=1, seed=SEED, jitter=False, housing_id_included=False,
                    abalone_sex_included=False, scaling="calibration_pool_only_ddof0", datasets=records,
                    preparation_source_sha256=digest(Path(__file__)))
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError("Prepared manifest drift; use a new data directory")
    path.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    prepare(parser.parse_args().output)
