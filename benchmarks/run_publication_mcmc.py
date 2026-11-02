"""Frozen five-arm benchmark: paired trials, atomic receipts, bounded CPU use.

Production settings are taken from the October 8 validated MCMC experiment,
not the historical AIS benchmark. ``smoke`` uses its own labelled manifest.
No command in this module launches a remote benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.metadata as md
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
import time
import tomllib
import warnings

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS")
for key in THREAD_VARIABLES:
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits, threadpool_info

from si_shap import selective_inference
from si_shap.inference import _spline_effect_basis
from si_shap.selection import (ShapSelector, LimeSelector, SplineScreeningSelector,
    MutualInformationSelector, target_from_selected_set)

CONFIG = ROOT / "benchmarks/publication_mcmc20_20261009.json"
SMOKE_CONFIG = ROOT / "benchmarks/publication_mcmc20_smoke.json"
CONFIG_HASH = "f8971b9b488e726ca3312cdb747ab7abc9b10ba7320ca49575627652df638650"
SMOKE_CONFIG_HASH = "53ee52f13d16eaeb8d2ac77383b7f12b9f08c42395fadfaa4b9cff818a56cd79"
FOCUSED_CONFIG = ROOT / "benchmarks/publication_mcmc20_focused_20261010.json"
FOCUSED_CONFIG_HASH = "10b3144687928a08ba54c67386cffe219e8061cc84efece15ed9a94514da042b"
METHODS = ("shap_mcmc", "lime_mcmc", "spline_mcmc", "mutual_information_mcmc", "spline_exact")
SELECTORS = dict(zip(METHODS, ("shap", "lime", "spline_screening", "mutual_information", "spline_screening")))
_DATA_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def encoded(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def reference(m):
    path = ROOT / m["reference"]
    require(digest(path) == m["reference_sha256"], "Authoritative reference snapshot changed")
    return read(path)


def selector(m, method):
    a = reference(m)["requested_manifest"]
    name = SELECTORS[method]
    if name == "shap":
        return ShapSelector(RandomForestRegressor(**a["rf_params"]), selection_decimals=a["selection_decimals"])
    if name == "lime":
        return LimeSelector(RandomForestRegressor(**a["rf_params"]), selection_decimals=a["selection_decimals"], **a["lime"])
    if name == "mutual_information":
        return MutualInformationSelector(**a["mutual_information"])
    return SplineScreeningSelector()


def mcmc(m):
    values = dict(reference(m)["requested_manifest"]["mcmc"])
    if m["smoke"]:
        values.update(m["smoke_mcmc"])
    return values


def read_config(path=CONFIG):
    path = Path(path).resolve()
    hashes = {CONFIG.resolve(): CONFIG_HASH, SMOKE_CONFIG.resolve(): SMOKE_CONFIG_HASH,
              FOCUSED_CONFIG.resolve(): FOCUSED_CONFIG_HASH}
    require(path in hashes, "Use a named, frozen study manifest")
    require(digest(path) == hashes[path],
            "Study manifest changed; freeze a newly named experiment instead")
    m = read(path)
    require(m["schema"] == 1 and m["methods"] == list(METHODS), "Schema/method drift")
    require(1 <= m["workers"] <= 32, "CPU worker limit must be in [1,32]")
    require(m["alpha_levels"] == [.01, .05, .1] and m["sigma"] == 1, "Testing convention drift")
    a = reference(m)
    # Freeze every core algorithm, including selector defaults and rank construction.
    for name, checksum in a["source_hashes"].items():
        if name.startswith("src/"):
            require(digest(ROOT/name) == checksum, f"Validated algorithm source drift: {name}")
    for method in METHODS:
        actual = clean(dict(selector(m, method).get_settings()))
        expected = a["selector_settings"][SELECTORS[method]]
        require(all(expected.get(k) == v for k, v in actual.items()), f"Validated selector settings drift: {method}")
    if not m["smoke"]:
        require("smoke_mcmc" not in m, "Reduced MCMC budget forbidden in production")
        require(mcmc(m)["replicas"] == 511 and mcmc(m)["steps"] == 20, "MCMC budget drift")
    else:
        require("smoke" in m["experiment"] and "smoke" in m["output_root"], "Smoke outputs must be labelled")
    require(len({c["id"] for c in m["null_conditions"]}) == len(m["null_conditions"]), "Duplicate null conditions")
    require(len({c["id"] for c in m["real_conditions"]}) == len(m["real_conditions"]), "Duplicate real conditions")
    for condition in m["null_conditions"]+m["real_conditions"]:
        require(condition["n"] >= 20 and condition["repetitions"] >= 1, "Invalid sample/replication count")
    for condition in m["real_conditions"]:
        if "datasets" in condition:
            names = condition["datasets"]
            require(bool(names) and len(names) == len(set(names)) and
                    set(names).issubset(m["real_datasets"]), "Invalid real-condition dataset restriction")
    if "primary_calibration_conditions" in m:
        names = m["primary_calibration_conditions"]
        require(bool(names) and len(names) == len(set(names)) and set(names).issubset(
            c["id"] for c in m["null_conditions"]), "Primary calibration must identify null conditions")
    return m


def source_hashes(m):
    files = [*sorted((ROOT/"src").rglob("*.py")), Path(__file__),
        ROOT/"benchmarks/prepare_publication_mcmc_data.py", ROOT/m["reference"],
        ROOT/"benchmarks/plot_publication_mcmc.py", ROOT/"tests/test_publication_mcmc_benchmark.py",
        CONFIG, SMOKE_CONFIG, ROOT/"benchmarks/publication_mcmc_requirements.txt",
        ROOT/"examples/publication_five_mcmc20_cpu32.ps1", ROOT/"pyproject.toml"]
    if m["experiment"] == "publication_five_mcmc20_focused_20261010":
        files.extend([FOCUSED_CONFIG, ROOT/"examples/publication_five_mcmc20_focused_cpu32.ps1",
                      ROOT/"tests/test_publication_mcmc_focused.py"])
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def prepared_manifest(m):
    path = ROOT/m["prepared_root"]/"manifest.json"
    data = read(path)
    require(data["jitter"] is False and data["housing_id_included"] is False, "Prepared data policy drift")
    for name in m["real_datasets"]:
        item = data["datasets"][name]
        require(digest(path.parent/item["file"]) == item["sha256"], f"Prepared data corruption: {name}")
        require(digest(ROOT/"benchmarks/dataset"/item["raw_file"]) == item["raw_sha256"], f"Raw source drift: {name}")
        require(item["training_n"] >= max(c["n"] for c in m["real_conditions"]), f"Training pool too small: {name}")
    return data


def provenance(m):
    requirements = tomllib.loads((ROOT/"pyproject.toml").read_text())["project"]["dependencies"]
    versions = {name: md.version(name) for name in [r.split("==")[0] for r in requirements]+["threadpoolctl"]}
    if not m["smoke"]:
        for req in requirements:
            name, version = req.split("==")
            require(versions[name] == version, f"Pinned dependency drift: {name}")
        require(sys.version_info[:2] == (3, 11), "Production requires the project's CPU-server Python 3.11 environment")
    return dict(source_hashes=source_hashes(m), prepared_data=prepared_manifest(m),
                runtime=dict(python=sys.version, packages=versions, platform=platform.platform(),
                             numerical_threads=1, rf_jobs=1))


def lock(m):
    out = ROOT/m["output_root"]
    require(out.resolve().is_relative_to((ROOT/"outputs/benchmarks").resolve())
            or (m["smoke"] and out.resolve().is_relative_to((ROOT/"tmp").resolve())),
            "Output must stay under outputs/benchmarks (or workspace tmp for smoke tests)")
    for name, data in [("requested_manifest.json", m), ("provenance.json", provenance(m))]:
        path = out/name
        if path.exists():
            require(read(path) == data, f"{name} drift; use a new experiment directory")
        else:
            atomic(path, data)
    return out


@contextmanager
def execution_lock(m):
    """Prevent concurrent direct-Python controllers as well as launcher races."""
    out = ROOT/m["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    with (out/"controller.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def conditions(m):
    for i, c in enumerate(m["null_conditions"]):
        yield dict(c, study="null", dataset="gaussian_design", condition_index=i, dataset_index=0)
    for d, name in enumerate(m["real_datasets"]):
        for i, c in enumerate(m["real_conditions"]):
            if name not in c.get("datasets", m["real_datasets"]):
                continue
            yield dict(c, study="real", dataset=name, condition_index=i, dataset_index=d)


def jobs(m):
    for c in conditions(m):
        for trial in range(c["repetitions"]):
            yield c, trial


def identity(m, c, trial):
    values = [m["master_seed"], 0 if c["study"] == "null" else 1, c["dataset_index"], c["condition_index"], trial]
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(values).spawn(4)]
    return dict(study=c["study"], dataset=c["dataset"], condition=c["id"], trial=trial,
                n=c["n"], data_seed=seeds[0], response_seed=seeds[1], target_seed=seeds[2], sampler_seed=seeds[3],
                auxiliary_u=float(np.random.default_rng(seeds[2]).random()))


def dataset(m, name):
    key = (m["prepared_root"], name)
    if key not in _DATA_CACHE:
        meta = read(ROOT/m["prepared_root"]/"manifest.json")["datasets"][name]
        path = ROOT/m["prepared_root"]/meta["file"]
        require(digest(path) == meta["sha256"], f"Prepared data corruption: {name}")
        with np.load(path, allow_pickle=False) as saved:
            _DATA_CACHE[key] = {k: saved[k].copy() for k in saved.files}
    return _DATA_CACHE[key]


def generated(m, c, trial):
    ident = identity(m, c, trial)
    rng = np.random.default_rng(ident["data_seed"])
    noise = np.random.default_rng(ident["response_seed"])
    if c["study"] == "null":
        covariance = c["rho"] ** np.abs(np.arange(c["p"])[:, None]-np.arange(c["p"])[None, :])
        X = rng.normal(size=(c["n"], c["p"])) @ np.linalg.cholesky(covariance).T
        mean = np.zeros(c["n"])
        y = noise.normal(size=c["n"])
        return dict(X=X, y=y, mean=mean, row_indices=np.arange(c["n"]), injected_feature=np.array(-1)), ident
    pool = dataset(m, c["dataset"])
    rows = rng.choice(pool["training_rows"], c["n"], replace=False)
    X = pool["X"][rows].copy()
    # Uniform independently prespecified stimulus, never chosen from y. All
    # predictors receive the same sampling law across independent replications.
    injected = int(rng.integers(X.shape[1])) if c["mode"] == "gaussian_signal" else -1
    mean = np.full(c["n"], np.nan) if c["mode"] == "observed" else np.zeros(c["n"])
    if injected >= 0:
        effect = X[:, injected]+.5*X[:, injected]**2
        effect -= effect.mean()
        rms = np.sqrt(np.mean(effect**2))
        require(rms > 0, "Injected predictor is constant; retain trial as an input failure")
        mean = c["beta"]*effect/rms
    y = pool["y"][rows] if c["mode"] == "observed" else mean+noise.normal(size=c["n"])
    return dict(X=X, y=y, mean=mean, row_indices=rows, injected_feature=np.array(injected)), ident


def trial_dir(m, c, trial):
    return ROOT/m["output_root"]/c["study"]/c["dataset"]/c["id"]/f"trial_{trial:06d}"


def save_input(m, c, trial):
    path = trial_dir(m, c, trial)
    values, ident = generated(m, c, trial)
    request = dict(identity=ident, manifest_sha256=value_hash(m))
    if (path/"input.json").exists():
        saved, old_ident, checksum = load_input(m, c, trial)
        require(old_ident == ident, "Input identity drift")
        for key, value in values.items():
            np.testing.assert_array_equal(saved[key], value)
        return saved, ident, checksum
    path.mkdir(parents=True, exist_ok=True)
    tmp = path/"input.tmp.npz"
    np.savez_compressed(tmp, **values)
    tmp.replace(path/"input.npz")
    checksum = digest(path/"input.npz")
    atomic(path/"input.json", dict(request=request, sha256=checksum))
    return values, ident, checksum


def load_input(m, c, trial):
    path = trial_dir(m, c, trial)
    receipt = read(path/"input.json")
    ident = identity(m, c, trial)
    require(receipt["request"] == dict(identity=ident, manifest_sha256=value_hash(m)), "Input request drift")
    require(digest(path/"input.npz") == receipt["sha256"], f"Corrupt input: {path}")
    with np.load(path/"input.npz", allow_pickle=False) as saved:
        values = {k: saved[k].copy() for k in saved.files}
    return values, ident, receipt["sha256"]


def arm_request(m, ident, method, checksum):
    return dict(manifest_sha256=value_hash(m), identity=ident, method=method, input_sha256=checksum)


def load_arm(path, request):
    saved = read(path)
    require(saved["request"] == request, f"Arm request drift: {path}")
    require(saved["sha256"] == value_hash(saved["payload"]), f"Corrupt result: {path}")
    return saved["payload"]


def inference_args(m, ident, method):
    a = reference(m)["requested_manifest"]
    mc = mcmc(m)
    return dict(k_select=a["design"]["k_select"], sigma=m["sigma"], selector=selector(m, method),
        test_statistic="spline", selection_event=a["selection_event"], target_rule=a["target_rule"],
        auxiliary_u=ident["auxiliary_u"], target_seed=ident["target_seed"], multiplicity=a["multiplicity"],
        selection_decimals=a["selection_decimals"], inference_method="exact_spline" if method == "spline_exact" else "mcmc_rank",
        ais_seed=ident["sampler_seed"], max_final_samples=mc["replicas"], mcmc_steps=mc["steps"],
        mcmc_step_scales=mc["step_scales"], mcmc_global_jump_probability=mc["global_jump_probability"])


def predictive(m, c, values, selected):
    if c["study"] != "real" or c["mode"] != "observed":
        return {}
    pool = dataset(m, c["dataset"])
    evaluation = pool["evaluation_rows"]
    model = RandomForestRegressor(**reference(m)["requested_manifest"]["rf_params"])
    model.fit(values["X"][:, selected], values["y"])
    y_test = pool["y"][evaluation]
    pred = model.predict(pool["X"][evaluation][:, selected])
    baseline = np.full(len(y_test), values["y"].mean())
    return dict(holdout_n=len(evaluation), rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        mae=float(mean_absolute_error(y_test, pred)), r2=float(r2_score(y_test, pred)),
        baseline_rmse=float(np.sqrt(mean_squared_error(y_test, baseline))))


def execute(m, c, trial, method, values, ident, checksum):
    path = trial_dir(m, c, trial)/f"{method}.json"
    request = arm_request(m, ident, method, checksum)
    if path.exists():
        return load_arm(path, request)
    started, cpu = time.perf_counter(), time.process_time()
    base = dict(**ident, method=method, input_sha256=checksum,
        primary=c["primary"], mode="gaussian_null" if c["study"] == "null" else c["mode"],
        beta=c.get("beta"), inference_failed=True, error_type=None, error_message=None,
        p_value=None, unadjusted_p_value=None, prediction_failed=False)
    settings = None
    # Preserve the observed selection/truth even if inference fails.
    try:
        observed = selector(m, method).select(values["X"], values["y"], 2)
        selected = sorted(map(int, observed.selected_features))
        target = target_from_selected_set(selected, ident["auxiliary_u"])
        basis = _spline_effect_basis(values["X"][:, target])
        known = np.isfinite(values["mean"]).all()
        projection = float(np.linalg.norm(basis.T@values["mean"])) if known else None
        base.update(selected=selected, target=target, rank=basis.shape[1],
            radial_t_obs=float(np.linalg.norm(basis.T@values["y"])/m["sigma"]),
            unadjusted_p_value=float(stats.chi.sf(np.linalg.norm(basis.T@values["y"])/m["sigma"], df=basis.shape[1])),
            projection_mean_norm=projection, truth_known=bool(known),
            validity_scope="known_gaussian_response" if known else "exploratory_working_scale_unknown_noise",
            projection_null=bool(projection <= 1e-8*max(1., np.linalg.norm(values["mean"]))) if known else None,
            injected_feature=int(values["injected_feature"]),
            targets_injected=bool(target == int(values["injected_feature"])))
        inference_start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Exact-set conditioning can be rare")
            result = selective_inference(values["X"], values["y"], **inference_args(m, ident, method))
        row = result["feature_results"].iloc[0].to_dict()
        require(sorted(map(int, result["observed_selected_features"])) == selected, "Inconsistent observed selection")
        require(int(result["observed_target_feature"]) == target, "Inconsistent randomized target")
        # These legacy diagnostics are binomial summaries conditional on the hub,
        # not uncertainty intervals for the stationary selective tail probability.
        row["hub_tail_se"] = row.pop("mc_se", None) if method != "spline_exact" else None
        row["hub_tail_ci_lower"] = row.pop("mc_ci_95_lower", None) if method != "spline_exact" else None
        row["hub_tail_ci_upper"] = row.pop("mc_ci_95_upper", None) if method != "spline_exact" else None
        for key in ("denominator_ess", "tail_ess", "tail_probability_mc_ci_95_lower", "tail_probability_mc_ci_95_upper", "mc_se", "mc_ci_95_lower", "mc_ci_95_upper"):
            row.pop(key, None)
        p = row["p_value"]
        require(np.isfinite(p) and 0 <= p <= 1, "Nonfinite/invalid selective p-value")
        base.update(clean(row), inference_failed=False, inference_seconds=time.perf_counter()-inference_start)
        settings = clean(result["settings"])
    except Exception as error:
        base.update(error_type=type(error).__name__, error_message=str(error), inference_failed=True, p_value=None)
    if "selected" in base:
        try:
            base.update(predictive(m, c, values, base["selected"]))
        except Exception as error:
            base.update(prediction_failed=True, prediction_error=str(error))
    base.update(elapsed_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
    payload = dict(result=clean(base), settings=settings)
    atomic(path, dict(request=request, payload=payload, sha256=value_hash(payload)))
    return payload


def worker_init(workers):
    # Bound native threads even if a numerical library ignores environment flags.
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))[:32]
        os.sched_setaffinity(0, allowed)
    require(all(x["num_threads"] <= 1 for x in threadpool_info()), "Native thread cap failed")


def execute_trial(m, c, trial):
    try:
        values, ident, checksum = save_input(m, c, trial)
    except Exception as error:
        # Infrastructure/input failures are not fabricated statistical results.
        atomic(trial_dir(m, c, trial)/"input_failure.json", dict(type=type(error).__name__, message=str(error)))
        raise
    for method in METHODS:
        execute(m, c, trial, method, values, ident, checksum)
    return c["study"], c["dataset"], c["id"], trial


def run(m):
    out = lock(m)
    # One persistent pool; at most two waves queued, no 350,000-future list.
    iterator = iter(jobs(m))
    failures = []
    done_count = 0
    with ProcessPoolExecutor(max_workers=m["workers"], initializer=worker_init, initargs=(m["workers"],)) as pool:
        pending = {}
        def submit_next():
            try:
                c, t = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(execute_trial, m, c, t)] = (c, t)
            return True
        for _ in range(2*m["workers"]):
            submit_next()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                c, t = pending.pop(future)
                try:
                    result = future.result()
                    done_count += 1
                    print("Completed paired trial", result, flush=True)
                except Exception as error:
                    failures.append(dict(study=c["study"], dataset=c["dataset"], condition=c["id"], trial=t, error=str(error)))
                submit_next()
            atomic(out/"progress.json", dict(completed_paired_trials=done_count, infrastructure_failures=len(failures)))
    atomic(out/"infrastructure_failures.json", failures)
    require(not failures, "Infrastructure/input failures remain; inspect and Resume")
    verify(m)


def interval(k, n, confidence=.95):
    if not n:
        return None, None
    q = (1-confidence)/2
    return (0. if k == 0 else float(stats.beta.ppf(q, k, n-k+1)),
            1. if k == n else float(stats.beta.ppf(1-q, k+1, n-k)))


def bootstrap_mean(values, m, key):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    seed = int.from_bytes(hashlib.sha256(encoded([m["bootstrap_seed"], key])).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    means = np.empty(m["bootstrap_replicates"])
    # Keep temporary resampling memory independent of publication sample size.
    for i in range(len(means)):
        means[i] = values[rng.integers(len(values), size=len(values))].mean()
    return tuple(map(float, np.quantile(means, [.025, .975])))


def summaries(frame, m):
    keys = ["study", "dataset", "condition", "method"]
    rates, diagnostics, predictions, stability, calibration = [], [], [], [], []
    def primary_calibration(c):
        if "primary_calibration_conditions" in m:
            return c["study"] == "null" and c.get("id", c.get("condition")) in m["primary_calibration_conditions"]
        return c["study"] == "null" or c.get("mode") == "gaussian_null"
    calibration_cells = sum(primary_calibration(c) for c in conditions(m))*len(METHODS)
    for identity_values, g in frame.groupby(keys, sort=True):
        ident = dict(zip(keys, identity_values))
        p = pd.to_numeric(g.p_value, errors="coerce")
        valid = np.isfinite(p) & p.between(0, 1)
        n, v = len(g), int(valid.sum())
        is_null = g["mode"].iloc[0] == "gaussian_null"
        in_primary_family = is_null and primary_calibration(dict(ident, mode=g["mode"].iloc[0]))
        if is_null:
            grid = sorted(set(np.linspace(0, 1, 513).tolist()+m["alpha_levels"]))
            for threshold in grid:
                # CDF convention <= is distinct from the prespecified rejection
                # convention <; record both instead of hiding rank-grid effects.
                k = int((valid & p.le(threshold)).sum())
                lo, hi = interval(k, v)
                calibration.append(dict(**ident, threshold=threshold, attempted=n, valid=v,
                    cdf_valid=k/v if v else None, cdf_pointwise_ci95_lower=lo, cdf_pointwise_ci95_upper=hi,
                    cdf_missing_lower=k/n, cdf_missing_upper=(k+n-v)/n,
                    theoretical_superuniform_bound=threshold))
        for alpha in m["alpha_levels"]:
            k = int((valid & p.lt(alpha)).sum())
            lo, hi = interval(k, v)
            # Missing outcomes widen the full-attempt interval in both directions.
            bounded_lo = interval(k, n)[0]
            bounded_hi = interval(k+n-v, n)[1]
            upper = float(stats.beta.ppf(1-.05/calibration_cells, k+n-v+1, n-k-n+v)) if k+n-v < n else 1.
            rates.append(dict(**ident, alpha=alpha, attempted=n, valid=v, failures=n-v, rejected=k,
                valid_only_fraction=k/v if v else None, valid_ci95_lower=lo, valid_ci95_upper=hi,
                all_attempt_lower=k/n, all_attempt_upper=(k+n-v)/n,
                missing_ci95_lower=bounded_lo, missing_ci95_upper=bounded_hi,
                interpretation="type_I_error" if is_null else "working_model_rejection" if g["mode"].iloc[0] == "observed" else "marginal_detection",
                binomial_inflation_p=float(stats.binomtest(k, n, alpha, alternative="greater").pvalue) if is_null and v == n else None,
                primary_family_simultaneous_upper=upper if in_primary_family and alpha == m["primary_alpha"] else None,
                calibration_tolerance_supported=bool(upper <= alpha+m["calibration_tolerance"]) if in_primary_family and alpha == m["primary_alpha"] else None,
                rank_no_tie_grid_bound=(np.ceil(alpha*(mcmc(m)["replicas"]+1))-1)/(mcmc(m)["replicas"]+1) if ident["method"] != "spline_exact" else alpha))
        # Power conditional on the prespecified injected feature being targeted;
        # no redraw-until-selected procedure and no structural-null shortcuts.
        if g["mode"].iloc[0] == "gaussian_signal":
            target = g.targets_injected.fillna(False).astype(bool)
            nt = int(target.sum())
            kt = int((target & valid & p.lt(m["primary_alpha"])).sum())
            mt = int((target & ~valid).sum())
            lo, hi = interval(kt, nt)
            diagnostics.append(dict(**ident, metric="injected_target_detection", targeted=nt,
                targeting_probability=nt/n, detections=kt, unresolved=mt,
                conditional_lower=kt/nt if nt else None, conditional_upper=(kt+mt)/nt if nt else None,
                conditional_missing_ci95_lower=lo, conditional_missing_ci95_upper=interval(kt+mt, nt)[1],
                marginal_lower=kt/n, marginal_upper=(kt+mt)/n))
        for metric in ("elapsed_seconds", "cpu_seconds", "inference_seconds", "proposals",
                       "mcmc_replicate_acceptance_rate", "mcmc_hub_acceptance_rate", "mcmc_unique_replicas", "mcmc_moved_replicas"):
            x = pd.to_numeric(g.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            diagnostics.append(dict(**ident, metric=metric, count=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(x.median()) if len(x) else None, p95=float(x.quantile(.95)) if len(x) else None,
                total=float(x.sum()) if len(x) else None))
        if g["mode"].iloc[0] == "observed":
            for metric in ("rmse", "mae", "r2", "baseline_rmse"):
                x = pd.to_numeric(g.get(metric, pd.Series(index=g.index, dtype=float)), errors="coerce")
                lo, hi = bootstrap_mean(x, m, [*identity_values, metric])
                predictions.append(dict(**ident, metric=metric, attempted=n, valid=int(x.notna().sum()),
                    mean=float(x.mean()), conditional_mc_ci95_lower=lo, conditional_mc_ci95_upper=hi))
            selected = [set(x) for x in g.selected if isinstance(x, list)]
            for feature in sorted(set().union(*selected)) if selected else []:
                k = sum(feature in x for x in selected)
                lo, hi = interval(k, len(selected))
                stability.append(dict(**ident, feature=feature, count=k, valid_selections=len(selected),
                    frequency=k/len(selected), conditional_mc_ci95_lower=lo, conditional_mc_ci95_upper=hi))
    return {"rates.csv": pd.DataFrame(rates), "diagnostics.csv": pd.DataFrame(diagnostics),
            "prediction.csv": pd.DataFrame(predictions), "selection_frequency.csv": pd.DataFrame(stability),
            "calibration.csv": pd.DataFrame(calibration)}


def paired_summaries(frame, m):
    records = []
    for ids, group in frame.groupby(["study", "dataset", "condition"], sort=True):
        arms = {method: g.set_index("trial") for method, g in group.groupby("method")}
        for left, right in itertools.combinations(METHODS, 2):
            a, b = arms[left], arms[right]
            require(a.index.equals(b.index), "Unpaired trial indexes")
            pa, pb = pd.to_numeric(a.p_value, errors="coerce"), pd.to_numeric(b.p_value, errors="coerce")
            valid = pa.notna() & pb.notna()
            n = len(a)
            for alpha in m["alpha_levels"]:
                delta = (pa[valid].lt(alpha).astype(int)-pb[valid].lt(alpha).astype(int)).to_numpy()
                lo, hi = bootstrap_mean(delta, m, [*ids, left, right, alpha])
                d = float(delta.sum())
                uncertain = n-int(valid.sum())
                rec = dict(study=ids[0], dataset=ids[1], condition=ids[2], left=left, right=right,
                    metric="rejection_difference", alpha=alpha, attempted=n, complete_pairs=int(valid.sum()),
                    complete_pair_difference=float(delta.mean()) if len(delta) else None,
                    conditional_mc_ci95_lower=lo, conditional_mc_ci95_upper=hi,
                    all_attempt_lower=(d-uncertain)/n, all_attempt_upper=(d+uncertain)/n)
                plus, minus = int((delta == 1).sum()), int((delta == -1).sum())
                rec["discordant_left_only"] = plus
                rec["discordant_right_only"] = minus
                rec["mcnemar_exact_p"] = float(stats.binomtest(plus, plus+minus, .5).pvalue) if plus+minus else 1.
                # Difference = P(left-only) - P(right-only). Bonferroni CP
                # bounds on the two binomial marginals remain conservative
                # with sparse/zero discordance, unlike a degenerate bootstrap.
                complete_n = int(valid.sum())
                rec["paired_complete_ci95_lower"] = interval(plus, complete_n, .975)[0]-interval(minus, complete_n, .975)[1] if complete_n else None
                rec["paired_complete_ci95_upper"] = interval(plus, complete_n, .975)[1]-interval(minus, complete_n, .975)[0] if complete_n else None
                rec["paired_missing_ci95_lower"] = interval(plus, n, .975)[0]-interval(minus+uncertain, n, .975)[1]
                rec["paired_missing_ci95_upper"] = interval(plus+uncertain, n, .975)[1]-interval(minus, n, .975)[0]
                if left == "spline_mcmc" and right == "spline_exact":
                    errors = pa[valid]-pb[valid]
                    rec.update(p_difference_mean=float(errors.mean()), p_abs_difference_median=float(errors.abs().median()),
                               p_abs_difference_p95=float(errors.abs().quantile(.95)))
                records.append(rec)
            for metric in ("rmse", "mae", "r2", "elapsed_seconds"):
                if metric not in a or metric not in b:
                    continue
                delta = pd.to_numeric(a[metric], errors="coerce")-pd.to_numeric(b[metric], errors="coerce")
                if not delta.notna().any():
                    continue
                lo, hi = bootstrap_mean(delta, m, [*ids, left, right, metric])
                records.append(dict(study=ids[0], dataset=ids[1], condition=ids[2], left=left, right=right,
                    metric=metric+"_difference", attempted=n, complete_pairs=int(delta.notna().sum()),
                    complete_pair_difference=float(delta.mean()), conditional_mc_ci95_lower=lo, conditional_mc_ci95_upper=hi))
        # Naive same-target inference is a diagnostic, not a sixth selection arm.
        for method, g in arms.items():
            p = pd.to_numeric(g.unadjusted_p_value, errors="coerce")
            for alpha in m["alpha_levels"]:
                k, n = int(p.lt(alpha).sum()), len(g)
                records.append(dict(study=ids[0], dataset=ids[1], condition=ids[2], left=method, right="naive_same_target",
                    metric="naive_rejection_fraction", alpha=alpha, attempted=n, complete_pairs=int(p.notna().sum()),
                    complete_pair_difference=k/n))
    return pd.DataFrame(records)


def audit_arm(m, c, values, ident, checksum, method, payload, replay=False):
    row, settings = payload["result"], payload["settings"]
    for key, value in ident.items():
        require(row[key] == value, f"Identity mismatch: {key}")
    require(row["method"] == method and row["input_sha256"] == checksum, "Unmatched result")
    if "selected" in row:
        selected = row["selected"]
        require(selected == sorted(set(selected)) and len(selected) == 2 and all(0 <= j < values["X"].shape[1] for j in selected), "Invalid selected set")
        require(row["target"] == target_from_selected_set(selected, ident["auxiliary_u"]), "Wrong randomized target")
        basis = _spline_effect_basis(values["X"][:, row["target"]])
        require(row["rank"] == basis.shape[1], "Test basis rank drift")
        require(np.isclose(row["radial_t_obs"], np.linalg.norm(basis.T@values["y"])/m["sigma"], rtol=1e-10, atol=1e-10), "Wrong statistic")
        known = np.isfinite(values["mean"]).all()
        require(row["truth_known"] == bool(known), "Wrong truth flag")
        if known:
            projection = np.linalg.norm(basis.T@values["mean"])
            require(np.isclose(row["projection_mean_norm"], projection, atol=1e-10), "Wrong fixed-design truth")
            require(row["projection_null"] == bool(projection <= 1e-8*max(1., np.linalg.norm(values["mean"]))), "Wrong projection-null flag")
        if replay:
            observed = selector(m, method).select(values["X"], values["y"], 2)
            require(selected == sorted(map(int, observed.selected_features)), "Observed selector replay mismatch")
    if row["inference_failed"]:
        require(row["p_value"] is None and row["error_type"] is not None, "Failure not explicitly retained")
        return
    require(settings is not None, "Missing successful inference settings")
    require(row["feature"] == row["target"] and row["test_rank"] == row["rank"], "Wrong tested feature/rank")
    expected = dict(n_samples=c["n"], n_features=values["X"].shape[1], k_select=2,
        sigma=m["sigma"], selection_event="exact_set", target_rule="uniform_from_selected", multiplicity="none",
        inference_method="exact_spline" if method == "spline_exact" else "mcmc_rank", test_statistic="spline",
        target_seed=ident["target_seed"], ais_seed=ident["sampler_seed"], auxiliary_u=ident["auxiliary_u"],
        max_final_samples=mcmc(m)["replicas"], mcmc_steps=mcmc(m)["steps"],
        mcmc_step_scales=mcmc(m)["step_scales"], mcmc_global_jump_probability=mcmc(m)["global_jump_probability"],
        stop_when_ess_met=False, ais_strategy="baseline", observed_target_feature=row["target"])
    expected.update(clean(dict(selector(m, method).get_settings())))
    require(all(settings.get(k) == v for k, v in expected.items()), "Inference/selector settings drift")
    require(row["finite_sample_valid"] and 0 <= row["p_value"] <= 1, "Invalid p-value")
    require("denominator_ess" not in row and "tail_ess" not in row and "mc_ci_95_lower" not in row, "Mislabelled sampler diagnostics")
    if method == "spline_exact":
        require(row["proposals"] == 0 and row["p_value_method"] == "exact_spline_selection_region", "Wrong exact engine")
    else:
        mc = mcmc(m)
        require(row["p_value"] == (row["tail_samples"]+1)/(mc["replicas"]+1), "Wrong rank formula")
        require(row["proposals"] == (mc["replicas"]+1)*mc["steps"] and row["mcmc_replicates"] == mc["replicas"] and row["mcmc_steps"] == mc["steps"], "Wrong transition budget")
        require(row["p_value_method"] == "parallel_reversible_mcmc_rank", "Wrong MCMC construction")


def collect(m, replay=False):
    rows = []
    for c, trial in jobs(m):
        values, ident, checksum = load_input(m, c, trial)
        regenerated, _ = generated(m, c, trial)
        for key, value in values.items():
            np.testing.assert_allclose(value, regenerated[key], rtol=1e-12, atol=1e-12, equal_nan=True)
        pair = {}
        for method in METHODS:
            payload = load_arm(trial_dir(m, c, trial)/f"{method}.json", arm_request(m, ident, method, checksum))
            # Replay all selectors on a prespecified 1% audit sample; every
            # spline pair is still checked for observed selection and statistic.
            audit_arm(m, c, values, ident, checksum, method, payload, replay=replay or trial % 100 == 0)
            rows.append(payload["result"])
            pair[method] = payload["result"]
        a, b = pair["spline_mcmc"], pair["spline_exact"]
        if "selected" in a and "selected" in b:
            require(a["selected"] == b["selected"] and a["target"] == b["target"] and a["rank"] == b["rank"], "Spline comparison is unpaired")
            require(np.isclose(a["radial_t_obs"], b["radial_t_obs"], atol=1e-12), "Spline statistic mismatch")
    frame = pd.DataFrame(rows)
    for column in ("selected", "targets_injected", "rmse", "mae", "r2", "baseline_rmse"):
        if column not in frame:
            frame[column] = None
    require(not frame.duplicated(["study", "dataset", "condition", "trial", "method"]).any(), "Duplicate result keys")
    return frame


def verify(m, downloaded=False, replay=False):
    out = ROOT/m["output_root"] if downloaded else lock(m)
    require(read(out/"requested_manifest.json") == m, "Saved configuration drift")
    # Download verification compares data and recorded hashes; host platform
    # and Python minor version are permitted to differ, never overwritten.
    if downloaded:
        saved = read(out/"provenance.json")
        require(saved["source_hashes"] == source_hashes(m), "Downloaded source differs from executed source")
        require(saved["prepared_data"] == prepared_manifest(m), "Downloaded prepared data differ")
    frame = collect(m, replay)
    outputs = summaries(frame, m)
    outputs["paired_comparisons.csv"] = paired_summaries(frame, m)
    export = frame.copy()
    export["selected"] = export["selected"].map(lambda x: json.dumps(x) if isinstance(x, list) else "")
    outputs["results.csv"] = export
    for filename, data in outputs.items():
        path = out/filename
        if downloaded:
            from io import StringIO
            old, new = pd.read_csv(path), pd.read_csv(StringIO(data.to_csv(index=False)))
            pd.testing.assert_frame_equal(old, new, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
        else:
            tmp = path.with_suffix(".csv.tmp")
            data.to_csv(tmp, index=False)
            tmp.replace(path)
    report = dict(tests=len(frame), expected_tests=sum(c["repetitions"] for c in conditions(m))*5,
        inference_failures=int(frame.inference_failed.sum()), missing_trials=0, duplicates=0,
        spline_pairing_passed=True, statistical_control_asserted=False, smoke=m["smoke"],
        output_sha256={filename: digest(out/filename) for filename in outputs})
    require(report["tests"] == report["expected_tests"], "Incomplete result set")
    if downloaded:
        require(read(out/"verification.json") == report, "Downloaded completion report mismatch")
        require(read(out/"COMPLETED") == report, "Downloaded completion marker mismatch")
        atomic(out/"local_verification.json", dict(status="passed", original_provenance_preserved=True, tests=len(frame)))
    else:
        atomic(out/"verification.json", report)
        atomic(out/"COMPLETED", report)
    print(json.dumps(report, indent=2))
    return report


def status(m):
    out = ROOT/m["output_root"]
    for study in ("null", "real"):
        expected = sum(c["repetitions"] for c in conditions(m) if c["study"] == study)*5
        count = sum(1 for p in (out/study).glob("*/*/trial_*/*.json") if p.stem in METHODS)
        print(f"{study}: {count}/{expected} committed arm receipts")
    if (out/"progress.json").exists():
        print(json.dumps(read(out/"progress.json")))
    if (out/"COMPLETED").exists():
        print("Computational completion verified; inspect failures/calibration before scientific conclusions.")


def check(m):
    data = prepared_manifest(m)
    counts = {study: sum(c["repetitions"] for c in conditions(m) if c["study"] == study)*5 for study in ("null", "real")}
    transition = (mcmc(m)["replicas"]+1)*mcmc(m)["steps"]
    print(json.dumps(dict(experiment=m["experiment"], tests=counts, total_tests=sum(counts.values()),
        mcmc_tests=sum(counts.values())*4//5, transitions_per_mcmc_test=transition,
        total_transitions=sum(counts.values())*4//5*transition, workers=m["workers"],
        estimated_data_n=sum(x["n"] for x in data["datasets"].values()), smoke=m["smoke"]), indent=2))


def validate_archive(path, m):
    seen = set()
    with tarfile.open(path) as archive:
        for entry in archive.getmembers():
            p = PurePosixPath(entry.name)
            require(p.parts and p.parts[0] == m["experiment"] and not p.is_absolute()
                and ".." not in p.parts and "\\" not in entry.name and ":" not in entry.name
                and (entry.isfile() or entry.isdir()) and entry.name not in seen, f"Unsafe archive entry: {entry.name}")
            seen.add(entry.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "run", "status", "verify", "verify-downloaded", "validate-archive"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--replay-all-selectors", action="store_true")
    args = parser.parse_args()
    m = read_config(args.config)
    if args.action == "run":
        with execution_lock(m):
            run(m)
    elif args.action == "check":
        check(m)
    elif args.action == "status":
        status(m)
    elif args.action.startswith("verify"):
        with execution_lock(m):
            verify(m, downloaded=args.action == "verify-downloaded", replay=args.replay_all_selectors)
    else:
        validate_archive(args.archive, m)


if __name__ == "__main__":
    main()
