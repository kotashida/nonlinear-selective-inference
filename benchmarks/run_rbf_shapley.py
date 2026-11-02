"""Paired selected-set and full-candidate additive RBF Shapley experiments.

Adapted from run_shapley_targets.py; frozen separately to preserve prior runs.
Known sigma, X-only hidden layer, unpenalized output fits, exact coalitions.
Reversible hub MCMC p-values and analytic truncated-chi reference p-values.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from functools import lru_cache
import hashlib
import importlib.metadata as metadata
from itertools import combinations, product
import json
from math import comb
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
import time

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_NUM_THREADS", "NUMBA_NUM_THREADS",
               "POLARS_MAX_THREADS", "BLIS_NUM_THREADS")
for key in THREAD_KEYS:
    os.environ[key] = "1"

import numpy as np
from rbf_network import AdditiveRBFNetwork
from scipy import stats
from scipy.special import logsumexp
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "benchmarks/rbf_shapley_mcmc20_cpu32_250_20261022.json"
TARGETS = ("marginal", "model", "selected_shapley", "full_shapley")
# Filled with canonical manifest hashes, including a separately labelled smoke run.
FROZEN = {'rbf_shapley_mcmc20_cpu32_250_20261022': 'f261c2ddf5da48262b2a3aae376d5b9a7f18c7e123f6d9eb349363da068b160b', 'rbf_shapley_mcmc20_cpu32_smoke_20261022': '85c03f0501a8946c251e6c67f778ca3470ca5b4c124afa6e14dbbcc83e042664'}
PACKAGED = ('benchmarks/run_rbf_shapley.py', 'benchmarks/rbf_network.py', 'benchmarks/rbf_shapley_mcmc20_cpu32_250_20261022.json', 'benchmarks/rbf_shapley_mcmc20_cpu32_smoke_20261022.json', 'examples/rbf_shapley_cpu32.ps1', 'docs/rbf_shapley_study.md', 'tests/test_rbf_shapley.py', 'docs/commands.md')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(obj):
    return hashlib.sha256(encoded(obj)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_config(path=DEFAULT):
    m = read(path)
    require(sha(m) in FROZEN.values(), "Manifest changed: create a newly named frozen experiment")
    require(len(m["targets"]) == len(set(m["targets"])) and set(m["targets"]) <= set(TARGETS), "Invalid targets")
    require(1 <= m["workers"] <= 32 and m["threads_per_worker"] == 1, "32-core budget exceeded")
    require(m["p"] == 4 and m["k"] == 2 and m["n"] == 100, "Geometry drift")
    require(m["sigma"] == 1 and m["rbf_units"] == 3, "Inference convention drift")
    require(m["bases"] == ["additive_rbf"] and m["rbf_units"] == len(m["center_quantiles"]), "Basis drift")
    require(m["targets"] == ["selected_shapley", "full_shapley"], "Target drift")
    require(("smoke" in m["experiment"]) == m["smoke"], "Smoke labelling mismatch")
    require(m["output_root"] == "outputs/benchmarks/" + m["experiment"], "Unsafe output root")
    return m


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PACKAGED}


def provenance(m):
    for package, expected in m["required_versions"].items():
        require(metadata.version(package) == expected, f"Dependency mismatch: {package}")
    return dict(manifest_sha256=sha(m), sources=source_hashes(),
                versions={name: metadata.version(name) for name in (*m["required_versions"], "threadpoolctl")},
                python=sys.version, platform=platform.platform())


def seed(m, stream, design_index, iteration=0):
    return int(np.random.SeedSequence([m["seed"], stream, design_index, iteration]).generate_state(1)[0])


def orthobasis(a, tol=1e-10):
    if a.shape[1] == 0:
        return np.empty((a.shape[0], 0))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, s > tol * max(1.0, float(s[0]))]


@lru_cache(maxsize=12)
def design(manifest_json, design_index):
    m = json.loads(manifest_json)
    name = m["designs"][design_index]
    if name == "orthogonal_factorial":
        X = np.array(list(product((-1.5, -.5, .5, 1.5), repeat=m["p"])))
    else:
        rng = np.random.default_rng(seed(m, 1, design_index))
        Z = rng.normal(size=(m["n"], m["p"]))
        if name == "correlated_gaussian":
            cov = m["correlation"] ** np.abs(np.subtract.outer(np.arange(m["p"]), np.arange(m["p"])))
            X = Z @ np.linalg.cholesky(cov).T
        else:
            X = Z
    network = AdditiveRBFNetwork(m["center_quantiles"], m["width_multiplier"]).fit_basis(X)
    return X, network.transform_blocks(X)


class Geometry:
    def __init__(self, blocks, tol=1e-10):
        self.tol, self.p = tol, len(blocks)
        self.Q = orthobasis(np.column_stack(blocks), tol)
        self.blocks = [self.Q.T @ b for b in blocks]
        self.r = self.Q.shape[1]
        self.projectors = {}
        for mask in range(1 << self.p):
            cols = [self.blocks[j] for j in range(self.p) if mask & (1 << j)]
            b = orthobasis(np.column_stack(cols), tol) if cols else np.empty((self.r, 0))
            self.projectors[mask] = b @ b.T
        self.increments = {}
        self.directions = {}
        for j in range(self.p):
            for mask in range(1 << self.p):
                if mask & (1 << j):
                    continue
                directions = orthobasis((np.eye(self.r) - self.projectors[mask]) @ self.blocks[j], tol)
                H = directions @ directions.T
                np.testing.assert_allclose(H, self.projectors[mask | (1 << j)] - self.projectors[mask], atol=1e-8)
                self.increments[j, mask] = H
                self.directions[j, mask] = directions
        self.selector = np.array([self.target("full_shapley", (1 << self.p) - 1, j)[0]
                                  for j in range(self.p)])
        np.testing.assert_allclose(self.selector.sum(axis=0), np.eye(self.r), atol=1e-8)

    @lru_cache(maxsize=None)
    def target(self, arm, M, j, uniform=False):
        if arm == "marginal":
            masks = [0]
        elif arm == "model":
            masks = [M & ~(1 << j)]
        else:
            U = M if arm == "selected_shapley" else (1 << self.p) - 1
            masks = [s for s in range(1 << self.p) if s & ~U == 0 and not s & (1 << j)]
        size = M.bit_count() if arm == "selected_shapley" else self.p
        if arm in ("marginal", "model"):
            weights = [1.0]
        elif uniform:
            weights = [1 / len(masks)] * len(masks)
        else:
            weights = [1 / (size * comb(size - 1, s.bit_count())) for s in masks]
        A = sum(w * self.increments[j, s] for w, s in zip(weights, masks))
        B = orthobasis(np.column_stack([self.directions[j, s] for s in masks]), self.tol)
        np.testing.assert_allclose(B @ B.T @ A, A, atol=1e-8)
        return A, B

    def select(self, y, k=2):
        scores = np.einsum("i,jik,k->j", y, self.selector, y)
        ranking = np.lexsort((np.arange(self.p), -scores))
        return tuple(sorted(int(j) for j in ranking[:k]))


@lru_cache(maxsize=12)
def geometry(manifest_json, design_index, basis):
    m = json.loads(manifest_json)
    require(basis == "additive_rbf", "Unsupported basis")
    _, blocks = design(manifest_json, design_index)
    return Geometry(blocks, m["rank_tolerance"])


def mean_response(m, di, condition):
    X, _ = design(encoded(m).decode(), di)
    if condition["mean"] == "zero":
        return np.zeros(m["n"])
    require(condition["mean"] == "quadratic_x0", "Unknown mean")
    f = X[:, 0]**2
    f -= f.mean()
    return condition["rms"] * f / np.sqrt(np.mean(f**2))


def positive_roots(a, b, c):
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0:
        return []
    a, b, c = a / scale, b / scale, c / scale
    if a == 0:
        return [-c / b] if b != 0 and -c / b > 0 else []
    disc = b*b - 4*a*c
    if disc <= 0:
        return []
    q = -.5 * (b + np.copysign(np.sqrt(disc), b))
    roots = [q / a]
    if q != 0:
        roots.append(c / q)
    return [float(r) for r in roots if r > 0 and np.isfinite(r)]


def selection_intervals(g, z, u, selected, sigma=1.0):
    """All open positive-radius components preserving the selected top-k set."""
    boundaries = [0.0]
    inequalities = []
    for j in selected:
        for ell in range(g.p):
            if ell in selected:
                continue
            D = g.selector[j] - g.selector[ell]
            a, b, c = (float(sigma**2 * u @ D @ u),
                       float(2*sigma * u @ D @ z), float(z @ D @ z))
            # Orthogonal blocks give exact zeros that may be represented as
            # ~1e-16 after SVD. Clean coefficients at their rounding-error scale,
            # before root solving. Scaling by c alone can invent huge roots.
            eps = 64*np.finfo(float).eps*max(1., float(np.linalg.norm(D, 2)))
            znorm = float(np.linalg.norm(z))
            if abs(a) <= eps*sigma**2:
                a = 0.
            if abs(b) <= eps*2*sigma*max(1., znorm):
                b = 0.
            if abs(c) <= eps*max(1., znorm**2):
                c = 0.
            inequalities.append((a, b, c, j < ell))
            boundaries.extend(positive_roots(a, b, c))
    bounds = sorted(set(boundaries)) + [np.inf]
    intervals = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mid = lo + max(1.0, lo) if np.isinf(hi) else lo + (hi-lo)/2
        # Evaluate score differences, avoiding cancellation between two large
        # individual quadratic scores at a distant interval midpoint.
        values = [((a*mid+b)*mid+c, tie) for a, b, c, tie in inequalities]
        if all(value > 0 or (value == 0 and tie) for value, tie in values):
            if intervals and intervals[-1][1] == lo:
                intervals[-1][1] = hi
            else:
                intervals.append([lo, hi])
    return intervals


def log_difference(a, b):
    if b == -np.inf:
        return a
    if b >= a:
        return -np.inf
    return float(a + np.log(-np.expm1(b-a)))


def log_mass(lo, hi, rank):
    if lo >= hi:
        return -np.inf
    # Choose the numerically accurate tail, then fall back to the other one.
    sf = log_difference(float(stats.chi.logsf(lo, rank)), float(stats.chi.logsf(hi, rank)))
    cdf = log_difference(float(stats.chi.logcdf(hi, rank)), float(stats.chi.logcdf(lo, rank)))
    return sf if stats.chi.cdf(lo, rank) >= .5 else cdf if np.isfinite(cdf) else sf


def radial_test(g, y, B, selected, sigma=1.0):
    rank = B.shape[1]
    if rank == 0:
        return dict(p_selective=1.0, p_naive=1.0, radius=0.0, log_selection_mass=0.0, intervals=1)
    projected = B @ (B.T @ y)
    radius = float(np.linalg.norm(projected) / sigma)
    require(radius > 0, "Zero observed radius in nonzero-rank space")
    u, z = projected / (sigma*radius), y - projected
    intervals = selection_intervals(g, z, u, selected, sigma)
    require(any(lo-1e-7 <= radius <= hi+1e-7 for lo, hi in intervals), "Observed radius lost from selection region")
    logden = float(logsumexp([log_mass(lo, hi, rank) for lo, hi in intervals]))
    lognum = float(logsumexp([log_mass(max(lo, radius), hi, rank) for lo, hi in intervals]))
    require(np.isfinite(logden), "Selection probability underflow; do not silently drop trial")
    p = float(np.exp(min(0.0, lognum-logden)))
    require(np.isfinite(p), "Nonfinite p-value")
    return dict(p_selective=p, p_naive=float(stats.chi.sf(radius, rank)), radius=radius,
                log_selection_mass=logden, intervals=len(intervals))


def mcmc_rank_test(g, y, B, selected, m, rng):
    """Reversible hub construction: independent branches reset to one hub.

    Updates are vectorized across independent branches. Every acceptance check
    reevaluates the same fixed quadratic selector, including index tie breaks.
    """
    rank = B.shape[1]
    projected = B @ (B.T @ y)
    radius = float(np.linalg.norm(projected) / m["sigma"])
    require(rank > 0 and radius > 0, "Degenerate radial test")
    u, z = projected/(m["sigma"]*radius), y-projected

    def inside(r):
        candidates = z[None, :] + m["sigma"]*r[:, None]*u[None, :]
        scores = np.einsum("bi,jik,bk->bj", candidates, g.selector, candidates)
        order = np.argsort(-scores, axis=1, kind="stable")[:, :m["k"]]
        return np.all(np.sort(order, axis=1) == np.asarray(selected), axis=1)

    def advance(initial):
        r = np.asarray(initial, dtype=float).copy()
        accepted = 0
        for _ in range(m["mcmc_steps"]):
            local = rng.uniform(size=len(r)) < m["local_probability"]
            scales = rng.choice(m["local_scales"], size=len(r))
            proposal = np.where(local, np.abs(r+rng.normal(size=len(r))*scales),
                                np.sqrt(rng.chisquare(rank, size=len(r))))
            logratio = np.minimum(0., stats.chi.logpdf(proposal, rank)-stats.chi.logpdf(r, rank))
            accept = inside(proposal) & ((~local) | (np.log(rng.uniform(size=len(r))) < logratio))
            accepted += int(accept.sum())
            r[accept] = proposal[accept]
        return r, accepted

    hub, _ = advance([radius])
    endpoints, accepted = advance(np.full(m["mcmc_branches"], hub[0]))
    exceedances = int(np.count_nonzero(endpoints >= radius))
    return dict(p_mcmc=(1+exceedances)/(m["mcmc_branches"]+1),
                hub_radius=float(hub[0]), mcmc_exceedances=exceedances,
                branch_acceptance=accepted/(m["mcmc_steps"]*m["mcmc_branches"]),
                unchanged_endpoints=int(np.count_nonzero(endpoints == radius)))


def jobs(m):
    for di in range(len(m["designs"])):
        for ci in range(len(m["conditions"])):
            for start in range(0, m["iterations"], m["batch_size"]):
                yield di, ci, start, min(start+m["batch_size"], m["iterations"])


def job_name(job):
    di, ci, start, end = job
    return f"d{di}_c{ci}_{start:05}_{end:05}.json"


def trial_rows(m, di, ci, iteration):
    condition = m["conditions"][ci]
    mu = mean_response(m, di, condition)
    noise_seed, target_seed = seed(m, 2, di, iteration), seed(m, 3, di, iteration)
    noise = np.random.default_rng(noise_seed).normal(size=m["n"])
    y = mu + m["sigma"] * noise
    auxiliary_u = float(np.random.default_rng(target_seed).uniform())
    data_hash = hashlib.sha256(y.tobytes()).hexdigest()
    rows = []
    for basis in m["bases"]:
        g = geometry(encoded(m).decode(), di, basis)
        yr, mur = g.Q.T @ y, g.Q.T @ mu
        selected = g.select(yr, m["k"])
        j = selected[min(int(auxiliary_u * len(selected)), len(selected)-1)]
        M = sum(1 << a for a in selected)
        for arm in m["targets"]:
            A, B = g.target(arm, M, j)
            delta = float(mur @ A @ mur)
            signal_norm = float(np.linalg.norm(B.T @ mur))
            null_true = signal_norm <= m["null_tolerance"] * max(1.0, np.linalg.norm(mu))
            result = radial_test(g, yr, B, selected, m["sigma"])
            exact_p = result["p_selective"]
            mcmc_seed = seed(m, 10 + m["targets"].index(arm), di, ci*m["iterations"]+iteration)
            mc = mcmc_rank_test(g, yr, B, selected, m, np.random.default_rng(mcmc_seed))
            result["p_selective"] = mc.pop("p_mcmc")
            row = dict(p_exact=exact_p, mcmc_seed=mcmc_seed, **mc, design=m["designs"][di], design_index=di, condition=condition["id"],
                       condition_index=ci, iteration=iteration, basis=basis, arm=arm,
                       selected=list(selected), selected_mask=M, target=j,
                       noise_seed=noise_seed, target_seed=target_seed, auxiliary_u=auxiliary_u,
                       data_sha256=data_hash, rank=B.shape[1], full_rank=g.r,
                       mean_contribution=max(0.0, delta), projected_mean_norm=signal_norm,
                       null_true=bool(null_true), target_is_x0=(j == 0), **result)
            if m.get("weight_diagnostics", True) and arm in ("selected_shapley", "full_shapley"):
                Au, Bu = g.target(arm, M, j, uniform=True)
                error = float(np.max(np.abs(B @ B.T - Bu @ Bu.T)))
                require(error < 1e-8, "Uniform weights changed support unexpectedly")
                # A different allocation statistic, identical radius and event.
                pu = radial_test(g, yr, Bu, selected, m["sigma"])["p_selective"]
                row.update(uniform_mean_contribution=max(0.0, float(mur @ Au @ mur)),
                           uniform_support_error=error, uniform_p_difference=abs(pu-result["p_selective"]))
                require(row["uniform_p_difference"] < 1e-7, "Weight-invariance diagnostic failed")
            rows.append(row)
    return rows


def run_job(m, job, out):
    di, ci, start, end = job
    rows = []
    with threadpool_limits(limits=1):
        for iteration in range(start, end):
            rows.extend(trial_rows(m, di, ci, iteration))
        require(all(t["num_threads"] <= 1 for t in threadpool_info()), "Numerical thread cap failed")
    receipt = dict(manifest_sha256=sha(m), job=list(job), rows=rows, rows_sha256=sha(rows))
    atomic(Path(out) / "jobs" / job_name(job), receipt)
    return job_name(job)


@contextmanager
def exclusive(out):
    out.mkdir(parents=True, exist_ok=True)
    stream = (out / "controller.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if stream.read(1) == b"":
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def setup_output(m, out):
    prov = provenance(m)
    path = out / "requested_manifest.json"
    if path.exists():
        require(read(path) == m, "Refusing to reuse output with changed settings")
        prior = read(out / "provenance.json")
        require(prior["sources"] == prov["sources"] and prior["versions"] == prov["versions"],
                "Source/dependency drift on resume")
    else:
        require(not any(out.glob("jobs/*.json")), "Orphan results without manifest")
        atomic(path, m)
        atomic(out / "provenance.json", prov)
    (out / "jobs").mkdir(exist_ok=True)


def validate_receipt(m, job, receipt):
    require(receipt["manifest_sha256"] == sha(m) and receipt["job"] == list(job), "Receipt manifest/job mismatch")
    rows = receipt["rows"]
    require(sha(rows) == receipt["rows_sha256"], "Receipt checksum mismatch")
    di, ci, start, end = job
    expected = {(i, b, a) for i in range(start, end) for b in m["bases"] for a in m["targets"]}
    actual = {(r["iteration"], r["basis"], r["arm"]) for r in rows}
    require(actual == expected and len(rows) == len(expected), "Missing/duplicate trial arms")
    paired = {}
    for r in rows:
        require(r["design_index"] == di and r["condition_index"] == ci, "Condition mismatch")
        require(r["design"] == m["designs"][di] and r["condition"] == m["conditions"][ci]["id"], "Label mismatch")
        require(r["noise_seed"] == seed(m, 2, di, r["iteration"]) and
                r["target_seed"] == seed(m, 3, di, r["iteration"]), "Seed mismatch")
        u = float(np.random.default_rng(r["target_seed"]).uniform())
        require(r["auxiliary_u"] == u and r["selected"] == sorted(r["selected"]), "Target randomization mismatch")
        require(len(r["selected"]) == m["k"] and len(set(r["selected"])) == m["k"], "Selected set invalid")
        require(r["target"] == r["selected"][int(u*m["k"])], "Target mismatch")
        require(r["selected_mask"] == sum(1 << j for j in r["selected"]), "Selected mask mismatch")
        key = (r["iteration"], r["basis"])
        match = (r["selected_mask"], r["target"], r["data_sha256"], u)
        require(key not in paired or paired[key] == match, "Unmatched target arms")
        paired[key] = match
        for field in ("p_selective", "p_naive", "p_exact"):
            require(np.isfinite(r[field]) and 0 <= r[field] <= 1, "Invalid p-value")
        require(r["mcmc_seed"] == seed(m, 10+m["targets"].index(r["arm"]), di, ci*m["iterations"]+r["iteration"]), "MCMC seed mismatch")
        require(r["p_selective"] == (1+r["mcmc_exceedances"])/(m["mcmc_branches"]+1), "MCMC rank mismatch")
        require(0 <= r["branch_acceptance"] <= 1, "Invalid acceptance rate")
        require(0 < r["rank"] <= r["full_rank"] <= m["p"]*m["rbf_units"], "Rank mismatch")
        require(np.isfinite(r["log_selection_mass"]) and r["log_selection_mass"] <= 1e-7, "Invalid selection mass")
        if m.get("weight_diagnostics", True) and r["arm"] in ("selected_shapley", "full_shapley"):
            require(r["uniform_p_difference"] < 1e-7 and r["uniform_support_error"] < 1e-8, "Weight check failed")
    for i in range(start, end):
        require(len({r["data_sha256"] for r in rows if r["iteration"] == i}) == 1, "Unmatched basis data")
    return rows


def write_csv(path, rows):
    fields = sorted({key for r in rows for key in r})
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    os.replace(tmp, path)


def binomial(k, n):
    if n == 0:
        return dict(n=0, rejections=0, rate=None, ci_lower=None, ci_upper=None)
    lower = float(stats.beta.ppf(.025, k, n-k+1)) if k else 0.0
    upper = float(stats.beta.ppf(.975, k+1, n-k)) if k < n else 1.0
    return dict(n=n, rejections=k, rate=k/n, ci_lower=lower, ci_upper=upper)


def summaries(m, rows, out):
    grouped = {}
    for r in rows:
        key = (r["design"], r["basis"], r["condition"], r["arm"])
        grouped.setdefault(key, []).append(r)
    summary, agreement, geometry_rows = [], [], []
    for key, group in grouped.items():
        labels = dict(zip(("design", "basis", "condition", "arm"), key))
        for null in (True, False):
            subset = [r for r in group if r["null_true"] == null]
            for alpha in m["alpha_levels"]:
                for method in ("selective", "exact", "naive"):
                    counts = binomial(sum(r["p_"+method] < alpha for r in subset), len(subset))
                    summary.append(dict(**labels, null_true=null, method=method, alpha=alpha,
                                        fraction_of_all_trials=len(subset)/m["iterations"], **counts))
        for j in range(m["p"]):
            subset = [r for r in group if r["target"] == j]
            geometry_rows.append(dict(**labels, target=j, target_count=len(subset),
                targeting_probability=len(subset)/m["iterations"],
                true_null_count=sum(r["null_true"] for r in subset),
                target_and_reject=sum(r["p_selective"] < .05 for r in subset)/m["iterations"]))
    pairgroups = {}
    for r in rows:
        key = (r["design"], r["basis"], r["condition"], r["iteration"])
        pairgroups.setdefault(key, {})[r["arm"]] = r
    counts = {}
    for key, arms in pairgroups.items():
        for a, b in combinations(m["targets"], 2):
            ck = (*key[:3], a, b)
            value = counts.setdefault(ck, dict(n=0, null_status_disagrees=0, decisions_disagree=0))
            value["n"] += 1
            value["null_status_disagrees"] += arms[a]["null_true"] != arms[b]["null_true"]
            value["decisions_disagree"] += (arms[a]["p_selective"] < .05) != (arms[b]["p_selective"] < .05)
    for key, value in counts.items():
        agreement.append(dict(zip(("design", "basis", "condition", "arm_a", "arm_b"), key), **value))
    write_csv(out / "calibration_power_summary.csv", summary)
    write_csv(out / "targeting_summary.csv", geometry_rows)
    write_csv(out / "paired_target_comparisons.csv", agreement)
    write_csv(out / "results.csv", rows)


def verify(m, out, export=False):
    require(read(out / "requested_manifest.json") == m, "Saved manifest mismatch")
    prov = read(out / "provenance.json")
    require(prov["sources"] == source_hashes() and prov["manifest_sha256"] == sha(m), "Source provenance mismatch")
    expected = {job_name(j) for j in jobs(m)}
    require({p.name for p in (out / "jobs").glob("*.json")} == expected, "Incomplete or extra job receipts")
    rows, hashes = [], {}
    for job in jobs(m):
        path = out / "jobs" / job_name(job)
        receipt = read(path)
        batch = validate_receipt(m, job, receipt)
        rows.extend(batch)
        hashes[path.name] = receipt["rows_sha256"]
        # Recreate one entire matched trial per committed batch, not merely a checksum.
        recreated = trial_rows(m, job[0], job[1], job[2])
        saved = [r for r in batch if r["iteration"] == job[2]]
        for a, b in zip(recreated, saved):
            for field in ("basis", "arm", "selected", "target", "rank", "null_true"):
                require(a[field] == b[field], f"Recomputed {field} mismatch")
            for field in ("p_selective", "p_exact", "p_naive", "mean_contribution", "radius"):
                require(np.isclose(a[field], b[field], rtol=1e-7, atol=1e-9), f"Recomputed {field} mismatch")
    # Classify truth for every arm from the fixed model, independently of saved labels.
    truths = {}
    for r in rows:
        key = (r["design_index"], r["condition_index"], r["basis"], r["selected_mask"], r["target"], r["arm"])
        if key not in truths:
            di, ci, basis, M, j, arm = key
            g = geometry(encoded(m).decode(), di, basis)
            A, B = g.target(arm, M, j)
            mu = mean_response(m, di, m["conditions"][ci])
            mur = g.Q.T @ mu
            signal = float(np.linalg.norm(B.T @ mur))
            truths[key] = (signal <= m["null_tolerance"]*max(1, np.linalg.norm(mu)), B.shape[1], max(0., float(mur @ A @ mur)))
        truth, rank, delta = truths[key]
        require(r["null_true"] == truth and r["rank"] == rank, "False null/rank label")
        require(np.isclose(r["mean_contribution"], delta, rtol=1e-7, atol=1e-9), "Contribution mismatch")
    count = len(m["designs"])*len(m["conditions"])*m["iterations"]*len(m["bases"])*len(m["targets"])
    require(len(rows) == count, "Total count mismatch")
    report = dict(status="verified", total_tests=count, trials=count//(len(m["bases"])*len(m["targets"])),
                  manifest_sha256=sha(m), receipts_sha256=sha(hashes),
                  max_uniform_p_difference=(max(r.get("uniform_p_difference", 0) for r in rows)
                                            if m.get("weight_diagnostics", True) else None),
                  statistical_calibration="Report pointwise binomial intervals; no stochastic pass/fail cutoff")
    if export:
        summaries(m, rows, out)
        atomic(out / "verification.json", report)
        exported = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.csv")}
        atomic(out / "COMPLETED", dict(**report, export_sha256=exported))
    else:
        done = read(out / "COMPLETED")
        require(done["receipts_sha256"] == report["receipts_sha256"], "Completion digest mismatch")
        for name, expected_sha in done["export_sha256"].items():
            require(hashlib.sha256((out / name).read_bytes()).hexdigest() == expected_sha, "Export checksum mismatch")
    return report


def run(m, out):
    with exclusive(out):
        setup_output(m, out)
        if (out / "COMPLETED").exists():
            print(json.dumps(verify(m, out)), flush=True)
            return
        if hasattr(os, "sched_getaffinity"):
            available = sorted(os.sched_getaffinity(0))
            require(len(available) >= m["workers"], "Insufficient available CPU cores")
            os.sched_setaffinity(0, available[:m["workers"]])
        atomic(out / "runtime.json", dict(pid=os.getpid(), workers=m["workers"],
               affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               thread_limits={key: os.environ[key] for key in THREAD_KEYS}, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        pending = []
        for job in jobs(m):
            receipt = out / "jobs" / job_name(job)
            if receipt.exists():
                validate_receipt(m, job, read(receipt))
            else:
                pending.append(job)
        print(f"Pending batches: {len(pending)} / {len(list(jobs(m)))}; workers={m['workers']}", flush=True)
        with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=m["workers"]) as pool:
            futures = [pool.submit(run_job, m, job, str(out)) for job in pending]
            for i, future in enumerate(as_completed(futures), 1):
                name = future.result()
                print(f"Committed {i}/{len(pending)}: {name}", flush=True)
        print(json.dumps(verify(m, out, export=True)), flush=True)


def validate_archive(m, path):
    prefix = m["experiment"]
    names = set()
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            require(not name.is_absolute() and ".." not in name.parts and "\\" not in entry.name,
                    "Unsafe archive path")
            require(name.parts and name.parts[0] == prefix, "Wrong archive root")
            require(entry.isfile() or entry.isdir(), "Links/special files forbidden")
            require(entry.name not in names, "Duplicate archive path")
            names.add(entry.name)
        require(f"{prefix}/COMPLETED" in names and f"{prefix}/requested_manifest.json" in names,
                "Incomplete result archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "status", "verify", "validate-archive"))
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    m = read_config(args.config)
    out = args.output or ROOT / m["output_root"]
    if args.action == "check":
        info = provenance(m)
        for di in range(len(m["designs"])):
            for basis in m["bases"]:
                g = geometry(encoded(m).decode(), di, basis)
                require(g.r == m["p"]*m["rbf_units"], "Unexpected design rank")
        print(json.dumps(dict(status="preflight_ok", workers=m["workers"],
                             total_tests=len(m["designs"])*len(m["conditions"])*m["iterations"]*len(m["bases"])*len(m["targets"]),
                             manifest_sha256=sha(m), sources=info["sources"]), indent=2))
    elif args.action == "run":
        run(m, out)
    elif args.action == "status":
        print(json.dumps(dict(committed_batches=len(list((out / "jobs").glob("*.json"))),
                              expected_batches=len(list(jobs(m))), complete=(out / "COMPLETED").exists())))
    elif args.action == "verify":
        with exclusive(out):
            print(json.dumps(verify(m, out), indent=2))
    else:
        validate_archive(m, args.archive)
        print("Archive paths and member types validated")


if __name__ == "__main__":
    main()
