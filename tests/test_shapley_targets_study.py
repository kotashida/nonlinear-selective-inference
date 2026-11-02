"""Independent RSS, selection-boundary, and resumability checks for the study."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

PATH = Path(__file__).resolve().parents[1] / "benchmarks/run_shapley_targets.py"
spec = importlib.util.spec_from_file_location("shapley_targets_study", PATH)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_exact_coalition_scores_match_independent_least_squares():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(35, 4))
    X[:, 1] += .8*X[:, 0]
    blocks = [np.column_stack([X[:, j], X[:, j]**2]) for j in range(4)]
    blocks = [b-b.mean(axis=0) for b in blocks]
    g = study.Geometry(blocks)
    y = rng.normal(size=35)
    reduced = g.Q.T @ y
    for j in range(4):
        expected = 0.
        for mask in range(16):
            if mask & (1 << j):
                continue
            def rss(s):
                a = np.column_stack([np.ones(35), *[blocks[k] for k in range(4) if s & (1 << k)]])
                e = y-a @ np.linalg.lstsq(a, y, rcond=None)[0]
                return e @ e
            expected += (rss(mask)-rss(mask | (1 << j)))/(4*study.comb(3, mask.bit_count()))
        assert reduced @ g.selector[j] @ reduced == pytest.approx(expected, abs=1e-10)


def test_correlated_example_distinguishes_nulls_and_weights_preserve_support():
    rng = np.random.default_rng(6)
    centered = rng.normal(size=(30, 2))
    centered -= centered.mean(axis=0)
    q, _ = np.linalg.qr(centered)
    rho = .6
    x1, x2 = q[:, 0], rho*q[:, 0]+np.sqrt(1-rho*rho)*q[:, 1]
    g = study.Geometry([x1[:, None], x2[:, None]])
    mu = g.Q.T @ x2
    expected = dict(marginal=rho**2, model=0., selected_shapley=rho**2/2, full_shapley=rho**2/2)
    for arm, value in expected.items():
        A, B = g.target(arm, 3, 0)
        assert mu @ A @ mu == pytest.approx(value, abs=1e-12)
        if arm in ("selected_shapley", "full_shapley"):
            assert B.shape[1] == 2
    m = study.read_config(study.ROOT / "benchmarks/shapley_targets_cpu12_smoke_20261016.json")
    g = study.geometry(study.encoded(m).decode(), 2, "cubic_spline")
    for arm in ("selected_shapley", "full_shapley"):
        A, B = g.target(arm, 3, 0)
        Au, Bu = g.target(arm, 3, 0, uniform=True)
        np.testing.assert_allclose(B@B.T, Bu@Bu.T, atol=1e-10)
    assert np.linalg.norm(A-Au) > 1e-3  # Different full-set allocations, same null.


def test_factorial_control_all_targets_coincide():
    m = study.read_config(study.ROOT / "benchmarks/shapley_targets_cpu12_smoke_20261016.json")
    for basis in m["bases"]:
        g = study.geometry(study.encoded(m).decode(), 0, basis)
        expected = g.target("marginal", 3, 0)[0]
        for arm in study.TARGETS:
            np.testing.assert_allclose(g.target(arm, 3, 0)[0], expected, atol=1e-10)


def test_exact_intervals_match_direct_selection_and_numerical_integral():
    rng = np.random.default_rng(80)
    X = rng.normal(size=(40, 4))
    X[:, 1] += X[:, 0]
    X -= X.mean(axis=0)
    g = study.Geometry([X[:, j:j+1] for j in range(4)])
    for _ in range(8):
        y = rng.normal(size=g.r)
        selected = g.select(y)
        A, B = g.target("marginal", sum(1<<j for j in selected), selected[0])
        projected = B @ (B.T @ y)
        r = np.linalg.norm(projected)
        u, z = projected/r, y-projected
        intervals = study.selection_intervals(g, z, u, selected)
        for candidate in np.r_[np.geomspace(.001, 30, 600), r]:
            inside = any(lo < candidate < hi for lo, hi in intervals)
            assert inside == (g.select(z+candidate*u) == selected)
        rank = B.shape[1]
        den = sum(integrate.quad(lambda s: stats.chi.pdf(s, rank), lo, hi)[0] for lo, hi in intervals)
        num = sum(integrate.quad(lambda s: stats.chi.pdf(s, rank), max(lo, r), hi)[0]
                  for lo, hi in intervals if hi > r)
        result = study.radial_test(g, y, B, selected)
        assert result["p_selective"] == pytest.approx(num/den, abs=1e-8)


def test_nuisance_mean_has_zero_x0_marginal_projection():
    m = study.read_config(study.ROOT / "benchmarks/shapley_targets_cpu12_smoke_20261016.json")
    for di in range(3):
        mu = study.mean_response(m, di, m["conditions"][-1])
        assert np.sqrt(np.mean(mu*mu)) == pytest.approx(.5)
        for basis in m["bases"]:
            g = study.geometry(study.encoded(m).decode(), di, basis)
            _, B = g.target("marginal", 3, 0)
            assert np.linalg.norm(B.T @ g.Q.T @ mu) < 1e-10


def test_receipts_reject_corruption_and_resumption_reuses_completed_batches(tmp_path):
    m = study.read_config(study.ROOT / "benchmarks/shapley_targets_cpu12_smoke_20261016.json")
    study.setup_output(m, tmp_path)
    job = next(study.jobs(m))
    study.run_job(m, job, tmp_path)
    path = tmp_path / "jobs" / study.job_name(job)
    receipt = study.read(path)
    study.validate_receipt(m, job, receipt)
    before = path.read_bytes()
    study.setup_output(m, tmp_path)
    assert before == path.read_bytes()
    receipt["rows"][0]["p_selective"] = .123
    with pytest.raises(ValueError, match="checksum"):
        study.validate_receipt(m, job, receipt)


def test_manifest_changes_and_worker_overrides_are_rejected(tmp_path):
    m = study.read_config(study.ROOT / "benchmarks/shapley_targets_cpu12_smoke_20261016.json")
    m["workers"] = 13
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Manifest changed"):
        study.read_config(path)


def test_upper_tail_interval_mass_remains_finite():
    value = study.log_mass(12., 13., 3)
    assert np.isfinite(value)
    assert np.exp(value) == pytest.approx(stats.chi.sf(12, 3)-stats.chi.sf(13, 3), rel=1e-10, abs=1e-45)


def test_roundoff_does_not_invent_distant_roots_for_orthogonal_control():
    m = study.read_config(study.ROOT / "benchmarks/shapley_targets_cpu12_smoke_20261016.json")
    rows = study.trial_rows(m, 0, 5, 3)
    for basis in m["bases"]:
        p = [r["p_selective"] for r in rows if r["basis"] == basis]
        np.testing.assert_allclose(p, p[0], atol=1e-10)


def test_final_pilot_has_200_repetitions_and_1600_paired_tests():
    m = study.read_config(study.ROOT / "benchmarks/shapley_targets_pilot_cpu12_200_20261016.json")
    assert m["iterations"] == 200
    assert m["workers"] == 12
    assert m["bases"] == ["cubic_spline"]
    assert m["targets"] == ["selected_shapley", "full_shapley"]
    assert len(m["designs"])*len(m["conditions"])*m["iterations"] == 800
    assert len(list(study.jobs(m))) == 32
    rows = study.trial_rows(m, 1, 1, 0)
    assert len(rows) == 2
    assert rows[0]["selected"] == rows[1]["selected"]
    assert rows[0]["target"] == rows[1]["target"]
    assert rows[0]["data_sha256"] == rows[1]["data_sha256"]


def test_original_basic_covers_all_four_targets_on_200_null_datasets():
    m = study.read_config(study.ROOT / "benchmarks/four_targets_basic_null_200_20261016.json")
    assert m["iterations"] == 200
    assert m["workers"] == 8
    assert m["designs"] == ["correlated_gaussian"]
    assert m["bases"] == ["cubic_spline"]
    assert m["targets"] == list(study.TARGETS)
    assert m["conditions"] == [{"id": "global_null", "mean": "zero", "rms": 0.0}]
    assert len(list(study.jobs(m))) == 8
    rows = study.trial_rows(m, 0, 0, 0)
    assert len(rows) == 4
    assert all(r["null_true"] for r in rows)
    assert len({(r["data_sha256"], r["selected_mask"], r["target"]) for r in rows}) == 1


def test_cpu32_revision_changes_only_resources_and_matches_existing_trials():
    old = study.read_config(study.ROOT / "benchmarks/four_targets_basic_null_200_20261016.json")
    new = study.read_config()
    assert new["workers"] == 32
    assert new["batch_size"] == 5
    assert len(list(study.jobs(new))) == 40
    ignore = {"workers", "batch_size", "experiment", "output_root"}
    assert {k: v for k, v in old.items() if k not in ignore} == {k: v for k, v in new.items() if k not in ignore}
    assert study.trial_rows(old, 0, 0, 7) == study.trial_rows(new, 0, 0, 7)
