"""Independent regression, CV-event geometry and experiment lifecycle checks."""
from pathlib import Path
import copy
import io
import sys
import tarfile

import numpy as np
import pytest
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
import run_rbf_ridge_cv_study as study
import rbf_ridge_cv as model


def smoke():
    return study.read_config(ROOT/'benchmarks'/f'{study.SMOKE}.json')


def test_sampled_centers_widths_and_fold_local_cv_against_direct_regression():
    m = smoke()
    saved = study.inputs(m, 'synthetic', 0, 25)
    X, y = np.asarray(saved['X']), np.asarray(saved['y_base'])
    state = model.basis(X, .1, saved['basis_seed'])
    assert len(state['indices']) == 3
    np.testing.assert_array_equal(state['centers'].T, X[state['indices']])
    for j in range(4):
        d = np.abs(X[:, j, None]-X[None, :, j])[np.triu_indices(25, 1)]
        assert state['widths'][j] == np.median(d)
    matrices, folds, centers = model.cv_matrices(X, .1, m['lambdas'], 5, saved['basis_seed'], saved['fold_seed'])
    for li, lam in enumerate(m['lambdas']):
        squared_error = 0.
        for fi, valid in enumerate(folds):
            train = np.setdiff1d(np.arange(len(X)), valid)
            assert set(centers[fi]) <= set(train) and not set(centers[fi]) & set(valid)
            fitted = model.basis(X[train], .1, np.random.SeedSequence([saved['basis_seed'], fi, 1]))
            Z = np.column_stack([np.ones(len(train)), *model.transform(X[train], fitted)])
            V = np.column_stack([np.ones(len(valid)), *model.transform(X[valid], fitted)])
            penalty = np.diag([0.]+[len(train)*lam]*(Z.shape[1]-1))
            beta = np.linalg.solve(Z.T@Z+penalty, Z.T@y[train])
            squared_error += np.sum((y[valid]-V@beta)**2)
        assert y@matrices[li]@y == pytest.approx(squared_error/len(y), rel=1e-8, abs=1e-10)


def test_ridge_scores_equal_penalized_objective_and_are_psd():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(25, 4)), rng.normal(size=25)
    blocks = model.transform(X, model.basis(X, .2, 8))
    lam = .01
    g = model.Geometry(blocks, lam)
    losses = []
    for mask in range(16):
        Z = np.column_stack([np.ones(25), *[b for j, b in enumerate(blocks) if mask & (1 << j)]])
        penalty = np.diag([0.]+[25*lam]*(Z.shape[1]-1))
        beta = np.linalg.solve(Z.T@Z+penalty, Z.T@y)
        losses.append(np.sum((y-Z@beta)**2)+beta@penalty@beta)
    yr = g.Q.T@y
    for j in range(4):
        for mask in range(16):
            if mask & (1 << j): continue
            D = g.smoothers[mask | (1 << j)]-g.smoothers[mask]
            assert np.linalg.eigvalsh(D).min() >= -1e-9
            assert yr@D@yr == pytest.approx(losses[mask]-losses[mask | (1 << j)], abs=1e-9)


def test_joint_cv_and_topk_intervals_and_chi_quadrature():
    m = smoke()
    for iteration in range(3):
        saved = study.inputs(m, 'synthetic', iteration, 25)
        X, y = np.asarray(saved['X']), np.asarray(saved['y_base'])
        blocks = model.transform(X, model.basis(X, .1, saved['basis_seed']))
        C, _, _ = model.cv_matrices(X, .1, m['lambdas'], 5, saved['basis_seed'], saved['fold_seed'])
        chosen = int(np.argmin([y@c@y for c in C]))
        g = model.Geometry(blocks, m['lambdas'][chosen])
        selected = g.select(g.Q.T@y)
        for arm in m['targets']:
            _, B = g.target(arm, sum(1 << j for j in selected), selected[0])
            W = g.Q@B
            projected = W@(W.T@y)
            radius = np.linalg.norm(projected)
            u, z = projected/radius, y-projected
            intervals = model.joint_intervals(g, z, u, selected, 1., C, chosen)
            for r in np.geomspace(.001, 100, 180):
                candidate = z+r*u
                direct = (g.select(g.Q.T@candidate) == selected and
                          int(np.argmin([candidate@c@candidate for c in C])) == chosen)
                assert any(lo < r < hi for lo, hi in intervals) == direct
            den = sum(integrate.quad(lambda r: stats.chi.pdf(r, B.shape[1]), lo, hi)[0] for lo, hi in intervals)
            num = sum(integrate.quad(lambda r: stats.chi.pdf(r, B.shape[1]), max(lo, radius), hi)[0]
                      for lo, hi in intervals if hi > radius)
            result = model.radial_test(g, y, B, selected, 1., C, chosen)
            assert result['p_value'] == pytest.approx(num/den, abs=1e-8)


def test_counts_matching_nested_samples_and_manifest_freezing(tmp_path):
    m = study.read_config()
    assert study.expected_tests(m) == 732000
    assert [int(np.ceil(.1*n)) for n in m['sample_sizes']] == [3, 5, 8]
    assert m['workers'] == 32 and m['threads_per_worker'] == 1
    assert study.expected_tests(smoke()) == 204
    for dataset in ('synthetic', 'housing'):
        small = study.inputs(m, dataset, 0, 25)
        large = study.inputs(m, dataset, 0, 75)
        assert small['auxiliary_u'] == large['auxiliary_u']
        if dataset == 'synthetic':
            np.testing.assert_array_equal(small['X'], large['X'][:25])
            np.testing.assert_array_equal(small['y_base'], large['y_base'][:25])
        else:
            assert set(small['indices']) <= set(large['indices'])
    changed = dict(m, workers=33)
    path = tmp_path/'bad.json'
    study.core.atomic(path, changed)
    with pytest.raises(ValueError, match='Manifest changed'): study.read_config(path)


def test_receipts_full_smoke_lifecycle_tampering_and_exports(tmp_path):
    m = smoke()
    study.setup_output(m, tmp_path)
    for job in study.jobs(m): study.run_job(m, job, tmp_path)
    original = {p.name: p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}
    study.setup_output(m, tmp_path)
    assert study.verify(m, tmp_path, export=True)['total_tests'] == 204
    assert study.verify(m, tmp_path)['status'] == 'verified'
    assert original == {p.name: p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}
    job = study.jobs(m)[0]
    receipt = study.core.read(tmp_path/'jobs'/study.job_name(job))
    broken = copy.deepcopy(receipt)
    broken['rows'][0]['p_value'] = .123
    with pytest.raises(ValueError, match='checksum'): study.validate_receipt(m, job, broken)
    broken['rows_sha256'] = study.core.sha(broken['rows'])
    with pytest.raises(ValueError): study.validate_receipt(m, job, broken, recompute=True)
    with (tmp_path/'summary.csv').open('a') as f: f.write('corrupt\n')
    with pytest.raises(ValueError, match='checksum'): study.verify(m, tmp_path)


def test_zero_distance_fallback_and_unsafe_archive(tmp_path):
    X = np.column_stack([np.ones(25), np.r_[np.zeros(24), 1.]])
    fitted = model.basis(X, .1, 2)
    assert np.isfinite(fitted['widths']).all() and (fitted['widths'] > 0).all()
    assert not model.transform(X, fitted)[0].any()
    path = tmp_path/'bad.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        entry = tarfile.TarInfo('../escape'); entry.size = 1
        archive.addfile(entry, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe archive'): study.core.validate_archive(smoke(), path)


def test_launcher_packages_exact_sources_and_has_process_guards():
    source = (ROOT/'examples/rbf_ridge_cv_n25_50_75_cpu32.ps1').read_text()
    for path in study.PACKAGED: assert path in source
    for token in ('Preflight', 'Launch', 'Status', 'Resume', 'Stop', 'Verify', 'Download',
                  'launcher.startticks', 'kill -TERM -- "-$PID"', 'flock -n 8', 'flock -n 7',
                  'Refusing to overwrite', 'validate-archive', 'run_rbf_ridge_cv_study.py run'):
        assert token in source
