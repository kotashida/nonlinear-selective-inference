"""Independent regression checks and experiment lifecycle validation."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from rbf_network import AdditiveRBFNetwork
import run_rbf_shapley as study


def test_fixed_hidden_layer_prediction_and_nested_rss():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(50, 4))
    y = X[:, 0]**2 + rng.normal(size=50)
    net = AdditiveRBFNetwork().fit_basis(X)
    centers = net.centers_.copy()
    widths = net.widths_.copy()
    small = net.design_matrix(X, [1])
    large = net.design_matrix(X, [0, 1])
    np.testing.assert_allclose(small[:, 1:], large[:, 4:])
    net.fit(X, y, [0, 1])
    np.testing.assert_allclose(net.predict(X), large @ np.linalg.lstsq(large, y, rcond=None)[0])
    rss_large = np.sum((y-net.predict(X))**2)
    net.fit(X, -y, [1])
    np.testing.assert_array_equal(net.centers_, centers)
    np.testing.assert_array_equal(net.widths_, widths)
    rss_small = np.sum((-y-net.predict(X))**2)
    assert rss_large <= rss_small + 1e-10
    # New observations use training centering, not their own column means.
    np.testing.assert_allclose(net.transform_blocks(X[:3])[0], net.transform_blocks(X)[0][:3])


def test_both_shapley_scores_match_independent_raw_rbf_regressions():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 4))
    X[:, 1] += .8*X[:, 0]
    net = AdditiveRBFNetwork().fit_basis(X)
    g = study.Geometry(net.transform_blocks(X))
    y = rng.normal(size=40)
    yr = g.Q.T @ y
    # Construct uncentered Gaussian units independently; intercept absorbs means.
    raw = [np.exp(-.5*((X[:, j, None]-np.quantile(X[:, j], [.2, .5, .8]))/X[:, j].std())**2)
           for j in range(4)]
    def rss(mask):
        a = np.column_stack([np.ones(40), *[raw[j] for j in range(4) if mask & (1 << j)]])
        e = y-a @ np.linalg.lstsq(a, y, rcond=None)[0]
        return e @ e
    for arm, universe in [('selected_shapley', 3), ('full_shapley', 15)]:
        q = universe.bit_count()
        expected = sum((rss(s)-rss(s | 1))/(q*study.comb(q-1, s.bit_count()))
                       for s in range(16) if not s & ~universe and not s & 1)
        A, B = g.target(arm, 3, 0)
        assert yr @ A @ yr == pytest.approx(expected, abs=1e-9)
        np.testing.assert_allclose(B @ B.T @ A, A, atol=1e-10)


def test_selection_intervals_and_analytic_tail_against_direct_integration():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(45, 4))
    g = study.Geometry(AdditiveRBFNetwork().fit_basis(X).transform_blocks(X))
    y = rng.normal(size=g.r)
    selected = g.select(y)
    mask = sum(1 << j for j in selected)
    for arm in ('selected_shapley', 'full_shapley'):
        _, B = g.target(arm, mask, selected[0])
        py = B @ B.T @ y
        radius = np.linalg.norm(py)
        z, u = y-py, py/radius
        intervals = study.selection_intervals(g, z, u, selected)
        for r in np.r_[np.geomspace(.001, 30, 300), radius]:
            assert any(lo < r < hi for lo, hi in intervals) == (g.select(z+r*u) == selected)
        rank = B.shape[1]
        den = sum(integrate.quad(lambda r: stats.chi.pdf(r, rank), lo, hi)[0] for lo, hi in intervals)
        num = sum(integrate.quad(lambda r: stats.chi.pdf(r, rank), max(lo, radius), hi)[0]
                  for lo, hi in intervals if hi > radius)
        assert study.radial_test(g, y, B, selected)['p_selective'] == pytest.approx(num/den, abs=1e-8)


def test_identity_kernel_counts_ties_and_resets_branches():
    m = study.read_config(ROOT/'benchmarks/rbf_shapley_mcmc20_cpu32_smoke_20261022.json')
    g = study.geometry(study.encoded(m).decode(), 0, 'additive_rbf')
    y = np.random.default_rng(6).normal(size=g.r)
    selected = g.select(y)
    _, B = g.target('selected_shapley', sum(1 << j for j in selected), selected[0])
    # Identity is reversible and yields the conservative p=1 regardless of data.
    m = dict(m, mcmc_steps=1)
    m['local_probability'] = 1.
    m['local_scales'] = [0.]
    result = study.mcmc_rank_test(g, y, B, selected, m, np.random.default_rng(8))
    assert result['p_mcmc'] == 1.
    assert result['mcmc_exceedances'] == 511
    assert result['unchanged_endpoints'] == 511


def test_frozen_design_pairing_and_corrupt_receipt_rejection(tmp_path):
    m = study.read_config(ROOT/'benchmarks/rbf_shapley_mcmc20_cpu32_smoke_20261022.json')
    study.setup_output(m, tmp_path)
    job = next(study.jobs(m))
    study.run_job(m, job, tmp_path)
    path = tmp_path/'jobs'/study.job_name(job)
    before = path.read_bytes()
    study.setup_output(m, tmp_path)
    assert before == path.read_bytes()
    receipt = study.read(path)
    rows = study.validate_receipt(m, job, receipt)
    assert len(rows) == 4 and all(r['null_true'] for r in rows)
    assert rows[0]['data_sha256'] == rows[1]['data_sha256']
    assert rows[0]['target'] == rows[1]['target']
    assert rows[0]['mcmc_seed'] != rows[1]['mcmc_seed']
    receipt['rows'][0]['p_selective'] = .123
    with pytest.raises(ValueError, match='checksum'):
        study.validate_receipt(m, job, receipt)
    m['width_multiplier'] = 2
    config = tmp_path/'modified.json'
    config.write_text(study.encoded(m).decode())
    with pytest.raises(ValueError, match='Manifest changed'):
        study.read_config(config)


def test_production_allocation():
    m = study.read_config()
    assert m['iterations'] == 250 and m['workers'] == 32
    assert len(list(study.jobs(m))) == 50
    assert len(m['conditions'])*m['iterations']*len(m['targets']) == 1000


@pytest.mark.parametrize('kwargs', [dict(width_multiplier=0), dict(center_quantiles=[.5,.5]), dict(center_quantiles=[1.])])
def test_invalid_basis_settings(kwargs):
    with pytest.raises(ValueError):
        AdditiveRBFNetwork(**kwargs)
