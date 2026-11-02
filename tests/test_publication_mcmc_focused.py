"""Focused-study checks; every inference smoke uses the full validated budget."""
import copy

import numpy as np
import pandas as pd
from scipy import stats

from benchmarks import run_publication_mcmc as e


def test_frozen_workload_and_identical_algorithms():
    original = e.read_config()
    focused = e.read_config(e.FOCUSED_CONFIG)
    cells = list(e.conditions(focused))
    assert sum(c['repetitions'] for c in cells if c['study'] == 'null') == 6000
    assert sum(c['repetitions'] for c in cells if c['study'] == 'real') == 8000
    assert len(list(e.jobs(focused))) == 14000
    assert len(cells) == 16
    assert e.mcmc(focused) == e.mcmc(original)
    assert e.mcmc(focused)['replicas'] == 511
    assert e.mcmc(focused)['steps'] == 20
    for method in e.METHODS:
        assert e.selector(focused, method).get_settings() == e.selector(original, method).get_settings()
        a = e.inference_args(original, {'auxiliary_u': .4, 'target_seed': 3, 'sampler_seed': 4}, method)
        b = e.inference_args(focused, {'auxiliary_u': .4, 'target_seed': 3, 'sampler_seed': 4}, method)
        a.pop('selector')
        b.pop('selector')
        assert a == b
    simulated = [c for c in cells if c['study'] == 'real' and c['mode'] != 'observed']
    assert {c['dataset'] for c in simulated} == {'heating_load', 'red_wine'}
    assert len([c for c in cells if c['study'] == 'real' and c['mode'] == 'observed']) == 8
    assert e.prepared_manifest(focused) == e.prepared_manifest(original)


def test_primary_calibration_family_and_failure_uncertainty():
    m = e.read_config(e.FOCUSED_CONFIG)
    m['bootstrap_replicates'] = 20
    rows = []
    for c in e.conditions(m):
        if c['study'] == 'real' and c['mode'] != 'gaussian_null':
            continue
        for method in e.METHODS:
            for trial in range(20):
                rows.append(dict(study=c['study'], dataset=c['dataset'], condition=c['id'],
                    trial=trial, method=method, mode='gaussian_null',
                    p_value=None if trial == 19 else .01 if trial == 0 else .5))
    rates = e.summaries(pd.DataFrame(rows), m)['rates.csv']
    primary = rates[(rates.study == 'null') & (rates.condition == 'iid_n100') & (rates.alpha == .05)]
    assert len(primary) == 5
    # Five primary arms; missing outcome is counted as a rejection in the upper bound.
    np.testing.assert_allclose(primary.primary_family_simultaneous_upper, stats.beta.ppf(.99, 3, 18))
    secondary = rates.drop(primary.index)
    assert secondary.primary_family_simultaneous_upper.isna().all()
    assert secondary.calibration_tolerance_supported.isna().all()
    assert rates.failures.eq(1).all()
    np.testing.assert_allclose(rates[rates.alpha == .05].all_attempt_lower, .05)
    np.testing.assert_allclose(rates[rates.alpha == .05].all_attempt_upper, .1)
    # The earlier experiment keeps its original all-null-cell family of 55.
    legacy = e.summaries(pd.DataFrame(rows), e.read_config())['rates.csv']
    legacy_null = legacy[legacy.alpha == .05]
    np.testing.assert_allclose(legacy_null.primary_family_simultaneous_upper,
                               stats.beta.ppf(1-.05/55, 3, 18))


def test_full_budget_smoke_selection_truth_pairing_and_resume(tmp_path):
    m = copy.deepcopy(e.read_config(e.FOCUSED_CONFIG))
    # Smoke changes workload/output only; inference retains 511 replicas and 20 steps.
    m.update(smoke=True, experiment=m['experiment']+'_smoke', output_root=str(tmp_path),
             smoke_mcmc={'replicas': 511, 'steps': 20})
    e.lock(m)
    cells = list(e.conditions(m))
    cases = [cells[0], cells[1]] + [c for c in cells if c['dataset'] == 'heating_load']
    assert len(cases) == 6
    for c in cases:
        values, ident, checksum = e.save_input(m, c, 0)
        # All five selectors are exercised with their actual validated settings.
        selections = {}
        for method in e.METHODS:
            selections[method] = sorted(e.selector(m, method).select(values['X'], values['y'], 2).selected_features.tolist())
        assert selections['spline_mcmc'] == selections['spline_exact']
        payloads = {}
        for method in ('spline_mcmc', 'mutual_information_mcmc', 'spline_exact'):
            payload = e.execute(m, c, 0, method, values, ident, checksum)
            e.audit_arm(m, c, values, ident, checksum, method, payload, replay=True)
            row = payload['result']
            assert not row['inference_failed']
            assert row['proposals'] == (0 if method == 'spline_exact' else 10240)
            assert row['selected'] == selections[method]
            assert row['input_sha256'] == checksum
            if c['study'] == 'null' or c['mode'] == 'gaussian_null':
                assert row['projection_null']
            if c.get('mode') == 'observed':
                assert row['projection_null'] is None
                assert np.isfinite(row['rmse'])
            payloads[method] = row
        assert payloads['spline_mcmc']['target'] == payloads['spline_exact']['target']
        np.testing.assert_allclose(payloads['spline_mcmc']['radial_t_obs'], payloads['spline_exact']['radial_t_obs'])
        receipts = {p: p.read_bytes() for p in e.trial_dir(m, c, 0).glob('*.json')}
        for method in ('spline_mcmc', 'mutual_information_mcmc', 'spline_exact'):
            e.execute(m, c, 0, method, values, ident, checksum)
        assert all(p.read_bytes() == contents for p, contents in receipts.items())
    # Recover an interrupted arm from its original input and seed.
    c = cases[0]
    values, ident, checksum = e.save_input(m, c, 0)
    arm = e.trial_dir(m, c, 0)/'spline_mcmc.json'
    before = e.read(arm)['payload']['result']
    arm.unlink()
    after = e.execute(m, c, 0, 'spline_mcmc', values, ident, checksum)['result']
    for key in ('p_value', 'selected', 'target', 'tail_samples', 'mcmc_hub'):
        assert before[key] == after[key]
