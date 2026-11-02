import copy
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest

from benchmarks import run_publication_mcmc as e
from benchmarks import prepare_publication_mcmc_data as prep


def small(tmp_path):
    m = copy.deepcopy(e.read_config(e.SMOKE_CONFIG))
    m['output_root'] = str(tmp_path)
    m['null_conditions'][0]['repetitions'] = 1
    m['bootstrap_replicates'] = 20
    return m


def test_authoritative_settings_and_counts():
    m = e.read_config()
    assert sum(c['repetitions'] for c in e.conditions(m) if c['study'] == 'null')*5 == 70000
    assert sum(c['repetitions'] for c in e.conditions(m) if c['study'] == 'real')*5 == 280000
    assert e.mcmc(m)['replicas'] == 511 and e.mcmc(m)['steps'] == 20
    assert e.selector(m, 'spline_mcmc').get_settings() == e.selector(m, 'spline_exact').get_settings()
    assert e.reference(m)['requested_manifest']['lime']['num_explanations'] == 40
    assert e.reference(m)['requested_manifest']['mutual_information']['bandwidth'] == 2.5


def test_preparation_roles_groups_and_no_jitter():
    m = e.read_config(e.SMOKE_CONFIG)
    meta = e.prepared_manifest(m)
    assert meta['datasets']['housing']['p'] == 6
    raw = prep.raw_tasks()
    for name in meta['datasets']:
        # Check all eight prepared tasks, not just smoke's Energy task.
        x, y, _ = raw[name]
        X = np.asarray(x, dtype=float)
        cal, train, evaluation = prep.partition(X, name)
        assert not (set(cal) & set(train) or set(cal) & set(evaluation) or set(train) & set(evaluation))
        if name != 'gas_turbine':
            key = X[:, :5] if 'load' in name else X
            _, groups = np.unique(key, axis=0, return_inverse=True)
            assert not (set(groups[cal]) & set(groups[train]) or set(groups[train]) & set(groups[evaluation]))
        with np.load(e.ROOT/m['prepared_root']/f'{name}.npz') as saved:
            scale = X[cal].std(axis=0)
            scale[scale == 0] = 1
            np.testing.assert_array_equal(saved['X'], (X-X[cal].mean(axis=0))/scale)
            assert len(train) >= 200


def test_all_methods_receipts_truth_resume_and_portable_verifier(tmp_path):
    m = small(tmp_path)
    e.lock(m)
    for c, trial in e.jobs(m):
        e.execute_trial(m, c, trial)
    receipts = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    for c, trial in e.jobs(m):
        e.execute_trial(m, c, trial)
    assert all(p.read_bytes() == content for p, content in receipts.items())
    report = e.verify(m, replay=True)
    assert report['tests'] == 20 and report['inference_failures'] == 0
    e.verify(m, downloaded=True, replay=True)
    summary = tmp_path/'rates.csv'
    original_summary = summary.read_bytes()
    corrupted = pd.read_csv(summary)
    corrupted.loc[0,'rejected'] += 1
    corrupted.to_csv(summary,index=False)
    with pytest.raises(AssertionError):
        e.verify(m, downloaded=True)
    summary.write_bytes(original_summary)
    frame = e.collect(m)
    assert (frame.groupby(['study','dataset','condition','trial']).input_sha256.nunique() == 1).all()
    assert frame[frame['mode'] == 'observed'].projection_null.isna().all()
    assert frame[frame['mode'] == 'gaussian_null'].projection_null.all()
    assert frame[frame.method != 'spline_exact'].proposals.eq(16).all()
    assert frame[frame.method == 'spline_exact'].proposals.eq(0).all()
    assert 'denominator_ess' not in frame and 'mc_ci_95_lower' not in frame
    # Resume one interrupted arm from the same saved input.
    c, trial = next(e.jobs(m))
    arm = e.trial_dir(m, c, trial)/'spline_mcmc.json'
    old_payload = e.read(arm)['payload']['result']
    arm.unlink()
    e.execute_trial(m, c, trial)
    new_payload = e.read(arm)['payload']['result']
    for key in ['p_value','mcmc_hub','tail_samples','selected','target']:
        assert new_payload[key] == old_payload[key]
    changed = copy.deepcopy(m)
    changed['master_seed'] += 1
    with pytest.raises(ValueError, match='requested_manifest.json drift'):
        e.lock(changed)
    # Self-consistent hash tampering must still fail the statistical verifier.
    envelope = e.read(arm)
    envelope['payload']['result']['p_value'] = .123
    envelope['sha256'] = e.value_hash(envelope['payload'])
    e.atomic(arm, envelope)
    with pytest.raises(ValueError, match='rank formula'):
        e.collect(m)


def test_retained_inference_failure_and_missing_outcome_bounds(tmp_path, monkeypatch):
    m = small(tmp_path)
    c, trial = next(e.jobs(m))
    values, ident, checksum = e.save_input(m, c, trial)
    def fail(*args, **kwargs):
        raise FloatingPointError('deliberate smoke failure')
    monkeypatch.setattr(e, 'selective_inference', fail)
    rows = []
    for method in e.METHODS:
        payload = e.execute(m, c, trial, method, values, ident, checksum)
        e.audit_arm(m, c, values, ident, checksum, method, payload)
        rows.append(payload['result'])
    frame = pd.DataFrame(rows)
    rates = e.summaries(frame, m)['rates.csv']
    assert rates.failures.eq(1).all()
    assert rates.all_attempt_lower.eq(0).all() and rates.all_attempt_upper.eq(1).all()
    assert rates.valid_only_fraction.isna().all()
    assert not rates.calibration_tolerance_supported.fillna(False).any()
    assert frame.p_value.isna().all() and frame.unadjusted_p_value.notna().all()


def test_input_and_summary_corruption(tmp_path):
    m = small(tmp_path)
    c, trial = next(e.jobs(m))
    e.save_input(m, c, trial)
    path = e.trial_dir(m, c, trial)/'input.npz'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='Corrupt input'):
        e.save_input(m, c, trial)


def test_archive_paths_duplicates_and_links(tmp_path):
    m = e.read_config(e.SMOKE_CONFIG)
    for name, link in [('wrong/file', False), (m['experiment']+'/../bad', False),
                       (m['experiment']+'/C:bad', False), (m['experiment']+'/link', True)]:
        archive = tmp_path/'bad.tar'
        with tarfile.open(archive, 'w') as tar:
            entry = tarfile.TarInfo(name)
            if link:
                entry.type = tarfile.SYMTYPE
                entry.linkname = '../bad'
            tar.addfile(entry)
        with pytest.raises(ValueError, match='Unsafe archive'):
            e.validate_archive(archive, m)


def test_same_trial_pairing_uses_differences_not_independent_error_bars():
    m = e.read_config(e.SMOKE_CONFIG)
    rows = []
    for trial in range(4):
        for method in e.METHODS:
            rows.append(dict(study='null',dataset='gaussian_design',condition='test',trial=trial,
                             method=method,p_value=.01 if trial == 0 else .5, unadjusted_p_value=.005))
    comparisons = e.paired_summaries(pd.DataFrame(rows), m)
    delta = comparisons[comparisons.metric == 'rejection_difference']
    assert delta.complete_pair_difference.eq(0).all()
    assert delta.conditional_mc_ci95_lower.eq(0).all() and delta.conditional_mc_ci95_upper.eq(0).all()
    # Zero observed discordance is not proof of zero population difference.
    assert delta.paired_missing_ci95_lower.lt(0).all()
    assert delta.paired_missing_ci95_upper.gt(0).all()


def test_production_mcmc_budget_on_fast_selectors_only(tmp_path):
    m = small(tmp_path)
    # One known-null input, keeping the full 511 x 20 construction. Tree/LIME
    # are exercised separately with the explicitly labelled reduced smoke.
    m['smoke_mcmc'] = {'replicas': 511, 'steps': 20}
    c, trial = next(e.jobs(m))
    values, ident, checksum = e.save_input(m, c, trial)
    for method in ('spline_mcmc','mutual_information_mcmc','spline_exact'):
        payload = e.execute(m, c, trial, method, values, ident, checksum)
        assert not payload['result']['inference_failed']
        e.audit_arm(m, c, values, ident, checksum, method, payload, replay=True)
        assert payload['result']['proposals'] == (0 if method == 'spline_exact' else 10240)
