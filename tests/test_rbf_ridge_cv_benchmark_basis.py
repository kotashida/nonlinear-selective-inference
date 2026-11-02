"""Benchmark sensitivity coverage, reference matching, and result integrity."""
import copy
import csv
import io
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
sys.path.insert(0, str(ROOT/'examples'))
import run_rbf_ridge_cv_benchmark_basis as study
import run_rbf_ridge_cv_study as reference
from plot_rbf_ridge_cv_benchmark_basis import plot


def smoke():
    return study.read_config(ROOT/'benchmarks'/f'{study.SMOKE}.json')


def test_frozen_design_counts_and_exact_reference_rows(tmp_path):
    m = study.read_config()
    assert study.expected_tests(m) == 1440000
    assert study.expected_tests(smoke()) == 72
    assert m['workers'] == 32 and m['threads_per_worker'] == 1
    assert [study.settings(m, job) for job in study.jobs(m)] == [(m['basis_fractions'], [0.])]*16000
    assert {job[:2] for job in study.jobs(m)} == {('real', d) for d in m['datasets']}
    old = reference.read_config()
    for dataset in m['datasets']:
        small = study.inputs(m, dataset, 1, 25)
        large = study.inputs(m, dataset, 1, 75)
        assert small == reference.inputs(old, dataset, 1, 25)
        assert set(small['indices']) <= set(large['indices'])
        X, y = study.load_data(str(ROOT/m['input_path']), m['input_sha256'])[dataset]
        np.testing.assert_array_equal(small['X'], X[small['indices']])
        np.testing.assert_array_equal(small['y_base'], y[small['indices']])
        assert len(small['X'][0]) == X.shape[1] > 4
    bad = tmp_path/'bad.json'
    study.core.atomic(bad, dict(m, basis_fractions=[.1]))
    with pytest.raises(ValueError, match='Manifest changed'):
        study.read_config(bad)


def test_matched_fractions_and_reference_10_percent_inference():
    m = smoke()
    saved = study.inputs(m, 'housing', 0, 25)
    rows = study.evaluate(m, saved, m['basis_fractions'], [0.])
    assert len(rows) == 6
    assert len({r['input_sha256'] for r in rows}) == 1
    assert len({str(r['folds']) for r in rows}) == 1
    assert len({r['auxiliary_u'] for r in rows}) == 1
    for fraction in m['basis_fractions']:
        pair = [r for r in rows if r['fraction'] == fraction]
        assert pair[0]['target'] == pair[1]['target']
        assert all(r['null_true'] is None and r['projected_mean_norm'] is None for r in pair)
        assert all(r['centers_per_feature'] == int(np.ceil(25*fraction)) for r in pair)
    centers = [r['basis_centers'] for r in rows[::2]]
    assert centers[0] == centers[-1][:len(centers[0])]
    assert centers[1] == centers[-1][:len(centers[1])]
    for fold in range(m['cv_folds']):
        for row in rows[::2]:
            prefix = row['fold_centers'][fold]
            assert prefix == rows[-1]['fold_centers'][fold][:len(prefix)]
    study.compare_rows([r for r in rows if r['fraction'] == .1],
                       reference.evaluate(reference.read_config(), saved, [.1], [0.]))
    with pytest.raises(ValueError, match='beta sweep'):
        study.evaluate(m, saved, [.1], [.5])


def test_smoke_receipts_resume_replay_exports_plots_and_tampering(tmp_path):
    m = smoke()
    study.setup_output(m, tmp_path)
    for job in study.jobs(m):
        study.run_job(m, job, tmp_path)
    prior = {p.name: p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}
    study.setup_output(m, tmp_path)
    report = study.verify(m, tmp_path, export=True)
    assert report['total_tests'] == 72
    assert study.verify(m, tmp_path)['status'] == 'verified'
    assert prior == {p.name: p.read_bytes() for p in (tmp_path/'jobs').glob('*.json')}
    with (tmp_path/'summary.csv').open(newline='') as stream:
        summary = list(csv.DictReader(stream))
    assert {r['metric'] for r in summary} == {'rejection_rate'}
    assert len(summary) == 108
    output = plot(tmp_path)
    assert len(list(output.glob('*.png'))) == 3
    assert len(list(output.glob('*.pdf'))) == 3
    assert 'truth is unknown' in (output/'index.html').read_text()
    job = study.jobs(m)[0]
    receipt = study.core.read(tmp_path/'jobs'/study.job_name(job))
    broken = copy.deepcopy(receipt)
    broken['rows'].pop()
    broken['rows_sha256'] = study.core.sha(broken['rows'])
    with pytest.raises(ValueError, match='coverage'):
        study.validate_receipt(m, job, broken)
    broken = copy.deepcopy(receipt)
    broken['rows'][0]['p_value'] = .123
    with pytest.raises(ValueError, match='checksum'):
        study.validate_receipt(m, job, broken)
    broken['rows_sha256'] = study.core.sha(broken['rows'])
    with pytest.raises(ValueError):
        study.validate_receipt(m, job, broken, recompute=True)
    with (tmp_path/'summary.csv').open('a') as stream:
        stream.write('corrupted\n')
    with pytest.raises(ValueError, match='checksum'):
        study.verify(m, tmp_path)


def test_launcher_sources_and_process_download_guards():
    text = (ROOT/'examples/rbf_ridge_cv_benchmark_basis_cpu32.ps1').read_text()
    for path in study.PACKAGED:
        assert (ROOT/path).is_file()
        assert path in text
    for token in ('launcher.startticks', 'kill -TERM -- "-$PID"', 'flock -n 8',
                  'flock -n 7', 'Refusing to overwrite', 'validate-archive',
                  '[r]un_rbf_ridge_cv_benchmark_basis.py run',
                  '[r]un_rbf_ridge_cv_study.py run', 'scp', 'sha256sum -c'):
        assert token in text


def test_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path/'unsafe.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo('../escaped')
        member.size = 1
        stream.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe archive'):
        study.core.validate_archive(smoke(), archive)
