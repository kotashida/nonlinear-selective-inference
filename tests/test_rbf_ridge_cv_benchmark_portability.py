"""Integrity remains strict when numerical replay is audited separately."""
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
import audit_rbf_ridge_cv_benchmark_download as audit


@pytest.fixture(scope='module')
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('benchmark_portability')
    m = audit.study.read_config(ROOT/'benchmarks'/f'{audit.study.SMOKE}.json')
    output = root/m['experiment']
    output.mkdir()
    audit.study.setup_output(m, output)
    for job in audit.study.jobs(m):
        audit.study.run_job(m, job, output)
    audit.study.verify(m, output, export=True)
    archive = root/'results.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(output, arcname=output.name)
    return m, output, archive


def test_streamed_aggregation_matches_original_exports(smoke_run):
    m, output, _ = smoke_run
    groups = {}
    for job in audit.study.jobs(m):
        receipt = audit.study.core.read(output/'jobs'/audit.study.job_name(job))
        audit.accumulate(groups, m, receipt['rows'])
    for name, expected in audit.derived_exports(m, groups).items():
        with (output/name).open(newline='') as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(expected)
        for left, right in zip(saved, expected):
            audit.compare_csv_row(left, right)


def test_integrity_pass_does_not_mislabel_replay_discrepancies(smoke_run, monkeypatch):
    m, output, archive = smoke_run
    monkeypatch.setattr(audit, 'ProcessPoolExecutor', ThreadPoolExecutor)
    original = audit.validate_job

    def replay_difference(*args):
        rows, checksum, differences, counts = original(*args)
        if counts['replayed_tests']:
            differences.append(dict(dataset=args[1][1], changes={'log_selection_mass': dict(local=-4., saved=-5.)}))
        return rows, checksum, differences, counts

    monkeypatch.setattr(audit, 'validate_job', replay_difference)
    report = audit.audit(m, output, archive, workers=1)
    assert report['integrity_status'] == 'passed'
    assert report['total_tests'] == 72
    assert report['replayed_tests'] == 72
    assert report['numerical_replay_status'] == 'mismatches_detected'
    assert report['differing_replayed_tests'] == 4
    assert report['archive_integrity']['status'] == 'passed'
    assert report['saved_server_results_unchanged'] is True


def test_export_tampering_still_fails(smoke_run):
    m, output, archive = smoke_run
    path = output/'summary.csv'
    original = path.read_bytes()
    try:
        path.write_bytes(original+b'corrupt\n')
        with pytest.raises(ValueError, match='Export checksum mismatch'):
            audit.audit(m, output, archive, workers=1)
    finally:
        path.write_bytes(original)


def test_only_last_bit_confidence_limits_may_differ():
    rounding = dict(fields=0, max_absolute_difference=0.)
    audit.compare_csv_row(dict(ci_lower='0.07068650804529249', count='10'),
                          dict(ci_lower=.07068650804529247, count=10), rounding)
    assert rounding['fields'] == 1
    for key in ('p_value', 'count'):
        with pytest.raises(ValueError, match='Derived export mismatch'):
            audit.compare_csv_row({key: '0.07068650804529249'},
                                  {key: .07068650804529247}, rounding)
    with pytest.raises(ValueError, match='Derived export mismatch'):
        audit.compare_csv_row(dict(ci_lower='0.071'), dict(ci_lower=.07), rounding)
