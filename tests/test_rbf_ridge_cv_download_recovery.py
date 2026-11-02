"""Do not turn cross-platform numerical discrepancies into silent passes."""
from pathlib import Path
import copy
import io
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
import recover_rbf_ridge_cv_download as recovery


def row(p):
    return dict(dataset='synthetic', iteration=0, n=25, fraction=.1, beta=0.,
                arm='full_shapley', p_value=p, rank=4, elapsed_seconds=0.)


def test_mismatch_is_reported_with_original_tolerance_and_alpha_changes():
    differences = []
    counts = dict(replayed_tests=0, tests_with_changed_alpha_decisions=0, max_absolute_p_difference=0.)
    compare = recovery.replay_auditor([.01, .05, .1], differences, counts)
    compare([row(.29995946168817733)], [row(.30447785693210605)])
    assert len(differences) == 1
    assert differences[0]['changes']['p_value']['saved'] == .30447785693210605
    assert counts['max_absolute_p_difference'] > .0045
    assert not differences[0]['changed_alpha_decisions']
    compare([row(.051)], [row(.049)])
    assert differences[-1]['changed_alpha_decisions'] == [.05]
    assert counts['tests_with_changed_alpha_decisions'] == 1


def test_exact_match_passes_and_input_identity_stays_mandatory():
    differences = []
    counts = dict(replayed_tests=0, tests_with_changed_alpha_decisions=0, max_absolute_p_difference=0.)
    compare = recovery.replay_auditor([.05], differences, counts)
    saved = row(.5)
    before = copy.deepcopy(saved)
    compare([row(.5)], [saved])
    assert differences == [] and saved == before
    wrong = dict(saved, iteration=1)
    with pytest.raises(ValueError, match='identity'): compare([wrong], [saved])


def test_frozen_function_clone_does_not_replace_original_globals():
    namespace = dict(vars(recovery.study))
    original = recovery.study.compare_rows
    namespace['compare_rows'] = lambda *args: None
    cloned = recovery.clone(recovery.study.validate_receipt, namespace)
    assert cloned.__globals__['compare_rows'] is namespace['compare_rows']
    assert recovery.study.validate_receipt.__globals__['compare_rows'] is original


def test_archive_requires_matching_bytes_and_only_exempts_controller_lock(tmp_path):
    m = recovery.study.read_config()
    output = tmp_path/m['experiment']
    output.mkdir()
    (output/'COMPLETED').write_text('{}')
    recovery.study.core.atomic(output/'requested_manifest.json', m)
    (output/'controller.lock').write_bytes(b'')
    (output/'results.csv').write_text('original')
    archive = tmp_path/'cached.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(output, arcname=output.name)
    archive.with_name(archive.name+'.sha256').write_text(recovery.study.digest(archive)+'  cached.tar.gz\n')
    (output/'controller.lock').write_bytes(b'0')
    assert recovery.audit_archive(m, archive, output)['files_compared'] == 3
    (output/'results.csv').write_text('tampered')
    with pytest.raises(ValueError, match='bytes mismatch'):
        recovery.audit_archive(m, archive, output)


def test_csv_rounding_exception_is_limited_to_derived_confidence_limits():
    rounding = dict(fields=0, max_absolute_difference=0.)
    proxy = recovery.csv_proxy(rounding)
    def read(text): return list(proxy.DictReader(io.StringIO(text)))
    assert read('ci_lower,count\n0.07068650804529247,10\n') == read('ci_lower,count\n0.07068650804529249,10\n')
    assert rounding['fields'] == 1
    assert not (read('p_value\n0.07068650804529247\n') == read('p_value\n0.07068650804529249\n'))
    assert not (read('count\n10\n') == read('count\n11\n'))
    assert not (read('ci_lower\n0.07\n') == read('ci_lower\n0.071\n'))
