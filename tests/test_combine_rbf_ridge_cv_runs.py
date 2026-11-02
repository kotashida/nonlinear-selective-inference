"""Check disjoint trial selection, duplicate handling, provenance, and plot order."""
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('combine_rbf', ROOT/'examples/combine_rbf_ridge_cv_runs.py')
combine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combine)


def write_csv(path, rows):
    fields = sorted(rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixture_runs(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    (repo/'frozen.py').write_text('frozen source\n')
    manifests = [combine.read(ROOT/'benchmarks'/f'{name}.json') for name in (combine.ORIGINAL, combine.EXTENSION)]
    for m in manifests:
        m.update(sample_sizes=[25], datasets=['housing'], real_iterations=2)
    manifests[0].update(synthetic_fractions=[.1, .2], null_iterations=2, power_iterations=1, betas=[.25])
    manifests[1].update(basis_fractions=[.1, .2])
    sources = []
    for index, (name, m) in enumerate(zip((combine.ORIGINAL, combine.EXTENSION), manifests)):
        folder = repo/'outputs'/'benchmarks'/name
        (folder/'analysis').mkdir(parents=True)
        rows = []
        conditions = []
        if index == 0:
            conditions.extend(('synthetic', f, b, arm, 2 if b == 0 else 1)
                              for f, b, arm in itertools.product([.1, .2], [0., .25], m['targets']))
        conditions.extend(('housing', f, 0., arm, 2)
                          for f, arm in itertools.product([.1] if index == 0 else [.1, .2], m['targets']))
        # Match real-data ordering after filtering the extension's extra fraction.
        for iteration in range(2):
            for dataset, fraction, beta, arm, count in conditions:
                if iteration < count:
                    rows.append(dict(dataset=dataset, n=25, fraction=fraction, beta=beta,
                                     arm=arm, iteration=iteration, p_value=.02 if iteration == 0 else .2,
                                     elapsed_seconds=1.+index))
        write_csv(folder/'results.csv', rows)
        for filename, extras in (
            ('summary.csv', dict(alpha=.05, metric='rejection_rate', rate=.5)),
            ('lambda_summary.csv', dict(chosen_lambda=.01)),
            ('target_summary.csv', dict(target=0)),
            ('rank_summary.csv', dict(rank=1)),
        ):
            summary = [dict(dataset=d, n=25, fraction=f, beta=b, arm=a, **extras)
                       for d, f, b, a, count in conditions]
            write_csv(folder/filename, summary)
        combine.write_json(folder/'requested_manifest.json', m)
        checksum = combine.canonical_sha(m)
        done = dict(status='verified', errors=0, manifest_sha256=checksum,
                    total_tests=len(rows), receipts_sha256=f'receipt_hash_{index}',
                    export_sha256={f: combine.sha(folder/f) for f in combine.TABLES})
        combine.write_json(folder/'COMPLETED', done)
        combine.write_json(folder/'verification.json', dict(status='verified'))
        combine.write_json(folder/'provenance.json', dict(manifest_sha256=checksum,
                           sources={'frozen.py': combine.sha(repo/'frozen.py')}))
        audit_path = (repo/'output/audits/rbf_ridge_cv_download_20261030/report.json' if index == 0
                      else repo/'output/audits'/name/'report.json')
        audit_path.parent.mkdir(parents=True)
        combine.write_json(audit_path, dict(integrity_status='passed', numerical_replay_status='mismatches_detected',
                           manifest_sha256=checksum, receipts_sha256=done['receipts_sha256'],
                           total_tests=len(rows), replayed_tests=2, differing_replayed_tests=1,
                           tests_with_changed_alpha_decisions=0, max_absolute_p_difference=.001))
        if index == 0:
            (folder/'analysis/05_appendix_power_n75.png').write_bytes(b'original plot bytes')
            (folder/'analysis/index.html').write_text(
                '<html><head><title>RBF ridge/CV experiments</title></head><body>'
                '<h1>RBF ridge/CV experiments</h1><aside>Old source warning</aside>'
                '<h2>Last original plot</h2><img style="max-width:100%" src="05_appendix_power_n75.png">'
                '</body></html>')
        else:
            for suffix in ('.png', '.pdf'):
                (folder/'analysis'/('01_benchmark_basis_n25'+suffix)).write_bytes(b'extension plot bytes')
            (folder/'analysis/benchmark_basis_table.csv').write_text('dataset,n\nhousing,25\n')
        sources.append(dict(name=name, folder=folder, done=done, manifest=m))
    return repo, sources, manifests


def test_build_removes_overlapping_trials_and_appends_gallery_without_source_changes(fixture_runs):
    repo, sources, manifests = fixture_runs
    before = {str(path.relative_to(repo)): combine.sha(path) for source in sources
              for path in source['folder'].rglob('*') if path.is_file()}
    output = combine.build(repo)
    report = combine.read(output/'verification.json')
    assert report['total_tests'] == 20
    assert report['overlap_tests'] == 4
    assert report['selected_tests'] == {combine.ORIGINAL: 12, combine.EXTENSION: 8}
    assert report['omitted_tests'] == {combine.ORIGINAL: 4, combine.EXTENSION: 0}
    assert report['overlap_rows_identical_except_elapsed_seconds'] is True
    with (output/'results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len({(*combine.condition(r), r['iteration']) for r in rows}) == 20
    assert all(r['source_experiment'] == (combine.ORIGINAL if r['dataset'] == 'synthetic' else combine.EXTENSION)
               for r in rows)
    assert {r['p_value'] for r in rows} == {'0.02', '0.2'}
    gallery = (output/'analysis/index.html').read_text()
    images = re.findall(r'<img[^>]*src="([^"]+)"', gallery)
    assert images == ['05_appendix_power_n75.png', '06_appendix_benchmark_basis_n25.png']
    assert (output/'analysis/06_appendix_benchmark_basis_n25.png').read_bytes() == b'extension plot bytes'
    assert 'Old source warning' not in gallery
    assert 'local numerical replay has reported differences' in gallery
    assert combine.read(output/'COMPLETED')['numerical_replay_status'] == 'see_separate_source_audits'
    assert all(combine.sha(repo/name) == digest for name, digest in before.items())
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        combine.build(repo)


def test_mismatched_duplicate_reruns_are_not_silently_deduplicated(fixture_runs):
    repo, sources, manifests = fixture_runs
    path = sources[1]['folder']/'results.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    next(r for r in rows if float(r['fraction']) == .1)['p_value'] = '0.04'
    write_csv(path, rows)
    destination = repo/'staging'
    destination.mkdir()
    with pytest.raises(ValueError, match='Overlapping 10% benchmark trials differ'):
        combine.combine_results(sources, destination, combine.layout(*manifests))


def test_duplicate_trial_is_rejected(fixture_runs):
    repo, sources, manifests = fixture_runs
    path = sources[1]['folder']/'results.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    write_csv(path, rows+[rows[0]])
    sources[1]['done']['total_tests'] += 1
    destination = repo/'staging'
    destination.mkdir()
    with pytest.raises(ValueError, match='Duplicate combined trial'):
        combine.combine_results(sources, destination, combine.layout(*manifests))


def test_tampered_source_export_stops_before_combination(fixture_runs):
    repo, sources, _ = fixture_runs
    with (sources[0]['folder']/'summary.csv').open('a') as stream:
        stream.write('tampered\n')
    with pytest.raises(ValueError, match='Source export checksum mismatch'):
        combine.build(repo)
    assert not (repo/'outputs/benchmarks'/combine.COMBINED).exists()


def test_plotter_restyling_is_recorded_but_computational_source_drift_is_rejected(fixture_runs):
    repo, sources, _ = fixture_runs
    folder = sources[0]['folder']
    relative = 'examples/plot_rbf_ridge_cv_study.py'
    plotter = repo/relative
    plotter.parent.mkdir()
    plotter.write_text('old styling\n')
    provenance = combine.read(folder/'provenance.json')
    provenance['sources'][relative] = combine.sha(plotter)
    combine.write_json(folder/'provenance.json', provenance)
    plotter.write_text('new styling\n')
    audit = repo/'output/audits/rbf_ridge_cv_download_20261030/report.json'
    result = combine.inspect_source(repo, combine.ORIGINAL, audit)
    assert relative in result['plotter_differences']
    (repo/'frozen.py').write_text('changed computation\n')
    with pytest.raises(ValueError, match='Frozen source drift: frozen.py'):
        combine.inspect_source(repo, combine.ORIGINAL, audit)
