"""Combine frozen RBF runs without reusing the overlapping real-data trials.

Stream original synthetic rows plus extension benchmark rows into a separate
derived dataset. Copy existing plots, appending benchmark basis sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = 'rbf_ridge_cv_n25_50_75_cpu32_20261028'
EXTENSION = 'rbf_ridge_cv_benchmark_basis_cpu32_20261031'
COMBINED = 'rbf_ridge_cv_combined_n25_50_75_20261101'
TABLES = ('results.csv', 'summary.csv', 'lambda_summary.csv', 'target_summary.csv', 'rank_summary.csv')
COMMON = ('sample_sizes', 'k', 'sigma', 'primary_alpha', 'alpha_levels', 'real_iterations',
          'datasets', 'seed', 'targets', 'lambdas', 'cv_folds', 'cv_score', 'objective',
          'coalition_score', 'selection_event', 'centers', 'widths', 'rank_tolerance',
          'null_tolerance', 'input_path', 'input_sha256', 'required_versions')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def inspect_source(repo, name, audit_path):
    folder = repo/'outputs'/'benchmarks'/name
    manifest = read(folder/'requested_manifest.json')
    done, provenance, audit = (read(folder/'COMPLETED'), read(folder/'provenance.json'), read(audit_path))
    require(manifest['experiment'] == name, 'Source experiment identity mismatch')
    require(done['status'] == 'verified' and done['errors'] == 0, 'Source completion failed')
    identity = canonical_sha(manifest)
    require(done['manifest_sha256'] == provenance['manifest_sha256'] == identity, 'Source manifest hash mismatch')
    require(audit['integrity_status'] == 'passed' and audit['manifest_sha256'] == identity
            and audit['receipts_sha256'] == done['receipts_sha256']
            and audit['total_tests'] == done['total_tests'], 'Source audit identity mismatch')
    require(set(done['export_sha256']) == set(TABLES), 'Source export coverage mismatch')
    plotter_differences = {}
    for relative, expected in provenance['sources'].items():
        actual = sha(repo/relative)
        if actual != expected:
            # Existing presentation plots may have been restyled after the run.
            # This builder copies those files; it never executes either plotter.
            require(relative in ('examples/plot_rbf_ridge_cv_study.py',
                                 'examples/plot_rbf_ridge_cv_benchmark_basis.py'),
                    f'Frozen source drift: {relative}')
            plotter_differences[relative] = dict(recorded_sha256=expected, current_sha256=actual,
                                                use='Existing plot files copied; plotter not executed')
    for filename, expected in done['export_sha256'].items():
        require(sha(folder/filename) == expected, f'Source export checksum mismatch: {name}/{filename}')
    return dict(name=name, folder=folder, manifest=manifest, done=done, provenance=provenance,
                audit=audit, audit_path=audit_path, plotter_differences=plotter_differences)


def condition(row):
    return row['dataset'], int(row['n']), float(row['fraction']), float(row['beta']), row['arm']


def layout(original, extension):
    result = {}
    for n in original['sample_sizes']:
        for fraction in original['synthetic_fractions']:
            for beta in [0., *original['betas']]:
                for arm in original['targets']:
                    result[('synthetic', n, fraction, beta, arm)] = (
                        original['null_iterations'] if beta == 0 else original['power_iterations'])
    for dataset in extension['datasets']:
        for n in extension['sample_sizes']:
            for fraction in extension['basis_fractions']:
                for arm in extension['targets']:
                    result[(dataset, n, fraction, 0., arm)] = extension['real_iterations']
    return result


def hash_row(digest, row, fields):
    # Cell strings, including p-values, must match exactly; execution time is
    # intentionally excluded because these are separately executed matched runs.
    digest.update(json.dumps([row[k] for k in fields], separators=(',', ':')).encode())
    digest.update(b'\n')


def combine_results(sources, destination, expected):
    seen = {key: bytearray(count) for key, count in expected.items()}
    overlap_hashes, overlap_counts, selected, omitted = [], [], {}, {}
    fields = None
    total = 0
    with (destination/'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for source in sources:
            name = source['name']
            digest, overlap, kept, dropped, read_count = hashlib.sha256(), 0, 0, 0, 0
            with (source['folder']/'results.csv').open(newline='', encoding='utf-8') as input_stream:
                reader = csv.DictReader(input_stream)
                require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'Invalid CSV schema')
                if fields is None:
                    fields = reader.fieldnames
                    require('source_experiment' not in fields, 'Already combined input')
                    writer = csv.DictWriter(stream, fieldnames=[*fields, 'source_experiment'])
                    writer.writeheader()
                require(reader.fieldnames == fields, 'Source trial schemas differ')
                overlap_fields = [key for key in fields if key != 'elapsed_seconds']
                for row in reader:
                    read_count += 1
                    key = condition(row)
                    if row['dataset'] != 'synthetic' and float(row['fraction']) == .1:
                        hash_row(digest, row, overlap_fields)
                        overlap += 1
                    use = row['dataset'] == 'synthetic' if name == ORIGINAL else row['dataset'] != 'synthetic'
                    if not use:
                        dropped += 1
                        continue
                    require(key in seen, f'Unexpected condition: {key}')
                    iteration = int(row['iteration'])
                    flags = seen[key]
                    require(0 <= iteration < len(flags), 'Unexpected repetition index')
                    require(flags[iteration] == 0, 'Duplicate combined trial')
                    flags[iteration] = 1
                    writer.writerow(dict(row, source_experiment=name))
                    kept += 1
                    total += 1
                    if total % 250000 == 0:
                        print(f'Combined {total:,} unique tests', flush=True)
            require(read_count == source['done']['total_tests'], 'Source trial count mismatch')
            overlap_hashes.append(digest.hexdigest())
            overlap_counts.append(overlap)
            selected[name], omitted[name] = kept, dropped
    require(all(all(flags) for flags in seen.values()), 'Incomplete combined condition coverage')
    require(overlap_counts[0] == overlap_counts[1] and overlap_hashes[0] == overlap_hashes[1],
            'Overlapping 10% benchmark trials differ beyond execution time')
    return dict(total_tests=total, selected_tests=selected, omitted_tests=omitted,
                overlap_tests=overlap_counts[0], overlap_rows_identical_except_elapsed_seconds=True,
                overlap_content_sha256=overlap_hashes[0], condition_count=len(seen))


def combine_table(filename, sources, destination, expected):
    extra_keys = {'summary.csv': ('alpha', 'metric'), 'lambda_summary.csv': ('chosen_lambda',),
                  'target_summary.csv': ('target',), 'rank_summary.csv': ('rank',)}[filename]
    seen, output_count, fields = set(), 0, None
    with (destination/filename).open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for source in sources:
            with (source['folder']/filename).open(newline='', encoding='utf-8') as input_stream:
                reader = csv.DictReader(input_stream)
                if fields is None:
                    fields = reader.fieldnames
                    writer = csv.DictWriter(stream, fieldnames=[*fields, 'source_experiment'])
                    writer.writeheader()
                require(fields == reader.fieldnames, f'Summary schema mismatch: {filename}')
                for row in reader:
                    use = row['dataset'] == 'synthetic' if source['name'] == ORIGINAL else row['dataset'] != 'synthetic'
                    if not use:
                        continue
                    base = condition(row)
                    require(base in expected, 'Unexpected summary condition')
                    key = (*base, *(row[k] for k in extra_keys))
                    require(key not in seen, f'Duplicate derived row: {filename}')
                    seen.add(key)
                    writer.writerow(dict(row, source_experiment=source['name']))
                    output_count += 1
    return output_count


def copy_records(sources, destination):
    for source in sources:
        folder = destination/'source_records'/source['name']
        folder.mkdir(parents=True)
        for name in ('requested_manifest.json', 'provenance.json', 'verification.json', 'COMPLETED'):
            shutil.copyfile(source['folder']/name, folder/name)
        shutil.copyfile(source['audit_path'], folder/'portability_audit.json')


def copy_gallery(sources, destination, report):
    original, extension = sources
    output = destination/'analysis'
    shutil.copytree(original['folder']/'analysis', output)
    content = (output/'index.html').read_text(encoding='utf-8')
    # Keep the source figures in their existing order and replace the old-run
    # warning with source-specific audit links for the derived combination.
    content = re.sub(r'<aside\b.*?</aside>', '', content, flags=re.S)
    content = content.replace('<title>RBF ridge/CV experiments</title>', '<title>Combined RBF ridge/CV experiments</title>')
    content = content.replace('<h1>RBF ridge/CV experiments</h1>', '<h1>Combined RBF ridge/CV experiments</h1>')
    notice = ('Combined saved server results: '+format(report['total_tests'], ',')+
              ' unique tests. Synthetic results come from the original run; benchmark results '
              'come from the extension. The overlapping 10% benchmark reruns are included once. '
              'Both source integrity audits passed; local numerical replay has reported differences. '
              'Server p-values are unchanged.')
    links = ' '.join(f'<a href="../source_records/{s["name"]}/portability_audit.json">{html.escape(s["name"])} audit</a>' for s in sources)
    note = '<aside style="padding:16px;background:#fff3cd;border:1px solid #c69c33;margin:16px 0">'+html.escape(notice)+'<p>'+links+'</p></aside>'
    content = content.replace('</h1>', '</h1>'+note+
                              '<p><a href="../combined_manifest.json">Combined provenance</a> | '
                              '<a href="../verification.json">Combination checks</a> | '
                              '<a href="../summary.csv">Summary CSV</a></p>', 1)
    extra = ''
    appended = []
    for n in extension['manifest']['sample_sizes']:
        source_name, name = f'01_benchmark_basis_n{n}', f'06_appendix_benchmark_basis_n{n}'
        for suffix in ('.png', '.pdf'):
            shutil.copyfile(extension['folder']/'analysis'/(source_name+suffix), output/(name+suffix))
        extra += (f'<h2>Appendix: benchmark basis-count sensitivity, n={n}</h2>'
                  f'<p><a href="{name}.pdf">PDF</a></p>'
                  f'<img style="max-width:100%" src="{name}.png">')
        appended.append(name)
    anchor = list(re.finditer(r'<img\b[^>]*\bsrc=[\"\']05_appendix_power_n75\.png[\"\'][^>]*>', content))
    require(len(anchor) == 1, 'Expected one final synthetic appendix plot in the original gallery')
    position = anchor[0].end()
    content = content[:position]+extra+content[position:]
    (output/'index.html').write_text(content, encoding='utf-8')
    (output/'PORTABILITY.txt').write_text(notice+'\nSee ../source_records/ for separate source audit reports.\n', encoding='utf-8')
    shutil.copyfile(extension['folder']/'analysis'/'benchmark_basis_table.csv', output/'benchmark_basis_table.csv')
    for image in re.findall(r'<img\b[^>]*\bsrc=[\"\']([^\"\']+)[\"\']', content):
        require((output/image).is_file(), f'Missing gallery image: {image}')
    return appended


def build(repo=ROOT):
    repo = Path(repo)
    target = repo/'outputs'/'benchmarks'/COMBINED
    require(not target.exists(), f'Refusing to overwrite existing combined results: {target}')
    audit_paths = (repo/'output/audits/rbf_ridge_cv_download_20261030/report.json',
                   repo/'output/audits'/EXTENSION/'report.json')
    print('Checking frozen source files, source audit identities, and export hashes.', flush=True)
    sources = [inspect_source(repo, name, audit) for name, audit in zip((ORIGINAL, EXTENSION), audit_paths)]
    original, extension = [source['manifest'] for source in sources]
    for key in COMMON:
        require(original[key] == extension[key], f'Incompatible settings: {key}')
    require(original['synthetic_fractions'] == extension['basis_fractions'], 'Incompatible basis fractions')
    expected = layout(original, extension)
    staging = repo/'tmp'/'combined_rbf_runs'/uuid.uuid4().hex
    staging.mkdir(parents=True)
    report = combine_results(sources, staging, expected)
    report['derived_table_rows'] = {name: combine_table(name, sources, staging, expected) for name in TABLES[1:]}
    report.update(status='combined_exports_verified', dataset_kind='derived_combination',
                  new_experiment_run=False, saved_server_pvalues_unchanged=True,
                  plotter_source_differences={s['name']: s['plotter_differences'] for s in sources
                                              if s['plotter_differences']},
                  numerical_replay_status='see_separate_source_audits',
                  source_audits={s['name']: {k: s['audit'][k] for k in
                                ('integrity_status', 'numerical_replay_status', 'replayed_tests',
                                 'differing_replayed_tests', 'tests_with_changed_alpha_decisions',
                                 'max_absolute_p_difference')} for s in sources})
    copy_records(sources, staging)
    report['appended_plots'] = copy_gallery(sources, staging, report)
    report['plot_sha256'] = {str(path.relative_to(staging)): sha(path) for path in (staging/'analysis').iterdir()
                            if path.suffix in ('.png', '.pdf')}
    manifest = dict(schema=1, name=COMBINED, kind='derived_combination',
                    selection='original dataset=synthetic; extension dataset!=synthetic',
                    total_tests=report['total_tests'], source_column='source_experiment',
                    sources=[dict(experiment=s['name'], folder=str(s['folder'].relative_to(repo)),
                                  manifest_sha256=canonical_sha(s['manifest']),
                                  receipts_sha256=s['done']['receipts_sha256'],
                                  completed_sha256=sha(s['folder']/'COMPLETED'),
                                  export_sha256=s['done']['export_sha256'],
                                  audit_sha256=sha(s['audit_path'])) for s in sources],
                    note='Source receipt directories remain in their original runs. '
                         'This is a derived dataset, not a new numerical-verification pass.')
    write_json(staging/'combined_manifest.json', manifest)
    report['export_sha256'] = {name: sha(staging/name) for name in TABLES}
    report['builder_sha256'] = sha(__file__)
    write_json(staging/'verification.json', report)
    write_json(staging/'COMPLETED', dict(kind='derived_combination', status='combined_exports_verified',
                                       total_tests=report['total_tests'], export_sha256=report['export_sha256'],
                                       numerical_replay_status='see_separate_source_audits'))
    (staging/'README.md').write_text(
        '# Combined RBF ridge/CV results\n\n'
        f'{report["total_tests"]:,} tests: original synthetic trials plus extension benchmark trials. '
        f'{report["overlap_tests"]:,} overlapping original benchmark trials were omitted.\n\n'
        'Each CSV row retains its original cell values and adds source_experiment. '
        'The overlapping 10% benchmark trials agree exactly except for elapsed_seconds. '
        'The original folders and backing job receipts are unchanged.\n\n'
        'Open analysis/index.html. The three 06_appendix_benchmark_basis figures follow '
        '05_appendix_power_n75. Synthetic calibration/power and descriptive benchmark rejection '
        'rates retain their separate meanings.\n\n'
        'Source integrity checks passed; numerical replay differences remain documented in '
        'source_records/. No replay differences were converted into passes. '
        'See combined_manifest.json and verification.json for provenance and merge checks.\n', encoding='utf-8')
    require(not target.exists(), 'Combined output appeared during build; staging is preserved')
    staging.rename(target)
    print(f'Combined {report["total_tests"]:,} tests. Gallery: {target/"analysis/index.html"}', flush=True)
    return target


if __name__ == '__main__':
    build()
