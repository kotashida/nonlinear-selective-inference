"""Match the October 24 paper/presentation figures using saved ridge/CV summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
import run_rbf_ridge_cv_study as study
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ORDER = ['concrete', 'heating_load', 'cooling_load', 'gas_turbine', 'housing', 'red_wine', 'white_wine', 'abalone']
NAMES = dict(zip(ORDER, ['Concrete', 'Heating load', 'Cooling load', 'Gas turbine', 'Housing', 'Red wine', 'White wine', 'Abalone']))
ARMS = ['selected_shapley', 'full_shapley']
LABELS = ['Selected-Features Shapley Effect', 'All-Features Shapley Effect']
COLORS = ['#2674a6', '#d58122']
NS = [25, 50, 75]
REFERENCE = ROOT/'outputs/benchmarks/rbf_shapley_realdata_and_null_cpu64_10000_20261024/analysis/paper_presentation'


def rate_text(value):
    text = f'{value:.3f}'
    return text[1:] if text.startswith('0.') else text


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create(source):
    m = study.core.read(source/'requested_manifest.json')
    done = study.core.read(source/'COMPLETED')
    study.core.require(study.core.sha(m) == done['manifest_sha256'] and done['status'] == 'verified', 'Completion/manifest mismatch')
    study.core.require(m['sample_sizes'] == NS and m['primary_fraction'] == .1, 'Unexpected primary design')
    study.core.require(study.digest(source/'summary.csv') == done['export_sha256']['summary.csv'], 'Summary checksum mismatch')
    with (source/'summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for key in ('n', 'trials', 'rejections'):
            r[key] = int(r[key])
        for key in ('fraction', 'alpha', 'beta', 'rate', 'ci_lower', 'ci_upper'):
            r[key] = float(r[key]) if r[key] else np.nan
    primary = [r for r in rows if r['fraction'] == .1 and r['alpha'] == .05]

    def row(dataset, n, arm, metric='rejection_rate', beta=0.):
        matched = [r for r in primary if r['dataset'] == dataset and r['n'] == n and r['arm'] == arm
                   and r['metric'] == metric and r['beta'] == beta]
        study.core.require(len(matched) == 1, f'Missing/duplicate summary: {dataset}/{n}/{arm}/{metric}/{beta}')
        r = matched[0]
        expected = m['real_iterations'] if dataset != 'synthetic' else (m['null_iterations'] if beta == 0 else m['power_iterations'])
        if metric in ('rejection_rate', 'targeting_probability', 'marginal_signal_detection'):
            study.core.require(r['trials'] == expected, 'Wrong summary denominator')
        if r['trials']:
            study.core.require(abs(r['rate']-r['rejections']/r['trials']) < 1e-14, 'Inconsistent rate/count')
        return r

    real = [row(d, n, a) for d in ORDER for n in NS for a in ARMS]
    null = [row('synthetic', n, a) for n in NS for a in ARMS]
    audit_path = ROOT/'output/audits/rbf_ridge_cv_download_20261030/report.json'
    audit = study.core.read(audit_path) if audit_path.exists() else None
    if audit:
        study.core.require(audit['integrity_status'] == 'passed' and audit['receipts_sha256'] == done['receipts_sha256'], 'Audit does not match these results')
    warning = ('Saved-result integrity passed. Numerical replay remains flagged: '
               f'{audit["differing_replayed_tests"]} of {audit["replayed_tests"]} checked tests differ; '
               f'maximum absolute p-value difference {audit["max_absolute_p_difference"]:.5f}. '
               'Figures use unchanged server p-values. No checked alpha=.01,.05,.10 decisions changed.'
               if audit and audit['numerical_replay_status'] != 'passed' else
               'Figures use saved server p-values; no new numerical reproducibility claim is made here.')
    out = source/'analysis/paper_presentation'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11,
                        'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 10,
                        'axes.spines.top': False, 'axes.spines.right': False,
                        'svg.fonttype': 'none', 'mathtext.fontset': 'dejavuserif'})
    figures = []

    def save(fig, name, title, caption):
        for ext in ('png', 'svg'):
            fig.savefig(out/f'{name}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures.append(dict(name=name, title=title, caption=caption))

    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.set_axis_off()
    xs = np.r_[.035, np.linspace(.14, .94, 8)]
    ax.text(.005, .99, 'Real-data rejection rates', fontsize=13, va='top', transform=ax.transAxes)
    ax.text(.005, .87, 'Each cell: Selected-Features Shapley Effect / All-Features Shapley Effect.',
            fontsize=9.5, va='top', transform=ax.transAxes)
    for y in (.72, .55, .12):
        ax.plot([0, 1], [y, y], color='black', lw=1 if y != .55 else .65, transform=ax.transAxes)
    for x, label in zip(xs, ['n']+[NAMES[d] for d in ORDER]):
        ax.text(x, .635, label, ha='center', va='center', fontsize=11, transform=ax.transAxes)
    compact = []
    latex = [r'\begin{tabular}{r'+'c'*8+'}', r'\hline', 'n & '+' & '.join(NAMES[d] for d in ORDER)+r' \\', r'\hline']
    for n, y in zip(NS, (.46, .33, .20)):
        ax.text(xs[0], y, str(n), ha='center', va='center', transform=ax.transAxes)
        record, cells = {'n': n}, []
        for j, dataset in enumerate(ORDER):
            a, b = [row(dataset, n, arm)['rate'] for arm in ARMS]
            aa, bb = rate_text(a), rate_text(b)
            record[dataset] = aa+'/'+bb
            left = r'\mathbf{'+aa+'}' if a >= b else aa
            right = r'\mathbf{'+bb+'}' if b >= a else bb
            ax.text(xs[j+1], y, '$'+left+r'\,/\,'+right+'$', ha='center', va='center', fontsize=11.5, transform=ax.transAxes)
            cells.append((r'\textbf{'+aa+'}' if a >= b else aa)+'/'+(r'\textbf{'+bb+'}' if b >= a else bb))
        compact.append(record)
        latex.append(str(n)+' & '+' & '.join(cells)+r' \\')
    latex += [r'\hline', r'\end{tabular}', '% Penalized-objective Shapley effects; real-data truth unknown; sigma=1 working scale.']
    (out/'table1.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')
    write_csv(out/'table1_compact.csv', compact)
    write_csv(out/'table1_detailed.csv', real)
    save(fig, '01_table1', 'Real-data rejection rates',
         'n=25,50,75; 10% centers; 10,000 tests per dataset/n/effect. Larger rates are bold, with both bold in ties. '
         'The effects test different hypotheses; bold does not establish better feature identification.')

    for index, n in enumerate(NS, 2):
        fig, ax = plt.subplots(figsize=(14, 5), layout='constrained')
        for j, (arm, color, label) in enumerate(zip(ARMS, COLORS, LABELS)):
            ax.plot(np.arange(8)+(j-.5)*.14, [row(d, n, arm)['rate'] for d in ORDER],
                    linestyle='none', marker='x', markersize=7, markeredgewidth=1.5, color=color, label=label)
        ax.set(xticks=range(8), xticklabels=[NAMES[d] for d in ORDER], ylim=(0, 1.04),
               ylabel='Working-model rejection fraction',
               title=f'Observed-response benchmarks (n = {n}, alpha = 0.05; 10,000 subsamples/task)')
        ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.12), fontsize=10)
        save(fig, f'{index:02}_observed_response_n{n}', f'Observed-response benchmarks: n = {n}',
             'Offset x markers, dataset order, serif typography and 0-1 scale match the reference. '
             'Pointwise confidence intervals and counts are in the detailed table. Real-data truth is unknown; sigma=1 is a working scale.')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for ax, dataset in zip(axes.flat, ORDER):
        for arm, color, label in zip(ARMS, COLORS, LABELS):
            group = [row(dataset, n, arm) for n in NS]
            ax.plot(NS, [r['rate'] for r in group], 'o-', color=color, label=label, lw=1.6, ms=4)
            ax.fill_between(NS, [r['ci_lower'] for r in group], [r['ci_upper'] for r in group], color=color, alpha=.12)
        ax.set(title=NAMES[dataset], xticks=NS, ylim=(0, 1.04), xlabel='Sample size (n)')
        ax.grid(alpha=.17)
    for ax in axes[:, 0]: ax.set_ylabel('Selective rejection rate')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle('Selective rejection rates as sample size increases', fontsize=16)
    fig.tight_layout(rect=(0, .065, 1, .94))
    save(fig, '05_sample_size_curves', 'Sample-size trends with confidence intervals',
         'Pointwise 95% exact binomial intervals. Samples are nested within repetition; the effects share the actual target at each n. '
         'These working-model rejection rates are not known power or Type I error.')

    write_csv(out/'type1_error_rates.csv', null)

    def null_panel(ax, n, short=False):
        for j, color in enumerate(COLORS):
            r = row('synthetic', n, ARMS[j])
            ax.errorbar(j, r['rate'], yerr=[[r['rate']-r['ci_lower']], [r['ci_upper']-r['rate']]],
                        fmt='x', markersize=9, markeredgewidth=1.8, capsize=5, color=color)
        ax.axhline(.05, color='#888888', ls='--', lw=.9)
        ax.set(xticks=[0, 1], xticklabels=['Selected-Features\nShapley Effect', 'All-Features\nShapley Effect'],
               xlim=(-.6, 1.6), ylim=(0, .10), ylabel='Type I error rate',
               title=f'n = {n}' if short else f'Type I error rate (n = {n})')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, n in zip(axes, NS): null_panel(ax, n, short=True)
    fig.suptitle('Type I error rate', fontsize=16)
    caption = ('Fresh iid standard Gaussian features and Y=N(0,I); p=4, k=2, known sigma=1, alpha=.05. '
               '10,000 trials per n/effect; 10% centers, five-fold CV. Markers and pointwise 95% exact intervals match the reference style.')
    save(fig, '06_type1_error_rate', 'Type I error rate: synthetic global null', caption)
    for letter, n in zip('abc', NS):
        fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
        null_panel(ax, n)
        save(fig, f'06{letter}_type1_error_rate_n{n}', f'Type I error rate: n = {n}', caption)

    metrics = ['targeting_probability', 'conditional_signal_detection', 'marginal_signal_detection']
    metric_titles = ['Probability of targeting', 'Conditional power', 'Marginal detection power']
    power_rows = []
    for index, n in enumerate(NS, 7):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
        for ax, metric, title in zip(axes, metrics, metric_titles):
            for arm, color, label in zip(ARMS, COLORS, LABELS):
                group = [row('synthetic', n, arm, metric, beta) for beta in [0., *m['betas']]]
                power_rows.extend(group)
                ax.plot([r['beta'] for r in group], [r['rate'] for r in group], 'x-', color=color, label=label, ms=6, lw=1.5)
                ax.fill_between([r['beta'] for r in group], [r['ci_lower'] for r in group], [r['ci_upper'] for r in group], color=color, alpha=.1)
            ax.set(title=title, xlabel='Signal strength (beta)', xticks=[0, .25, .5, .75, 1], ylim=(0, 1.04))
            ax.grid(alpha=.17)
        axes[0].set_ylabel('Probability')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .005))
        fig.suptitle(f'RBF ridge/CV power (n = {n}, p = 4, k = 2, alpha = 0.05)', fontsize=14)
        fig.tight_layout(rect=(0, .08, 1, .93))
        save(fig, f'{index:02}_power_n{n}', f'Power experiment: n = {n}',
             '10% centers; 1,000 matched trials per positive beta; beta=0 uses the 10,000-trial global-null baseline. '
             'Signal: standardized x + 0.5(x^2-1) in feature 0. Shading gives pointwise 95% exact intervals. '
             'The two targeting curves coincide because selection and the actual target are shared.')
    write_csv(out/'power_detailed.csv', power_rows)

    intro = ('RBF ridge/CV, 10% sampled centers per feature, median-distance widths, five-fold CV; '
             'n=25,50,75 and alpha=.05. The main real-data table contains 480,000 tests. '
             'Both effects use penalized-objective coalition improvements and joint CV/top-k conditioning. '
             'The previous OLS RSS effects are a different estimand.')
    sections = []
    for item in figures:
        sections.append('<section><h2>'+html.escape(item['title'])+'</h2><p>'+html.escape(item['caption'])+
                        '</p><a href="'+item['name']+'.svg"><img src="'+item['name']+'.png" alt="'+html.escape(item['title'])+'"></a></section>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>RBF ridge/CV: paper and presentation formats</title><style>'
            'body{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}'
            'section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}'
            'aside{background:#fff3cd;padding:16px;border:1px solid #c69c33}</style>'
            '<h1>RBF ridge/CV: paper and presentation figures</h1><p>'+html.escape(intro)+'</p><aside>'+html.escape(warning)+'</aside>'
            '<p><a href="table1_compact.csv">Compact Table 1 CSV</a> | <a href="table1_detailed.csv">Detailed table</a> | '
            '<a href="table1.tex">LaTeX table</a></p>'+''.join(sections)+'</html>')
    (out/'index.html').write_text(page, encoding='utf-8')
    (out/'README.md').write_text('# Paper and presentation figures\n\n'+intro+'\n\n'+warning+'\n\n'+
        '\n\n'.join(f'## {v["title"]}\n\n{v["caption"]}\n\n![{v["title"]}]({v["name"]}.png)' for v in figures)+'\n', encoding='utf-8')
    provenance = dict(manifest_sha256=done['manifest_sha256'], receipts_sha256=done['receipts_sha256'],
                      inputs={'summary.csv': study.digest(source/'summary.csv')},
                      script_sha256=study.digest(__file__), reference=str(REFERENCE),
                      primary_fraction=.1, sample_sizes=NS, numerical_replay_note=warning,
                      figures=figures, outputs={p.name: study.digest(p) for p in out.iterdir() if p.is_file() and p.name != 'analysis_provenance.json'})
    study.core.atomic(out/'analysis_provenance.json', provenance)
    print(f'Created {len(figures)} PNG/SVG figures and tables: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'outputs/benchmarks'/study.NAME)
    args = parser.parse_args()
    create(args.input.resolve())
