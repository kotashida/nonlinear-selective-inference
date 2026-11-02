"""Paper-style Type I error figures from the downloaded, verified null tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'outputs/benchmarks/selected_five_null_fpr_10000_cpu32_20260928'
METHODS = ['shap_ais', 'lime_40_ais', 'spline_screening_ais',
           'spline_screening_exact', 'mutual_information_kde_2p5_ais']
LABELS = ['SHAP (AIS)', 'LIME (AIS)', 'Spline (AIS)', 'Spline (exact)', 'KDE-MI (AIS)']
COLORS = ['#2878B5', '#E78B27', '#3C9B58', '#8866AF', '#23A7A1']


def bounds(k, n):
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k)))


def rates(frame, alpha, column='p_value'):
    rows = []
    for method, label in zip(METHODS, LABELS):
        group = frame[frame.method.eq(method)]
        p = pd.to_numeric(group[column], errors='coerce')
        valid = np.isfinite(p) & p.between(0, 1)
        n, missing = len(group), int((~valid).sum())
        k = int((valid & (p < alpha)).sum())
        # Envelope covers either extreme treatment of the unresolved tests.
        lo, _ = bounds(k, n)
        _, hi = bounds(k + missing, n)
        rows.append(dict(method=method, label=label, alpha=alpha, p_value_kind=column,
            n_attempted=n, n_failed=missing, n_rejected=k, rate=k/n,
            missing_bound_lower=k/n, missing_bound_upper=(k+missing)/n,
            ci95_envelope_lower=lo, ci95_envelope_upper=hi))
    return pd.DataFrame(rows)


def style(ax, upper, xlabel='Selection method / inference engine', ticks=None):
    ax.set_ylim(0, upper)
    ax.set_yticks(ticks if ticks is not None else [0, .05, .10])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: '0' if value == 0 else f'{value:.2f}'))
    ax.yaxis.set_minor_locator(MultipleLocator(.01 if upper <= .12 else .05))
    ax.set_ylabel('Type I error rate')
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', color='0.9', linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', which='major', length=5, pad=7)
    ax.tick_params(axis='y', which='minor', length=2.5, color='0.55')
    for value, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if np.isclose(value, .05):
            label.set_color('#D62728')
            label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_color('0.3')
        spine.set_linewidth(.7)


def points(ax, data, *, offset=0, color=None, marker='o', label=None, annotate=False):
    for i, row in enumerate(data.itertuples()):
        ax.errorbar(i+offset, row.rate,
            yerr=[[row.rate-row.ci95_envelope_lower], [row.ci95_envelope_upper-row.rate]],
            color=color or COLORS[i], marker=marker, markersize=5.5,
            capsize=4, linewidth=1.25, linestyle='none', label=label if i == 0 else None)
        if annotate:
            ax.annotate(f'{100*row.rate:.2f}%', (i+offset, row.ci95_envelope_upper),
                xytext=(0, 8), textcoords='offset points', ha='center', fontsize=10)
    ax.set_xticks(np.arange(5), LABELS)


def save(fig, output, stem):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{stem}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input / 'analysis/paper_style_fpr'
    output.mkdir(parents=True, exist_ok=True)
    verification = json.loads((args.input / 'portable_verification.json').read_text())
    assert verification['all_regenerated_array_hashes_match_origin']
    frame = pd.read_csv(args.input / 'results.csv')
    assert len(frame) == verification['verified_rows'] == 50000
    assert not frame.duplicated(['repetition', 'method']).any()
    assert set(frame.method) == set(METHODS)
    assert all(frame[frame.method.eq(method)].repetition.nunique() == 10000 for method in METHODS)
    selective, naive = rates(frame, .05), rates(frame, .05, 'naive_p_value')
    # Cross-check plotting counts against the experiment's original summary.
    summary = pd.read_csv(args.input / 'summary.csv')
    for data, kind in [(selective, 'selective'), (naive, 'naive_same_target')]:
        original = summary[(summary.alpha == .05) & (summary.p_value_kind == kind)].set_index('method')
        for row in data.itertuples():
            assert row.n_rejected == original.loc[row.method, 'n_rejected']
            assert row.n_failed == original.loc[row.method, 'n_failed']
    primary = pd.concat([selective, naive], ignore_index=True)
    primary.to_csv(output / 'type1_error_5percent_data.csv', index=False)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    design = r'$n=100$, $p=20$; 10,000 paired global-null replications per method'
    foot = ('Points: observed rejections / 10,000 attempts. Bars: 95% exact binomial interval envelopes\n'
            'covering both extreme treatments of missing p-values. Failed tests: SHAP 3; LIME 4; spline AIS 2; spline exact 0; KDE-MI 4.')

    fig, ax = plt.subplots(figsize=(9.1, 4.9))
    fig.subplots_adjust(left=.085, right=.975, top=.79, bottom=.25)
    fig.suptitle('Type I error at the 5% significance level', y=.965, fontsize=15)
    fig.text(.5, .89, design, ha='center', fontsize=10)
    points(ax, selective, annotate=True)
    ax.axhline(.05, color='#D62728', linestyle='--', linewidth=1.3, label=r'Significance level $\alpha=0.05$')
    style(ax, .10)
    ax.legend(loc='lower right', frameon=False)
    fig.text(.085, .055, foot, fontsize=8, linespacing=1.5)
    save(fig, output, 'type1_error_5percent')

    fig, ax = plt.subplots(figsize=(9.1, 4.9))
    fig.subplots_adjust(left=.085, right=.975, top=.79, bottom=.25)
    fig.suptitle('Selective inference corrects selection-induced false positives', y=.965, fontsize=14)
    fig.text(.5, .89, design, ha='center', fontsize=10)
    points(ax, selective, offset=-.10, color='#2878B5', label='Selective test')
    points(ax, naive, offset=.10, color='#E78B27', marker='x', label='Naive test of the same selected target', annotate=True)
    ax.axhline(.05, color='#D62728', linestyle='--', linewidth=1.3, label=r'Significance level $\alpha=0.05$')
    style(ax, .5, ticks=[0, .05, .10, .20, .30, .40, .50])
    ax.legend(loc='upper left', frameon=False)
    fig.text(.085, .055, foot, fontsize=8, linespacing=1.5)
    save(fig, output, 'selective_vs_naive_5percent')

    levels = np.arange(.005, .1001, .005)
    curves = pd.concat([rates(frame, float(level)) for level in levels], ignore_index=True)
    curves.to_csv(output / 'calibration_curve_data.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.subplots_adjust(left=.12, right=.97, top=.81, bottom=.29)
    fig.suptitle('Selective-test calibration under the global null', y=.965, fontsize=14)
    fig.text(.5, .90, design, ha='center', fontsize=9)
    for method, label, color in zip(METHODS, LABELS, COLORS):
        group = curves[curves.method.eq(method)]
        ax.plot(group.alpha, group.rate, color=color, linewidth=1.25,
                marker='x', markersize=3.5, markevery=3, label=label)
    ax.plot([0,.1], [0,.1], color='#D62728', linestyle='--', linewidth=1.2, label=r'Nominal rate $=\alpha$')
    ax.axvline(.05, color='0.5', linestyle=':', linewidth=.9)
    ax.text(.051, .009, 'Primary level: 5%', color='0.4', fontsize=8)
    ax.set_xlim(0, .105)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    style(ax, .12, 'Significance level', ticks=[0, .02, .04, .05, .06, .08, .10, .12])
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.53,.11),
               ncol=3, frameon=False, columnspacing=1.5)
    fig.text(.12, .035, 'Rates use all 10,000 attempts; at most 4 missing p-values per method.\n'
             'Changing significance levels reuses the same tests; sample size remains fixed.', fontsize=8)
    save(fig, output, 'selective_calibration_curve')
    manifest = {'input': str(args.input.resolve()), 'raw_results_sha256': hashlib.sha256((args.input/'results.csv').read_bytes()).hexdigest(),
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'primary_alpha': .05, 'replications_per_method': 10000,
        'y_axis': 'Type I error probability, explicit zero baseline; nominal 0.05 tick highlighted',
        'intervals': 'Envelope of 95% Clopper-Pearson intervals for k and k+missing out of N attempts',
        'reference': 'references/papers/2025_Shiraishi_SelectiveInferenceFeatureSelectionPipelines.pdf, Figures 3 and 9',
        'interpretation': 'Compatible with nominal calibration in this Gaussian global-null setting; no proof of a universal FPR bound.'}
    (output/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'README.md').write_text('''# Type I error figures

The figures adapt the reference paper's clean axes, markers, and red nominal-level
reference line. Only n=100 and p=20 were run, so no sample-size or dimension sweep
is implied. Each method has 10,000 paired Gaussian global-null replications.

- `type1_error_5percent`: main 5% comparison, with conservative interval envelopes.
- `selective_vs_naive_5percent`: same selected targets, selective versus naive tests.
- `selective_calibration_curve`: rejection fractions over significance levels .005-.10.

PNG (300 dpi), vector PDF and SVG are supplied for each figure. Companion CSVs
contain the plotted counts and rates. At 5%, rates are 5.01%, 5.17%, 5.28%, 5.21%,
and 4.96% in the displayed method order. All primary uncertainty envelopes include
5%. The data support compatibility with nominal calibration in the tested setting.

Point estimates count observed rejections divided by all attempted tests. There
are 13 unresolved insufficient-ESS failures, retained in the denominator. Error
bars span the lower bound of the 95% exact binomial interval for k rejections
and the upper bound for k+missing rejections. Thus failures cannot artificially
narrow the displayed uncertainty. These are pointwise, not simultaneous intervals.
The paired arms are not independent replications of different methods. Naive
same-target p-values are diagnostic and do not change the inference pipeline.
''', encoding='utf-8')
    print(f'Created three figures in PNG/PDF/SVG and plotting data: {output}')


if __name__ == '__main__':
    main()
