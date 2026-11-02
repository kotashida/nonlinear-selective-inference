"""Plot the verified four-target global-null experiment using existing output styles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/four_targets/four_targets_basic_null_cpu32_200_20261018'
ARMS = ['marginal', 'model', 'selected_shapley', 'full_shapley']
LABELS = ['Marginal effect', 'Selected-model effect', 'Selected-set Shapley', 'Full-set Shapley']
COLORS = ['#2878B5', '#E78B27', '#3C9B58', '#8866AF']
REFERENCES = [
    'outputs/benchmarks/selected_five_null_fpr_10000_cpu32_20260928/analysis/paper_style_fpr/selective_vs_naive_5percent.png',
    'outputs/null_calibration/random_u_seed_123/shard_0/p_value_histograms.png',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(k, n):
    return (float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k < n else 1.)


def save(fig, output, name):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{name}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='.90', linewidth=.6)
    for spine in ax.spines.values():
        spine.set_color('.3')
        spine.set_linewidth(.7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output or source / 'analysis/plots'
    manifest = json.loads((source / 'requested_manifest.json').read_text())
    completed = json.loads((source / 'COMPLETED').read_text())
    for name, digest in completed['export_sha256'].items():
        require(Path(name).name == name, 'Unsafe export filename')
        require(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest,
                f'Export checksum mismatch: {name}')
    frame = pd.read_csv(source / 'results.csv')
    n = manifest['iterations']
    require(manifest['conditions'] == [{'id': 'global_null', 'mean': 'zero', 'rms': 0.0}],
            'These plots require the global-null experiment')
    require(len(manifest['designs']) == len(manifest['bases']) == 1, 'Expected one design and basis')
    require(len(frame) == completed['total_tests'] == n*4, 'Incomplete result coverage')
    require(set(frame.arm) == set(ARMS) and frame.null_true.eq(True).all(), 'Unexpected arms or null labels')
    require(not frame.duplicated(['iteration', 'arm']).any(), 'Duplicate trial/arm')
    require(set(frame.iteration) == set(range(n)), 'Missing trial indices')
    for column in ['p_selective', 'p_naive']:
        require(np.isfinite(frame[column]).all() and frame[column].between(0, 1).all(), 'Invalid p-values')
    for _, group in frame.groupby('iteration'):
        require(set(group.arm) == set(ARMS), 'Missing paired arm')
        for column in ['data_sha256', 'selected', 'target', 'noise_seed', 'target_seed']:
            require(group[column].nunique() == 1, f'Unmatched paired field: {column}')

    records = []
    for alpha in manifest['alpha_levels']:
        for arm in ARMS:
            group = frame[frame.arm.eq(arm)]
            for method in ['selective', 'naive']:
                k = int((group[f'p_{method}'] < alpha).sum())
                lo, hi = interval(k, n)
                records.append(dict(arm=arm, method=method, alpha=alpha, n=n,
                                    rejections=k, rate=k/n, ci_lower=lo, ci_upper=hi))
    rates = pd.DataFrame(records)
    original = pd.read_csv(source / 'calibration_power_summary.csv')
    for row in rates.itertuples():
        saved = original[original.arm.eq(row.arm) & original.method.eq(row.method)
                         & np.isclose(original.alpha, row.alpha) & original.null_true.eq(True)]
        require(len(saved) == 1, 'Missing original summary row')
        for key in ['n', 'rejections', 'rate', 'ci_lower', 'ci_upper']:
            require(np.isclose(saved.iloc[0][key], getattr(row, key)), f'Summary mismatch: {key}')
    output.mkdir(parents=True, exist_ok=True)
    rates.to_csv(output / 'rejection_rates.csv', index=False)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    subtitle = (rf'$n={manifest["n"]}$, $p=4$, $k=2$; correlated Gaussian design ($\rho=0.6$)'
                f'\nCubic splines; {n} paired global-null trials per target')
    stems = []

    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(left=.09, right=.98, top=.74, bottom=.25)
    fig.suptitle('Type I error across four target definitions', y=.97, fontsize=16)
    fig.text(.5, .86, subtitle, ha='center', linespacing=1.6)
    for method, offset, color, marker in [('selective', -.12, '#2878B5', 'o'), ('naive', .12, '#E78B27', 'x')]:
        subset = rates[rates.method.eq(method) & np.isclose(rates.alpha, .05)].set_index('arm').loc[ARMS]
        ax.errorbar(np.arange(4)+offset, subset.rate,
                    yerr=[subset.rate-subset.ci_lower, subset.ci_upper-subset.rate],
                    color=color, marker=marker, linestyle='none', markersize=6,
                    capsize=4, linewidth=1.3, label='Selective test' if method == 'selective' else 'Naive same-target test')
        for index, row in enumerate(subset.itertuples()):
            ax.annotate(f'{row.rate:.1%}', (index+offset, row.ci_upper),
                        xytext=(0, 6), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(.05, color='#D62728', ls='--', lw=1.3, label=r'Nominal $\alpha=0.05$')
    ax.set(xticks=range(4), xticklabels=LABELS, ylim=(0, .25), ylabel='Type I error rate', xlabel='Tested target')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(frameon=False, loc='upper left', ncol=3, fontsize=8)
    style(ax)
    fig.text(.09, .055, 'Bars: pointwise 95% exact binomial intervals; 200 trials per target, no missing tests.\n'
             'The four targets share the same responses, selected sets, and target indices.', fontsize=9, linespacing=1.6)
    save(fig, output, 'type1_error_5percent'); stems.append('type1_error_5percent')

    levels = np.linspace(0, .20, 81)
    curves = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.72, bottom=.26, wspace=.14)
    fig.suptitle('Global-null calibration curves', y=.97, fontsize=16)
    fig.text(.5, .86, subtitle, ha='center', linespacing=1.6)
    for ax, method in zip(axes, ['selective', 'naive']):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            p = frame.loc[frame.arm.eq(arm), f'p_{method}'].to_numpy()
            values = (p[:, None] < levels).mean(axis=0)
            ax.plot(levels, values, color=color, label=label, lw=1.5)
            curves.extend(dict(arm=arm, method=method, alpha=a, rate=r, n=n) for a, r in zip(levels, values))
        ax.plot([0,.2], [0,.2], '--', color='#D62728', lw=1.2, label='Nominal rate')
        ax.set(title='Selective tests' if method == 'selective' else 'Naive same-target tests',
               xlabel='Significance level', xlim=(0,.2), xticks=[0,.05,.10,.15,.20])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); style(ax)
    curve_upper = np.ceil((max(row['rate'] for row in curves)+.025)/.05)*.05
    axes[0].set_ylim(0, curve_upper)
    axes[0].set_ylabel('Empirical rejection rate')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.52,.085), ncol=3, frameon=False)
    fig.text(.08, .025, 'Each curve reuses the same 200 trials; levels are not independent experiments.', fontsize=9)
    pd.DataFrame(curves).to_csv(output/'calibration_curve_data.csv', index=False)
    save(fig, output, 'calibration_curves'); stems.append('calibration_curves')

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.76, bottom=.17, hspace=.38, wspace=.15)
    fig.suptitle('P-value distributions under the global null', y=.97, fontsize=16)
    fig.text(.5, .88, subtitle, ha='center', linespacing=1.6)
    histograms = []
    bins = np.linspace(0, 1, 11)
    for ax, arm, label in zip(axes.flat, ARMS, LABELS):
        for method, color in [('selective', '#2878B5'), ('naive', '#E78B27')]:
            p = frame.loc[frame.arm.eq(arm), f'p_{method}']
            counts, _ = np.histogram(p, bins)
            density = counts/(n*np.diff(bins))
            ax.stairs(density, bins, color=color, lw=1.8, label=method.capitalize())
            histograms.extend(dict(arm=arm, method=method, left=l, right=r, count=int(c), density=d)
                              for l,r,c,d in zip(bins[:-1], bins[1:], counts, density))
        ax.axhline(1, ls='--', color='#D62728', lw=1.1)
        ax.set(title=label, xlim=(0,1)); style(ax)
    axes.flat[0].set_ylim(0, max(row['density'] for row in histograms)*1.12)
    for ax in axes[-1]: ax.set_xlabel('P-value')
    for ax in axes[:,0]: ax.set_ylabel('Density')
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.07), ncol=2, frameon=False)
    fig.text(.09, .025, 'Ten equal-width bins. Red dashed line: Uniform(0, 1) density. Each panel contains 200 paired tests.', fontsize=9)
    pd.DataFrame(histograms).to_csv(output/'histogram_data.csv', index=False)
    save(fig, output, 'p_value_histograms'); stems.append('p_value_histograms')

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.1, right=.96, top=.77, bottom=.17, hspace=.35, wspace=.17)
    fig.suptitle('Uniform Q–Q plots of null p-values', y=.97, fontsize=16)
    fig.text(.5, .89, subtitle, ha='center', linespacing=1.6)
    quantiles = (np.arange(n)+.5)/n
    lo = beta.ppf(.025, np.arange(1,n+1), np.arange(n,0,-1))
    hi = beta.ppf(.975, np.arange(1,n+1), np.arange(n,0,-1))
    qq = []
    for ax, arm, label in zip(axes.flat, ARMS, LABELS):
        ax.fill_between(quantiles, lo, hi, color='.90', label='95% pointwise uniform envelope')
        ax.plot([0,1],[0,1], '--', color='#D62728', lw=1)
        for method, color in [('selective', '#2878B5'), ('naive', '#E78B27')]:
            observed = np.sort(frame.loc[frame.arm.eq(arm), f'p_{method}'])
            ax.plot(quantiles, observed, color=color, lw=1.2, label=method.capitalize())
            qq.extend(dict(arm=arm, method=method, uniform_quantile=q, observed_p=p, ci_lower=l, ci_upper=h)
                      for q,p,l,h in zip(quantiles, observed, lo, hi))
        ax.set(title=label, xlim=(0,1), ylim=(0,1)); style(ax)
    for ax in axes[-1]: ax.set_xlabel('Uniform quantile')
    for ax in axes[:,0]: ax.set_ylabel('Ordered p-value')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.065), ncol=3, frameon=False, fontsize=8)
    fig.text(.1, .025, 'Envelope uses Beta order-statistic quantiles for 200 independent Uniform(0, 1) draws; not simultaneous.', fontsize=8)
    pd.DataFrame(qq).to_csv(output/'qq_data.csv', index=False)
    save(fig, output, 'uniform_qq_plots'); stems.append('uniform_qq_plots')

    paired = frame.pivot(index='iteration', columns='arm', values='p_selective').loc[:,ARMS]
    decisions = paired.to_numpy() < .05
    counts = (decisions[:,:,None] != decisions[:,None,:]).sum(axis=0)
    original_pairs = pd.read_csv(source/'paired_target_comparisons.csv')
    for row in original_pairs.itertuples():
        require(counts[ARMS.index(row.arm_a), ARMS.index(row.arm_b)] == row.decisions_disagree,
                'Paired decision summary mismatch')
    pd.DataFrame(counts, index=ARMS, columns=ARMS).to_csv(output/'paired_disagreement_counts.csv')
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    fig.subplots_adjust(left=.25, right=.90, top=.74, bottom=.24)
    fig.suptitle('Paired selective-test rejection disagreements', y=.97, fontsize=15)
    fig.text(.5, .87, subtitle, ha='center', linespacing=1.6)
    im = ax.imshow(counts/n, cmap='Blues', vmin=0, vmax=max(.10, counts.max()/n))
    short = ['Marginal', 'Selected\nmodel', 'Selected-set\nShapley', 'Full-set\nShapley']
    ax.set(xticks=range(4), yticks=range(4), xticklabels=short, yticklabels=short)
    ax.tick_params(axis='both', length=0, pad=8, labelsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{counts[i,j]} / {n}\n{counts[i,j]/n:.1%}', ha='center', va='center',
                    fontsize=10, color='white' if counts[i,j]/n>.055 else '#17324D')
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    cb.set_label('Fraction of paired trials')
    fig.text(.1, .075, 'At α = 5%, one target rejects and the other does not. Each cell uses the same 200 responses.\n'
             'Different target definitions test different projection nulls; all are true in this global-null run.', fontsize=9, linespacing=1.6)
    save(fig, output, 'paired_rejection_disagreements'); stems.append('paired_rejection_disagreements')

    plot_manifest = dict(input=str(source), results_sha256=hashlib.sha256((source/'results.csv').read_bytes()).hexdigest(),
                         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         reference_plots=REFERENCES, trials_per_target=n, total_tests=len(frame),
                         figures=stems, formats=['png','pdf','svg'], checks='Export SHA256, paired coverage, finite p-values, original rates/CIs and disagreement counts')
    (output/'plot_manifest.json').write_text(json.dumps(plot_manifest, indent=2)+'\n', encoding='utf-8')
    primary = rates[np.isclose(rates.alpha,.05)].pivot(index='arm', columns='method', values='rate').loc[ARMS]
    lines = ['# Four-target global-null results', '',
             '200 paired trials; 800 tests. Correlated Gaussian design, n=256, p=4, k=2, cubic splines.', '',
             '| Target | Selective rejection rate | Naive same-target rate |', '| --- | ---: | ---: |']
    lines += [f'| {label} | {primary.loc[arm,"selective"]:.1%} | {primary.loc[arm,"naive"]:.1%} |' for arm,label in zip(ARMS,LABELS)]
    lines += ['', 'All four selective-test 95% exact binomial intervals include the nominal 5% level. '
              'With 200 trials per target these are exploratory calibration results, not evidence of a universal bound. '
              'All mean vectors are zero; this run does not estimate power.', '',
              'Full-set Shapley selective and naive p-values coincide to floating-point precision in this run. '
              'Target ranks are 3 (marginal), 3 (selected-model), 6 (selected-set Shapley), and 12 (full-set Shapley).', '',
              'Figures are supplied as 300 dpi PNG, vector PDF, and SVG, with companion CSVs. '
              'Intervals and Q–Q envelopes are pointwise, not simultaneous. No test treats the four arms as independent replications.', '',
              'Style references:', *[f'- `{p}`' for p in REFERENCES], '',
              'Recreate with `.venv/Scripts/python.exe examples/plot_four_targets_basic.py`.']
    (output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Created {len(stems)} figures in PNG/PDF/SVG, plotting data, and README: {output}')


if __name__ == '__main__':
    main()
