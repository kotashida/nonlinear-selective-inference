"""Make presentation-style figures from a verified ridge/CV result folder."""
import argparse
import csv
import html
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def null_basis_figure(rows):
    """Compare only null rejection rates, with discrete basis-count labels."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    fractions = (.05, .1, .2)
    bounds = [.05]
    for ax, n in zip(axes, (25, 50, 75)):
        for arm, label, offset, marker, color in (
            ('selected_shapley', 'Selected-Features', -.055, 'o', '#0072B2'),
            ('full_shapley', 'All-Features', .055, 's', '#D55E00'),
        ):
            subset = [r for r in rows if r['dataset'] == 'synthetic'
                      and r['n'] == n and r['beta'] == 0 and r['arm'] == arm
                      and r['alpha'] == .05 and r['metric'] == 'rejection_rate']
            by_fraction = {r['fraction']: r for r in subset}
            if len(subset) != 3 or set(by_fraction) != set(fractions):
                raise ValueError(f'Expected one null rejection rate per fraction: {n=}, {arm=}')
            values = [by_fraction[f] for f in fractions]
            lower = [float(r['ci_lower']) for r in values]
            upper = [float(r['ci_upper']) for r in values]
            bounds.extend(lower + upper)
            ax.errorbar([i + offset for i in range(3)], [r['rate'] for r in values],
                        yerr=[[r['rate'] - lo for r, lo in zip(values, lower)],
                              [hi - r['rate'] for r, hi in zip(values, upper)]],
                        fmt=marker, color=color, markersize=6, capsize=4,
                        elinewidth=1.5, label=label)
        ax.axhline(.05, color='0.4', linestyle='--', linewidth=1.2,
                   label='Nominal level (5%)')
        ax.set(title=f'n = {n}', xlim=(-.4, 2.4), xticks=range(3),
               xticklabels=[f'{math.ceil(f*n)}\n({f:.0%})' for f in fractions],
               xlabel='Basis functions per feature\n(center fraction)')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    low = min(.035, min(bounds) - .002)
    high = max(.065, max(bounds) + .002)
    axes[0].set(ylim=(low, high), ylabel='Type I error rate')
    from matplotlib.ticker import PercentFormatter
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.5, .045), ncol=3, frameon=False)
    fig.text(.5, .015, 'Synthetic global null; 10,000 repetitions per setting. '
             'Error bars: pointwise 95% exact binomial confidence intervals.',
             ha='center', fontsize=9)
    return fig


def plot(folder):
    with (folder/'summary.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('n', 'fraction', 'beta', 'alpha', 'rate'):
            r[k] = float(r[k]) if r[k] not in ('', 'None') else float('nan')
    output = folder/'analysis'
    output.mkdir(exist_ok=True)
    arms = ('selected_shapley', 'full_shapley')
    labels = ('Selected-Features', 'All-Features')
    figures = []

    def save(fig, name, title, rect=None):
        fig.suptitle(title)
        fig.tight_layout(rect=rect)
        fig.savefig(output/f'{name}.png', dpi=160)
        fig.savefig(output/f'{name}.pdf')
        plt.close(fig)
        figures.append((name, title))

    def curve(ax, subset, x, label):
        subset = sorted(subset, key=lambda r: r[x])
        valid = [r for r in subset if math.isfinite(r['rate'])]
        if not valid:
            return
        ax.errorbar([r[x] for r in valid], [r['rate'] for r in valid],
                    yerr=[[max(0., r['rate']-float(r['ci_lower'])) for r in valid],
                          [max(0., float(r['ci_upper'])-r['rate']) for r in valid]],
                    marker='o', capsize=2, label=label)

    primary = [r for r in rows if r['fraction'] == .1 and r['alpha'] == .05]
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, label in zip(arms, labels):
        curve(ax, [r for r in primary if r['dataset'] == 'synthetic' and r['beta'] == 0 and r['arm'] == arm and r['metric'] == 'rejection_rate'], 'n', label)
    ax.axhline(.05, color='gray', linestyle='--')
    ax.set(xlabel='Sample size n', ylabel='Type I error rate', xticks=[25, 50, 75])
    ax.legend()
    save(fig, '01_type1', 'Global-null calibration: RBF ridge + CV (10% centers)')

    metrics = ('targeting_probability', 'conditional_signal_detection', 'marginal_signal_detection')
    for n in (25, 50, 75):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, metrics):
            for arm, label in zip(arms, labels):
                curve(ax, [r for r in primary if r['dataset'] == 'synthetic' and r['n'] == n and r['arm'] == arm and r['metric'] == metric], 'beta', label)
            ax.set(xlabel='Signal strength beta', ylabel='Probability', title=metric.replace('_', ' '),
                   ylim=(0, 1), xlim=(-.03, 1.03), xticks=[0, .25, .5, .75, 1])
            ax.legend(fontsize=8)
        save(fig, f'02_power_n{n}', f'Power: n={n}, p=4, k=2, 10% centers')

    datasets = sorted({r['dataset'] for r in primary if r['dataset'] != 'synthetic'})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    table = []
    for ax, dataset in zip(axes.flat, datasets):
        for arm, label in zip(arms, labels):
            subset = [r for r in primary if r['dataset'] == dataset and r['arm'] == arm and r['metric'] == 'rejection_rate']
            curve(ax, subset, 'n', label)
            table.extend(subset)
        ax.set(title=dataset, xlabel='n', ylabel='Rejection rate', xticks=[25, 50, 75], ylim=(0, 1))
        ax.legend(fontsize=7)
    save(fig, '03_realdata', 'Real-data rejection rates (working sigma=1; truth unknown)')
    with (output/'realdata_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)

    save(null_basis_figure(rows), '04_appendix_null',
         'Type I error across different numbers of RBF basis functions',
         rect=(0, .15, 1, .94))
    for n in (25, 50, 75):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, metrics):
            for fraction in (.05, .1, .2):
                for arm, label in zip(arms, labels):
                    subset = [r for r in rows if r['dataset'] == 'synthetic' and r['n'] == n and r['arm'] == arm and r['fraction'] == fraction and r['alpha'] == .05 and r['metric'] == metric]
                    curve(ax, subset, 'beta', f'{label}, {fraction:.0%}')
            ax.set(title=metric.replace('_', ' '), xlabel='beta', ylabel='Probability',
                   ylim=(0, 1), xlim=(-.03, 1.03), xticks=[0, .25, .5, .75, 1])
            ax.legend(fontsize=6)
        save(fig, f'05_appendix_power_n{n}', f'Appendix: basis-count sensitivity, n={n}')
    gallery = '<html><meta charset="utf-8"><title>RBF ridge/CV experiments</title><body>'
    gallery += '<h1>RBF ridge/CV experiments</h1><p>Exact confidence intervals are in summary.csv. Real-data truth is unknown.</p>'
    for name, title in figures:
        gallery += f'<h2>{html.escape(title)}</h2><img style="max-width:100%" src="{name}.png">'
    (output/'index.html').write_text(gallery+'</body></html>', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    plot(parser.parse_args().input)
