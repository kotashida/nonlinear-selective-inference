"""Plot observed-response benchmark rejection rates by RBF basis count."""
import argparse
import csv
import html
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    with (folder/'requested_manifest.json').open(encoding='utf-8') as stream:
        import json
        manifest = json.load(stream)
    with (folder/'summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    output = folder/'analysis'
    output.mkdir(exist_ok=True)
    table = []
    figures = []
    fractions = manifest['basis_fractions']
    datasets = manifest['datasets']
    for n in manifest['sample_sizes']:
        ncols = min(4, len(datasets))
        nrows = math.ceil(len(datasets)/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.7*nrows),
                                 squeeze=False, sharey=True)
        for ax, dataset in zip(axes.flat, datasets):
            for arm, label, color, offset, marker in (
                ('selected_shapley', 'Selected-Features', '#0072B2', -.04, 'o'),
                ('full_shapley', 'All-Features', '#D55E00', .04, 's'),
            ):
                subset = [r for r in rows if r['dataset'] == dataset and int(r['n']) == n
                          and r['arm'] == arm and float(r['alpha']) == manifest['primary_alpha']
                          and r['metric'] == 'rejection_rate']
                by_fraction = {float(r['fraction']): r for r in subset}
                if len(subset) != len(fractions) or set(by_fraction) != set(fractions):
                    raise ValueError(f'Incomplete or duplicate conditions: {dataset}, {n}, {arm}')
                values = [by_fraction[f] for f in fractions]
                rates = [float(r['rate']) for r in values]
                ax.errorbar([i+offset for i in range(len(fractions))], rates,
                            yerr=[[max(0., float(r['rate'])-float(r['ci_lower'])) for r in values],
                                  [max(0., float(r['ci_upper'])-float(r['rate'])) for r in values]],
                            fmt=marker+'-', color=color, capsize=3, label=label)
                table.extend(dict(dataset=dataset, n=n, arm=arm, fraction=f,
                                  bases_per_feature=math.ceil(f*n), **{k:r[k] for k in
                                  ('alpha', 'trials', 'rejections', 'rate', 'ci_lower', 'ci_upper')})
                             for f, r in zip(fractions, values))
            ax.set(title=dataset.replace('_', ' '), ylim=(0, 1.02),
                   xticks=range(len(fractions)),
                   xticklabels=[f'{math.ceil(f*n)} ({f:.0%})' for f in fractions],
                   xlabel='Bases per feature (center fraction)', ylabel='Rejection rate')
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        for ax in list(axes.flat)[len(datasets):]:
            ax.set_visible(False)
        fig.suptitle(f'Benchmark RBF basis sensitivity: n={n}, alpha={manifest["primary_alpha"]:g}')
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(.5, .075), ncol=2, frameon=False)
        fig.text(.5, .015,
                 f'Observed responses; {manifest["real_iterations"]:,} matched subsamples per dataset. '
                 'Pointwise 95% exact binomial intervals.\n'
                 'Rejection rates are descriptive; truth is unknown.',
                 ha='center', va='bottom', fontsize=8.5)
        fig.tight_layout(rect=(0, .18, 1, .94))
        name = f'01_benchmark_basis_n{n}'
        fig.savefig(output/f'{name}.png', dpi=160)
        fig.savefig(output/f'{name}.pdf')
        plt.close(fig)
        figures.append((name, n))
    with (output/'benchmark_basis_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    content = '<!doctype html><meta charset="utf-8"><title>Benchmark RBF basis sensitivity</title>'
    content += '<h1>Benchmark RBF basis sensitivity</h1><p>Observed responses and all native features. '
    content += 'These are descriptive rejection rates, not known power or Type I error; truth is unknown. '
    content += 'See summary.csv for all alpha levels and rank_summary.csv for target ranks.</p>'
    for name, n in figures:
        content += f'<h2>n={n}</h2><a href="{html.escape(name)}.pdf">PDF</a>'
        content += f'<p><img src="{html.escape(name)}.png" style="max-width:100%"></p>'
    (output/'index.html').write_text(content, encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    print(plot(parser.parse_args().input))
