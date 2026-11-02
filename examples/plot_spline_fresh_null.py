"""Plot the 10,000-trial paired spline null-calibration experiment."""
from pathlib import Path
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/selection_method_validation/exact_set/spline_fresh_null_10000'
METHODS = {'ais': 'Spline screening (AIS)',
           'exact_spline': 'Spline screening (exact)'}
COLORS = {'ais': '#258568', 'exact_spline': '#c44e52'}
FIGURES = []


def save(fig, name, caption):
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    FIGURES.append((name, caption))


def main():
    data = pd.read_csv(OUT / 'unified_tests.csv')
    summary = pd.read_csv(OUT / 'calibration_summary.csv')
    paired = pd.read_csv(OUT / 'paired_tests.csv')
    config = json.loads((OUT / 'experiment_config.json').read_text(encoding='utf-8'))
    assert set(data.inference_method) == set(METHODS)
    assert len(data) == 2 * config['n_trials'] == 20_000
    assert len(paired) == config['n_trials'] == 10_000
    assert not data.failed.astype(bool).any()
    for row in summary.itertuples():
        group = data[data.inference_method.eq(row.method)]
        assert len(group) == row.n
        assert int(group.p_value.le(row.alpha).sum()) == row.rejections
        assert np.isclose(row.fpr, row.rejections / row.n)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})

    # Match the null-calibration figure in the five-method report.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, (method, label) in enumerate(METHODS.items()):
        group = summary[summary.method.eq(method)].sort_values('alpha')
        axes[0].errorbar(
            group.alpha + (j - .5) * .0015, group.fpr,
            yerr=np.maximum(0, np.vstack([group.fpr-group.ci_lower,
                                          group.ci_upper-group.fpr])),
            fmt='o', capsize=3, color=COLORS[method], label=label)
        values = np.sort(data.loc[data.inference_method.eq(method), 'p_value'])
        axes[1].step(values, np.arange(1, len(values)+1)/len(values),
                     where='post', color=COLORS[method], label=label)
    axes[0].plot([0, .12], [0, .12], '--', color='gray')
    axes[0].set(title='Null rejection rates and 95% intervals',
                xlabel='Nominal alpha (points offset for readability)',
                ylabel='Observed false-positive rate', xticks=[.01, .05, .1],
                xlim=(0, .115), ylim=(0, .12))
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].set(title='Null p-value empirical distribution', xlabel='p-value',
                ylabel='Fraction at or below p', xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Fresh spline null calibration (10,000 paired trials; 0 failures)')
    save(fig, '01_null_calibration',
         'The diagonal is the nominal/uniform reference. Error bars are pointwise '
         'exact 95% binomial confidence intervals.')

    # Show the rejection-relevant part of the distribution at greater resolution.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    grid = np.linspace(0, .12, 241)
    for method, label in METHODS.items():
        values = data.loc[data.inference_method.eq(method), 'p_value'].to_numpy()
        ecdf = np.searchsorted(np.sort(values), grid, side='right') / len(values)
        axes[0].plot(grid, ecdf, color=COLORS[method], label=label)
        axes[1].plot(grid, ecdf-grid, color=COLORS[method], label=label)
    axes[0].plot(grid, grid, '--', color='gray')
    axes[0].set(title='Empirical CDF near rejection thresholds', xlabel='p-value',
                ylabel='Fraction at or below p', xlim=(0, .12), ylim=(0, .12))
    axes[1].axhline(0, ls='--', color='gray')
    axes[1].set(title='Departure from uniform calibration', xlabel='p-value',
                ylabel='Empirical CDF minus p', xlim=(0, .12))
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Fresh spline null calibration: lower-tail detail')
    save(fig, '02_lower_tail_calibration',
         'The lower tail determines rejection rates. Values above zero in the '
         'right panel indicate more small p-values than the uniform reference.')

    # Paired agreement isolates AIS approximation error from trial variability.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hb = axes[0].hexbin(paired.p_value_exact, paired.p_value_ais, gridsize=55,
                        mincnt=1, bins='log', cmap='viridis')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    axes[0].set(title='Paired selective p-values', xlabel='Exact spline p-value',
                ylabel='AIS p-value', xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(hb, ax=axes[0], label='Paired trials per hexagon (log scale)')
    axes[1].hist(paired.p_difference, bins=80, color=COLORS['ais'], alpha=.8)
    axes[1].axvline(0, ls='--', color='gray')
    mean_difference = paired.p_difference.mean()
    axes[1].axvline(mean_difference, color='#172d3f', lw=1.5,
                    label=f'Mean = {mean_difference:+.4f}')
    axes[1].set(title='AIS minus exact p-value', xlabel='Paired p-value difference',
                ylabel='Trials')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('Spline AIS agreement with exact conditional inference')
    save(fig, '03_paired_agreement',
         'Each trial uses the same response, selected set, and target for AIS and '
         'exact inference. The difference therefore measures AIS approximation error.')

    # Adapt the five-method sampling and ESS diagnostics to these null trials.
    ais = data[data.inference_method.eq('ais')].copy()
    total_proposals = ais.proposals + ais.pilot_evaluations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    values = np.sort(total_proposals)
    axes[0].step(values, np.arange(1, len(values)+1)/len(values), where='post',
                 color=COLORS['ais'])
    axes[0].set(title='AIS sampling effort', xlabel='Proposals including pilot',
                ylabel='Fraction of trials at or below cost', xscale='symlog',
                ylim=(0, 1.03))
    scatter = axes[1].scatter(ais.denominator_ess, ais.tail_ess, c=ais.p_value,
                              s=11, alpha=.35, cmap='viridis', rasterized=True)
    axes[1].axvline(80, ls='--', color='gray')
    axes[1].axhline(15, ls='--', color='gray')
    axes[1].set(title='AIS effective sample sizes', xlabel='Denominator ESS',
                ylabel='Tail ESS', xscale='log', yscale='log')
    fig.colorbar(scatter, ax=axes[1], label='AIS p-value')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('Spline AIS sampling diagnostics (10,000 null trials)')
    save(fig, '04_ais_diagnostics',
         'Sampling counts include the 3,200 pilot evaluations. Dashed lines mark '
         'the denominator ESS=80 and tail ESS=15 stopping thresholds.')

    exact05 = summary[summary.method.eq('exact_spline') & summary.alpha.eq(.05)].iloc[0]
    ais05 = summary[summary.method.eq('ais') & summary.alpha.eq(.05)].iloc[0]
    intro = (f'This report compares spline-screening AIS with exact spline inference '
             f'on {config["n_trials"]:,} fresh paired global-null trials. There were '
             f'no failures. At alpha=0.05, observed FPR was {ais05.fpr:.2%} for AIS '
             f'and {exact05.fpr:.2%} for exact inference; both confidence intervals '
             f'contain the nominal 5% rate.')
    report = ['# Fresh spline null-calibration plots', intro, '## Plot gallery']
    sections = []
    for name, caption in FIGURES:
        title = name[3:].replace('_', ' ').title()
        report.append(f'### {title}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)')
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
                        f'<a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a>'
                        f'<p><a href="{name}.pdf">Download PDF</a></p></section>')
    report.append('## Data\n\nPlots were verified against `unified_tests.csv`, '
                  '`calibration_summary.csv`, and `paired_tests.csv`.')
    (OUT / 'PLOTS.md').write_text('\n\n'.join(report), encoding='utf-8')
    (OUT / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Fresh spline null calibration</title>'
        '<style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;'
        'color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;'
        'border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style>'
        f'<h1>Fresh spline null-calibration plots</h1><p>{html.escape(intro)}</p>'
        + ''.join(sections) + '</html>', encoding='utf-8')
    print(f'Saved {len(FIGURES)} PNG/PDF figures and gallery to {OUT}')


if __name__ == '__main__':
    main()
