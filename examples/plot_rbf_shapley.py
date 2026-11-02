"""Validate completed RBF Shapley exports and create analysis figures."""
from __future__ import annotations

import argparse
import hashlib
import html
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
DEFAULT = ROOT/'outputs/rbf_shapley/rbf_shapley_mcmc20_cpu32_250_20261022'
ARMS = ['selected_shapley', 'full_shapley']
LABELS = {'selected_shapley': 'Selected-Features Shapley Effect', 'full_shapley': 'All-Features Shapley Effect'}
METHODS = {'selective': ('MCMC selective', '#2878B5', 'o'),
           'exact': ('Analytic selective', '#259876', 's'),
           'naive': ('Naive', '#DA8431', '^')}
CONDITIONS = ['global_null', 'quadratic_signal']
TITLES = {'global_null': 'Global null', 'quadratic_signal': 'Quadratic signal (RMS 0.5)'}
ARM_COLORS = {'selected_shapley': '#2674a6', 'full_shapley': '#d58122'}
FIGURES = []


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def interval(k, n):
    return (float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k < n else 1.)


def load(source):
    manifest = json.loads((source/'requested_manifest.json').read_text())
    done = json.loads((source/'COMPLETED').read_text())
    require(done['manifest_sha256'] == canonical(manifest), 'Manifest digest mismatch')
    for name, sha in done['export_sha256'].items():
        require(Path(name).name == name, 'Unsafe export filename')
        require(digest(source/name) == sha, f'Export checksum mismatch: {name}')
    receipts = {}
    receipt_rows = []
    for path in sorted((source/'jobs').glob('*.json')):
        receipt = json.loads(path.read_text())
        require(receipt['manifest_sha256'] == canonical(manifest), 'Receipt manifest mismatch')
        require(canonical(receipt['rows']) == receipt['rows_sha256'], 'Receipt rows corrupted')
        receipts[path.name] = receipt['rows_sha256']
        receipt_rows.extend(receipt['rows'])
    require(canonical(receipts) == done['receipts_sha256'], 'Receipt completion digest mismatch')
    d = pd.read_csv(source/'results.csv')
    expected = manifest['iterations']*len(manifest['conditions'])*len(manifest['targets'])
    require(len(d) == len(receipt_rows) == done['total_tests'] == expected, 'Incomplete results')
    require(set(d.arm) == set(ARMS) and set(d.condition) == set(CONDITIONS), 'Unexpected experiment arms')
    require(len(manifest['designs']) == len(manifest['bases']) == 1, 'Expected one design and basis')
    keys = ['condition', 'iteration', 'arm']
    require(not d.duplicated(keys).any(), 'Duplicate test rows')
    reconstructed = pd.DataFrame(receipt_rows).sort_values(keys).reset_index(drop=True)
    saved = d.sort_values(keys).reset_index(drop=True)
    for field in ['p_selective', 'p_exact', 'p_naive', 'rank', 'target', 'mean_contribution']:
        require(np.allclose(saved[field], reconstructed[field], rtol=1e-12, atol=1e-14), f'Receipt/export mismatch: {field}')
    for _, group in d.groupby(['condition', 'iteration']):
        require(set(group.arm) == set(ARMS), 'Missing matched arm')
        for field in ['data_sha256', 'selected', 'target', 'noise_seed', 'target_seed']:
            require(group[field].nunique() == 1, f'Unpaired field: {field}')
    for (condition, arm), group in d.groupby(['condition', 'arm']):
        require(set(group.iteration) == set(range(manifest['iterations'])), 'Missing iterations')
        require(group.null_true.eq(condition == 'global_null').all(), 'Unexpected null truth; revise analysis strata')
    for method in METHODS:
        require(np.isfinite(d['p_'+method]).all() and d['p_'+method].between(0, 1).all(), 'Invalid p-value')
    return manifest, d


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, output, stem, note):
    fig.tight_layout(rect=(0, .02, 1, .94))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'{stem}.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    FIGURES.append(dict(name=stem, caption=note.replace('\n', ' ')))


def reference_panels(d, output):
    signal = d[d.condition == 'quadratic_signal']
    summaries = []
    for arm in ARMS:
        group = signal[signal.arm == arm]
        target = group[group.target == 0]
        detected = int((target.p_selective < .05).sum())
        for metric, k, n in [('Targeting feature 0', len(target), len(group)),
                             ('Detection given target 0', detected, len(target)),
                             ('Target 0 and detection', detected, len(group))]:
            lo, hi = interval(k, n)
            summaries.append(dict(arm=arm, metric=metric, count=k, n=n, rate=k/n,
                                  ci_lower=lo, ci_upper=hi))
    power = pd.DataFrame(summaries)
    power.to_csv(output/'signal_targeting_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, power.metric.unique()):
        subset = power[power.metric == metric].set_index('arm').loc[ARMS]
        ax.bar(range(2), subset.rate, color=[ARM_COLORS[a] for a in ARMS], width=.6)
        ax.errorbar(range(2), subset.rate,
                    yerr=[subset.rate-subset.ci_lower, subset.ci_upper-subset.rate],
                    fmt='none', ecolor='#203040', capsize=4)
        for x, r in enumerate(subset.itertuples()):
            ax.text(x, r.ci_upper+.025, f'{r.rate:.1%}\n{r.count}/{r.n}', ha='center', fontsize=9)
        if metric != 'Detection given target 0':
            ax.axhline(.5, ls='--', color='gray')
        ax.set(title=metric, xticks=range(2), xticklabels=['Selected-Features\nShapley Effect', 'All-Features\nShapley Effect'],
               ylim=(0,1.16), ylabel='Fraction')
        ax.yaxis.set_major_formatter(PercentFormatter(1)); style(ax)
    fig.suptitle('Parallel MCMC (20 steps): signal targeting and detection')
    save(fig, output, '07_signal_targeting', 'One signal strength: quadratic mean with RMS 0.5. Bars show targeting, detection conditional on targeting feature 0, and their joint frequency; intervals are pointwise 95% exact binomial intervals. The 50% line is the population targeting ceiling. Finite samples can exceed it. These feature-0 summaries differ from power over all false Shapley nulls.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    outcomes = []
    for ax, arm in zip(axes, ARMS):
        group = signal[signal.arm == arm]
        subsets = [group[group.target == 0], group]
        rates = [float((g.p_selective < .05).mean()) for g in subsets]
        ax.bar(range(2), rates, color='#258568', label='Detected', width=.6)
        ax.bar(range(2), 1-np.array(rates), bottom=rates, color='#bcc8d1',
               label='Resolved, not detected', width=.6)
        for x, (g, rate) in enumerate(zip(subsets, rates)):
            ax.text(x, 1.025, f'n={len(g)}', ha='center', fontsize=9)
            ax.text(x, rate/2, f'{rate:.1%}', ha='center', va='center', color='white')
            outcomes.append(dict(arm=arm, stratum=['target_0', 'all_false_nulls'][x],
                                 n=len(g), detected=int((g.p_selective < .05).sum()), unresolved=0))
        ax.set(title=LABELS[arm], xticks=range(2),
               xticklabels=['Targeting feature 0', 'All false Shapley nulls'], ylim=(0,1.13))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); style(ax)
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Signal outcomes: generating feature and Shapley hypotheses')
    pd.DataFrame(outcomes).to_csv(output/'signal_outcomes.csv', index=False)
    save(fig, output, '08_signal_outcomes', 'Strict MCMC p < 0.05. Every test is resolved. All 250 signal-condition Shapley nulls per arm are false, including tests of correlated proxy features. Rejection does not identify a unique or causal effect; Selected-Features Shapley Effect and All-Features Shapley Effect tests have different hypotheses.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arm in zip(axes, ARMS):
        group = d[(d.condition == 'global_null') & (d.arm == arm)]
        for method, (label, color, marker) in METHODS.items():
            values = np.sort(group['p_'+method].to_numpy())
            ax.plot((np.arange(len(values))+.5)/len(values), values,
                    color=color, label=label, linewidth=1.5)
        ax.plot([0,1], [0,1], '--', color='gray')
        ax.set(title=LABELS[arm], xlabel='Uniform reference quantile', ylabel='Observed p-value',
               xlim=(0,1), ylim=(0,1)); style(ax)
    axes[0].legend(fontsize=9)
    fig.suptitle('Null quantiles: MCMC, analytic and naive references')
    save(fig, output, '09_null_quantiles', '250 null trials per target. Values below the diagonal indicate smaller p-values than the uniform reference. The All-Features Shapley Effect analytic and naive curves coincide because radial scaling preserves selection in the full candidate space. MCMC rank p-values are discrete.')


def gallery(output):
    intro = ('RBF Shapley tests with parallel reversible MCMC rank inference: n=100, p=4, k=2; '
             '250 responses per condition, 500 paired datasets and 1,000 tests. '
             'Known sigma=1, fixed correlated design, exact-set conditioning. '
             'The two targets share responses, selected sets and tested features.')
    sections = []
    markdown = ['# RBF Shapley analysis', intro,
                'Selected-Features Shapley Effect and All-Features Shapley Effect tests address different hypotheses. '
                'The All-Features Shapley Effect space has rank 12; Selected-Features Shapley Effect spaces have rank 6. '
                'Higher rejection does not establish better identification of the generating feature. '
                'Pointwise intervals do not adjust for multiple comparisons. '
                'MCMC acceptance alone does not establish mixing.',
                '[Numerical findings](analysis_summary.md)']
    for item in FIGURES:
        title = item['name'][3:].replace('_', ' ').title()
        caption = item['caption']
        markdown.append(f'## {title}\n\n{caption}\n\n![{title}]({item["name"]}.png)')
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
                        f'<a href="{item["name"]}.png"><img loading="lazy" src="{item["name"]}.png" '
                        f'alt="{html.escape(title)}"></a></section>')
    markdown.append('## Reproduction\n\nRun `.venv/Scripts/python.exe examples/plot_rbf_shapley.py`. '
                    'Figures are exported as PNG, PDF and SVG from checksum-validated saved results.')
    (output/'README.md').write_text('\n\n'.join(markdown)+'\n', encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>RBF Shapley analysis</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:32px auto;padding:0 24px;'
        'color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;'
        'border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style>'
        '<h1>RBF Shapley analysis</h1><p>'+html.escape(intro)+'</p>'
        '<p>Selected-Features Shapley Effect and All-Features Shapley Effect tests address different hypotheses. '
        'See <a href="analysis_summary.md">numerical findings and interpretation</a>.</p>'
        +''.join(sections)+'</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output or source/'analysis'
    m, d = load(source)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    rows = []
    for (condition, arm), group in d.groupby(['condition', 'arm'], sort=False):
        for method in METHODS:
            k, n = int((group['p_'+method] < .05).sum()), len(group)
            lo, hi = interval(k, n)
            rows.append(dict(condition=condition, arm=arm, method=method, n=n, rejections=k,
                             rate=k/n, ci_lower=lo, ci_upper=hi))
    rates = pd.DataFrame(rows)
    rates.to_csv(output/'rejection_rates.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.suptitle('RBF Shapley tests: calibration and signal rejection', fontsize=15)
    for ax, condition in zip(axes, CONDITIONS):
        for offset, (method, (label, color, marker)) in zip([-.22, 0, .22], METHODS.items()):
            r = rates[(rates.condition == condition) & (rates.method == method)].set_index('arm').loc[ARMS]
            x = np.arange(2)+offset
            ax.errorbar(x, r.rate, yerr=[r.rate-r.ci_lower, r.ci_upper-r.rate],
                        fmt=marker, color=color, capsize=4, label=label, markersize=6)
            for xx, rr in zip(x, r.itertuples()):
                ax.annotate(f'{rr.rate:.1%}', (xx, rr.ci_upper), xytext=(0, 7),
                            textcoords='offset points', ha='center', fontsize=8)
        if condition == 'global_null':
            ax.axhline(.05, color='#A5414A', ls='--', lw=1, label='Nominal 5%')
        ax.set(title=TITLES[condition], xticks=[0, 1], xticklabels=['Selected-Features\nShapley Effect', 'All-Features\nShapley Effect'],
               xlim=(-.5, 1.5), ylim=(0, .27 if condition == 'global_null' else 1.12),
               ylabel='Type I error' if condition == 'global_null' else 'Rejection rate under signal')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc='upper right' if condition == 'global_null' else 'lower right', frameon=False)
        style(ax)
    save(fig, output, '01_rejection_rates', '250 paired trials per target and condition; pointwise 95% exact binomial intervals.\nThe two target definitions test different hypotheses. All signal-condition nulls are false in this fixed design.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Null calibration under the known Gaussian model', fontsize=15)
    null = d[d.condition == 'global_null']
    calibration = []
    for j, arm in enumerate(ARMS):
        group = null[null.arm == arm]
        p = group.p_selective.to_numpy()
        values, lower, upper = [], [], []
        for alpha in m['alpha_levels']:
            k, n = int((p < alpha).sum()), len(p)
            lo, hi = interval(k, n)
            values.append(k/n); lower.append(k/n-lo); upper.append(hi-k/n)
            calibration.append(dict(arm=arm, alpha=alpha, n=n, rejections=k,
                                    rate=k/n, ci_lower=lo, ci_upper=hi))
        axes[0].errorbar(np.array(m['alpha_levels'])+(j-.5)*.002, values,
                         yerr=[lower, upper], fmt='o', capsize=3,
                         color=ARM_COLORS[arm], label=LABELS[arm])
        axes[1].step(np.r_[0, np.sort(p), 1], np.r_[0, np.arange(1, len(p)+1)/len(p), 1],
                     where='post', color=ARM_COLORS[arm], label=LABELS[arm])
    axes[0].plot([0,.12], [0,.12], '--', color='gray')
    axes[0].set(title='Null rejection rates (250 trials)', xlabel='Nominal alpha (points offset)',
                ylabel='Rejection fraction', xticks=m['alpha_levels'], ylim=(0,.21))
    axes[0].legend(fontsize=9)
    axes[1].plot([0,1], [0,1], '--', color='gray')
    axes[1].set(title='Null p-value empirical distribution', xlabel='p-value',
                ylabel='Fraction at or below p', xlim=(0,1), ylim=(0,1))
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1)); style(ax)
    pd.DataFrame(calibration).to_csv(output/'calibration_summary.csv', index=False)
    save(fig, output, '02_null_calibration', 'MCMC rank tests; strict p < alpha rejection with pointwise 95% exact binomial intervals. The diagonal is a uniform reference. These sample sizes do not establish conditional calibration; both targets use the same 250 null datasets.')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    fig.suptitle('MCMC p-values versus analytic selective reference', fontsize=15)
    diagnostics = []
    for i, condition in enumerate(CONDITIONS):
        for j, arm in enumerate(ARMS):
            ax = axes[i,j]
            group = d[(d.condition == condition) & (d.arm == arm)]
            difference = group.p_selective-group.p_exact
            disagree = int(((group.p_selective < .05) != (group.p_exact < .05)).sum())
            diagnostics.append(dict(condition=condition, arm=arm, mean_absolute_p_difference=difference.abs().mean(),
                                    median_absolute_p_difference=difference.abs().median(),
                                    max_absolute_p_difference=difference.abs().max(), decisions_disagree=disagree,
                                    mean_acceptance=group.branch_acceptance.mean()))
            ax.scatter(group.p_exact, group.p_selective, s=13, alpha=.5, color='#2878B5', edgecolors='none')
            ax.plot([0,1], [0,1], '--', color='#7E8490', lw=1)
            ax.axvline(.05, color='#DA8431', lw=.7); ax.axhline(.05, color='#DA8431', lw=.7)
            ax.set(title=f'{TITLES[condition]} | {LABELS[arm]}', xlim=(0,1), ylim=(0,1),
                   xlabel='Analytic selective p-value', ylabel='MCMC rank p-value')
            ax.text(.04, .95, f'Mean absolute difference: {difference.abs().mean():.3f}\n5% decisions disagree: {disagree}/250',
                    transform=ax.transAxes, va='top', fontsize=8)
            style(ax)
    pd.DataFrame(diagnostics).to_csv(output/'mcmc_diagnostics.csv', index=False)
    save(fig, output, '03_mcmc_vs_analytic', 'MCMC: 20 updates to a hub, 511 independent 20-update branches. Orange lines mark 5%.\nThe MCMC rank test has additional hub conditioning; individual differences are not estimates of bias or invalidity.')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Paired comparison of the two Shapley targets', fontsize=15)
    disagreements = []
    for j, condition in enumerate(CONDITIONS):
        wide = d[d.condition == condition].pivot(index='iteration', columns='arm', values='p_selective')
        x, y = wide[ARMS[0]], wide[ARMS[1]]
        axes[0,j].scatter(x, y, s=15, alpha=.5, color='#6C62A8', edgecolors='none')
        axes[0,j].plot([0,1], [0,1], '--', color='#7E8490')
        axes[0,j].axvline(.05, color='#DA8431', lw=.8); axes[0,j].axhline(.05, color='#DA8431', lw=.8)
        axes[0,j].set(title=TITLES[condition], xlabel='Selected-Features Shapley Effect\nMCMC p-value',
                      ylabel='All-Features Shapley Effect\nMCMC p-value', xlim=(0,1), ylim=(0,1))
        style(axes[0,j])
        table = np.array([[int((((x < .05) == bool(a)) & ((y < .05) == bool(b))).sum())
                           for b in [0,1]] for a in [0,1]])
        ax = axes[1,j]
        ax.imshow(table, cmap='Blues', vmin=0, vmax=250)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f'{table[a,b]}\n({table[a,b]/250:.1%})', ha='center', va='center',
                        color='white' if table[a,b] > 125 else '#192333')
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Do not reject','Reject'],
               yticklabels=['Do not reject','Reject'], xlabel='All-Features Shapley Effect\ndecision', ylabel='Selected-Features Shapley Effect\ndecision')
        disagreements.append(dict(condition=condition, both_nonreject=int(table[0,0]), full_only=int(table[0,1]),
                                  selected_only=int(table[1,0]), both_reject=int(table[1,1])))
    pd.DataFrame(disagreements).to_csv(output/'paired_decisions.csv', index=False)
    save(fig, output, '04_paired_targets', 'Same responses, selected sets and tested feature for both arms. Decisions use p < 0.05.\nDisagreement is between different hypotheses, not an error count.')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('Sampler diagnostics and tested-space dimensions', fontsize=15)
    groups = [d[(d.condition == c) & (d.arm == a)] for c in CONDITIONS for a in ARMS]
    labels = ['Null\nSelected-Features\nShapley Effect', 'Null\nAll-Features\nShapley Effect', 'Signal\nSelected-Features\nShapley Effect', 'Signal\nAll-Features\nShapley Effect']
    axes[0].boxplot([g.branch_acceptance for g in groups], tick_labels=labels, widths=.55, showfliers=False)
    axes[0].set(title='Branch acceptance rates', ylabel='Accepted updates / proposed updates', ylim=(0,1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].boxplot([np.exp(g.log_selection_mass) for g in groups], tick_labels=labels, widths=.55, showfliers=False)
    axes[1].set(title='Conditional selection probability', ylabel='Null probability on the conditioned ray', ylim=(0,1.05))
    axes[2].boxplot([g['rank'] for g in groups], tick_labels=labels, widths=.55)
    axes[2].axhline(d.full_rank.iloc[0], color='#A5414A', ls='--', label='Full RBF space')
    axes[2].set(title='Tested-space rank', ylabel='Chi degrees of freedom', ylim=(0,14))
    axes[2].legend(frameon=False)
    for ax in axes: style(ax)
    save(fig, output, '05_sampler_and_geometry', 'Boxes: median and interquartile range; whiskers: 1.5 IQR. Outliers omitted in the first two panels.\nAcceptance does not measure mixing by itself. Rank 12 means testing the entire candidate RBF space.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Which features were tested?', fontsize=15)
    targeting = []
    for j, condition in enumerate(CONDITIONS):
        # Selection is shared across arms: count each paired trial once.
        group = d[(d.condition == condition) & (d.arm == ARMS[0])]
        heights = np.array([(group.target == target).mean() for target in range(m['p'])])
        axes[j].bar(np.arange(m['p']), heights, color=['#2878B5','#B8C3D1','#B8C3D1','#B8C3D1'], width=.65)
        for target, rate in enumerate(heights):
            axes[j].text(target, rate+.012, f'{rate:.1%}', ha='center', fontsize=9)
            targeting.append(dict(condition=condition, target=target, count=int((group.target == target).sum()), rate=rate))
        axes[j].set(title=TITLES[condition], xlabel='Tested feature', ylabel='Fraction of paired trials',
                    xticks=range(m['p']), ylim=(0,.60))
        axes[j].yaxis.set_major_formatter(PercentFormatter(1)); style(axes[j])
    pd.DataFrame(targeting).to_csv(output/'target_frequencies.csv', index=False)
    save(fig, output, '06_target_frequencies', 'One target is sampled uniformly from the selected pair. Feature 0 generates the quadratic signal.\nPopulation targeting probability is at most 50% per feature; finite-sample frequencies can exceed 50%.')

    reference_panels(d, output)

    lines = ['# RBF Shapley analysis', '',
             'Validated all export checksums, receipt checksums, manifest identity, trial coverage and arm pairing.', '',
             '500 paired datasets; 250 trials per target per condition; 1,000 tests. Rejection threshold: p < 0.05.', '',
             '| Condition | Target | MCMC rejection (95% CI) | Analytic | Naive |',
             '|---|---|---:|---:|---:|']
    for condition in CONDITIONS:
        for arm in ARMS:
            subset = rates[(rates.condition == condition) & (rates.arm == arm)].set_index('method')
            r = subset.loc['selective']
            lines.append(f'| {TITLES[condition]} | {LABELS[arm]} | {r.rate:.1%} ({r.ci_lower:.1%}-{r.ci_upper:.1%}) | {subset.loc["exact", "rate"]:.1%} | {subset.loc["naive", "rate"]:.1%} |')
    lines += ['', 'Intervals are pointwise exact binomial intervals. All null-condition hypotheses are true; all signal-condition hypotheses are false in this design.', '',
              'The All-Features Shapley Effect tested space has rank 12, the entire candidate RBF space; Selected-Features Shapley Effect tests have rank 6. The higher All-Features Shapley Effect signal rejection rate compares different hypotheses and should not be read as superior identification of the generating feature.', '',
              'Naive All-Features Shapley Effect and analytic selective p-values coincide here because the full-space radial scaling preserves the homogeneous quadratic top-k selector. This simplification does not apply to the Selected-Features Shapley Effect test.', '',
              'MCMC versus analytic differences are diagnostic: the rank test additionally conditions on a hub, so exact equality is not expected. High acceptance alone does not establish rapid mixing.', '',
              '## Figures', '']
    for path in sorted(output.glob('0*.png')):
        lines.append(f'- [{path.stem.replace("_", " ")}]({path.name}) (also PDF and SVG).')
    lines += ['', '## Reproduce', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe examples/plot_rbf_shapley.py', '```', '']
    (output/'analysis_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    gallery(output)
    exports = {p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name != 'analysis_provenance.json'}
    provenance = dict(input_results_sha256=digest(source/'results.csv'), manifest_sha256=canonical(m),
                      script_sha256=digest(Path(__file__)), validated_tests=len(d), files=exports,
                      style_reference='outputs/selection_method_validation/exact_set/presentation_four_selectors_mcmc20_250_20261008/analysis',
                      figures=FIGURES)
    (output/'analysis_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), tests=len(d), figures=len(FIGURES), formats=['png','pdf','svg']), indent=2))


if __name__ == '__main__':
    main()
