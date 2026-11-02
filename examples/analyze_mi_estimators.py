"""Audit and compare histogram/KDE MI with the existing kNN-30 benchmark."""
import contextlib
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_matched_ais_methods as matched
from analyze_ais_method_comparison import _read_raw
from verify_mi_estimators import verify
from verify_mi_neighbors_sweep import verify_variant

BASE = matched.BASE / 'mi_parameter_tuning'
RUN = BASE / 'estimators_histogram_kde_200_trials_20260912'
OUT = RUN / 'analysis'
ROOTS = {
    'kNN-30': BASE / 'neighbors_12_16_20_25_30_200_trials_20260911/neighbors_30',
    'Histogram (5 bins)': RUN / 'histogram',
    'Gaussian KDE': RUN / 'kde',
}
COLORS = dict(zip(ROOTS, ['#673e91', '#cc7832', '#198675']))
BETAS = [.3, .5, .75, 1.]
FIGURES = []


def load():
    with contextlib.redirect_stdout(io.StringIO()):
        verify(RUN)
        verify_variant(ROOTS['kNN-30'].parent, 30)
    audits, manifests, frames = [], {}, []
    for label, root in ROOTS.items():
        settings = json.loads((root / 'pooled_summary/validation_settings.json').read_text())
        manifests[label] = {'source': str(root.relative_to(matched.BASE)), 'settings': settings}
        for shard in range(32):
            for condition in ['null_fresh'] + [f'power_feature_0_beta_{str(b).replace(".", "p")}' for b in BETAS]:
                relative = Path(f'shard_{shard}/mutual_information/baseline/{condition}')
                a = json.loads((ROOTS['kNN-30'] / relative / 'settings.json').read_text())
                b = json.loads((root / relative / 'settings.json').read_text())
                differences = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
                assert set(differences) <= {'selector_settings'}, (label, relative, differences)
                audits.append(dict(option=label, shard=shard, condition=condition,
                                   differing_keys=','.join(differences), shared_settings_match=True))
                if condition == 'null_fresh':
                    np.testing.assert_array_equal(np.load(root / relative / 'fixed_design.npy'),
                                                  np.load(ROOTS['kNN-30'] / relative / 'fixed_design.npy'))
        n, t, f = _read_raw(root, 'ais', settings, 32)
        keys = ['selection_method', 'validation_shard', 'iteration', 'selection_event', 'signal_strength']
        t = t.merge(f[keys + ['feature', 't_obs', 'proposals', 'denominator_ess', 'tail_ess']],
                    on=keys, validate='one_to_one')
        for frame, scope in [(n, 'null_calibration'), (t, 'power')]:
            frame['scope'] = scope
            frame['option'] = label
            frame['source_run'] = str(root.relative_to(matched.BASE)).replace('\\', '/')
            frame['ais_setting'] = 'combined'
            frame['max_final_samples'] = 51200
            frame['pilot_iters'] = 10
            frame['pilot_samples'] = 320
            frame['total_proposals'] = frame.proposals + 3200
            frames.append(frame)
        for row in pd.read_csv(root / 'pooled_summary/power_decisions.csv').itertuples():
            g = t[t.signal_strength.eq(row.signal_strength)]
            assert len(g) == row.n_iterations == 200
            assert g.target_is_signal.sum() == row.n_signal_targets
            assert np.isclose(g.successful_detection.mean(), row.marginal_detection_power)
        for row in pd.read_csv(root / 'pooled_summary/calibration_decisions.csv').itertuples():
            assert len(n) == row.n_iterations == 200
            assert n.failed.sum() == row.n_failed
            assert (n.p_value.lt(row.alpha) & ~n.failed).sum() == row.n_rejected
        print('Verified and loaded', label, flush=True)
    data = pd.concat(frames, ignore_index=True)
    keys = ['option', 'scope', 'validation_shard', 'iteration', 'selection_event', 'signal_strength']
    assert len(data) == 3000 and not data.duplicated(keys).any()
    data.to_csv(OUT / 'unified_tests.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'configuration_audit.csv', index=False)
    (OUT / 'source_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    return data


def paired(data):
    audits, changes = [], []
    for label in ROOTS:
        for scope in ['null_calibration', 'power']:
            keys = ['validation_shard', 'iteration', 'selection_event'] + (['signal_strength'] if scope == 'power' else [])
            ref = data[data.option.eq('kNN-30') & data.scope.eq(scope)]
            other = data[data.option.eq(label) & data.scope.eq(scope)]
            pair = ref.merge(other, on=keys, suffixes=('_ref', '_new'), validate='one_to_one')
            assert len(pair) == len(ref)
            for field in ['target_seed', 'auxiliary_u', 'ais_seed']:
                if field + '_ref' in pair:
                    np.testing.assert_array_equal(pair[field + '_ref'], pair[field + '_new'])
            audits.append(dict(option=label, scope=scope, matched_trials=len(pair),
                               changed_targets=int(pair.target_feature_ref.ne(pair.target_feature_new).sum())))
            if scope == 'power':
                for beta, g in pair.groupby('signal_strength'):
                    old = g.successful_detection_ref.astype(bool)
                    new = g.successful_detection_new.astype(bool)
                    changes.append(dict(option=label, beta=beta, gained_detections=int((~old & new).sum()),
                                        lost_detections=int((old & ~new).sum()), net_detections=int(new.sum()-old.sum()),
                                        shared_signal_targets=int((g.target_is_signal_ref & g.target_is_signal_new).sum()),
                                        changed_targets=int(g.target_feature_ref.ne(g.target_feature_new).sum())))
    pd.DataFrame(audits).to_csv(OUT / 'matched_trial_audit.csv', index=False)
    result = pd.DataFrame(changes)
    result.to_csv(OUT / 'paired_detection_changes_vs_knn30.csv', index=False)
    return result


def save(fig, name, caption):
    fig.tight_layout(rect=(0, .08 if fig.legends else 0, 1, .95))
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    FIGURES.append((name, caption))


def plots(data, p, c, changes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [('conditional_power', 'Conditional power; 95% Wilson intervals'),
               ('marginal_power', 'Signal detections / all trials'),
               ('signal_target_rate', 'Signal targets / all trials'),
               ('signal_failure_rate', 'Failures / signal targets')]
    for label in ROOTS:
        g = p[p.method.eq(label)].sort_values('beta')
        for ax, (metric, title) in zip(axes.flat, metrics):
            ax.plot(g.beta, g[metric], '-o', color=COLORS[label], label=label)
            ax.set(title=title, xlabel='Signal strength (beta)', xticks=BETAS)
            ax.grid(alpha=.2)
        axes[0, 0].fill_between(g.beta, g.ci_lower, g.ci_upper, color=COLORS[label], alpha=.10)
    axes[0, 0].axhline(.8, ls='--', color='gray'); axes[0, 0].set_ylim(0, 1.03)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3)
    fig.suptitle('MI estimators under identical combined AIS')
    save(fig, '01_power_and_selection', '200 trials per beta. Failed signal tests count as misses. Conditional target populations differ.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, label in enumerate(ROOTS):
        row = p[p.method.eq(label) & p.beta.eq(.5)].iloc[0]
        axes[0].errorbar(row.conditional_power, i, xerr=[[row.conditional_power-row.ci_lower], [row.ci_upper-row.conditional_power]], fmt='o', capsize=4, color=COLORS[label])
        axes[0].annotate(f'{int(row.detections)}/{int(row.signal_targets)} = {row.conditional_power:.1%}', (row.conditional_power, i), xytext=(0, 12), textcoords='offset points', ha='center')
        axes[1].bar(i, row.detections, color=COLORS[label])
        axes[1].bar(i, row.signal_targets-row.detections, bottom=row.detections, color='#cbd2d8')
        axes[1].text(i, row.signal_targets+2, f'{int(row.signal_targets)} targets', ha='center')
    axes[0].set(yticks=range(3), yticklabels=list(ROOTS), xlim=(.35, 1), ylim=(-.6, 2.7), title='Conditional power at beta = 0.50')
    axes[0].axvline(.8, color='gray', ls='--'); axes[0].grid(axis='x', alpha=.2)
    axes[1].set(xticks=range(3), xticklabels=list(ROOTS), ylim=(0, 110), title='Detected signals and non-detections', ylabel='Count out of 200 trials')
    fig.suptitle('The 80% objective: compare both numerator and denominator')
    save(fig, '02_beta05_target', 'Colored bars: detected signals. Gray: signal targets not detected. All three options have zero signal-target failures at beta=0.50; histogram has one failed noise-target test.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, label in enumerate(ROOTS):
        g = c[c.method.eq(label)]
        axes[0].errorbar(g.alpha+i*.001-.001, g.fpr, yerr=np.maximum(0, np.vstack([g.fpr-g.ci_lower, g.ci_upper-g.fpr])), fmt='o-', color=COLORS[label], label=label, capsize=3)
        values = np.sort(data.loc[data.option.eq(label) & data.scope.eq('null_calibration') & ~data.failed, 'p_value'])
        axes[1].step(values, np.arange(1, len(values)+1)/len(values), where='post', color=COLORS[label], label=label)
    axes[0].plot([0, .12], [0, .12], '--', color='gray')
    axes[0].set(title='Null FPR with pointwise 95% intervals', xlabel='Nominal alpha', ylabel='Observed rejection rate')
    axes[1].plot([0, 1], [0, 1], '--', color='gray'); axes[1].set(title='Null p-value empirical CDF', xlabel='p-value', ylabel='Fraction at or below p')
    for ax in axes: ax.grid(alpha=.2)
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=3)
    fig.suptitle('Null calibration: 200 successful tests per estimator')
    save(fig, '03_null_calibration', 'All null tests succeeded. Finite-sample intervals do not establish exact calibration.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label in ROOTS:
        g = p[p.method.eq(label)].sort_values('beta')
        axes[0].plot(g.beta, g.mean_total_proposals, '-o', color=COLORS[label], label=label)
        axes[1].plot(g.beta, g.failures, '-o', color=COLORS[label], label=label)
    for ax in axes: ax.set(xlabel='Signal strength (beta)', xticks=BETAS); ax.grid(alpha=.2)
    axes[0].set(title='Mean proposals, including 3,200 pilot draws', ylabel='Proposals per trial')
    axes[1].set(title='Failed inference across all targets', ylabel='Failures / 200 trials')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3)
    fig.suptitle('Sampling cost and reliability')
    save(fig, '04_cost_and_failures', 'Proposal counts are not wall-clock costs: each estimator has a different cost per selector evaluation.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, label in zip(axes, list(ROOTS)[1:]):
        g = changes[changes.option.eq(label)].sort_values('beta')
        ax.bar(np.arange(4)-.16, g.gained_detections, width=.32, color='#198675', label='Gained detections')
        ax.bar(np.arange(4)+.16, -g.lost_detections, width=.32, color='#b54d54', label='Lost detections')
        ax.axhline(0, color='gray'); ax.set(title=label, xticks=range(4), xticklabels=BETAS, xlabel='Signal strength (beta)')
        ax.grid(axis='y', alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2)
    fig.suptitle('Paired signal-detection changes relative to kNN-30')
    save(fig, '05_paired_changes', 'Pairs share simulation seeds and target randomization draws. Selected targets may differ, so this compares complete procedures.')


def main():
    OUT.mkdir(exist_ok=True)
    data = load(); changes = paired(data)
    temp = data.copy(); temp['selection_method'] = temp.option
    previous = matched.OUT; matched.OUT = OUT
    p, c = matched.summarize(temp); matched.OUT = previous
    p['signal_target_rate'] = p.signal_targets / p.n
    p.to_csv(OUT / 'power_summary.csv', index=False)
    plots(data, p, c, changes)
    intro = 'Combined 3,000 test records: kNN-30, five-bin quantile histogram MI, and Gaussian KDE MI. Each has 200 null trials and 200 trials at each of four betas. The existing kNN-30 control is included once. Original experiment folders are preserved.'
    audit = 'All 480 low-level manifests match the kNN-30 simulation, seeds, AIS controls, iteration counts and software metadata, allowing only selector_settings to differ. Fixed null design arrays are identical. Trial keys are unique; available AIS/target seeds and auxiliary draws agree. Raw results reproduce the pooled counts.'
    finding = 'KDE is the strongest new candidate, but not a uniform improvement. At beta=0.50 it detects 71/90=78.89% versus kNN-30 at 67/87=77.01%; marginal detection improves from 33.5% to 35.5%. Both remain below the 80% objective. Histogram MI reaches only 34/58=58.62%, with 17.0% marginal detection.'
    tradeoff = 'At beta=0.30, KDE improves conditional power from 34.78% to 49.06%, with 26 detections versus 16. At beta=0.75, both detect 94 signals, but KDE has 101 signal targets versus 99 for kNN-30, giving 93.07% versus 94.95%. At beta=1.00, KDE detects 100/104 versus kNN-30 at 103/104 (96.15% versus 99.04%). Histogram MI is substantially weaker at beta>=0.50.'
    caveat = 'Power here counts failed signal tests as misses, unlike the completed-test conditional estimate in the original pooled tables. Null FPR divides by successful null tests; all 600 succeeded. At nominal 5%, kNN-30 and KDE each reject 10/200 (5%); histogram rejects 7/200 (3.5%). These counts do not prove calibration. Intervals are pointwise and do not account for repeatedly tuning on these seeds. Conditional power differences also reflect different selected-target populations.'
    recommendation = 'Retain kNN-30 if performance at the stronger signals must not degrade. KDE merits consideration for weak-signal performance and computational efficiency, but this experiment does not establish overall superiority or achieve 80% at beta=0.50. Histogram MI is not competitive here. Do not equate fewer proposals or an earlier local timing benchmark with measured full-run server speed.'
    reliability = 'KDE has zero failures across 800 power trials, versus one for kNN-30 and three for histogram MI. KDE uses fewer mean proposals at every beta (about 1-8% fewer, including pilot draws). At beta=0.50 the pointwise 95% conditional-power intervals are 69.4-86.0% for KDE and 67.1-84.6% for kNN-30. The paired comparison gains 16 detections and loses 12, a net gain of four; this small advantage does not establish superiority.'
    sections = [intro, audit, finding, tradeoff, reliability, caveat, recommendation]
    report = ['# MI estimator comparison under combined AIS', *sections, '## Power', matched.table(p[['method', 'beta', 'detections', 'signal_targets', 'conditional_power', 'ci_lower', 'ci_upper', 'marginal_power', 'failures', 'mean_total_proposals']]), '## Null calibration', matched.table(c), '## Paired changes', matched.table(changes), '## Plots']
    gallery = []
    for name, caption in FIGURES:
        report.append(f'### {name[3:].replace("_", " ")}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)')
        gallery.append(f'<section><h2>{html.escape(name[3:].replace("_", " "))}</h2><p>{html.escape(caption)}</p><img src="{name}.png" alt="{html.escape(caption)}"><p><a href="{name}.pdf">PDF</a></p></section>')
    (OUT / 'analysis_report.md').write_text('\n\n'.join(report), encoding='utf-8')
    (OUT / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MI estimator comparison</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#203040}img{width:100%}section{margin:40px 0}</style><h1>MI estimators: combined AIS</h1>' + ''.join('<p>'+html.escape(t)+'</p>' for t in sections) + ''.join(gallery) + '</html>', encoding='utf-8')
    print(p.to_string(index=False)); print(c[c.alpha.eq(.05)].to_string(index=False)); print(changes.to_string(index=False))
    print('Saved', OUT)


if __name__ == '__main__':
    main()
