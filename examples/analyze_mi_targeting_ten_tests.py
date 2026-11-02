"""Summarize the ten MI tests, including zero-score tie artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/selection_method_validation/exact_set/mi_parameter_tuning/targeting_ten_tests_20260920'
d = pd.read_csv(OUT / 'trials.csv')
assert len(d) == 8000 and d.configuration.nunique() == 10
assert d.k_select.eq(2).all()
assert d.groupby(['configuration', 'beta']).size().eq(200).all()
assert d.groupby(['shard', 'iteration', 'beta']).dataset_sha256.nunique().eq(1).all()
scores = np.array(d.scores.map(json.loads).tolist())
d['all_zero'] = (scores == 0).all(axis=1)
d['zero_cutoff'] = np.sort(scores, axis=1)[:, -2] == 0
d['signal_zero'] = scores[:, 0] == 0
d['selected_with_positive_score'] = d.signal_selected & ~d.signal_zero
# Average the current increasing-index tie-breaker over random relabelings.
# This is an audit calculation only; it does not change the tested target rule.
greater = (scores > scores[:, :1]).sum(axis=1)
equal = (scores == scores[:, :1]).sum(axis=1)
d['tie_averaged_inclusion'] = np.clip((2 - greater) / equal, 0, 1)
audit = d.groupby(['configuration', 'beta'], sort=False).agg(
    zero_signal_trials=('signal_zero', 'sum'), zero_cutoff_trials=('zero_cutoff', 'sum'),
    all_zero_trials=('all_zero', 'sum'), positive_score_inclusions=('selected_with_positive_score', 'sum'),
    tie_averaged_inclusion=('tie_averaged_inclusion', 'mean')).reset_index()
s = pd.read_csv(OUT / 'summary.csv').merge(audit, on=['configuration', 'beta'], validate='one_to_one')
s.to_csv(OUT / 'summary_with_tie_audit.csv', index=False)
previous = pd.read_csv(ROOT / 'outputs/selection_method_validation/exact_set/mi_parameter_tuning/estimators_histogram_kde_200_trials_20260912/analysis/power_summary.csv')
for beta, g in d[d.configuration.eq('kde_1')].groupby('beta'):
    old = previous[previous.method.eq('Gaussian KDE') & previous.beta.eq(beta)]
    assert len(old) == 1 and int(g.signal_targeted.sum()) == int(old.iloc[0].signal_targets)
print(s.to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for name, g in s.groupby('configuration', sort=False):
    for ax, column in zip(axes, ['selection_rate', 'targeting_rate', 'tie_averaged_inclusion']):
        ax.plot(g.beta, g[column], marker='o', label=name)
for ax, title in zip(axes, ['Signal inclusion, k=2', 'Realized signal targeting', 'Inclusion averaged over tie ordering']):
    ax.set(title=title, xlabel='Signal strength (beta)', ylim=(0, 1.03), xticks=[.3, .5, .75, 1.])
    ax.grid(alpha=.2)
axes[1].axhline(.5, color='gray', ls='--', lw=1)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5)
fig.suptitle('10 MI parameter tests: identical data, seeds, target rule, and k=2')
fig.tight_layout(rect=(0, .14, 1, .94))
fig.savefig(OUT / 'targeting_comparison.png', dpi=160)
plt.close(fig)
