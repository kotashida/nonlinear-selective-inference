"""Regenerate the downloaded fpr.png with explicit zero-based rate axes."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/benchmarks/selected_five_null_fpr_10000_cpu32_20260928'
METHODS = ['shap_ais', 'lime_40_ais', 'spline_screening_ais',
           'spline_screening_exact', 'mutual_information_kde_2p5_ais']


def main():
    summary = pd.read_csv(OUT / 'summary.csv')
    analysis = OUT / 'analysis'
    analysis.mkdir(exist_ok=True)
    backup = analysis / 'fpr_original_autoscaled.png'
    if not backup.exists():
        shutil.copy2(OUT / 'fpr.png', backup)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    fig.subplots_adjust(left=.065, right=.985, top=.86, bottom=.23, wspace=.28)
    for ax, alpha in zip(axes, [.01, .05, .10]):
        s = summary[(summary.alpha == alpha) & summary.p_value_kind.eq('selective')]
        s = s.set_index('method').loc[METHODS]
        x = np.arange(len(s))
        ax.errorbar(x, s.fpr_all_attempts,
            yerr=[s.fpr_all_attempts-s.ci95_all_lower, s.ci95_all_upper-s.fpr_all_attempts],
            fmt='o', capsize=4, color='#2878B5', markersize=5)
        ax.axhline(alpha, color='#D62728', linestyle='--', linewidth=1.3)
        ax.set_ylim(0, 2*alpha)
        ax.set_yticks(np.linspace(0, 2*alpha, 5))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: '0' if value == 0 else f'{value:.3f}'.rstrip('0').rstrip('.')))
        for value, label in zip(ax.get_yticks(), ax.get_yticklabels()):
            if np.isclose(value, alpha):
                label.set_color('#D62728')
                label.set_fontweight('bold')
        ax.set_xticks(x, ['SHAP', 'LIME', 'Spline AIS', 'Spline exact', 'MI'],
                      rotation=35, ha='right')
        ax.set_title(f'Significance level: {100*alpha:g}%', pad=12)
        ax.set_ylabel('Type I error rate')
        ax.grid(axis='y', color='0.9', linewidth=.6)
        ax.set_axisbelow(True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'fpr.{extension}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    (analysis / 'fpr_axis_revision.json').write_text(json.dumps({
        'change': 'Explicit zero baseline, readable fixed ticks, nominal-level tick highlighted, bottom captions removed; original point estimates and intervals retained.',
        'original_plot_backup': str(backup.relative_to(OUT)),
        'original_plot_sha256': hashlib.sha256(backup.read_bytes()).hexdigest(),
        'revised_plot_sha256': hashlib.sha256((OUT/'fpr.png').read_bytes()).hexdigest(),
        'summary_sha256': hashlib.sha256((OUT/'summary.csv').read_bytes()).hexdigest(),
    }, indent=2)+'\n')
    print(OUT / 'fpr.png')


if __name__ == '__main__':
    main()
