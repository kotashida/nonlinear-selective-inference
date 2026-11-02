"""Export manuscript-ready plots from verified results; never runs inference."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmarks import run_publication_mcmc as e

LABELS = ["Tree SHAP", "LIME", "Spline MCMC", "MI", "Spline exact"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#444444"]


def plot(m):
    out = e.ROOT/m['output_root']
    completion = e.read(out/'verification.json')
    e.require(e.read(out/'COMPLETED') == completion, 'Unverified/incomplete benchmark')
    for name, checksum in completion['output_sha256'].items():
        e.require(e.digest(out/name) == checksum, f'Corrupt summary: {name}')
    figdir = out/'figures'
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    def save(fig, stem):
        for ext in ('png','pdf','svg'):
            fig.savefig(figdir/f'{stem}.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    rates = pd.read_csv(out/'rates.csv', keep_default_na=False)
    rows = rates[(rates.study == 'null') & (rates.alpha == .05)]
    conditions = rows.condition.unique()
    fig, axes = plt.subplots(1,len(conditions),figsize=(max(5,4*len(conditions)),3.7),squeeze=False)
    for ax, cond in zip(axes[0], conditions):
        g = rows[rows.condition == cond].set_index('method')
        for i, (method, color) in enumerate(zip(e.METHODS, COLORS)):
            r = g.loc[method]
            p, lo, hi = float(r.all_attempt_lower), float(r.missing_ci95_lower), float(r.missing_ci95_upper)
            ax.errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='o',color=color,capsize=3)
            if r.failures:
                ax.plot([i,i],[float(r.all_attempt_lower),float(r.all_attempt_upper)],color=color,lw=4,alpha=.3)
        ax.axhline(.05,color='black',ls='--',lw=1)
        ax.set(title=f'{cond}; attempts/arm={int(g.at[e.METHODS[0],"attempted"])}',
               xticks=range(5),xticklabels=LABELS,ylabel='Rejection fraction (p < 0.05)')
        ax.tick_params(axis='x',rotation=40)
        ax.set_ylim(bottom=0)
    fig.suptitle(('SMOKE ONLY - ' if m['smoke'] else '')+'Null calibration; pointwise 95% intervals include missing-outcome uncertainty')
    fig.tight_layout()
    save(fig,'null_calibration')
    cdf = pd.read_csv(out/'calibration.csv', keep_default_na=False)
    fig, ax = plt.subplots(figsize=(5,4))
    for method, label, color in zip(e.METHODS,LABELS,COLORS):
        g = cdf[(cdf.study == 'null') & (cdf.condition == 'iid_n100') & (cdf.method == method)]
        ax.step(g.threshold,pd.to_numeric(g.cdf_valid,errors='coerce'),where='post',label=label,color=color)
    ax.plot([0,1],[0,1],ls='--',color='black',lw=1,label='Superuniform boundary')
    ax.set(xlim=(0,.15),ylim=(0,.15),xlabel='p-value threshold',ylabel='Empirical CDF (finite results)',
           title=('SMOKE ONLY - ' if m['smoke'] else '')+'IID Gaussian null, n=100')
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig,'null_lower_tail_cdf')
    pred = pd.read_csv(out/'prediction.csv', keep_default_na=False)
    pred = pred[(pred.condition == 'observed_n100') & (pred.metric == 'rmse')]
    if len(pred):
        names = pred.dataset.unique()
        fig, axes = plt.subplots(int(np.ceil(len(names)/4)),min(4,len(names)),
            figsize=(max(5,min(4,len(names))*3.4),max(3.6,int(np.ceil(len(names)/4))*3.6)),squeeze=False)
        for ax, name in zip(axes.flat,names):
            g = pred[pred.dataset == name].set_index('method')
            for i,(method,color) in enumerate(zip(e.METHODS,COLORS)):
                r = g.loc[method]
                if r['mean'] == '':
                    continue
                value = float(r['mean'])
                lo, hi = r.conditional_mc_ci95_lower,r.conditional_mc_ci95_upper
                if lo == '' or hi == '':
                    ax.plot(i,value,'o',color=color)
                else:
                    ax.errorbar(i,value,yerr=[[max(0,value-float(lo))],[max(0,float(hi)-value)]],fmt='o',color=color,capsize=3)
            ax.set(title=name,xticks=range(5),xticklabels=LABELS,ylabel='Holdout RMSE (calibration response scale)')
            ax.tick_params(axis='x',rotation=40)
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle(('SMOKE ONLY - ' if m['smoke'] else '')+'Predictive performance; uncertainty conditional on frozen data and split')
        fig.tight_layout()
        save(fig,'observed_holdout_rmse')
    print(f'Exported PDF, SVG and PNG figures to {figdir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=e.CONFIG)
    plot(e.read_config(parser.parse_args().config))
