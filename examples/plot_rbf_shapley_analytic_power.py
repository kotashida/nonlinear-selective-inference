"""Plot verified analytic RBF power results; never run inference."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import run_rbf_shapley_analytic_power as study
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

COLORS={'selected_shapley':'#2674a6','full_shapley':'#d58122'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/rbf_shapley'/study.NAME)
    args=parser.parse_args();source=args.input.resolve();out=source/'analysis'
    done=study.core.read(source/'COMPLETED');m=study.core.read(source/'requested_manifest.json')
    study.core.require(done['status']=='verified' and done['manifest_sha256']==study.core.sha(m),'Unverified run')
    for name,sha in done['export_sha256'].items():
        study.core.require(Path(name).name==name and study.digest(source/name)==sha,'Export checksum mismatch')
    d=pd.read_csv(source/'results.csv');power=pd.read_csv(source/'power_summary.csv')
    calibration=pd.read_csv(source/'calibration_summary.csv');truth=pd.read_csv(source/'null_truth_summary.csv')
    study.core.require(len(d)==m['iterations']*len(m['betas'])*2,'Incomplete results')
    out.mkdir(exist_ok=True);figures=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    betas=[b for b in m['betas'] if b>0]
    def percent(ax):
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.set_axisbelow(True)
    def save(fig,name,caption):
        fig.tight_layout(rect=(0,.02,1,.94))
        for ext in ['png','pdf','svg']:
            fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(fig);figures.append(dict(name=name,caption=caption))
    def curves(axes,metrics,titles):
        for arm,label in study.LABELS.items():
            for ax,metric in zip(axes,metrics):
                g=power[(power.arm==arm)&(power.metric==metric)].sort_values('beta')
                ax.plot(g.beta,g.rate,'o-',color=COLORS[arm],label=label)
                ax.fill_between(g.beta,g.ci_lower,g.ci_upper,color=COLORS[arm],alpha=.1)
        for ax,title in zip(axes,titles):
            ax.set(title=title,xlabel='Signal strength (beta)',xticks=betas,ylim=(0,1.03));percent(ax)

    fig,axes=plt.subplots(1,3,figsize=(17,5))
    curves(axes,['targeting_probability','conditional_signal_detection','marginal_signal_detection'],
           ['Probability of targeting feature 0','Conditional signal detection','Marginal signal detection'])
    for ax in [axes[0],axes[2]]: ax.axhline(.5,ls='--',color='gray')
    axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('Analytic selective inference: power and signal targeting')
    save(fig,'01_power_curves',f'{m["iterations"]:,} fresh paired inputs per beta; p < 0.05. Shading: pointwise 95% exact binomial intervals. Conditional detection counts rejections when target=0; marginal detection counts target=0 AND rejection among all trials. Both arms share selection and targets, so targeting curves coincide. Beta is the empirical signal SD and RMS. The population targeting ceiling is 50%; finite samples may exceed it. Different effect definitions test different hypotheses.')

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    curves(axes,['non_x0_rejection','all_signal_rejection'],
           ['Rejection when targeting another feature','Rejection across all signal-condition tests'])
    axes[0].legend(fontsize=8);fig.suptitle('Which signal is being detected?')
    save(fig,'02_other_target_rejection','The left panel conditions on target != 0; the right includes every tested feature. Neither is automatically a Type I error rate. Actual Shapley null truth is computed from the mean and tested space for every test. Population-independent features need not have orthogonal finite-sample RBF blocks. Full-range tests can detect signal outside the target feature.')

    fig,axes=plt.subplots(1,2,figsize=(13,5));null=d[d.beta==0]
    for j,(arm,label) in enumerate(study.LABELS.items()):
        g=calibration[calibration.arm==arm].sort_values('alpha')
        axes[0].errorbar(g.alpha+(j-.5)*.002,g.rate,yerr=[g.rate-g.ci_lower,g.ci_upper-g.rate],
                         fmt='o',capsize=3,color=COLORS[arm],label=label)
        p=np.sort(null[null.arm==arm].p_value)
        axes[1].step(np.r_[0,p,1],np.r_[0,np.arange(1,len(p)+1)/len(p),1],where='post',color=COLORS[arm])
    axes[0].plot([0,.12],[0,.12],'--',color='gray')
    axes[0].set(title=f'Type I error rates ({m["iterations"]:,} trials)',xlabel='Nominal alpha (points offset)',
                ylabel='Rejection fraction',xticks=m['alpha_levels'],ylim=(0,max(.16,float(calibration.ci_upper.max())+.02)))
    axes[0].legend(fontsize=8)
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(title='Null p-value empirical distribution',
        xlabel='p-value',ylabel='Fraction at or below p',xlim=(0,1),ylim=(0,1))
    for ax in axes: percent(ax)
    fig.suptitle('Type I error under the global null: beta = 0')
    save(fig,'03_null_calibration','At beta=0 the response is independent Gaussian noise and both effect nulls are true. Strict p < alpha rejection; intervals are pointwise 95% exact binomial intervals. Diagonals show nominal/uniform references. These are unconditional Monte Carlo summaries across fresh X, not a proof of conditional calibration.')

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for arm,label in study.LABELS.items():
        g=truth[(truth.arm==arm)&(truth.alpha==m['primary_alpha'])&truth.null_true].sort_values('beta')
        axes[0].plot(g.beta,g.n/(m['iterations']),'o-',color=COLORS[arm],label=label)
        g=d[d.arm==arm].groupby('beta')['rank'].agg(['min','max','mean']).reset_index()
        axes[1].plot(g.beta,g['mean'],'o-',color=COLORS[arm],label=label)
        axes[1].fill_between(g.beta,g['min'],g['max'],color=COLORS[arm],alpha=.1)
    axes[0].set(title='Fraction of tested nulls that are true',xlabel='Signal strength (beta)',ylim=(-.02,1.05));percent(axes[0]);axes[0].legend(fontsize=8)
    axes[1].set(title='Tested-space dimension',xlabel='Signal strength (beta)',ylabel='Rank (chi degrees of freedom)',ylim=(0,14));axes[1].grid(alpha=.2)
    fig.suptitle('Actual null truth and test geometry')
    save(fig,'04_null_truth_and_geometry','Truth means zero projection of the noiseless mean onto the tested Shapley space, using the saved numerical tolerance. Feature identity alone does not define truth. Rank curves show means with min/max shading. Full-candidate space has at most 12 RBF dimensions; rank is recorded for every test.')

    intro=(f'Analytic-only tests of Selected-Features Shapley Effect and All-Features Shapley Effect; '
           f'{m["iterations"]:,} repetitions per beta, {len(d):,} tests. n=100, p=4, k=2, independent Gaussian features; '
           'fresh X per repetition. Signal g(x)=x+0.5(x²−1), empirically centered and standardized, '
           'Y=beta·g_standardized(X0)+N(0,1). Inputs, noise and auxiliary target draws are shared across strengths and effects.')
    findings=[]
    for arm,label in study.LABELS.items():
        r=calibration[(calibration.arm==arm)&(calibration.alpha==.05)].iloc[0]
        findings.append(f'{label}: global-null Type I error {r.rate:.1%} ({int(r.rejections)}/{int(r.n)}), 95% interval {r.ci_lower:.1%}–{r.ci_upper:.1%}.')
    sections=[];md=['# Analytic RBF Shapley power study',intro,*findings,
        'The two effects have different null hypotheses. Higher full-space rejection does not establish superior identification of a unique feature effect. All intervals are pointwise.']
    for item in figures:
        title=item['name'][3:].replace('_',' ').title()
        md.append(f'## {title}\n\n{item["caption"]}\n\n![{title}]({item["name"]}.png)')
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(item["caption"])}</p><img src="{item["name"]}.png" alt="{html.escape(title)}"></section>')
    md.append('## Reproduction\n\n`.venv/Scripts/python.exe examples/plot_rbf_shapley_analytic_power.py --input PATH_TO_RESULTS`')
    (out/'README.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analytic RBF power study</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}</style><h1>Analytic RBF Shapley power study</h1><p>'+html.escape(intro)+'</p><ul>'+''.join('<li>'+html.escape(f)+'</li>' for f in findings)+'</ul>'+''.join(sections)+'</html>',encoding='utf-8')
    study.core.atomic(out/'analysis_provenance.json',dict(validated_tests=len(d),
        source_exports=done['export_sha256'],plot_script_sha256=study.digest(Path(__file__)),figures=figures,
        files_sha256={p.name:study.digest(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_provenance.json'}))
    print(f'Created {len(figures)} figures and gallery in {out}')


if __name__=='__main__': main()
