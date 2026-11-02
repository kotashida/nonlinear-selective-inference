"""Extend the analytic RBF plots with reference-style outcomes and comparisons.

Uses only verified downloaded results. Does not change frozen experiment sources.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import subprocess
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/rbf_shapley'/study.NAME)
    args=parser.parse_args();source=args.input.resolve();out=source/'analysis'
    # The original frozen plotter validates completion and all source exports.
    subprocess.run([sys.executable,str(ROOT/'examples/plot_rbf_shapley_analytic_power.py'),
                    '--input',str(source)],check=True)
    provenance=study.core.read(out/'analysis_provenance.json')
    figures=provenance['figures']
    m=study.core.read(source/'requested_manifest.json')
    d=pd.read_csv(source/'results.csv');power=pd.read_csv(source/'power_summary.csv')
    arms=list(study.LABELS);betas=[b for b in m['betas'] if b>0]
    colors=['#2674a6','#d58122']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    def percent(ax):
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.set_axisbelow(True)
    def save(fig,name,caption):
        fig.tight_layout(rect=(0,.02,1,.94))
        for ext in ['png','pdf','svg']:
            fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(fig);figures.append(dict(name=name,caption=caption))

    fig,axes=plt.subplots(1,2,figsize=(14,5));outcomes=[]
    for ax,arm in zip(axes,arms):
        group=d[(d.arm==arm)&(d.beta>0)&(d.target==0)]
        rates=[];counts=[]
        for beta in betas:
            g=group[group.beta==beta];n=len(g);k=int(g.rejected.sum())
            counts.append(n);rates.append(k/n if n else np.nan)
            outcomes.append(dict(arm=arm,beta=beta,n=n,detected=k,resolved_not_detected=n-k))
        x=np.arange(len(betas))
        ax.bar(x,rates,color='#258568',label='Detected')
        ax.bar(x,1-np.array(rates),bottom=rates,color='#bcc8d1',label='Resolved, not detected')
        for j,(rate,n) in enumerate(zip(rates,counts)):
            ax.text(j,1.025,f'n={n}',ha='center',fontsize=9)
            ax.text(j,rate/2,f'{rate:.1%}',ha='center',va='center',color='white',fontsize=9)
        ax.set(title=study.LABELS[arm],xticks=x,xticklabels=betas,ylim=(0,1.13),xlabel='Signal strength (beta)');percent(ax)
    axes[0].legend(loc='lower right',fontsize=8)
    fig.suptitle('Outcomes conditional on targeting the generating feature')
    pd.DataFrame(outcomes).to_csv(out/'signal_outcomes.csv',index=False)
    save(fig,'05_signal_outcomes','Only trials with target=0 are included. Counts above bars are the actual denominators; green is analytic p < 0.05, gray is resolved non-rejection. No computational failures were recorded. The two effect definitions have different null hypotheses.')

    fig,axes=plt.subplots(1,2,figsize=(15,5))
    subset=power[power.metric=='conditional_signal_detection']
    for ax,col,title in zip(axes,['rate','ci_lower'],['Conditional signal detection','Pointwise 95% lower confidence bound']):
        values=subset.pivot(index='arm',columns='beta',values=col).reindex(arms)[betas].to_numpy()
        ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
        ax.set(title=title,yticks=range(2),yticklabels=[study.LABELS[a].replace(' Shapley','\nShapley') for a in arms],
               xticks=range(len(betas)),xticklabels=betas,xlabel='Signal strength (beta)')
        for i,arm in enumerate(arms):
            for j,beta in enumerate(betas):
                r=subset[(subset.arm==arm)&(subset.beta==beta)].iloc[0]
                label=f'{r[col]:.1%}'
                if col=='rate': label+=f'\n{int(r.rejections)}/{int(r.n)}'
                ax.text(j,i,label,ha='center',va='center',color='white' if values[i,j]>.65 else '#203040',fontsize=10)
    fig.suptitle('Signal-target detection and its uncertainty')
    save(fig,'06_power_heatmap','Conditional detection among trials targeting feature 0. Counts are detections / feature-0 targets. Right panel shows pointwise 95% Clopper-Pearson lower bounds, with no multiplicity adjustment. Rates compare different Shapley null hypotheses.')

    paired=[]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,only_target0 in zip(axes,[False,True]):
        tables=[]
        for beta in betas:
            g=d[d.beta==beta]
            if only_target0: g=g[g.target==0]
            wide=g.pivot(index='iteration',columns='arm',values='rejected').reindex(columns=arms)
            a,b=wide[arms[0]].astype(bool),wide[arms[1]].astype(bool)
            counts=np.array([int((a&b).sum()),int((a&~b).sum()),int((~a&b).sum()),int((~a&~b).sum())])
            tables.append(counts/len(wide))
            paired.append(dict(beta=beta,stratum='target_0' if only_target0 else 'all_signal_tests',n=len(wide),
                               both_reject=int(counts[0]),selected_only=int(counts[1]),all_only=int(counts[2]),neither=int(counts[3])))
        tables=np.asarray(tables);bottom=np.zeros(len(betas))
        for j,(label,color) in enumerate(zip(['Both reject','Selected-Features only','All-Features only','Neither rejects'],
                                            ['#258568','#2674a6','#d58122','#bcc8d1'])):
            ax.bar(range(len(betas)),tables[:,j],bottom=bottom,color=color,label=label);bottom+=tables[:,j]
        ax.set(title='Targeting feature 0' if only_target0 else 'All signal-condition tests',
               xticks=range(len(betas)),xticklabels=betas,xlabel='Signal strength (beta)',ylim=(0,1.03));percent(ax)
    axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.17),ncol=2)
    fig.suptitle('Paired rejection outcomes for the two Shapley effects')
    pd.DataFrame(paired).to_csv(out/'paired_decisions.csv',index=False)
    save(fig,'07_paired_detection_comparisons','Each paired decision uses the same response, selected set and tested feature. Rejection is strict p < 0.05. Disagreements are between different hypotheses, not an error count. Left denominator is all 1,000 trials; right denominator is the number targeting feature 0.')

    fig,axes=plt.subplots(2,2,figsize=(12,10),sharex=True,sharey=True)
    for ax,beta in zip(axes.flat,betas):
        g=d[d.beta==beta]
        wide=g.pivot(index='iteration',columns='arm',values='p_value').reindex(columns=arms)
        targets=g[g.arm==arms[0]].set_index('iteration').target.reindex(wide.index)
        for is_x0,color,label in [(False,'#bcc8d1','Other feature'),(True,'#2674a6','Feature 0')]:
            mask=targets.eq(0).to_numpy()==is_x0
            ax.scatter(wide.loc[mask,arms[0]],wide.loc[mask,arms[1]],s=10,alpha=.4,color=color,label=label,rasterized=True)
        ax.plot([0,1],[0,1],'--',color='gray');ax.axvline(.05,lw=.8,color='#d58122');ax.axhline(.05,lw=.8,color='#d58122')
        ax.set(title=f'beta = {beta:g}',xlim=(0,1),ylim=(0,1),
               xlabel='Selected-Features Shapley Effect\np-value',ylabel='All-Features Shapley Effect\np-value');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Same-target analytic p-values')
    save(fig,'08_same_target_p_values','Blue points test feature 0; gray points test another feature. Each point is one paired trial. Orange lines mark alpha=0.05; the diagonal indicates equal p-values. Axes correspond to different effect nulls, so differences are not approximation errors.')

    fig,axes=plt.subplots(1,2,figsize=(13,5));null=d[d.beta==0]
    for ax,arm,color in zip(axes,arms,colors):
        p=np.sort(null[null.arm==arm].p_value.to_numpy())
        ax.plot((np.arange(len(p))+.5)/len(p),p,color=color)
        ax.plot([0,1],[0,1],'--',color='gray');ax.set(title=study.LABELS[arm],
            xlabel='Uniform reference quantile',ylabel='Observed p-value',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.2)
    fig.suptitle('Global-null p-value quantiles')
    save(fig,'09_null_quantiles','1,000 global-null tests per effect. Below-diagonal values indicate smaller p-values than the uniform reference. This is a descriptive finite-sample diagnostic; no inference is rerun.')

    # Keep the numerical findings from the original gallery; replace its figure list.
    readme=(out/'README.md').read_text(encoding='utf-8')
    intro=readme.split('## ',1)[0].rstrip()
    md=[intro];sections=[]
    for item in figures:
        title=item['name'][3:].replace('_',' ').title()
        md.append(f'## {title}\n\n{item["caption"]}\n\n![{title}]({item["name"]}.png)')
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(item["caption"])}</p><a href="{item["name"]}.png"><img loading="lazy" src="{item["name"]}.png" alt="{html.escape(title)}"></a></section>')
    md.append('## Reproduction\n\n`.venv/Scripts/python.exe examples/build_rbf_shapley_analytic_power_gallery.py`\n\nPNG, PDF and SVG exports use verified saved results; no inference is rerun.')
    (out/'README.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analytic RBF power study</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>Analytic RBF Shapley power study</h1><p>1,000 trials per beta and effect; 10,000 analytic tests. Fresh independent Gaussian features, n=100, p=4, k=2. Both effects share datasets, selection and actual targets. Their null hypotheses differ.</p>'+''.join(sections)+'</html>',encoding='utf-8')
    provenance.update(figures=figures,gallery_script_sha256=study.digest(Path(__file__)),
        style_reference='outputs/selection_method_validation/exact_set/presentation_four_selectors_mcmc20_250_20261008/analysis',
        files_sha256={p.name:study.digest(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_provenance.json'})
    study.core.atomic(out/'analysis_provenance.json',provenance)
    print(f'Created {len(figures)} figures and gallery in {out}')


if __name__=='__main__': main()
