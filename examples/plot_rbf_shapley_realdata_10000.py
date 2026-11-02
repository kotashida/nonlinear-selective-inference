"""Table 1-style summaries and sample-size curves from verified real-data tests."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import run_rbf_shapley_realdata_10000 as study
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/benchmarks'/study.NAME)
    args=parser.parse_args();source=args.input.resolve();out=source/'analysis'
    done=study.core.read(source/'COMPLETED');m=study.core.read(source/'requested_manifest.json')
    study.core.require(done['status']=='verified' and done['manifest_sha256']==study.core.sha(m),'Unverified experiment')
    for name,sha in done['export_sha256'].items():
        study.core.require(Path(name).name==name and study.digest(source/name)==sha,'Export checksum mismatch')
    d=pd.read_csv(source/'results.csv');s=pd.read_csv(source/'summary.csv')
    study.core.require(len(d)==study.expected_tests(m),'Incomplete tests')
    nd=d[d.dataset==study.NULL_DATASET].copy()
    d=d[d.dataset!=study.NULL_DATASET].copy()
    s=s[s.alpha==m['primary_alpha']];arms=list(study.LABELS);colors=['#2674a6','#d58122']
    out.mkdir(exist_ok=True);figures=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name,caption):
        fig.tight_layout(rect=(0,.02,1,.94))
        for ext in ('png','pdf','svg'): fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(fig);figures.append(dict(name=name,caption=caption))
    def panels():
        cols=min(4,len(m['datasets']));nr=(len(m['datasets'])+cols-1)//cols
        fig,axes=plt.subplots(nr,cols,figsize=(4*cols,4*nr),squeeze=False)
        for ax in axes.flat[len(m['datasets']):]: ax.set_visible(False)
        return fig,list(axes.flat)
    def format_ax(ax,name):
        ax.set(title=name.replace('_',' ').title(),xlabel='Sample size (n)',xticks=m['sample_sizes'])
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)

    fig,axes=panels()
    for ax,name in zip(axes,m['datasets']):
        for arm,color in zip(arms,colors):
            g=s[(s.dataset==name)&(s.arm==arm)].sort_values('n')
            ax.plot(g.n,g.resolved_rejection_rate,'o-',color=color,label=study.LABELS[arm])
            ax.fill_between(g.n,g.resolved_ci_lower,g.resolved_ci_upper,color=color,alpha=.12)
        format_ax(ax,name);ax.set(ylim=(0,1.04),ylabel='Selective rejection rate')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,fontsize=9)
    fig.suptitle('Real-data analytic Shapley tests: sample-size comparison')
    save(fig,'01_sample_size_curves',f'Analytic p < 0.05 among resolved tests; shading is pointwise 95% exact binomial intervals. Each cell attempts {m["repetitions"]:,} tests. Failures are reported separately. The curves compare different effect nulls. Real-data null truth is unknown, so these are selective rejection rates, not empirically calibrated power. Sample sizes are nested within repetition; increases are evaluated, not imposed.')

    table=s[['dataset','n','arm','attempted','resolved','failures','rejections','resolved_rejection_rate','resolved_ci_lower','resolved_ci_upper','detection_fraction','detection_ci_lower','detection_ci_upper','unresolved_upper_fraction']]
    table.to_csv(out/'table1_rejection_rates.csv',index=False)
    table1=[]
    for n in m['sample_sizes']:
        row={'n':n}
        for name in m['datasets']:
            values=[]
            for arm in arms:
                r=s[(s.dataset==name)&(s.n==n)&(s.arm==arm)].iloc[0]
                values.append(f'{r.resolved_rejection_rate:.3f} ({int(r.rejections)}/{int(r.resolved)}; failures={int(r.failures)})')
            row[name]=' / '.join(values)
        table1.append(row)
    pd.DataFrame(table1).to_csv(out/'table1_compact.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(max(9,len(m['datasets'])*1.6),7.5))
    for ax,arm in zip(axes,arms):
        mat=s[s.arm==arm].pivot(index='n',columns='dataset',values='resolved_rejection_rate').reindex(index=m['sample_sizes'],columns=m['datasets']).to_numpy()
        ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set(title=study.LABELS[arm],yticks=range(3),yticklabels=m['sample_sizes'],ylabel='Sample size (n)',
               xticks=range(len(m['datasets'])),xticklabels=[v.replace('_','\n') for v in m['datasets']])
        for i,n in enumerate(m['sample_sizes']):
            for j,name in enumerate(m['datasets']):
                r=s[(s.dataset==name)&(s.n==n)&(s.arm==arm)].iloc[0]
                label=f'{r.resolved_rejection_rate:.1%}\n{int(r.rejections)}/{int(r.resolved)}'
                ax.text(j,i,label,ha='center',va='center',color='white' if mat[i,j]>.65 else '#203040',fontsize=9)
    fig.suptitle('Table 1-style comparison across datasets and sample sizes')
    save(fig,'02_rejection_heatmap','Each cell reports rejection fraction and rejections / resolved tests. Long-form CSV includes pointwise intervals, intended denominators, failure counts and attempted-test detection fractions. Compact table cells list Selected-Features / All-Features results in that order.')

    fig,axes=panels()
    for ax,name in zip(axes,m['datasets']):
        for j,(arm,color) in enumerate(zip(arms,colors)):
            g=s[(s.dataset==name)&(s.arm==arm)].sort_values('n')
            ax.bar(g.n+(j-.5)*9,g.failures/g.attempted,width=8,color=color,label=study.LABELS[arm])
        format_ax(ax,name);ax.set(ylim=(0,max(.05,float((s.failures/s.attempted).max())*1.2)),ylabel='Unresolved / attempted')
    fig.suptitle('Computation failures retained in the trial accounting')
    save(fig,'03_failure_rates','Errors are retained as unresolved outcomes, not converted into statistical non-rejections. Detection fractions over all attempted trials and unresolved upper fractions are saved separately from resolved-test rejection rates. Failure messages are retained in results.csv and batch receipts.')

    changes=[];rng=np.random.default_rng(m['seed']+1)
    for name in m['datasets']:
        for arm in arms:
            g=d[(d.dataset==name)&(d.arm==arm)]
            values=g.pivot(index='repetition',columns='n',values='p_value')
            for lo,hi in [(100,150),(100,200),(150,200)]:
                pair=values[[lo,hi]].dropna();diff=(pair[hi].to_numpy()<.05).astype(int)-(pair[lo].to_numpy()<.05).astype(int)
                if len(diff):
                    counts=np.array([(diff==-1).sum(),(diff==0).sum(),(diff==1).sum()])
                    bootstrap=rng.multinomial(len(diff),counts/len(diff),size=10000)
                    means=(bootstrap[:,2]-bootstrap[:,0])/len(diff)
                    low,high=np.quantile(means,[.025,.975]);delta=float(diff.mean())
                else: delta=low=high=np.nan
                changes.append(dict(dataset=name,arm=arm,n_from=lo,n_to=hi,resolved_pairs=len(diff),
                                    excluded_pairs=m['repetitions']-len(diff),difference=delta,ci_lower=low,ci_upper=high))
    changes=pd.DataFrame(changes);changes.to_csv(out/'paired_sample_size_changes.csv',index=False)
    fig,axes=panels()
    for ax,name in zip(axes,m['datasets']):
        for j,(arm,color) in enumerate(zip(arms,colors)):
            g=changes[(changes.dataset==name)&(changes.arm==arm)]
            ax.errorbar(np.arange(3)+(j-.5)*.13,g.difference,yerr=[np.maximum(0,g.difference-g.ci_lower),np.maximum(0,g.ci_upper-g.difference)],fmt='o',capsize=3,color=color)
        ax.axhline(0,color='gray',ls='--');ax.set(title=name.replace('_',' ').title(),xticks=range(3),xticklabels=['150−100','200−100','200−150'],ylabel='Change in rejection fraction')
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    fig.suptitle('Paired changes as sample size increases')
    save(fig,'04_paired_sample_size_changes','Differences use repetitions resolved at both sample sizes. Pointwise percentile intervals use 10,000 paired bootstrap resamples, seed=master seed+1; multinomial sampling of the paired differences is equivalent to resampling repetitions. Intervals are not adjusted for multiple comparisons. Positive values indicate more frequent rejection at the larger sample size; no monotonicity is assumed.')

    null_summary=pd.read_csv(source/'type1_error_summary.csv')
    primary=null_summary[null_summary.alpha==m['primary_alpha']].set_index('arm').loc[arms]
    fig,ax=plt.subplots(figsize=(10,5))
    for j,(arm,color) in enumerate(zip(arms,colors)):
        r=primary.loc[arm]
        ax.errorbar(j,r.rate,yerr=[[r.rate-r.ci_lower],[r.ci_upper-r.rate]],fmt='x',
                    markersize=9,capsize=5,markeredgewidth=1.6,color=color)
    ax.axhline(.05,color='#888888',ls='--',lw=.9)
    ax.set(title='Type I error rate',ylabel='Type I error rate',
           ylim=(0,min(1.05,max(.10,float(primary.ci_upper.max())*1.12))),xlim=(-.6,1.6),
           xticks=[0,1],xticklabels=[study.LABELS[a].replace(' Shapley','\nShapley') for a in arms])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig,'05_type1_error_rate',f'Separate synthetic global-null validation: n=100, p=4, k=2; '
         f'{m["synthetic_null"]["repetitions"]:,} intended trials per effect, fresh iid N(0,1) features and independent N(0,1) responses. '
         'Analytic p < 0.05; pointwise 95% exact binomial intervals among resolved tests. '
         'Failures remain in type1_error_summary.csv accounting. This is not a real-data Type I error estimate or validation at n=150,200.')
    primary.reset_index().to_csv(out/'type1_error_rates.csv',index=False)

    intro=(f'Analytic RBF Shapley benchmark: {m["repetitions"]:,} attempts per dataset, sample size and effect; '
           f'n=100,150,200; {len(d):,} tests. k=2; sigma=1 is a working scale. '
           'Rows are nested across sample sizes within repetitions. Effects share selection and actual targets. '
           f'An additional {len(nd):,} synthetic global-null tests validate Type I error at n=100,p=4. '
           'Null truth is unknown for real data; full-space rejection need not identify a unique feature effect.')
    md=['# Real-data RBF Shapley benchmark',intro];sections=[]
    for item in figures:
        title=item['name'][3:].replace('_',' ').title()
        md.append(f'## {title}\n\n{item["caption"]}\n\n![{title}]({item["name"]}.png)')
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(item["caption"])}</p><img src="{item["name"]}.png" alt="{html.escape(title)}"></section>')
    (out/'README.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real-data RBF benchmark</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}</style><h1>Real-data RBF Shapley benchmark</h1><p>'+html.escape(intro)+'</p>'+''.join(sections)+'</html>',encoding='utf-8')
    study.core.atomic(out/'analysis_provenance.json',dict(source_exports=done['export_sha256'],plot_script_sha256=study.digest(Path(__file__)),
        bootstrap_seed=m['seed']+1,bootstrap_resamples=10000,figures=figures,
        files_sha256={p.name:study.digest(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_provenance.json'}))
    print(f'Created {len(figures)} figures and Table 1 summaries in {out}')


if __name__=='__main__': main()
