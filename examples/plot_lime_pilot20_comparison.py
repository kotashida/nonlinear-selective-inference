"""Five-configuration variant of plot_combined_ais_analysis; invoked by the audited builder."""
from pathlib import Path
import json
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _wilson_interval

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/selection_method_validation/exact_set/ais_comparison/analysis'
OUT = SOURCE/'combined_ais_only'
METHODS = {'shap':'SHAP', 'lime':'LIME (40 explanations)',
           'spline_screening':'Spline screening', 'mutual_information':'Mutual information'}
COLORS = dict(zip(METHODS, ['#2674a6','#d58122','#258568','#9256a0']))
BETAS = [.3,.5,.75,1.]
FIGURES = []
DATA_FILENAME = "combined_ais_tests.csv"


def save(fig, name, caption):
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    FIGURES.append((name,caption))


def plot_power_curves(p):
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for method,label in METHODS.items():
        g=p[p.method.eq(method)].sort_values('beta');color=COLORS[method]
        for ax,values in zip(axes,[g.signal_targets/200,g.conditional_power,g.marginal_power]):
            ax.plot(g.beta,values,'o-',color=color,label=label)
        axes[1].fill_between(g.beta,g.ci_lower,g.ci_upper,color=color,alpha=.10)
    for ax,title in zip(axes,['Probability of targeting the signal','Conditional power','Marginal detection power']):
        ax.set(title=title,xlabel='Signal strength (beta)',xticks=BETAS,ylim=(0,1.03));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Combined AIS: power and signal targeting')
    save(fig,'01_power_curves','Failed signal tests count as misses. Shading is the pointwise 95% Wilson interval for conditional power. Marginal power also accounts for how often the selected target is a signal.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    data=pd.read_csv(SOURCE/'unified_tests.csv')
    data=data[data.ais_setting.eq('combined')].copy()
    assert len(data)==5000 and data.max_final_samples.eq(51200).all()
    assert data.pilot_iters.eq(data.selection_method.map({m: (20 if m == 'lime_pilot20' else 10) for m in METHODS})).all() and data.pilot_samples.eq(320).all()
    n=data[data.scope.eq('null_calibration')].copy()
    t=data[data.scope.eq('power')].copy()
    t['target_is_signal']=t.target_is_signal.astype(bool)
    p=pd.read_csv(SOURCE/'power_summary.csv');p=p[p.ais.eq('combined')].copy()
    c=pd.read_csv(SOURCE/'calibration_summary.csv');c=c[c.ais.eq('combined')].copy()
    for row in p.itertuples():
        g=t[t.selection_method.eq(row.method)&t.signal_strength.eq(row.beta)]
        sig=g[g.target_is_signal]
        assert len(g)==200 and len(sig)==row.signal_targets
        assert int((sig.p_value.lt(.05)&~sig.failed).sum())==row.detections
        assert int(sig.failed.sum())==row.signal_failures
        assert np.isclose(g.total_proposals.mean(),row.mean_total_proposals)
    assert len(n)==1000 and not n.failed.any()
    data.to_csv(OUT/DATA_FILENAME,index=False)
    p.to_csv(OUT/'power_summary.csv',index=False);c.to_csv(OUT/'calibration_summary.csv',index=False)
    manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    (OUT/'source_manifest.json').write_text(json.dumps({k:manifest[k] for k in data.source_run.unique()},indent=2))

    plot_power_curves(p)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric,title in zip(axes,['conditional_power','ci_lower'],['Observed conditional power','95% lower confidence bound']):
        mat=p.pivot(index='method',columns='beta',values=metric).reindex(METHODS)[BETAS]
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for i,method in enumerate(METHODS):
            for j,beta in enumerate(BETAS):
                r=p[p.method.eq(method)&p.beta.eq(beta)].iloc[0];v=mat.iloc[i,j]
                extra=f'\n{int(r.detections)}/{int(r.signal_targets)}' if metric=='conditional_power' else ('\n>= 80%' if v>=.8 else '')
                ax.text(j,i,f'{v:.1%}'+extra,ha='center',va='center',color='white' if v>.65 else 'black',fontsize=10)
        ax.set(title=title,xticks=range(4),xticklabels=[f'{b:.2f}' for b in BETAS],yticks=range(len(METHODS)),yticklabels=list(METHODS.values()),xlabel='Signal strength (beta)')
    fig.suptitle('Combined AIS: the 80% power requirement')
    save(fig,'02_power_target_heatmap','Counts show signal detections / signal targets, with failures retained in the denominator. Confidence bounds are pointwise and do not adjust for selecting AIS settings using these seeds.')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for j,(method,label) in enumerate(METHODS.items()):
        g=c[c.method.eq(method)].sort_values('alpha')
        axes[0].errorbar(g.alpha+(j-(len(METHODS)-1)/2)*.0015,g.fpr,yerr=np.maximum(0,np.vstack([g.fpr-g.ci_lower,g.ci_upper-g.fpr])),fmt='o',capsize=3,color=COLORS[method],label=label)
        vals=np.sort(n[n.selection_method.eq(method)].p_value)
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[method],label=label)
    axes[0].plot([0,.12],[0,.12],'--',color='gray');axes[0].set(title='Null rejection rates and 95% intervals',xlabel='Nominal alpha (points offset for readability)',ylabel='Observed false-positive rate',xticks=[.01,.05,.1],ylim=(0,.18));axes[0].legend(fontsize=8)
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(title='Null p-value empirical distribution',xlabel='p-value',ylabel='Fraction at or below p',xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Combined AIS: null calibration (200 trials per method; zero failures)')
    save(fig,'03_null_calibration','The diagonal is the nominal/uniform reference. Small departures in 200 trials do not prove either miscalibration or exact calibration.')

    fig,axes=plt.subplots(3,2,figsize=(11,12))
    outcomes=[]
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=p[p.method.eq(method)].sort_values('beta');det=g.detections/g.signal_targets;fail=g.signal_failures/g.signal_targets;miss=1-det-fail
        x=np.arange(4)
        ax.bar(x,det,color='#258568',label='Detected');ax.bar(x,miss,bottom=det,color='#bcc8d1',label='Valid, not detected');ax.bar(x,fail,bottom=det+miss,color='#bc4545',label='Inference failed')
        for i,r in enumerate(g.itertuples()):
            ax.text(i,1.02,f'n={r.signal_targets}\nfailed={r.signal_failures}',ha='center',fontsize=8)
            outcomes.append(dict(method=method,beta=r.beta,signal_targets=r.signal_targets,detected=r.detections,valid_not_detected=r.signal_targets-r.detections-r.signal_failures,failed=r.signal_failures))
        ax.set(title=label,xticks=x,xticklabels=BETAS,ylim=(0,1.18),xlabel='Signal strength (beta)');ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[-1,-1].set_visible(False)
    axes[0,0].legend(fontsize=8,loc='lower right');fig.suptitle('Combined AIS: outcomes among signal targets')
    save(fig,'04_signal_outcomes','Separates statistical non-detection from unresolved inference. Each bar is conditional on the target being a true signal.')
    pd.DataFrame(outcomes).to_csv(OUT/'signal_outcomes.csv',index=False)

    fig,axes=plt.subplots(1,2,figsize=(12,5));costs=[]
    for method,label in METHODS.items():
        g=p[p.method.eq(method)].sort_values('beta');axes[0].plot(g.beta,g.mean_total_proposals,'o-',color=COLORS[method],label=label)
        vals=np.sort(t[t.selection_method.eq(method)].total_proposals)
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[method])
        for beta in BETAS:
            raw=t[t.selection_method.eq(method)&t.signal_strength.eq(beta)]
            costs.append(dict(method=method,beta=beta,mean=raw.total_proposals.mean(),median=raw.total_proposals.median(),p95=raw.total_proposals.quantile(.95),maximum=raw.total_proposals.max(),first_batch_fraction=raw.proposals.eq(800).mean(),cap_fraction=raw.proposals.eq(51200).mean()))
    axes[0].set(title='Mean cost by signal strength',xlabel='Signal strength (beta)',ylabel='Proposals including pilot',xticks=BETAS);axes[0].legend(fontsize=8)
    axes[1].set(title='Cost distribution across 800 power trials per method',xlabel='Proposals including pilot (log scale)',ylabel='Fraction of trials at or below cost',xscale='log',ylim=(0,1.03))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Combined AIS: sampling effort (3,200 / 6,400 pilot + final proposals)')
    save(fig,'05_sampling_cost','Counts measure sampling effort, not wall-clock time. Selector evaluation costs differ between methods. The 10-round configurations use 3,200 pilot draws (4,000 to 54,400 total proposals); LIME with 20 rounds uses 6,400 (7,200 to 57,600 total).')
    pd.DataFrame(costs).to_csv(OUT/'sampling_cost_summary.csv',index=False)

    fig,axes=plt.subplots(3,2,figsize=(11,12),sharex=True,sharey=True)
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=t[t.selection_method.eq(method)]
        for failed,marker,color in [(False,'o',COLORS[method]),(True,'x','#bf3232')]:
            q=g[g.failed.eq(failed)]
            ax.scatter(q.denominator_ess,q.tail_ess,s=15 if not failed else 40,alpha=.3 if not failed else .9,marker=marker,color=color,label='Failed' if failed else 'Successful')
        ax.axvline(80,ls='--',color='gray');ax.axhline(15,ls='--',color='gray')
        ax.set(title=f'{label}: {int(g.failed.sum())}/800 failures',xlabel='Denominator ESS',ylabel='Tail ESS',xscale='log',yscale='log');ax.grid(alpha=.15)
    axes[-1,-1].set_visible(False)
    axes[0,0].legend(fontsize=8);fig.suptitle('Combined AIS: final effective sample sizes across all power trials')
    save(fig,'06_ess_diagnostics','Dashed lines mark the unchanged denominator ESS=80 and tail ESS=15 thresholds. Failed inference is marked in red; both axes are logarithmic.')

    fig,ax=plt.subplots(figsize=(9,5))
    for method,label in METHODS.items():
        g=p[p.method.eq(method)&p.beta.eq(1)].iloc[0]
        ax.errorbar(g.mean_total_proposals,g.conditional_power,yerr=[[g.conditional_power-g.ci_lower],[g.ci_upper-g.conditional_power]],fmt='o',capsize=4,color=COLORS[method],label=label)
        ax.annotate(f'{g.conditional_power:.1%}',(g.mean_total_proposals,g.conditional_power),xytext=(7,0),textcoords='offset points',color=COLORS[method])
    ax.axhline(.8,ls='--',color='gray');ax.set(xlabel='Mean proposals per power trial, including pilot',ylabel='Conditional power',ylim=(.77,1.015),xlim=(4200,max(8200, float(p[p.beta.eq(1)].mean_total_proposals.max())+600)));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='lower left');ax.grid(alpha=.2)
    fig.suptitle('Combined AIS at beta=1: power versus sampling effort')
    save(fig,'07_power_cost_tradeoff','Vertical bars show pointwise 95% power intervals. Sampling counts are not measured runtimes, so this chart does not establish which selector is fastest.')

    intro='Combined AIS only: final cap 51,200; pilot 10 x 320 (LIME comparison: 20 x 320); batch 800; ESS thresholds 80/15; early stopping enabled. Five method/configuration combinations, 200 null trials and 200 power trials per beta per method. Both LIME configurations are fixed at 40 explanations.'
    strongest=p[p.beta.eq(1)].set_index('method')
    failures=t.groupby('selection_method').failed.sum()
    notes='At beta=1: '+', '.join(f'{METHODS[m]} {strongest.loc[m, "conditional_power"]:.1%}' for m in METHODS)+'. '
    notes+=f'There are {int(t.failed.sum())} failures across 4,000 power tests ('+', '.join(f'{METHODS[m]} {int(failures[m])}' for m in METHODS)+'), and zero failures among 1,000 null tests. '
    notes+='Power counts failed signal tests as misses. Confidence intervals are pointwise. These are exploratory results on reused seeds, not independent confirmation after tuning.'
    if 'mi_neighbors' in data and data.loc[data.selection_method.eq('mutual_information'),'mi_neighbors'].notna().any():
        intro+=f' MI uses {int(data.loc[data.selection_method.eq("mutual_information"),"mi_neighbors"].dropna().iloc[0])} neighbors.'

    report=['# Combined AIS analysis',intro,notes,'## Plot gallery']
    sections=[]
    for name,caption in FIGURES:
        report.append(f'### {name[3:].replace("_"," ").title()}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)')
        sections.append(f'<section><h2>{html.escape(name[3:].replace("_"," ").title())}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a><p><a href="{name}.pdf">Download PDF</a></p></section>')
    report.append(f'## Data\n\n{DATA_FILENAME} contains the 5,000 selected records from the audited parent dataset. It must not be appended to that parent dataset. source_manifest.json preserves the run parameters and software metadata. Rates were checked against the raw records. Sampling counts include 3,200 pilot draws, or 6,400 for LIME with 20 rounds; no wall-clock timing is inferred.')
    (OUT/'README.md').write_text('\n\n'.join(report),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Combined AIS analysis</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>Combined AIS analysis</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(notes)+'</p>'+''.join(sections)+'</html>',encoding='utf-8')
    print(f'Saved {len(FIGURES)} PNG/PDF figures, gallery and 5000 verified records to {OUT}')


if __name__=='__main__':main()
