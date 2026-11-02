"""Combined-AIS-only analysis from the audited, matched four-method dataset."""
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
           'spline_screening':'Spline screening (AIS)',
           'mutual_information':'Mutual information'}
COLORS = dict(zip(METHODS, ['#2674a6','#d58122','#258568','#9256a0']))
BETAS = [.3,.5,.75,1.]
FIGURES = []
DATA_FILENAME = "combined_ais_tests.csv"
TRIALS_PER_CONDITION = 200
REPORT_LABEL = "Combined AIS"
EXPECTED_PILOT_ITERS = (10,)
REPORT_INTRO = None
ALLOW_INCOMPLETE = False
MIN_CONDITIONAL_POWER = .8
MIN_SIGNAL_TARGETS = 30


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
        for ax,values in zip(axes,[g.signal_targets/g.n,g.conditional_power,g.marginal_power]):
            ax.plot(g.beta,values,'o-',color=color,label=label)
        axes[1].fill_between(g.beta,g.ci_lower,g.ci_upper,color=color,alpha=.10)
    for ax,title in zip(axes,['Probability of targeting the signal','Conditional power','Marginal detection power']):
        ax.set(title=title,xlabel='Signal strength (beta)',xticks=BETAS,ylim=(0,1.03));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    axes[0].axhline(.5,color='gray',ls='--');axes[2].axhline(.5,color='gray',ls='--');axes[0].legend(fontsize=8)
    fig.suptitle(f'{REPORT_LABEL}: power and signal targeting')
    save(fig,'01_power_curves','Failed signal tests count as misses. Shading is the pointwise 95% Wilson interval for conditional power. Marginal power also accounts for how often the selected target is a signal.')


def _power_decision(row):
    """Apply the strict report-level power rule used by the pooled validator."""
    evidence = bool(
        row.signal_targets >= MIN_SIGNAL_TARGETS
        and row.signal_failures == 0
    )
    passed = bool(evidence and row.ci_lower >= MIN_CONDITIONAL_POWER)
    return evidence, passed


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    data=pd.read_csv(SOURCE/'unified_tests.csv',low_memory=False)
    data=data[data.ais_setting.isin(['combined','exact'])].copy()
    expected_records=(1+len(BETAS))*TRIALS_PER_CONDITION*len(METHODS)
    assert (ALLOW_INCOMPLETE or len(data)==expected_records) and data.max_final_samples.eq(51200).all()
    assert data.pilot_iters.isin(EXPECTED_PILOT_ITERS).all() and data.pilot_samples.eq(320).all()
    n=data[data.scope.eq('null_calibration')].copy()
    t=data[data.scope.eq('power')].copy()
    t['target_is_signal']=t.target_is_signal.astype(bool)
    p=pd.read_csv(SOURCE/'power_summary.csv');p=p[p.ais.isin(['combined','exact'])].copy()
    c=pd.read_csv(SOURCE/'calibration_summary.csv');c=c[c.ais.isin(['combined','exact'])].copy()
    power_output_columns=p.columns.tolist()
    decisions=p.apply(_power_decision,axis=1,result_type='expand')
    p['power_evidence_sufficient']=decisions[0].astype(bool)
    p['enough_power']=decisions[1].astype(bool)
    for row in p.itertuples():
        g=t[t.selection_method.eq(row.method)&t.signal_strength.eq(row.beta)]
        sig=g[g.target_is_signal]
        assert len(g)==row.n and len(sig)==row.signal_targets
        assert int((sig.p_value.lt(.05)&~sig.failed).sum())==row.detections
        assert int(sig.failed.sum())==row.signal_failures
        assert np.isclose(g.total_proposals.mean(),row.mean_total_proposals)
    assert (ALLOW_INCOMPLETE or len(n)==TRIALS_PER_CONDITION*len(METHODS)) and (ALLOW_INCOMPLETE or not n.failed.any())
    data.to_csv(OUT/DATA_FILENAME,index=False)
    p[power_output_columns].to_csv(OUT/'power_summary.csv',index=False);c.to_csv(OUT/'calibration_summary.csv',index=False)
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
                if metric == 'conditional_power':
                    extra=f'\n{int(r.detections)}/{int(r.signal_targets)}'
                elif r.enough_power:
                    extra='\nPASS'
                elif not r.power_evidence_sufficient:
                    extra='\nUNRESOLVED'
                else:
                    extra='\nBELOW 80%'
                fontsize = 10 if metric == 'conditional_power' else 8
                ax.text(j,i,f'{v:.1%}'+extra,ha='center',va='center',color='white' if v>.65 else 'black',fontsize=fontsize)
        ax.set(title=title,xticks=range(len(BETAS)),xticklabels=[f'{b:.2f}' for b in BETAS],yticks=range(len(METHODS)),yticklabels=list(METHODS.values()),xlabel='Signal strength (beta)')
    fig.suptitle(f'{REPORT_LABEL}: the 80% power requirement')
    save(fig,'02_power_target_heatmap','Counts show signal detections / signal targets, with failures retained in the denominator. PASS requires a pointwise 95% lower bound of at least 80%, at least 30 signal targets, and no unresolved signal tests. UNRESOLVED means failed tests prevent a formal power decision. Confidence bounds are pointwise and do not adjust for selecting AIS settings using these seeds.')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for j,(method,label) in enumerate(METHODS.items()):
        g=c[c.method.eq(method)].sort_values('alpha')
        axes[0].errorbar(g.alpha+(j-(len(METHODS)-1)/2)*.0015,g.fpr,yerr=np.maximum(0,np.vstack([g.fpr-g.ci_lower,g.ci_upper-g.fpr])),fmt='o',capsize=3,color=COLORS[method],label=label)
        vals=np.sort(n[n.selection_method.eq(method)&~n.failed].p_value.dropna())
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[method],label=label)
    axes[0].plot([0,.12],[0,.12],'--',color='gray');axes[0].set(title='Null rejection rates and 95% intervals',xlabel='Nominal alpha (points offset for readability)',ylabel='Observed false-positive rate',xticks=[.01,.05,.1],ylim=(0,.18));axes[0].legend(fontsize=8)
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(title='Null p-value empirical distribution',xlabel='p-value',ylabel='Fraction at or below p',xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f'{REPORT_LABEL}: null calibration ({int(n.failed.sum())} failures; rates among valid tests)')
    save(fig,'03_null_calibration',f'The diagonal is the nominal/uniform reference. Small departures in {TRIALS_PER_CONDITION:,} trials do not prove either miscalibration or exact calibration.')

    ncols=3 if len(METHODS)>4 else min(2,len(METHODS))
    nrows=int(np.ceil(len(METHODS)/ncols))
    panel_size=(7,5) if ncols==1 else (15 if ncols==3 else 11,8)
    fig,axes=plt.subplots(nrows,ncols,figsize=panel_size,squeeze=False)
    outcomes=[]
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=p[p.method.eq(method)].sort_values('beta');det=g.detections/g.signal_targets;fail=g.signal_failures/g.signal_targets;miss=1-det-fail
        x=np.arange(len(BETAS))
        ax.bar(x,det,color='#258568',label='Detected');ax.bar(x,miss,bottom=det,color='#bcc8d1',label='Valid, not detected');ax.bar(x,fail,bottom=det+miss,color='#bc4545',label='Inference failed')
        for i,r in enumerate(g.itertuples()):
            ax.text(i,1.02,f'n={r.signal_targets}\nfailed={r.signal_failures}',ha='center',fontsize=8)
            outcomes.append(dict(method=method,beta=r.beta,signal_targets=r.signal_targets,detected=r.detections,valid_not_detected=r.signal_targets-r.detections-r.signal_failures,failed=r.signal_failures))
        ax.set(title=label,xticks=x,xticklabels=BETAS,ylim=(0,1.18),xlabel='Signal strength (beta)');ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes.flat[len(METHODS):]: ax.remove()
    axes[0,0].legend(fontsize=8,loc='lower right');fig.suptitle(f'{REPORT_LABEL}: outcomes among signal targets')
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
    axes[1].set(title='Cost distribution across available power trials',xlabel='Sampling proposals (symlog)',ylabel='Fraction of trials at or below cost',xscale='symlog',ylim=(0,1.03))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f'{REPORT_LABEL}: sampling effort')
    save(fig,'05_sampling_cost','Counts measure sampling proposals, not wall-clock time. Selector evaluation costs differ between methods.')
    pd.DataFrame(costs).to_csv(OUT/'sampling_cost_summary.csv',index=False)

    fig,axes=plt.subplots(nrows,ncols,figsize=panel_size,squeeze=False)
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=t[t.selection_method.eq(method)]
        if method.endswith('_exact'):
            ax.text(.5,.5,'Not applicable\nExact analytic inference\n(no importance samples)',ha='center',va='center',transform=ax.transAxes,fontsize=12)
            ax.set(title=f'{label}: 0/{len(BETAS) * TRIALS_PER_CONDITION:,} failures',xticks=[],yticks=[])
            continue
        for failed,marker,color in [(False,'o',COLORS[method]),(True,'x','#bf3232')]:
            q=g[g.failed.eq(failed)]
            ax.scatter(q.denominator_ess,q.tail_ess,s=15 if not failed else 40,alpha=.3 if not failed else .9,marker=marker,color=color,label='Failed' if failed else 'Successful')
        ax.axvline(80,ls='--',color='gray');ax.axhline(15,ls='--',color='gray')
        ax.set(title=f'{label}: {int(g.failed.sum())}/{len(g):,} failures',xlabel='Denominator ESS',ylabel='Tail ESS',xscale='log',yscale='log');ax.grid(alpha=.15)
    for ax in axes.flat[len(METHODS):]: ax.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{REPORT_LABEL}: effective sample sizes')
    save(fig,'06_ess_diagnostics','Dashed lines mark the unchanged denominator ESS=80 and tail ESS=15 thresholds. Failed inference is marked in red; both axes are logarithmic.')

    fig,ax=plt.subplots(figsize=(9,5))
    for method,label in METHODS.items():
        g=p[p.method.eq(method)&p.beta.eq(1)].iloc[0]
        ax.errorbar(g.mean_total_proposals,g.conditional_power,yerr=[[g.conditional_power-g.ci_lower],[g.ci_upper-g.conditional_power]],fmt='o',capsize=4,color=COLORS[method],label=label)
        ax.annotate(f'{g.conditional_power:.1%}',(g.mean_total_proposals,g.conditional_power),xytext=(7,0),textcoords='offset points',color=COLORS[method])
    ax.axhline(.8,ls='--',color='gray');ax.set(xlabel='Mean sampling proposals per power trial',ylabel='Conditional power',ylim=(max(0, p.ci_lower.min()-.05),1.015),xlim=(-250,max(8200,p.mean_total_proposals.max()*1.25)));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='lower left');ax.grid(alpha=.2)
    fig.suptitle(f'{REPORT_LABEL} at beta=1: power versus sampling effort')
    save(fig,'07_power_cost_tradeoff','Vertical bars show pointwise 95% power intervals. Sampling counts are not measured runtimes.')

    method_word = 'method' if len(METHODS) == 1 else 'methods'
    if all(method.endswith('_exact') for method in METHODS):
        intro=f'{REPORT_LABEL} inference uses analytic selective p-values and no importance-sampling proposals. {len(METHODS)} {method_word}, {TRIALS_PER_CONDITION:,} null trials and {TRIALS_PER_CONDITION:,} power trials per beta per method.'
    else:
        intro=f'{REPORT_LABEL} inference. AIS uses final cap 51,200; pilot 10 x 320; batch 800; ESS thresholds 80/15; early stopping enabled. {len(METHODS)} {method_word}, {TRIALS_PER_CONDITION:,} null trials and {TRIALS_PER_CONDITION:,} power trials per beta per method. LIME is fixed at 40 explanations.'
    if REPORT_INTRO is not None:
        intro = REPORT_INTRO
    strongest=p[p.beta.eq(1)].set_index('method')
    failures=t.groupby('selection_method').failed.sum()
    notes='At beta=1: '+', '.join(f'{METHODS[m]} {strongest.loc[m, "conditional_power"]:.1%}' for m in METHODS)+'. '
    notes+=f'There are {int(t.failed.sum())} failures across {len(t):,} power tests ('+', '.join(f'{METHODS[m]} {int(failures[m])}' for m in METHODS)+f'), and {int(n.failed.sum())} failures among {len(n):,} null tests. '
    notes+='Power counts failed signal tests as misses. Confidence intervals are pointwise. These are exploratory results on reused seeds, not independent confirmation after tuning.'
    if 'mi_neighbors' in data and data.loc[data.selection_method.eq('mutual_information'),'mi_neighbors'].notna().any():
        intro+=f' MI uses {int(data.loc[data.selection_method.eq("mutual_information"),"mi_neighbors"].dropna().iloc[0])} neighbors.'

    report=[f'# {REPORT_LABEL} analysis',intro,notes,'## Plot gallery']
    sections=[]
    for name,caption in FIGURES:
        report.append(f'### {name[3:].replace("_"," ").title()}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)')
        sections.append(f'<section><h2>{html.escape(name[3:].replace("_"," ").title())}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a><p><a href="{name}.pdf">Download PDF</a></p></section>')
    report.append(f'## Data\n\n{DATA_FILENAME} contains the {len(data):,} selected records from the audited source datasets. It must not be appended to those parent datasets. source_manifest.json preserves the run parameters and software metadata. Rates were checked against the raw records. No wall-clock timing is inferred.')
    (OUT/'README.md').write_text('\n\n'.join(report),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Combined AIS analysis</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>Combined AIS analysis</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(notes)+'</p>'+''.join(sections)+'</html>',encoding='utf-8')
    print(f'Saved {len(FIGURES)} PNG/PDF figures, gallery and {len(data)} verified records to {OUT}')


if __name__=='__main__':main()
