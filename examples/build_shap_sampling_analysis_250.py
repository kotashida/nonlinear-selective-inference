"""Adapt the selected-six gallery to the matched SHAP sampler experiment.

Reproduce: .venv/Scripts/python.exe examples/build_shap_sampling_analysis_250.py
Uses existing results only. The source experiment and its aggregates stay intact.
"""
from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

os.environ['MPLBACKEND']='Agg'
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'examples')]
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import plot_combined_ais_analysis as reference
from examples.validate_selection_methods import _clopper_pearson
from examples.run_shap_sampling_ten_250 import read_manifest, verify_downloaded

SOURCE=ROOT/'outputs/selection_method_validation/exact_set/shap_sampling_ten_250_20260928'
OUT=SOURCE/'analysis'
LABELS=dict(zip([f'test_{i:02d}' for i in range(1,11)],[
    'T01 AIS reference','T02 AIS double cap','T03 AIS pilot 15',
    'T04 AIS CDF/SF intervals','T05 AIS denominator mixture',
    'T06 AIS full stack','T07 AIS stable tail',
    'T08 MCMC 20 steps','T09 MCMC 100 steps','T10 MCMC 100 / jump .30']))
COLORS=dict(zip(LABELS,['#2674a6','#d58122','#258568','#c44e52','#9256a0',
                       '#6b6ecf','#8c564b','#17a2a4','#333333','#e377c2']))
BETAS=[.75,1.]
FIGURES=[]


def save(fig,name,caption):
    # Retain the existing reference gallery's rendering and export convention.
    reference.OUT=OUT
    reference.FIGURES=FIGURES
    reference.save(fig,name,caption)


def legend(fig):
    handles=[plt.Line2D([],[],color=COLORS[a],marker='o',label=LABELS[a]) for a in LABELS]
    fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=8,bbox_to_anchor=(.5,-.055))


def summarize(data,m):
    powers=[];null=[];costs=[]
    for a in LABELS:
        for b in BETAS:
            g=data[data.arm.eq(a)&data.beta.eq(b)]
            s=g[g.target_is_signal.eq(True)]
            k=int(s.p_value.lt(.05).sum());failed=int(s.failed.sum())
            lo,hi=_clopper_pearson(k,len(s))
            powers.append(dict(arm=a,beta=b,n=len(g),signal_targets=len(s),detections=k,
                signal_failures=failed,conditional_detection=k/len(s),ci_lower=lo,ci_upper=hi,
                unresolved_upper=(k+failed)/len(s),marginal_detection=k/len(g),
                targeting=len(s)/len(g),mean_effort=g.effort.mean()))
        g=data[data.arm.eq(a)&data.condition.eq('null')];v=g[g.p_value.notna()]
        k=int(v.p_value.lt(.05).sum());lo,hi=_clopper_pearson(k,len(v))
        null.append(dict(arm=a,n=len(g),valid=len(v),failures=int(g.failed.sum()),rejections=k,
            resolved_only_fpr=k/len(v),ci_lower=lo,ci_upper=hi,
            all_trial_rejection_lower=k/len(g),all_trial_rejection_upper=(k+g.failed.sum())/len(g)))
        for c,g in data[data.arm.eq(a)].groupby('condition'):
            config=next(x for x in m['arms'] if x['name']==a)
            costs.append(dict(arm=a,condition=c,n=len(g),mean=g.effort.mean(),median=g.effort.median(),
                p95=g.effort.quantile(.95),maximum=g.effort.max(),
                ais_cap_fraction=g.proposals.eq(config['max_final_samples']).mean() if a<'test_08' else np.nan))
    return pd.DataFrame(powers),pd.DataFrame(null),pd.DataFrame(costs)


def plot_reference_gallery(data,p,n):
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    shared=p[p.arm.eq('test_01')]
    axes[0].plot(shared.beta,shared.targeting,'o-',color='#2674a6')
    for r in shared.itertuples(): axes[0].annotate(f'{r.signal_targets}/250',(r.beta,r.targeting),xytext=(0,10),textcoords='offset points',ha='center')
    axes[0].text(.05,.10,'Identical targets across all 10 arms',transform=axes[0].transAxes)
    for a in LABELS:
        g=p[p.arm.eq(a)]
        axes[1].plot(g.beta,g.conditional_detection,'o-',color=COLORS[a])
        axes[1].fill_between(g.beta,g.ci_lower,g.ci_upper,color=COLORS[a],alpha=.07)
        axes[2].plot(g.beta,g.marginal_detection,'o-',color=COLORS[a])
    for ax,title in zip(axes,['Probability of targeting signal','Conditional signal detection','Marginal signal detection']):
        ax.set(title=title,xlabel='Signal strength (beta)',xticks=BETAS,ylim=(0,1.04));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    axes[0].axhline(.5,color='gray',ls='--');axes[2].axhline(.5,color='gray',ls='--')
    legend(fig);fig.suptitle('SHAP sampling comparison: power and signal targeting')
    save(fig,'01_power_curves','Reference layout. Reported signal detections / ALL signal targets: unresolved tests remain in the denominator as non-detections. This is operational detection, not an identified failure-free power estimate. Shading: pointwise 95% Clopper–Pearson intervals, matching this experiment; the selected-six reference used Wilson intervals.')

    fig,axes=plt.subplots(1,2,figsize=(14,8))
    for ax,metric,title in zip(axes,['conditional_detection','ci_lower'],['Reported detections / signal targets','Pointwise 95% lower bound']):
        mat=p.pivot(index='arm',columns='beta',values=metric).reindex(LABELS)[BETAS]
        ax.imshow(mat,vmin=.8,vmax=1,cmap='YlGnBu',aspect='auto')
        for i,a in enumerate(LABELS):
            for j,b in enumerate(BETAS):
                r=p[p.arm.eq(a)&p.beta.eq(b)].iloc[0]
                suffix=f'{int(r.detections)}/{int(r.signal_targets)}; failures={int(r.signal_failures)}' if metric=='conditional_detection' else ('UNRESOLVED' if r.signal_failures else ('PASS SCREEN' if r.ci_lower>=.8 else 'BELOW 80%'))
                ax.text(j,i,f'{mat.iloc[i,j]:.1%}\n{suffix}',ha='center',va='center',fontsize=8,color='white' if mat.iloc[i,j]>.91 else 'black')
        ax.set(title=title,xticks=[0,1],xticklabels=BETAS,yticks=range(10),yticklabels=LABELS.values(),xlabel='Signal strength (beta)')
    fig.suptitle('SHAP sampling comparison: detection and the reference 80% screening rule')
    save(fig,'02_power_target_heatmap','Adapts the reference screening display: lower bound >=80%, at least 30 signal targets, and no unresolved signal tests. A screen pass is not proof of calibration or superiority. Failures prevent a complete power decision; they are not silently excluded.')

    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for i,a in enumerate(LABELS):
        r=n[n.arm.eq(a)].iloc[0]
        axes[0].errorbar(r.resolved_only_fpr,i,xerr=[[r.resolved_only_fpr-r.ci_lower],[r.ci_upper-r.resolved_only_fpr]],fmt='o',capsize=3,color=COLORS[a])
        axes[0].text(.061,i,f'{int(r.rejections)}/{int(r.valid)}; failed={int(r.failures)}',va='center',fontsize=8)
        vals=np.sort(data[data.arm.eq(a)&data.condition.eq('null')].p_value.dropna())
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[a])
    axes[0].axvline(.05,color='gray',ls='--');axes[0].set(title='Null rejection at alpha=.05: 95% exact intervals',yticks=range(10),yticklabels=LABELS.values(),xlim=(0,.09),xlabel='Rejection fraction among resolved tests');axes[0].invert_yaxis();axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(title='Null p-value ECDF (resolved tests)',xlabel='p-value',ylabel='Fraction at or below p',xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('SHAP sampling comparison: null calibration; one unresolved null test')
    save(fig,'03_null_calibration','Only the requested alpha=.05 is tested. T04 has 6/249 resolved rejections and one missing outcome: the all-trial rejection fraction lies in [6/250,7/250]. Arms share the same 250 null datasets and must not be pooled as independent null trials. Conservative rank p-values may legitimately have ECDF below the uniform diagonal.')

    fig,axes=plt.subplots(2,5,figsize=(19,8))
    outcomes=[]
    for ax,a in zip(axes.flat,LABELS):
        g=p[p.arm.eq(a)];x=np.arange(2);det=g.detections/g.signal_targets;fail=g.signal_failures/g.signal_targets;miss=1-det-fail
        ax.bar(x,det,color='#258568',label='Detected');ax.bar(x,miss,bottom=det,color='#bcc8d1',label='Resolved, not detected');ax.bar(x,fail,bottom=det+miss,color='#bc4545',label='Unresolved')
        for j,r in enumerate(g.itertuples()):
            ax.text(j,1.015,f'n={r.signal_targets}\nfailed={r.signal_failures}',ha='center',fontsize=8)
            outcomes.append(dict(arm=a,beta=r.beta,signal_targets=r.signal_targets,detected=r.detections,valid_not_detected=r.signal_targets-r.detections-r.signal_failures,failed=r.signal_failures))
        ax.set(title=LABELS[a],xticks=x,xticklabels=BETAS,ylim=(0,1.17));ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=7,loc='lower left');fig.suptitle('SHAP sampling comparison: outcomes among signal targets')
    save(fig,'04_signal_outcomes','Same stacked outcome convention as the reference gallery; non-detection and failed inference remain separate.')
    pd.DataFrame(outcomes).to_csv(OUT/'signal_outcomes.csv',index=False)

    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for a in LABELS:
        g=p[p.arm.eq(a)];axes[0].plot(g.beta,g.mean_effort,'o-',color=COLORS[a])
        vals=np.sort(data[data.arm.eq(a)&data.beta.notna()].effort)
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[a])
    axes[0].set(title='Mean effort by signal strength',xlabel='Signal strength (beta)',ylabel='AIS draws incl. pilot / MCMC transitions',xticks=BETAS)
    axes[1].set(title='Effort distribution across 500 alternative trials',xlabel='AIS draws / MCMC transitions (log)',ylabel='Fraction at or below effort',xscale='log',ylim=(0,1.03))
    for ax in axes:ax.grid(alpha=.2)
    legend(fig);fig.suptitle('SHAP sampling comparison: sampling effort')
    save(fig,'05_sampling_cost','AIS includes pilot draws; MCMC includes the hub and every branch transition. These counts are not runtimes or exact uncached forest-fit counts. Every outcome, including cap failures, is included.')

    fig,axes=plt.subplots(3,3,figsize=(15,12))
    for ax,a in zip(axes.flat,list(LABELS)[:7]):
        g=data[data.arm.eq(a)&data.beta.notna()]
        for failed,marker,color in [(False,'o',COLORS[a]),(True,'x','#bf3232')]:
            q=g[g.failed.eq(failed)];ax.scatter(q.denominator_ess,q.tail_ess,s=15 if not failed else 45,alpha=.3 if not failed else 1,marker=marker,color=color)
        ax.axvline(80,ls='--',color='gray');ax.axhline(15,ls='--',color='gray');ax.set(title=f'{LABELS[a]}: {g.failed.sum()}/500 failures',xlabel='Denominator ESS',ylabel='Tail ESS',xscale='log',yscale='log');ax.grid(alpha=.15)
    for ax in axes.flat[7:]:ax.remove()
    fig.suptitle('SHAP sampling comparison: AIS effective sample sizes')
    save(fig,'06_ess_diagnostics','Reference ESS display for AIS only, across alternative trials; failed tests are red crosses. MCMC replica counts are not importance-weight ESS and are deliberately shown in their own diagnostics.')

    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,b in zip(axes,BETAS):
        for i,a in enumerate(LABELS):
            r=p[p.arm.eq(a)&p.beta.eq(b)].iloc[0]
            ax.errorbar(r.mean_effort,r.conditional_detection,yerr=[[r.conditional_detection-r.ci_lower],[r.ci_upper-r.conditional_detection]],fmt='o',capsize=3,color=COLORS[a])
            if a in ('test_01','test_08','test_09','test_10'):
                ax.annotate(a[-2:],(r.mean_effort,r.conditional_detection),xytext=(5,-12 if a=='test_09' else 8),textcoords='offset points',color=COLORS[a],fontsize=8)
        ax.set(title=f'beta={b}',xlabel='Mean AIS draws / MCMC transitions',ylabel='Conditional signal detection',ylim=(.84,1.02));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    legend(fig);fig.suptitle('SHAP sampling comparison: detection versus effort')
    save(fig,'07_power_cost_tradeoff','Adapts the reference tradeoff at both requested betas. Failure-inclusive detection and pointwise exact intervals; baseline and MCMC points are numbered, and all arms use the legend colors. T09/T10 coincide at beta=1. Overlapping intervals do not establish superiority.')


def diagnostics(data):
    failures=data[data.failed].copy()
    failures['failure_reason']=np.select([(failures.denominator_ess<80)&(failures.tail_ess>=15),(failures.denominator_ess>=80)&(failures.tail_ess<15)],['denominator ESS only','tail ESS only'],default='both ESS')
    failures.to_csv(OUT/'failed_tests.csv',index=False)
    hard=failures[['condition','shard','iteration']].drop_duplicates().sort_values(['condition','shard','iteration'])
    mat=[]
    for r in hard.itertuples():
        g=data[data.condition.eq(r.condition)&data.shard.eq(r.shard)&data.iteration.eq(r.iteration)].set_index('arm')
        mat.append([int(g.loc[a,'failed']) for a in LABELS])
    fig,axes=plt.subplots(1,2,figsize=(18,9),gridspec_kw={'width_ratios':[1,2]})
    counts=data.groupby(['arm','condition']).failed.sum().unstack().reindex(LABELS)[['null','beta_0.75','beta_1.0']]
    axes[0].imshow(counts,cmap='Reds',vmin=0,vmax=9,aspect='auto')
    for i in range(10):
        for j in range(3):axes[0].text(j,i,str(counts.iloc[i,j]),ha='center',va='center')
    axes[0].set(title='Unresolved tests / 250 in each cell',xticks=range(3),xticklabels=['Null','.75','1.00'],yticks=range(10),yticklabels=LABELS.values())
    axes[1].imshow(mat,cmap='Reds',vmin=0,vmax=1,aspect='auto')
    axes[1].set(title='The 20 input trials with any AIS failure',xticks=range(10),xticklabels=[a[-2:] for a in LABELS],yticks=range(len(hard)),yticklabels=[f'{r.condition}: shard {r.shard}, iter {r.iteration}' for r in hard.itertuples()],xlabel='Configuration number')
    fig.suptitle('SHAP sampling comparison: failures across matched inputs')
    save(fig,'08_failure_patterns','All 33 failures are retained: 32 denominator-only and one tail-only. Red means unresolved; white means a p-value was returned. The same failure count can conceal failures on different input trials.')

    fig,axes=plt.subplots(3,3,figsize=(15,11));mcmc=[]
    for row,a in enumerate(list(LABELS)[7:]):
        for j,(col,title) in enumerate([('mcmc_replicate_acceptance_rate','Acceptance fraction'),('mcmc_moved_replicas','Moved replicas / 511'),('mcmc_unique_replicas','Unique replicas / 511')]):
            for c,color in [('null','#2674a6'),('beta_0.75','#258568'),('beta_1.0','#d58122')]:
                g=data[data.arm.eq(a)&data.condition.eq(c)]
                values=g[col]/511 if 'replicas' in col else g[col]
                axes[row,j].hist(values,bins=20,histtype='step',color=color,label=c)
            axes[row,j].set(title=f'{LABELS[a]}: {title}',xlabel=title,ylabel='Trials');axes[row,j].grid(alpha=.15)
        for c,g in data[data.arm.eq(a)].groupby('condition'):
            mcmc.append(dict(arm=a,condition=c,n=len(g),no_movement=int(g.mcmc_moved_replicas.eq(0).sum()),
                median_acceptance=g.mcmc_replicate_acceptance_rate.median(),min_moved=g.mcmc_moved_replicas.min(),
                median_moved=g.mcmc_moved_replicas.median(),median_unique=g.mcmc_unique_replicas.median()))
    axes[0,0].legend(fontsize=8);fig.suptitle('SHAP sampling comparison: reversible MCMC movement')
    save(fig,'09_mcmc_diagnostics','Every MCMC test returned a p-value; none had zero moved replicas. Movement and uniqueness are diagnostics, not proof of independent stationary draws or a guarantee of power.')
    pd.DataFrame(mcmc).to_csv(OUT/'mcmc_summary.csv',index=False)

    paired=[]
    base=data[data.arm.eq('test_01')].set_index(['condition','shard','iteration'])
    for a in list(LABELS)[1:]:
        other=data[data.arm.eq(a)].set_index(['condition','shard','iteration'])
        for c in ['null','beta_0.75','beta_1.0']:
            l=base.loc[c];r=other.loc[c];signal=pd.Series(True,index=l.index) if c=='null' else l.target_is_signal.eq(True)
            valid=l.p_value.notna()&r.p_value.notna()&signal;x=l.p_value.lt(.05);y=r.p_value.lt(.05)
            paired.append(dict(arm=a,condition=c,target_trials=int(signal.sum()),complete_pairs=int(valid.sum()),
                gains=int((~x&y&valid).sum()),losses=int((x&~y&valid).sum()),
                baseline_failures=int((l.failed&signal).sum()),arm_failures=int((r.failed&signal).sum()),
                newly_unresolved=int((r.failed&~l.failed&signal).sum()),
                recovered_baseline_failures=int((l.failed&r.p_value.notna()&signal).sum()),
                recovered_and_rejected=int((l.failed&y&signal).sum())))
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired_comparisons.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    for ax,c in zip(axes,['beta_0.75','beta_1.0']):
        g=paired[paired.condition.eq(c)];x=np.arange(9)
        ax.barh(x-.18,g.gains,height=.35,color='#258568',label='Gains among complete pairs');ax.barh(x+.18,-g.losses,height=.35,color='#bc4545',label='Losses among complete pairs');ax.axvline(0,color='gray')
        for i,r in enumerate(g.itertuples()):ax.text(3.2,i,f'pairs={r.complete_pairs}; recovered={r.recovered_baseline_failures}; arm failed={r.arm_failures}',va='center',fontsize=7)
        ax.set(title=c,yticks=x,yticklabels=[LABELS[a] for a in g.arm],xlim=(-6,8),xlabel='Additional / fewer signal rejections vs T01');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8,bbox_to_anchor=(.5,-.03));fig.suptitle('SHAP sampling comparison: paired changes from baseline AIS')
    save(fig,'10_paired_decisions','Matched signal targets. Gains/losses use pairs where both p-values are available; excluded pairs are explicitly counted in paired_comparisons.csv. Recovered failures are separate from complete-pair gains.')

    timings=pd.read_csv(SOURCE/'bundle_timings.csv');timings['n']=timings.shard.map(lambda s:8 if s<26 else 7)
    runtime=timings.groupby('arm').agg(worker_seconds=('elapsed_seconds','sum'),n=('n','sum'))
    runtime['seconds_per_trial']=runtime.worker_seconds/runtime.n
    runtime['relative_to_T01']=runtime.seconds_per_trial/runtime.loc['test_01','seconds_per_trial']
    runtime.to_csv(OUT/'runtime_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(15,6));y=np.arange(10)
    axes[0].barh(y,runtime.reindex(LABELS).seconds_per_trial,color=[COLORS[a] for a in LABELS]);axes[0].set(yticks=y,yticklabels=LABELS.values(),xlabel='Recorded worker seconds / target test',title='Measured simulation + inference time')
    axes[1].barh(y,runtime.reindex(LABELS).relative_to_T01,color=[COLORS[a] for a in LABELS]);axes[1].set(yticks=y,yticklabels=LABELS.values(),xlabel='Relative to T01',title='Relative recorded worker time')
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('SHAP sampling comparison: measured computational cost')
    save(fig,'11_runtime_comparison','Sum of condition-bundle simulation/inference wall times divided by 750 trials per arm; excludes plotting. Workers ran concurrently under contention. This is a descriptive worker-time comparison, not total experiment elapsed time or a controlled speed benchmark.')


def report(data,p,n,m):
    summary='''All 7,500 tests are accounted for; 33 AIS outcomes are unresolved and all 2,250 MCMC tests returned p-values. No benchmark or simulation was rerun.

Baseline AIS (T01) detected 116/120 signal targets at beta=.75 (96.7%) and 110/115 at beta=1 (95.7%, with three unresolved tests). T02 doubled the final cap but still had three failures; T03 also had three failures. T04/T05/T06/T07 had 9/4/4/7 failures in total. No tested AIS modification demonstrates a reliability improvement. The failed input trials differ between arms, so equal counts do not mean identical failures. Of the 33 failures, 32 missed the denominator ESS threshold alone and one missed the tail threshold alone.

MCMC T08/T09/T10 had zero missing p-values and no zero-movement trials. T08 detected 114/120 and 108/115 signal targets (95.0%, 93.9%). T09 detected 114/120 and 113/115 (95.0%, 98.3%); T10 detected 115/120 and 113/115 (95.8%, 98.3%). At beta=1, T09/T10 agree with all baseline decisions on complete pairs, and reject on the three trials where baseline AIS failed. At beta=.75 T09 loses two baseline detections; T10 gains one and loses two. The small, paired differences do not establish a power winner.

T09 and T10 use about 10.6 times T01's recorded worker time; T08 uses about 2.1 times. AIS stops early on most trials. More MCMC steps improve beta=1 detection here, whereas increasing global jumps to .30 does not produce a clear benefit.

Null rejection is 6/250 for every complete arm except T10 (5/250). T04 has 6/249 resolved null rejections plus one missing outcome. For 6/250 the exact 95% interval is 0.89%–5.15%; for 5/250 it is 0.65%–4.61%. Conservative rejection rates can satisfy level control. These 250 matched null inputs per arm cannot establish validity, and the ten arms cannot be pooled as 2,500 independent null inputs.

Next steps:

1. Keep T01 as the fast AIS baseline. Do not adopt the larger cap, CDF/SF strategy, full stack or stable-tail variant as defaults from this screen. Independent sampling streams and only 250 trials limit causal claims about small differences.
2. Retain T09 (511 replicas, 100 steps, global jump .10) as the reliability candidate. Use it as a complete inference method in the next comparison. An automatic switch from AIS to MCMC after failure is a new composite testing procedure and requires its own calibration; individual sampler properties do not automatically validate that switch.
3. Use the 20 difficult synthetic input trials listed in hard_trials.csv to inspect selector regions, denominator coverage and weight concentration. Compare successful and failed sampler realizations on the same inputs; treat these as selected diagnostic cases, not a calibration sample.
4. Next compare only T01 and T09 on difficult real-data Energy heating/cooling cases using matched inputs and the existing benchmark protocol. Evaluate completion, disagreement, movement and cost; real-data rejection rates are not Type I error or known power. Run a small matched stress pilot before another full benchmark.
5. Before choosing a production default, run fresh independent synthetic confirmation with at least 1,000 null trials per arm and both requested betas. For tighter evidence near the 5% level, use more null trials. Include the lower-signal settings later to expose mixing-related power loss that beta=.75/1 can hide. Predefine criteria and report every unresolved outcome.
'''
    (OUT/'SUMMARY.md').write_text(summary,encoding='utf-8')
    methods=pd.DataFrame(m['arms']);methods.insert(1,'label',methods.name.map(LABELS));methods.to_csv(OUT/'configuration_summary.csv',index=False)
    failures=data[data.failed][['condition','shard','iteration']].drop_duplicates().sort_values(['condition','shard','iteration']);failures.to_csv(OUT/'hard_trials.csv',index=False)
    intro='SHAP-only: 10 matched sampler configurations; 250 null trials and 250 alternative trials at each beta (.75, 1.00); alpha=.05. The seven reference gallery categories are retained, followed by failure, MCMC, paired-outcome and timing diagnostics.'
    methodology='The original downloaded files are preserved. Figures use the reference presentation with the experiment\'s Clopper–Pearson intervals. Failure-inclusive signal detection is an operational lower bound on unidentified power; its confidence interval describes the recorded detection outcome, not missing true decisions. Resolved-only rates remain in the original sampling_summary.csv. Null ECDFs exclude the one unresolved value and disclose it. Costs distinguish AIS pilot+final draws from MCMC transitions; MCMC replica counts are not AIS ESS. Intervals are pointwise; no ranking significance or post-tuning confirmation is claimed.'
    table_columns=['name','label','inference_method','ais_strategy','pilot_iters','max_final_samples','mcmc_steps','mcmc_global_jump_probability']
    table='| '+' | '.join(table_columns)+' |\n| '+' | '.join(['---']*len(table_columns))+' |\n'
    table+='\n'.join('| '+' | '.join(str(value) for value in row)+' |' for row in methods[table_columns].itertuples(index=False,name=None))
    readme=['# SHAP sampler comparison',intro,methodology,'[Findings and next steps](SUMMARY.md)','## Configurations',table,'## Plot gallery']
    sections=[]
    for name,caption in FIGURES:
        readme.append(f'### {name[3:].replace("_"," ").title()}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)')
        sections.append(f'<section><h2>{html.escape(name[3:].replace("_"," ").title())}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png"></a><a href="{name}.pdf">PDF</a></section>')
    readme.append('## Reproduction\n\nRun `.venv/Scripts/python.exe examples/build_shap_sampling_analysis_250.py` from the repository. The copied analyze_results.py is also runnable. Existing inference is never rerun. See analysis_manifest.json and the original local_verification_report.json for provenance.')
    (OUT/'README.md').write_text('\n\n'.join(readme),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SHAP sampler comparison</title><style>body{font:17px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%}a{color:#17669b}</style><h1>SHAP sampler comparison</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(methodology)+'</p><p><a href="SUMMARY.md">Findings and next steps</a></p>'+''.join(sections),encoding='utf-8')


def build(m,data):
    assert len(data)==7500 and not data.duplicated(['arm','condition','shard','iteration']).any()
    data['beta']=data.condition.map({'beta_0.75':.75,'beta_1.0':1.})
    data['effort']=data.proposals+data.pilot_evaluations.fillna(0)
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    p,n,c=summarize(data,m)
    p.to_csv(OUT/'power_summary.csv',index=False);n.to_csv(OUT/'calibration_summary.csv',index=False);c.to_csv(OUT/'sampling_cost_summary.csv',index=False)
    plot_reference_gallery(data,p,n);diagnostics(data);report(data,p,n,m)
    source=Path(__file__).resolve();copy=OUT/'analyze_results.py'
    if source!=copy.resolve():shutil.copyfile(source,copy)
    manifest=dict(records=7500,raw_bundles=960,missing_bundles=0,sampling_failures=int(data.failed.sum()),
        figures=[name for name,_ in FIGURES],source_files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SOURCE/'unified_tests.csv',SOURCE/'requested_manifest.json',SOURCE/'source_hashes.json',source,ROOT/'examples/plot_combined_ais_analysis.py']},
        reference='outputs/selection_method_validation/exact_set/selected_six_with_linear_shapley_300_20260928',
        inference_rerun=False)
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {len(FIGURES)} PNG/PDF figure pairs to {OUT}')


def main():
    m=read_manifest();verify_downloaded(m)
    data=pd.read_csv(SOURCE/'unified_tests.csv',keep_default_na=False,na_values=[''],low_memory=False)
    build(m,data)


if __name__=='__main__':main()
