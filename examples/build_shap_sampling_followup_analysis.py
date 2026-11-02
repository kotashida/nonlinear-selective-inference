"""Build PNG-only reference-style analysis of the verified SHAP follow-up.

Run: .venv/Scripts/python.exe examples/build_shap_sampling_followup_analysis.py
Reads saved results and traces; never reruns inference or modifies source data.
"""
from pathlib import Path
import hashlib
import html
import json
import os
import sys
os.environ['MPLBACKEND']='Agg'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
from examples import run_shap_sampling_followup as exp

SOURCE=ROOT/'outputs/selection_method_validation/exact_set/shap_sampling_followup_20261005'
OUT=SOURCE/'analysis'
ARMS={
 'ais_reference':'AIS reference (early stop)',
 'ais_stable':'AIS stable (early stop)',
 'ais_preserve':'AIS preserve (early stop)',
 'ais_reference_fixed':'AIS reference (12,800 draws)',
 'mcmc_parallel_20':'MCMC parallel (20 steps)',
 'mcmc_parallel_100':'MCMC parallel (100 steps)',
 'mcmc_serial_100':'MCMC serial (100 steps)'}
COLORS=dict(zip(ARMS,['#2674a6','#d58122','#258568','#9256a0','#c44e52','#6b6ecf','#66452e']))
AIS=list(ARMS)[:4]
MCMC=list(ARMS)[4:]
BETAS=[.35,.6]
FIGURES=[]

def save(fig,name,caption,group='Overview'):
    fig.tight_layout(rect=(0,.02,1,.95))
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    FIGURES.append(dict(name=name,caption=caption,group=group))

def percent(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.18)

def panels(n,width=15,height=4):
    rows=(n+2)//3
    fig,axes=plt.subplots(rows,min(3,n),figsize=(width,height*rows),squeeze=False)
    for ax in axes.flat[n:]:ax.remove()
    return fig,list(axes.flat[:n])

def interval(k,n):
    return exp.binomial_interval(int(k),int(n))

def grouped_rows(f):
    records=[]
    for (arm,condition),g in f.groupby(['arm','condition']):
        signal=g if condition=='null' else g[g.target_is_signal]
        k=int(signal.rejected.sum());n=len(signal);failed=int(signal.failed.sum());lo,hi=interval(k,n)
        records.append(dict(arm=arm,condition=condition,beta=float(g.beta.iloc[0]),n=len(g),signal_targets=n,
            detections=k,signal_failures=failed,conditional_power=k/n,ci_lower=lo,ci_upper=hi,
            targeting=n/len(g),marginal_power=k/len(g),missing_lower=k/n,missing_upper=(k+failed)/n,
            mean_proposals=g.proposals.mean(),mean_worker_seconds=g.elapsed_seconds.mean()))
    return pd.DataFrame(records)

def heatmap(ax,values,labels,columns,title,annotations=None,vmin=0,vmax=1,cmap='YlGnBu'):
    im=ax.imshow(values,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(len(columns)),xticklabels=columns,title=title)
    for i in range(len(labels)):
        for j in range(len(columns)):
            value=values[i,j]
            label=annotations[i][j] if annotations is not None else f'{value:.1%}'
            ax.text(j,i,label,ha='center',va='center',fontsize=9,
                    color='white' if np.isfinite(value) and value>(vmin+vmax)*.65 else '#203040')
    return im

def overview(f,p):
    power=p[p.beta.gt(0)]
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for arm,label in ARMS.items():
        g=power[power.arm.eq(arm)].sort_values('beta')
        for ax,col in zip(axes,['targeting','conditional_power','marginal_power']):
            ax.plot(g.beta,g[col],'o-',color=COLORS[arm],label=label)
        axes[1].fill_between(g.beta,g.ci_lower,g.ci_upper,color=COLORS[arm],alpha=.08)
    for ax,title in zip(axes,['Probability of targeting the signal','Conditional signal detection','Marginal signal detection']):
        ax.set(title=title,xlabel='Signal strength (beta)',xticks=BETAS,ylim=(0,1.03));percent(ax)
    axes[0].axhline(.5,color='gray',ls='--');axes[2].axhline(.5,color='gray',ls='--')
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('SHAP sampling follow-up: power and signal targeting')
    save(fig,'01_power_curves','250 paired inputs per beta. Unresolved signal tests count as non-detections. Shading: pointwise 95% Clopper–Pearson intervals. Lines join only two tested effect sizes; shared target selection makes targeting identical across arms.')
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,col,title in zip(axes,['conditional_power','ci_lower'],['Conditional signal detection','Pointwise 95% lower bound']):
        mat=power.pivot(index='arm',columns='beta',values=col).reindex(ARMS)[BETAS].to_numpy()
        annotations=[]
        for arm in ARMS:
            cells=[]
            for beta in BETAS:
                r=power[power.arm.eq(arm)&power.beta.eq(beta)].iloc[0]
                note=f'{r.detections:.0f}/{r.signal_targets:.0f}; unresolved={r.signal_failures:.0f}' if col=='conditional_power' else ('PASS' if r.ci_lower>=.8 and r.signal_targets>=30 and r.signal_failures==0 else 'UNRESOLVED' if r.signal_failures else 'BELOW 80%')
                cells.append(f'{r[col]:.1%}\n{note}')
            annotations.append(cells)
        heatmap(ax,mat,list(ARMS.values()),BETAS,title,annotations)
        ax.set_xlabel('Signal strength (beta)')
    fig.suptitle('Conditional signal detection and the reference 80% evidence rule')
    save(fig,'02_power_target_heatmap','PASS requires a 95% lower bound ≥80%, at least 30 signal targets, and no unresolved signal tests. This descriptive rule uses pointwise intervals and is not adjusted for multiple comparisons.')
    null=f[f.beta.eq(0)]
    calibration=[]
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for j,(arm,label) in enumerate(ARMS.items()):
        g=null[null.arm.eq(arm)];resolved=g.p_value.dropna();n=len(g)
        for alpha in [.01,.05,.1]:
            k=int(g.p_value.lt(alpha).sum());lo,hi=interval(k,n)
            calibration.append(dict(arm=arm,alpha=alpha,n=n,rejections=k,unresolved=int(g.failed.sum()),operational_rate=k/n,lower=lo,upper=hi,missing_upper=(k+g.failed.sum())/n))
        c=pd.DataFrame(calibration).query('arm==@arm')
        axes[0].errorbar(c.alpha+(j-3)*.0015,c.operational_rate,yerr=[c.operational_rate-c.lower,c.upper-c.operational_rate],fmt='o',capsize=2,color=COLORS[arm],label=label)
        vals=np.sort(resolved)
        for ax in axes[1:]:ax.step(np.r_[0,vals,1],np.r_[0,np.arange(1,len(vals)+1)/n,len(vals)/n],where='post',color=COLORS[arm])
    axes[0].plot([0,.12],[0,.12],'--',color='gray');axes[0].set(title='Null rejection rates (500 inputs)',xlabel='Nominal alpha (points offset)',ylabel='Operational rejection fraction',ylim=(0,.16));axes[0].legend(fontsize=7)
    for ax in axes[1:]:ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='p-value',ylabel='Fraction of all null tests',ylim=(0,1),xlim=(0,1))
    axes[1].set_title('Null p-value subdistribution');axes[2].set(title='Lower-tail detail',xlim=(0,.12),ylim=(0,.12))
    for ax in axes:percent(ax)
    fig.suptitle('Null calibration: unresolved outcomes retained in the denominator')
    save(fig,'03_null_calibration','Strict p < alpha rejection. Pointwise 95% exact binomial intervals. The empirical subdistribution divides by all 500 tests; unresolved outcomes have no finite p-value. Diagonal is a nominal reference, not proof of calibration. Rank tests may be conservative.')
    pd.DataFrame(calibration).to_csv(OUT/'calibration_summary.csv',index=False)
    fig,axes=panels(7)
    outcomes=[]
    for ax,(arm,label) in zip(axes,ARMS.items()):
        g=power[power.arm.eq(arm)].sort_values('beta');det=g.conditional_power;fail=g.signal_failures/g.signal_targets
        ax.bar(range(2),det,color='#258568',label='Detected');ax.bar(range(2),1-det-fail,bottom=det,color='#bcc8d1',label='Resolved, not detected');ax.bar(range(2),fail,bottom=1-fail,color='#bc4545',label='Unresolved')
        for j,r in enumerate(g.itertuples()):
            ax.text(j,1.02,f'n={r.signal_targets}; unresolved={r.signal_failures}',ha='center',fontsize=8)
            outcomes.append(dict(arm=arm,beta=r.beta,signal_targets=r.signal_targets,detected=r.detections,unresolved=r.signal_failures,resolved_not_detected=r.signal_targets-r.detections-r.signal_failures))
        ax.set(title=label,xticks=range(2),xticklabels=BETAS,ylim=(0,1.15),xlabel='Signal strength (beta)');percent(ax)
    axes[-1].legend(fontsize=8,loc='lower right');fig.suptitle('Outcomes conditional on the selected target being the signal')
    save(fig,'04_signal_outcomes','Statistical non-detection and unresolved inference are distinct outcomes; all signal targets remain in the denominator.')
    pd.DataFrame(outcomes).to_csv(OUT/'signal_outcomes.csv',index=False)
    costs=f.groupby(['arm','condition']).agg(tests=('arm','size'),mean_proposals=('proposals','mean'),median_proposals=('proposals','median'),p95_proposals=('proposals',lambda x:x.quantile(.95)),mean_worker_seconds=('elapsed_seconds','mean'),mean_uncached_fits=('uncached_selector_fits','mean')).reset_index()
    costs.to_csv(OUT/'sampling_cost_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for ax,subset,title in [(axes[0],f[f.beta.eq(0)],'Null trials'),(axes[1],f[f.beta.gt(0)],'Power trials (both betas)')]:
        ax.boxplot([subset[subset.arm.eq(a)].proposals for a in ARMS],orientation='horizontal',tick_labels=list(ARMS.values()),showfliers=False)
        ax.set(xscale='log',xlabel='Final proposals / MCMC transitions (pilot excluded)',title=title);ax.grid(alpha=.2)
    fig.suptitle('Sampling cost distributions (whiskers: 1.5 IQR; outliers omitted)')
    save(fig,'05_sampling_cost','AIS common pilots cost 3,200 evaluations per input and are counted once, separately. MCMC has no inference need for that pilot. Final AIS draws and MCMC transitions count different constructions; neither is ESS. Raw costs include all unresolved runs.')
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,arm in zip(axes.flat,AIS):
        g=f[f.arm.eq(arm)]
        for failed,marker,color in [(False,'o',COLORS[arm]),(True,'x','#bf3232')]:
            q=g[g.failed.eq(failed)];ax.scatter(q.denominator_ess.clip(lower=.1),q.tail_ess.clip(lower=.1),s=12 if not failed else 60,alpha=.25 if not failed else 1,marker=marker,color=color,label='Unresolved' if failed else 'Resolved')
        ax.axvline(80,ls='--',color='gray');ax.axhline(15,ls='--',color='gray');ax.set(xscale='log',yscale='log',title=f'{ARMS[arm]}: {int(g.failed.sum())}/1,000 unresolved',xlabel='Denominator ESS',ylabel='Tail ESS');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('Fresh AIS ESS diagnostics')
    save(fig,'06_ess_diagnostics','Thresholds: denominator ESS=80 and tail ESS=15. Zero ESS is displayed at 0.1 for logarithmic axes. MCMC replica counts are not labelled as ESS.')
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,beta in zip(axes,BETAS):
        for arm,label in ARMS.items():
            r=power[power.arm.eq(arm)&power.beta.eq(beta)].iloc[0]
            ax.errorbar(r.mean_proposals,r.conditional_power,yerr=[[r.conditional_power-r.ci_lower],[r.ci_upper-r.conditional_power]],fmt='o',capsize=3,color=COLORS[arm],label=label)
        ax.set(xscale='log',title=f'beta={beta}',xlabel='Mean final proposals / transitions per input',ylabel='Conditional signal detection',ylim=(.35,1.02));ax.axhline(.8,color='gray',ls='--');percent(ax)
    axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('Signal detection versus sampling cost')
    save(fig,'07_power_cost_tradeoff','Vertical bars are pointwise 95% exact binomial intervals. Shared pilot excluded; measured worker times are shown separately. Points share inputs and targets, so uncertainty in differences requires paired analysis.')
    return power

def paired_analysis(f):
    base=f[f.arm.eq('ais_reference')]
    records=[]
    fig,axes=panels(6)
    for ax,arm in zip(axes,list(ARMS)[1:]):
        pair=base.merge(f[f.arm.eq(arm)],on=['case_id','condition'],suffixes=('_base','_arm'),validate='one_to_one')
        complete=pair[~pair.failed_base & ~pair.failed_arm]
        for beta in [0,*BETAS]:
            g=complete[complete.beta_base.eq(beta)]
            ax.scatter(g.p_value_base,g.p_value_arm,s=12,alpha=.35,label=f'beta={beta}')
            allg=pair[pair.beta_base.eq(beta)]
            for group in ['all','signal']:
                q=allg if group=='all' else allg[allg.target_is_signal_base]
                if not len(q):continue
                gain=int((~q.rejected_base & q.rejected_arm).sum());loss=int((q.rejected_base & ~q.rejected_arm).sum())
                records.append(dict(arm=arm,beta=beta,target_group=group,n=len(q),gains=gain,losses=loss,net_difference=(gain-loss)/len(q),discordant_pairs=gain+loss,unresolved_baseline=int(q.failed_base.sum()),unresolved_arm=int(q.failed_arm.sum()),descriptive_mcnemar_p=stats.binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.))
        ax.plot([0,1],[0,1],'--',color='gray');ax.axhline(.05,color='gray',lw=.7);ax.axvline(.05,color='gray',lw=.7)
        ax.set(title=f'{ARMS[arm]}\n{len(complete)}/{len(pair)} jointly resolved',xlabel='AIS reference p-value',ylabel='Comparator p-value',xlim=(0,1),ylim=(0,1))
    axes[0].legend(fontsize=8);fig.suptitle('Matched fresh inputs: p-value agreement with AIS reference')
    save(fig,'08_paired_p_values','Only jointly resolved p-values are plotted; counts show exclusions. MCMC rank p-values and AIS tail-probability estimates have different constructions, so disagreement alone does not identify an error.')
    paired=pd.DataFrame(records);paired.to_csv(OUT/'paired_decisions.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(17,6))
    for ax,beta in zip(axes,[0,*BETAS]):
        q=paired[paired.beta.eq(beta)&paired.target_group.eq('all' if beta==0 else 'signal')].set_index('arm').reindex(list(ARMS)[1:])
        y=np.arange(6);ax.barh(y-.16,q.gains,height=.3,color='#258568',label='Comparator only');ax.barh(y+.16,-q.losses,height=.3,color='#c44e52',label='Reference only');ax.axvline(0,color='gray')
        ax.set(yticks=y,yticklabels=[ARMS[a] for a in q.index],title='Null: all 500 inputs' if beta==0 else f'beta={beta}: signal targets',xlabel='Paired rejection discordances');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Matched rejection gains and losses (unresolved = non-rejection)')
    save(fig,'09_paired_decisions','Each input contributes to paired rejection indicators. Unresolved results are operational non-rejections. Exploratory exact McNemar p-values are in paired_decisions.csv, with no multiplicity adjustment.')
    fixed=f[f.arm.eq('ais_reference_fixed')].merge(base,on='case_id',suffixes=('_fixed','_early'),validate='one_to_one')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    q=fixed.dropna(subset=['estimate_p_value_fixed','estimate_p_value_early'])
    axes[0].scatter(q.estimate_p_value_early,q.estimate_p_value_fixed,s=13,alpha=.3);axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='Early-stop raw ratio',ylabel='12,800-draw raw ratio',title='Matched proposal and RNG prefix')
    axes[1].scatter(q.proposals_early,q.estimate_p_value_fixed-q.estimate_p_value_early,s=13,alpha=.35);axes[1].axhline(0,color='gray');axes[1].set(xscale='log',xlabel='Early-stop final draws',ylabel='Fixed minus early raw ratio',title='Change with continuation')
    statuses=pd.crosstab(fixed.failed_early,fixed.failed_fixed).reindex(index=[False,True],columns=[False,True],fill_value=0)
    heatmap(axes[2],statuses.to_numpy(),['Early resolved','Early unresolved'],['Fixed resolved','Fixed unresolved'],'Resolution transitions',[[str(v) for v in row] for row in statuses.to_numpy()],vmax=max(1,statuses.to_numpy().max()),cmap='Blues')
    fig.suptitle('AIS early stopping versus fixed work: 1,000 matched inputs')
    save(fig,'10_ais_continuation','Raw importance-weight ratios include unresolved estimates and are diagnostic only. Reference/fixed draws have an audited identical prefix; this comparison isolates continuation at 12,800 draws, while early stopping can continue to 51,200.')

def mechanics(f,diagnostic,geometry):
    fig,axes=plt.subplots(1,3,figsize=(17,6))
    for j,(column,title) in enumerate([('mcmc_local_accepts','Local transition acceptance'),('mcmc_global_accepts','Global proposal acceptance')]):
        attempt='mcmc_local_attempts' if j==0 else 'mcmc_global_attempts'
        axes[j].boxplot([(f[f.arm.eq(a)][column]/f[f.arm.eq(a)][attempt]).dropna() for a in MCMC],tick_labels=[ARMS[a].replace('MCMC ','') for a in MCMC],showfliers=False)
        axes[j].set(title=title,ylabel='Accepted / attempted');percent(axes[j]);axes[j].tick_params(axis='x',rotation=15)
    axes[2].boxplot([f[f.arm.eq(a)].mcmc_unique_replicas/511 for a in MCMC],tick_labels=[ARMS[a].replace('MCMC ','') for a in MCMC],showfliers=False);axes[2].set(title='Distinct endpoint fraction',ylabel='Unique saved endpoints / 511',ylim=(0,1.02));percent(axes[2]);axes[2].tick_params(axis='x',rotation=15)
    fig.suptitle('Fresh MCMC movement diagnostics')
    save(fig,'11_mcmc_movement','Whiskers: 1.5 IQR; outliers omitted. Acceptance and distinct endpoints do not establish mixing or effective sample size. Parallel endpoints share a hub; serial states are dependent.')
    rows=[]
    for cohort,g in [('fresh',f),('diagnostic',diagnostic)]:
        for arm in MCMC:
            q=g[g.arm.eq(arm)]
            rows.append(dict(cohort=cohort,arm=arm,tests=len(q),mean_local_acceptance=(q.mcmc_local_accepts/q.mcmc_local_attempts).mean(),mean_global_acceptance=(q.mcmc_global_accepts/q.mcmc_global_attempts).mean(),median_unique_endpoints=q.mcmc_unique_replicas.median(),zero_moved=int(q.mcmc_moved_replicas.eq(0).sum()) if 'parallel' in arm else None))
    pd.DataFrame(rows).to_csv(OUT/'mcmc_movement_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,col,title in zip(axes,['elapsed_seconds','uncached_selector_fits'],['Measured sampler worker time','Uncached selector fits']):
        ax.boxplot([f[f.arm.eq(a)][col] for a in ARMS],orientation='horizontal',tick_labels=list(ARMS.values()),showfliers=False);ax.set(xscale='log',xlabel='Seconds' if col=='elapsed_seconds' else 'Forest fits',title=title);ax.grid(alpha=.2)
    fig.suptitle('Measured incremental costs with shared caches')
    save(fig,'12_measured_cost','Worker times exclude the common pilot, setup, and other arms. Fixed continuation reuses cached reference queries. These are incremental costs under concurrent execution, not standalone timings or elapsed server runtime.')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for arm in AIS:
        g=f[f.arm.eq(arm)];axes[0].scatter(g.max_normalized_weight,g.denominator_ess,s=12,alpha=.25,color=COLORS[arm],label=ARMS[arm]);axes[1].scatter(g.t_obs,g.max_normalized_weight,s=12,alpha=.25,color=COLORS[arm])
        bad=g[g.failed];axes[0].scatter(bad.max_normalized_weight,bad.denominator_ess,marker='x',s=55,color='#bf3232');axes[1].scatter(bad.t_obs,bad.max_normalized_weight,marker='x',s=55,color='#bf3232')
    axes[0].set(xscale='log',yscale='log',xlabel='Maximum normalized weight',ylabel='Denominator ESS',title='Weight concentration and ESS');axes[0].axhline(80,color='gray',ls='--');axes[0].legend(fontsize=8)
    axes[1].set(yscale='log',xlabel='Observed radial statistic',ylabel='Maximum normalized weight',title='Concentration versus observed radius');fig.suptitle('Fresh AIS weight diagnostics (red crosses: unresolved)')
    save(fig,'13_weight_concentration','Weight concentration describes the observed sample, not unvisited selection regions. High acceptance counts need not imply high ESS.')
    replay_arms=[a for a in ARMS if a!='ais_reference_fixed'];cases=list(diagnostic.case_id.unique())
    fig,axes=plt.subplots(1,2,figsize=(18,8))
    for ax,metric,title in zip(axes,['p_value','denominator_ess'],['Reported p-values (X = unresolved)','AIS denominator ESS (MCMC = not applicable)']):
        values=np.full((len(cases)*2,6),np.nan);annotations=[];labels=[]
        for i,case in enumerate(cases):
            for repeat in range(2):
                labels.append(case+f' / seed {repeat+1}');cells=[]
                for j,arm in enumerate(replay_arms):
                    r=diagnostic[diagnostic.case_id.eq(case)&diagnostic.repeat.eq(repeat)&diagnostic.arm.eq(arm)].iloc[0];v=r[metric]
                    values[i*2+repeat,j]=v if metric=='p_value' else np.log10(max(v,.1)) if np.isfinite(v) else np.nan
                    cells.append('X' if metric=='p_value' and r.failed else '—' if not np.isfinite(v) else (f'{v:.1e}' if 0<v<.001 else f'{v:.3f}') if metric=='p_value' else f'{v:.1f}')
                annotations.append(cells)
        heatmap(ax,values,labels,[ARMS[a].replace(' (early stop)','').replace('MCMC ','').replace('AIS ','') for a in replay_arms],title,annotations,vmin=0,vmax=1 if metric=='p_value' else 4)
        ax.tick_params(axis='x',rotation=35,labelsize=8)
    fig.suptitle('Selected-case replays: two seeds per fixed input')
    save(fig,'14_selected_case_replays','96 replays on eight deliberately selected cases; not a calibration or power sample. Diagnostic AIS uses fixed 12,800 draws for all three variants. Cells marked X retain unresolved outcomes; the raw ratios are shown in the per-case traces.')
    fig,axes=plt.subplots(1,2,figsize=(16,9));g=geometry.sort_values('case_id');y=np.arange(len(g));colors=np.where(g.stabilized,'#258568','#c44e52')
    axes[0].barh(y,g.denominator_relative_change.clip(lower=1e-8),color=colors);axes[0].axvline(.05,color='gray',ls='--');axes[0].set(xscale='log',yticks=y,yticklabels=g.case_id,title='Selection-mass change between grid passes',xlabel='Absolute relative change (zero shown at 1e-8)');axes[0].invert_yaxis()
    axes[1].scatter(g.previous_diagnostic_p,g.diagnostic_p,c=colors,s=35);axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(xlabel='First-pass diagnostic ratio',ylabel='Final-grid diagnostic ratio',title=f'{int(g.stabilized.sum())}/{len(g)} met stabilization rule; none certified',xlim=(0,1),ylim=(0,1));fig.suptitle('Finite-grid selection geometry: convergence diagnostics')
    save(fig,'15_geometry_stability','Green: met the predefined stabilization rule; red: did not. All maps remain uncertified. Discovered interval counts and grid-integrated tail ratios may miss narrow islands; they cannot serve as exact reference p-values.')


def case_figures(diagnostic,geometry):
    (OUT/'regions').mkdir(exist_ok=True);(OUT/'replays').mkdir(exist_ok=True)
    geom=geometry.set_index('case_id')
    for case,r in geom.iterrows():
        with np.load(SOURCE/'geometry'/case/'trace.npz',allow_pickle=False) as z:
            grid=z['grid_z'];selected=z['grid_selected'];margin=z['query_margin'];query=z['query_z']
        fig,axes=plt.subplots(3,1,figsize=(13,8))
        intervals=json.loads(r.intervals)
        for lo,hi in intervals:axes[0].axvspan(lo,hi,color='#258568',alpha=.3)
        axes[0].scatter(grid,np.where(selected,.8,.2),s=3,c=np.where(selected,'#258568','#c44e52'),rasterized=True)
        axes[0].set(yticks=[.2,.8],yticklabels=['Not selected','Exact set selected'],title='Saved grid states and inferred intervals',ylim=(0,1))
        axes[1].scatter(query,margin,s=3,alpha=.5);axes[1].axhline(0,color='gray',lw=.7);axes[1].set(ylabel='Selected / excluded importance gap',title='Selector score margin on evaluated radii')
        axes[2].plot(grid,stats.chi.logpdf(grid,3)/np.log(10),color='#2674a6',label='Unconditional chi density (rank 3)')
        for lo,hi in intervals:axes[2].axvspan(lo,hi,color='#258568',alpha=.2)
        axes[2].set(ylabel='log10 chi density',xlabel='Radial statistic z',title='Unconditional mass scale; shading is finite-grid selection geometry');axes[2].legend(fontsize=8)
        for ax in axes:ax.axvline(r.t_obs,color='#9256a0',ls='--',label='Observed radius');ax.set_xlim(0,r.z_max);ax.grid(alpha=.15)
        fig.suptitle(f'{case}\nFinite grid: {r.grid_evaluations} evaluations; stabilized={r.stabilized}; diagnostic ratio={r.diagnostic_p:.4g}; NOT certified',fontsize=12)
        save(fig,'regions/'+case,'Saved selector evaluations and inferred intervals. Dashed line: observed radius. Grid ratios and interval boundaries are diagnostics, not exact integration; all maps are uncertified. Rank 3 verified from the spline basis.','Selection-region maps')
    for case,g in diagnostic.groupby('case_id',sort=True):
        fig,axes=plt.subplots(2,2,figsize=(15,10))
        for repeat,ls in [(0,'-'),(1,'--')]:
            for arm in AIS[:3]:
                with np.load(SOURCE/'runs/diagnostic'/case/f'repeat_{repeat}'/arm/'trace.npz',allow_pickle=False) as z:
                    points=z['checkpoints'];weights=np.exp(z['selected_log_weights']-np.max(z['selected_log_weights'])) if len(z['selected_log_weights']) else np.array([])
                    axes[0,0].plot(points[:,0],points[:,1],ls,color=COLORS[arm],label=ARMS[arm].replace(' (early stop)','')+f' / seed {repeat+1}')
                    axes[0,1].plot(points[:,0],points[:,2],ls,color=COLORS[arm]);axes[1,0].plot(points[:,0],points[:,3],ls,color=COLORS[arm])
                    if repeat==0 and len(weights):axes[1,1].hist(z['selected_z'],bins=70,weights=weights/weights.sum(),histtype='step',color=COLORS[arm],label=arm.replace('ais_','AIS '))
        for arm in MCMC:
            for repeat in range(2):
                r=g[g.arm.eq(arm)&g.repeat.eq(repeat)].iloc[0];axes[0,0].axhline(r.p_value,color=COLORS[arm],ls=':' if repeat==0 else '-.',alpha=.65,label=ARMS[arm]+f' / seed {repeat+1}')
        axes[0,0].set(title='AIS raw ratio trajectories; horizontal MCMC ranks',xlabel='Final AIS draws',ylabel='Tail ratio / rank p-value',ylim=(-.02,1.02));axes[0,0].axhline(.05,color='gray',lw=.7);axes[0,0].legend(fontsize=6,ncol=2)
        axes[0,1].set(title='Denominator ESS trajectories',xlabel='Final AIS draws',ylabel='Denominator ESS',yscale='log');axes[0,1].axhline(80,color='gray',ls='--')
        axes[1,0].set(title='Tail ESS trajectories',xlabel='Final AIS draws',ylabel='Tail ESS',yscale='log');axes[1,0].axhline(15,color='gray',ls='--')
        axes[1,1].set(title='Seed 1: weighted selected AIS radii',xlabel='Radial statistic z',ylabel='Normalized sampled weight per bin');axes[1,1].axvline(geom.loc[case,'t_obs'],ls='--',color='gray');axes[1,1].legend(fontsize=8)
        for ax in axes.flat:ax.grid(alpha=.15)
        fig.suptitle(f'{case}: fixed-work diagnostic replays',fontsize=13)
        save(fig,'replays/'+case,'AIS trajectories show raw ratios even when ESS criteria fail; these unresolved ratios are not reported p-values. MCMC horizontal lines are reversible rank tests, not independent estimates of the AIS estimand. Weighted histograms summarize visited radii only.','Selected-case traces')


def reports(f,p,diagnostic,geometry):
    summary=[]
    for arm in ARMS:
        n=p[p.arm.eq(arm)&p.beta.eq(0)].iloc[0];strong=p[p.arm.eq(arm)&p.beta.eq(.6)].iloc[0]
        summary.append(f'{ARMS[arm]}: null {n.detections}/{n.n} ({n.conditional_power:.1%}); beta=0.6 signal detection {strong.detections}/{strong.signal_targets} ({strong.conditional_power:.1%}); fresh unresolved {int(f[f.arm.eq(arm)].failed.sum())}/1,000.')
    intro='Seven inference arms on 1,000 matched fresh inputs: 500 global-null trials and 250 trials at each beta (0.35, 0.60). Fixed Tree SHAP selector, exact-set conditioning, spline statistic, and uniformly randomized selected target. Four AIS arms and three reversible MCMC rank constructions. Diagnostic replays and region maps use deliberately selected inputs and remain separate from fresh performance estimates.'
    limits='Signal detection conditions on targeting feature 0; marginal detection includes target randomization. Other selected features under alternatives are not automatically projection-null controls. Unresolved AIS results remain visible and count as operational non-detections; missing-outcome bounds are in power_summary.csv. Intervals are pointwise 95% Clopper–Pearson intervals. MCMC ranks are not identical estimands to AIS conditional-tail estimates. Only 4/28 finite-grid maps stabilized; none is certified. No calibration conclusion follows from completion or these plots.'
    findings=[
        'Fresh beta=0.35 signal detection is 54.3–56.2%; beta=0.60 detection is 91.8–95.1%. With 105 and 122 signal targets respectively, these small between-arm differences have substantial sampling uncertainty; the paired plots show the actual discordances.',
        'Fresh global-null operational rejection rates are 3.0–4.0% at alpha=0.05. No arm shows an elevated point estimate here; 500 null trials do not establish conditional calibration.',
        'All three MCMC arms resolve all 1,000 fresh tests and all 16 selected-case replays per arm. Fresh AIS unresolved counts are reference=2, stable=1, preserve=1 and fixed reference=4. Diagnostic fixed-work AIS unresolved counts are 7/16, 7/16 and 8/16 for reference, stable and preserve. Selected-case failures remain substantial.',
        'The average early-stop reference uses 1,004 final draws per fresh input. Parallel MCMC uses 10,240 or 51,200 transitions; serial MCMC uses 51,100. AIS additionally needs the shared 3,200-evaluation pilot. Compare standalone requirements separately from the actual shared-cache worker timings.',
        'The two historical disagreement cases remain informative: parallel MCMC ranks can be much larger than AIS ratios, and serial ranks vary between seeds. Reversible-rank validity does not imply rapid mixing or equality to the conditional-tail probability; the plots do not identify a unique correct tail ratio.',
        'Only 4/28 region maps meet the stabilization criterion. No map is certified, and its diagnostic integration ratio must not be used as ground truth for judging AIS or MCMC.'
    ]
    md=['# SHAP sampling follow-up analysis',intro,limits,'## Findings',*['- '+line for line in findings],'## Results',*['- '+line for line in summary]]
    sections=[]
    for group in ['Overview','Selection-region maps','Selected-case traces']:
        md.append('## '+group)
        for entry in FIGURES:
            if entry['group']!=group:continue
            name=entry['name'];title=name.split('/')[-1].replace('_',' ')
            md.append(f'### {title}\n\n{entry["caption"]}\n\n![{title}]({name}.png)')
            sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(entry["caption"])}</p><a href="{name}.png"><img loading="lazy" src="{name}.png" alt="{html.escape(title)}"></a></section>')
    (OUT/'README.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SHAP sampling follow-up analysis</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>SHAP sampling follow-up analysis</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(limits)+'</p><h2>Findings</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in findings)+'</ul><h2>Results by arm</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in summary)+'</ul>'+''.join(sections)+'</html>',encoding='utf-8')
    files=['unified_tests.csv','sampling_summary.csv','geometry_summary.csv','requested_manifest.json','source_hashes.json','local_verification_report.json']
    manifest=dict(source=SOURCE.relative_to(ROOT).as_posix(),source_sha256={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files},plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=FIGURES,format='PNG only',fresh_rows=len(f),diagnostic_rows=len(diagnostic),geometry_rows=len(geometry),local_verification=json.loads((SOURCE/'local_verification_report.json').read_text()),regeneration=json.loads((SOURCE/'local_input_regeneration_report.json').read_text()))
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (OUT/'COMPLETED').write_text(f'Created {len(FIGURES)} PNG figures from 7,000 fresh outcomes, 96 selected-case replays and 28 diagnostic maps. No inference rerun.\n')


def main():
    require=exp.require
    require((SOURCE/'COMPLETED').exists(),'Source results incomplete')
    require(json.loads((SOURCE/'local_verification_report.json').read_text())['status']=='passed','Run VerifyLocal before building analysis')
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    frame=pd.read_csv(SOURCE/'unified_tests.csv',keep_default_na=False,na_values=[''])
    require(len(frame)==7096 and not frame.duplicated(['cohort','case_id','repeat','arm']).any(),'Missing/duplicate source rows')
    f=frame[frame.cohort.eq('fresh')].copy();diagnostic=frame[frame.cohort.eq('diagnostic')].copy()
    require(len(f)==7000 and len(diagnostic)==96,'Cohort count mismatch')
    require(f.groupby(['arm','condition']).size().to_dict()=={(a,c):n for a in ARMS for c,n in [('null',500),('beta_0.35',250),('beta_0.6',250)]},'Condition count drift')
    for _,g in f.groupby('case_id'):
        require(len(g)==7 and g[['input_hash','target_feature','t_obs','auxiliary_u']].nunique().eq(1).all(),'Unmatched input/target')
    require(frame.failed.eq(frame.p_value.isna()).all() and frame.rejected.eq(frame.p_value.lt(.05)).all(),'Outcome mismatch')
    geometry=pd.read_csv(SOURCE/'geometry_summary.csv')
    require(len(geometry)==28 and not geometry.regions_certified.any(),'Unexpected map certification')
    require(frame.test_rank.eq(3).all(),'Region density rank assumption changed')
    p=grouped_rows(f)
    recorded=pd.read_csv(SOURCE/'sampling_summary.csv',keep_default_na=False,na_values=[''])
    for r in p.itertuples():
        old=recorded[recorded.arm.eq(r.arm)&recorded.condition.eq(r.condition)].iloc[0]
        require(r.detections==old.detections_or_rejections and r.signal_failures==old.unresolved_targets and np.isclose(r.conditional_power,old.operational_fraction),'Summary disagrees with verified source')
    p.to_csv(OUT/'power_summary.csv',index=False)
    overview(f,p);paired_analysis(f);mechanics(f,diagnostic,geometry);case_figures(diagnostic,geometry);reports(f,p,diagnostic,geometry)
    print(f'Created {len(FIGURES)} PNG figures, CSV summaries and gallery in {OUT}')

if __name__=='__main__':main()
