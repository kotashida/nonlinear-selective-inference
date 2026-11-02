"""PNG-only comprehensive analysis of the four presentation selectors.

Run with .venv/Scripts/python.exe examples/build_presentation_four_mcmc20_analysis.py.
Uses verified saved results only; no inference is rerun.
"""
from pathlib import Path
import hashlib
import html
import json
import os
import sys
os.environ['MPLBACKEND']='Agg'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
from examples import run_presentation_four_selectors_mcmc20_250_20261008 as run
SOURCE=ROOT/'outputs/selection_method_validation/exact_set/presentation_four_selectors_mcmc20_250_20261008'
OUT=SOURCE/'analysis'
METHODS={'shap':'Tree SHAP','lime':'LIME (40)','spline_screening':'Spline screening','mutual_information':'KDE-MI (2.5)'}
COLORS=dict(zip(METHODS,['#2674a6','#d58122','#258568','#9256a0']))
BETAS=[.25,.5,.75,1.]
FIGURES=[]

def save(fig,name,caption):
    fig.tight_layout(rect=(0,.02,1,.94))
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig);FIGURES.append(dict(name=name,caption=caption))

def percent(ax):ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)

def heat(ax,values,rows,cols,title,annotations=None,vmin=0,vmax=1):
    im=ax.imshow(values,aspect='auto',vmin=vmin,vmax=vmax,cmap='YlGnBu')
    ax.set(title=title,yticks=range(len(rows)),yticklabels=rows,xticks=range(len(cols)),xticklabels=cols)
    for i in range(len(rows)):
        for j in range(len(cols)):
            value=values[i,j];label=annotations[i][j] if annotations is not None else f'{value:.1%}'
            ax.text(j,i,label,ha='center',va='center',fontsize=10,color='white' if np.isfinite(value) and value>.65*vmax else '#203040')
    return im

def load():
    run.require(json.loads((SOURCE/'local_verification_report.json').read_text())['status']=='passed','Run VerifyLocal before analysis')
    m=run.read_manifest();f=pd.read_csv(SOURCE/'unified_tests.csv',keep_default_na=False,na_values=[''])
    run.require(len(f)==5000 and set(f.method)==set(METHODS),'Result count/method drift')
    run.require(not f.duplicated(['method','condition','trial']).any(),'Duplicate trials')
    run.require(f.groupby(['method','beta']).size().eq(250).all() and len(f.groupby(['method','beta']))==20,'Condition counts changed')
    run.require(f.rejected.eq(f.p_value.lt(.05)).all() and f.failed.eq(f.p_value.isna()).all(),'Outcome mismatch')
    run.require(f.proposals.eq(10240).all() and f.mcmc_replicates.eq(511).all() and f.mcmc_steps.eq(20).all(),'MCMC construction mismatch')
    for _,g in f.groupby(['beta','trial']):
        run.require(len(g)==4 and g[['input_sha256','data_seed','target_seed','sampler_seed','auxiliary_u']].nunique().eq(1).all(),'Dataset/seed pairing changed')
    summary,calibration=run.summaries(f,m)
    for name,expected in [('power_summary.csv',summary[summary.beta.gt(0)]),('null_summary.csv',summary[summary.beta.eq(0)]),('calibration_summary.csv',calibration)]:
        import io
        actual=pd.read_csv(SOURCE/name,keep_default_na=False,na_values=[''])
        roundtrip=pd.read_csv(io.StringIO(expected.to_csv(index=False)),keep_default_na=False,na_values=[''])
        pd.testing.assert_frame_equal(actual,roundtrip,check_dtype=False,rtol=1e-12,atol=1e-12)
    p=summary[summary.beta.gt(0)].copy()
    targeting=[]
    for r in p.itertuples():
        lo,hi=run.ci(int(r.signal_targets),int(r.tests));mlo,mhi=run.ci(int(r.detections),int(r.tests))
        targeting.append(dict(method=r.method,beta=r.beta,signal_targets=r.signal_targets,n=r.tests,rate=r.targeting_probability,lower=lo,upper=hi,marginal_lower=mlo,marginal_upper=mhi))
    t=pd.DataFrame(targeting);p=p.merge(t[['method','beta','marginal_lower','marginal_upper']],on=['method','beta'],validate='one_to_one')
    p.to_csv(OUT/'power_summary.csv',index=False);summary[summary.beta.eq(0)].to_csv(OUT/'null_summary.csv',index=False);calibration.to_csv(OUT/'calibration_summary.csv',index=False);t.to_csv(OUT/'targeting_summary.csv',index=False)
    return f,p,calibration,t

def reference_plots(f,p,c,t):
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for method,label in METHODS.items():
        g=p[p.method.eq(method)].sort_values('beta')
        for ax,column in zip(axes,['targeting_probability','operational_fraction','marginal_signal_detection']):ax.plot(g.beta,g[column],'o-',color=COLORS[method],label=label)
        axes[1].fill_between(g.beta,g.ci_lower,g.ci_upper,alpha=.10,color=COLORS[method]);axes[2].fill_between(g.beta,g.marginal_lower,g.marginal_upper,alpha=.07,color=COLORS[method])
    for ax,title in zip(axes,['Probability of targeting feature 0','Conditional signal detection','Marginal signal detection']):
        ax.set(title=title,xlabel='Signal strength (beta)',xticks=BETAS,ylim=(0,1.03));percent(ax)
    axes[0].axhline(.5,ls='--',color='gray');axes[2].axhline(.5,ls='--',color='gray');axes[0].legend(fontsize=9,loc='lower left');fig.suptitle('Parallel MCMC (20 steps): power and signal targeting')
    save(fig,'01_power_curves','250 paired synthetic inputs per beta and method. Shading: pointwise 95% Clopper-Pearson intervals. Conditional detection divides by signal targets; marginal detection divides by all 250 trials. The population targeting ceiling is 50%; a finite sample can exceed it. No unresolved outcomes.')
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,col,title in zip(axes,['operational_fraction','ci_lower'],['Conditional signal detection','Pointwise 95% lower confidence bound']):
        mat=p.pivot(index='method',columns='beta',values=col).reindex(METHODS)[BETAS].to_numpy();annotations=[]
        for method in METHODS:
            cells=[]
            for beta in BETAS:
                r=p[p.method.eq(method)&p.beta.eq(beta)].iloc[0]
                note=f'{r.detections:.0f}/{r.signal_targets:.0f}' if col=='operational_fraction' else 'PASS' if r.ci_lower>=.8 and r.signal_targets>=30 and r.unresolved==0 else 'UNRESOLVED' if r.unresolved else 'BELOW 80%'
                cells.append(f'{r[col]:.1%}\n{note}')
            annotations.append(cells)
        heat(ax,mat,list(METHODS.values()),BETAS,title,annotations);ax.set_xlabel('Signal strength (beta)')
    fig.suptitle('The reference 80% conditional-power evidence rule')
    save(fig,'02_power_target_heatmap','PASS requires a pointwise 95% lower bound of at least 80%, at least 30 signal targets, and no unresolved signal tests. This descriptive rule is not adjusted for multiple comparisons. Counts show detections / signal targets.')
    null=f[f.beta.eq(0)]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for j,(method,label) in enumerate(METHODS.items()):
        g=c[c.method.eq(method)].sort_values('alpha');axes[0].errorbar(g.alpha+(j-1.5)*.0015,g.fraction,yerr=[g.fraction-g.ci_lower,g.ci_upper-g.fraction],fmt='o',capsize=3,color=COLORS[method],label=label)
        vals=np.sort(null[null.method.eq(method)].p_value)
        for ax in axes[1:]:ax.step(np.r_[0,vals,1],np.r_[0,np.arange(1,251)/250,1],where='post',color=COLORS[method])
    axes[0].plot([0,.12],[0,.12],'--',color='gray');axes[0].set(title='Null rejection rates (250 trials)',xlabel='Nominal alpha (points offset)',ylabel='Rejection fraction',xticks=[.01,.05,.1],ylim=(0,.18));axes[0].legend(fontsize=8)
    for ax in axes[1:]:ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='p-value',ylabel='Fraction at or below p',xlim=(0,1),ylim=(0,1));percent(ax)
    axes[1].set_title('Null p-value empirical distribution');percent(axes[0]);fig.suptitle('Null calibration under the known Gaussian model')
    save(fig,'03_null_calibration','Strict p < alpha rejection with pointwise 95% exact binomial intervals. All 250 tests per method are resolved. The diagonal is a uniform reference; rank p-values can be conservative. These sample sizes do not establish conditional calibration.')
    fig,axes=plt.subplots(2,2,figsize=(13,8));outcomes=[]
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=p[p.method.eq(method)].sort_values('beta');det=g.operational_fraction;fail=g.unresolved/g.signal_targets;x=np.arange(4)
        ax.bar(x,det,color='#258568',label='Detected');ax.bar(x,1-det-fail,bottom=det,color='#bcc8d1',label='Resolved, not detected');ax.bar(x,fail,bottom=1-fail,color='#c44e52',label='Unresolved')
        for j,r in enumerate(g.itertuples()):
            ax.text(j,1.02,f'n={r.signal_targets:.0f}',ha='center',fontsize=9);outcomes.append(dict(method=method,beta=r.beta,signal_targets=r.signal_targets,detected=r.detections,resolved_not_detected=r.signal_targets-r.detections-r.unresolved,unresolved=r.unresolved))
        ax.set(title=label,xticks=x,xticklabels=BETAS,ylim=(0,1.12),xlabel='Signal strength (beta)');percent(ax)
    handles,labels=axes[0,0].get_legend_handles_labels()
    legend_entries=[(handle,label) for handle,label in zip(handles,labels) if label!='Unresolved' or p.unresolved.gt(0).any()]
    axes[0,0].legend([entry[0] for entry in legend_entries],[entry[1] for entry in legend_entries],fontsize=8,loc='lower right');fig.suptitle('Outcomes conditional on targeting the signal')
    save(fig,'04_signal_outcomes','Every signal target is retained in the denominator. No unresolved outcomes were observed; statistical non-detection is shown separately.')
    pd.DataFrame(outcomes).to_csv(OUT/'signal_outcomes.csv',index=False)
    costs=f.groupby(['method','beta']).agg(tests=('method','size'),mean_transitions=('proposals','mean'),mean_worker_seconds=('elapsed_seconds','mean'),median_worker_seconds=('elapsed_seconds','median'),p95_worker_seconds=('elapsed_seconds',lambda x:x.quantile(.95)),total_worker_seconds=('elapsed_seconds','sum'),total_cpu_seconds=('cpu_seconds','sum')).reset_index();costs.to_csv(OUT/'sampling_cost_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5));y=np.arange(4)
    for ax,col,title in zip(axes,['proposals','elapsed_seconds'],['Equal transition budgets','Different measured worker costs']):
        vals=[f[f.method.eq(method)][col].mean() for method in METHODS];ax.barh(y,vals,color=list(COLORS.values()));ax.set(yticks=y,yticklabels=list(METHODS.values()),xlabel='Transitions / trial' if col=='proposals' else 'Mean worker seconds / trial (log scale)',title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        if col=='elapsed_seconds':ax.set_xscale('log');ax.set_xlim(.5,max(vals)*2)
        for i,v in enumerate(vals):ax.text(v*1.03,i,f'{v:,.0f}' if col=='proposals' else f'{v:.2f} s',va='center',fontsize=9)
    fig.suptitle('Sampling budget versus measured computation')
    save(fig,'05_sampling_cost','Every test uses 10,240 transitions: one 20-step hub and 511 20-step spokes. Worker times include per-trial selector setup, inference and final test-basis construction, before output writes. They reflect concurrent execution, not elapsed server runtime.')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,column,title in zip(axes.flat,['mcmc_hub_acceptance_rate','mcmc_replicate_acceptance_rate','mcmc_moved_replicas','mcmc_unique_replicas'],['Hub transition acceptance','Spoke transition acceptance','Spokes moving away from hub','Distinct saved spoke endpoints']):
        values=[f[f.method.eq(method)][column]/(511 if column in ['mcmc_moved_replicas','mcmc_unique_replicas'] else 1) for method in METHODS]
        ax.boxplot(values,tick_labels=list(METHODS.values()),showfliers=True,flierprops=dict(markersize=2,alpha=.2));ax.set(title=title,ylim=(-.03,1.03),ylabel='Acceptance fraction' if 'acceptance' in column else 'Fraction of 511 replicas');percent(ax)
    fig.suptitle('MCMC movement diagnostics across all 1,250 trials per selector')
    save(fig,'06_mcmc_diagnostics','Replaces the reference AIS ESS plot. Acceptance, moved replicas, and distinct endpoints are descriptive diagnostics, not ESS or proof of mixing. All spokes share a hub. The raw endpoint paths were not saved, so no chain trace or autocorrelation estimate is manufactured.')
    move=f.groupby(['method','beta']).agg(tests=('method','size'),mean_hub_acceptance=('mcmc_hub_acceptance_rate','mean'),mean_spoke_acceptance=('mcmc_replicate_acceptance_rate','mean'),median_moved=('mcmc_moved_replicas','median'),zero_moved=('mcmc_moved_replicas',lambda x:int(x.eq(0).sum())),median_unique=('mcmc_unique_replicas','median')).reset_index();move.to_csv(OUT/'mcmc_diagnostics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,beta in zip(axes,[.5,1.]):
        for method,label in METHODS.items():
            r=p[p.method.eq(method)&p.beta.eq(beta)].iloc[0];cost=costs[costs.method.eq(method)&costs.beta.eq(beta)].mean_worker_seconds.iloc[0]
            ax.errorbar(cost,r.operational_fraction,yerr=[[r.operational_fraction-r.ci_lower],[r.ci_upper-r.operational_fraction]],fmt='o',color=COLORS[method],capsize=4,label=label)
        ax.set(xscale='log',xlabel='Mean measured worker seconds / trial',ylabel='Conditional signal detection',title=f'beta={beta:g}',ylim=(.6,1.015));ax.axhline(.8,ls='--',color='gray');percent(ax)
    axes[0].legend(fontsize=9,loc='lower right');fig.suptitle('Power versus computation with the same MCMC budget')
    save(fig,'07_power_cost_tradeoff','All methods have identical transition budgets, so this plot uses actual measured worker time. Vertical bars are pointwise 95% exact binomial intervals. Methods can select different targets; this is an end-to-end pipeline comparison.')
    return costs

def additional_plots(f,p,t,costs):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,sub,title in [(axes[0],f[f.beta.eq(0)],'Null trials'),(axes[1],f[f.beta.gt(0)],'Power trials pooled across betas')]:
        ax.boxplot([sub[sub.method.eq(m)].elapsed_seconds for m in METHODS],orientation='horizontal',tick_labels=list(METHODS.values()),showfliers=True,flierprops=dict(markersize=2,alpha=.2));ax.set(xscale='log',xlabel='Measured worker seconds / trial',title=title);ax.grid(alpha=.2)
    fig.suptitle('Worker-time distributions')
    save(fig,'08_runtime_distributions','Whiskers are 1.5 IQR; outliers are retained as small dots. Includes selector setup and inference. Summed worker/CPU times are resource costs, not elapsed wall time. Power pooling is descriptive; condition-specific summaries are in the CSV.')
    # Pair end-to-end detection indicators across all shared inputs. Conditioning
    # separately on each selector's signal targets would destroy that pairing.
    records=[];rng=np.random.default_rng(2026100901);base=f[f.method.eq('shap')]
    for method in list(METHODS)[1:]:
        pair=base.merge(f[f.method.eq(method)],on=['beta','trial'],suffixes=('_base','_arm'),validate='one_to_one')
        for beta in [0,*BETAS]:
            q=pair[pair.beta.eq(beta)];left=q.rejected_base.to_numpy();right=q.rejected_arm.to_numpy()
            if beta>0:left=left & q.target_is_signal_base.to_numpy();right=right & q.target_is_signal_arm.to_numpy()
            delta=right.astype(int)-left.astype(int);gain=int((delta==1).sum());loss=int((delta==-1).sum());boot=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
            records.append(dict(method=method,beta=beta,n=len(q),gains=gain,losses=loss,net_fraction=delta.mean(),bootstrap_lower=lo,bootstrap_upper=hi,matched_targets=int(q.target_feature_base.eq(q.target_feature_arm).sum()),mcnemar_p=stats.binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.,metric='null_rejection' if beta==0 else 'marginal_signal_detection'))
    paired=pd.DataFrame(records);paired.to_csv(OUT/'paired_comparisons.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for j,method in enumerate(list(METHODS)[1:]):
        g=paired[paired.method.eq(method)&paired.beta.gt(0)].sort_values('beta');axes[0].errorbar(g.beta+(j-1)*.012,g.net_fraction,yerr=[np.maximum(0,g.net_fraction-g.bootstrap_lower),np.maximum(0,g.bootstrap_upper-g.net_fraction)],fmt='o-',capsize=3,color=COLORS[method],label=METHODS[method])
        q=paired[paired.method.eq(method)&paired.beta.eq(0)].iloc[0];axes[1].bar(j-.16,q.gains,width=.3,color='#258568',label='Comparator only' if j==0 else None);axes[1].bar(j+.16,-q.losses,width=.3,color='#c44e52',label='Tree SHAP only' if j==0 else None)
    axes[0].axhline(0,color='gray',ls='--');axes[0].set(xticks=BETAS,xlabel='Signal strength (beta)',ylabel='Comparator minus Tree SHAP',title='Paired marginal signal-detection differences');percent(axes[0]);axes[0].legend(fontsize=8)
    axes[1].axhline(0,color='gray');axes[1].set(xticks=range(3),xticklabels=[METHODS[x] for x in list(METHODS)[1:]],ylabel='Discordant null rejections',title='Null: paired rejection gains / losses');axes[1].legend(fontsize=8);fig.suptitle('Paired end-to-end comparisons on shared datasets')
    save(fig,'09_paired_detection_comparisons','Power indicators are target=feature 0 AND rejection, compared across all 250 paired inputs. Error bars: 20,000 paired bootstrap resamples, seed 2026100901; pointwise and exploratory. Exact McNemar p-values are in the CSV, without multiplicity adjustment. This is not a paired difference of separately conditioned power rates.')
    fig,axes=plt.subplots(1,3,figsize=(16,5));agreement=[]
    for ax,method in zip(axes,list(METHODS)[1:]):
        pair=base.merge(f[f.method.eq(method)],on=['beta','trial'],suffixes=('_base','_arm'),validate='one_to_one');same=pair[pair.target_feature_base.eq(pair.target_feature_arm)]
        for beta in [0,*BETAS]:
            q=same[same.beta.eq(beta)];ax.scatter(q.p_value_base,q.p_value_arm,s=10,alpha=.3,label=f'beta={beta:g}');agreement.append(dict(method=method,beta=beta,matched_target_pairs=len(q),total_pairs=250,median_abs_p_difference=(q.p_value_base-q.p_value_arm).abs().median()))
        ax.plot([0,1],[0,1],'--',color='gray');ax.axhline(.05,color='gray',lw=.5);ax.axvline(.05,color='gray',lw=.5);ax.set(xlabel='Tree SHAP p-value',ylabel='Comparator p-value',xlim=(0,1),ylim=(0,1),title=f'{METHODS[method]}\n{len(same)}/1,250 same-target inputs')
    axes[0].legend(fontsize=7);fig.suptitle('Same-target p-value comparisons')
    save(fig,'10_same_target_p_values','Only pairs with the same target feature are plotted. Their selector-specific conditioning events still differ, so disagreement does not establish an inference error. This selected subset is not a calibration sample.')
    pd.DataFrame(agreement).to_csv(OUT/'same_target_agreement.csv',index=False)
    pairrecords=[]
    for beta in [0,*BETAS]:
        subset=f[f.beta.eq(beta)]
        for left in METHODS:
            for right in METHODS:
                q=subset[subset.method.eq(left)].merge(subset[subset.method.eq(right)],on='trial',suffixes=('_left','_right'),validate='one_to_one')
                same_sets=np.array([set(json.loads(a))==set(json.loads(b)) for a,b in zip(q.observed_selected_features_left,q.observed_selected_features_right)])
                pairrecords.append(dict(beta=beta,left=left,right=right,target_agreement=q.target_feature_left.eq(q.target_feature_right).mean(),selected_set_agreement=same_sets.mean()))
    a=pd.DataFrame(pairrecords);a.to_csv(OUT/'selector_agreement.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    for ax,beta in zip(axes.flat,[0,*BETAS]):
        mat=a[a.beta.eq(beta)].pivot(index='left',columns='right',values='selected_set_agreement').reindex(index=METHODS,columns=METHODS).to_numpy();heat(ax,mat,list(METHODS.values()),list(METHODS.values()),'Null' if beta==0 else f'beta={beta:g}');ax.tick_params(axis='x',rotation=20,labelsize=8)
    axes.flat[-1].remove();fig.suptitle('Agreement on the exact selected set')
    save(fig,'11_selector_agreement','Each cell is the fraction of 250 shared inputs with exactly the same selected pair. Target agreement is separately recorded in selector_agreement.csv. Shared randomization does not force agreement when selected sets differ.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for method,label in METHODS.items():
        g=t[t.method.eq(method)].sort_values('beta');axes[0].plot(g.beta,g.rate,'o-',color=COLORS[method],label=label);axes[0].fill_between(g.beta,g.lower,g.upper,color=COLORS[method],alpha=.1)
        power=f[f.method.eq(method)&f.beta.gt(0)];axes[1].scatter(power.radial_t_obs,power.p_value,s=10,alpha=.2,color=COLORS[method],label=label)
    axes[0].axhline(.5,color='gray',ls='--');axes[0].set(title='Targeting frequency and sampling uncertainty',xlabel='Signal strength (beta)',ylabel='Fraction targeting feature 0',xticks=BETAS,ylim=(0,.7));percent(axes[0]);axes[0].legend(fontsize=8)
    axes[1].axhline(.05,color='gray',ls='--');axes[1].set(title='Power trials: observed radius versus selective rank',xlabel='Observed spline radius',ylabel='MCMC rank p-value',yscale='log');fig.suptitle('Target selection and observed test statistics')
    save(fig,'12_targeting_and_statistics','Targeting intervals are pointwise 95% exact binomial intervals. The 50% population ceiling follows from uniform choice of one of two selected features; finite samples can exceed it. Right panel pools all alternative targets, including features that are not automatically projection-null.')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=f[f.method.eq(method)];ax.scatter(g.p_value,g.mc_se_conditional_hub,s=10,alpha=.2,color=COLORS[method]);ax.set(title=label,xlabel='Rank p-value',ylabel='Conditional-hub binomial SE',ylim=(0,.025));ax.grid(alpha=.15)
    fig.suptitle('Conditional-hub Monte Carlo variability')
    save(fig,'13_conditional_hub_variability','The SE describes the spoke tail fraction conditional on the realized hub. It excludes variability from the hub and does not measure error relative to the exact conditional-tail probability. It is not a confidence interval for an individual rank p-value.')
    fig,axes=plt.subplots(2,2,figsize=(12,10));ranks=np.arange(1,251);q=(ranks-.5)/250;lower=stats.beta.ppf(.025,ranks,251-ranks);upper=stats.beta.ppf(.975,ranks,251-ranks)
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        vals=np.sort(f[f.method.eq(method)&f.beta.eq(0)].p_value);ax.fill_between(q,lower,upper,color='#bcc8d1',alpha=.5,label='Uniform order-statistic reference');ax.plot(q,vals,color=COLORS[method]);ax.plot([0,1],[0,1],'--',color='gray');ax.set(title=label,xlabel='Uniform quantile',ylabel='Observed null p-value',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7);fig.suptitle('Null p-value quantile diagnostics')
    save(fig,'14_null_quantiles','Shading is the pointwise 95% order-statistic interval for 250 independent continuous uniform values. It is a descriptive uniform reference, not a simultaneous test band; discrete reversible-rank p-values may be conservative.')
    fig,axes=plt.subplots(2,2,figsize=(13,9));allout=[]
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=f[f.method.eq(method)&f.beta.gt(0)];signal=[];other=[];no=[]
        for beta in BETAS:
            q=g[g.beta.eq(beta)];s=int((q.rejected&q.target_is_signal).sum());o=int((q.rejected&~q.target_is_signal).sum());signal.append(s/250);other.append(o/250);no.append(1-(s+o)/250);allout.append(dict(method=method,beta=beta,signal_target_rejections=s,other_target_rejections=o,nonrejections=250-s-o,n=250))
        x=np.arange(4);ax.bar(x,signal,color='#258568',label='Signal target rejected');ax.bar(x,other,bottom=signal,color='#9256a0',label='Other target rejected');ax.bar(x,no,bottom=np.array(signal)+other,color='#bcc8d1',label='Not rejected');ax.set(title=label,xticks=x,xticklabels=BETAS,ylim=(0,1.02),xlabel='Signal strength (beta)');percent(ax)
    axes[0,0].legend(fontsize=7,loc='lower right');fig.suptitle('All power-trial outcomes, classified by selected target')
    save(fig,'15_all_target_outcomes','Denominator: all 250 trials per method/beta. Rejections of other targets are not labelled false positives: their mean response can have nonzero projection onto the target spline space in a finite design. These outcomes must not be substituted for signal-detection power.')
    pd.DataFrame(allout).to_csv(OUT/'all_target_outcomes.csv',index=False)
    return paired

def report(f,p,c,costs,paired):
    mean=f.groupby('method').elapsed_seconds.mean();null=c[c.alpha.eq(.05)].set_index('method');strong=p[p.beta.eq(1)].set_index('method');mid=p[p.beta.eq(.5)].set_index('method')
    findings=[
        'All 5,000 tests resolved; each used 511 spokes, 20 steps and 10,240 transitions. These are synthetic Gaussian simulations, not the eight real benchmark datasets.',
        'Null rejection at alpha=0.05: '+', '.join(f'{METHODS[m]} {int(null.loc[m,"rejections"])}/250 ({null.loc[m,"fraction"]:.1%})' for m in METHODS)+'. Pointwise intervals include 5%; this does not establish exact conditional calibration.',
        'At beta=0.50, conditional signal detection: '+', '.join(f'{METHODS[m]} {mid.loc[m,"operational_fraction"]:.1%}' for m in METHODS)+'. At beta=1.00: '+', '.join(f'{METHODS[m]} {strong.loc[m,"operational_fraction"]:.1%}' for m in METHODS)+'. Differences should be assessed with sampling uncertainty and selector-dependent target sets.',
        'Mean measured worker time per input: '+', '.join(f'{METHODS[m]} {mean[m]:.2f} seconds' for m in METHODS)+'. Spline screening and KDE-MI are substantially cheaper than the two random-forest selectors in this run, at an identical transition budget.',
        'All methods meet the reference 80% lower-confidence-bound criterion at beta=0.75 and 1.00; none does at beta=0.25 or 0.50. Conditional point estimates above 80% at beta=0.50 are not enough to meet that evidence rule.',
        'Spline screening is a promising default for this synthetic setting because of its computation cost and observed detection performance. KDE-MI is another inexpensive option. These data do not establish a universal selector ranking or benchmark-data superiority.'
    ]
    intro='Four presentation selectors with parallel reversible MCMC rank inference: 250 fresh null inputs and 250 fresh power inputs at beta=0.25, 0.50, 0.75 and 1.00. n=100, p=20, k=2; known sigma=1; exact-set conditioning; one uniformly randomized selected target. Inputs and auxiliary target draws are shared across methods; selected sets and actual targets may differ.'
    caveats='Power means rejection conditional on targeting the simulated signal feature, plus the corresponding marginal detection rate. All methods use the same spline projection statistic but condition on their own selection events. Intervals are pointwise and do not adjust for multiple comparisons. MCMC acceptance is not ESS or proof of mixing. Small null samples do not establish calibration. Summed worker times are not elapsed server runtime. No AIS ESS, chain traces or selection-region maps are available in this run.'
    md=['# Four-selector MCMC analysis',intro,caveats,'## Findings',*['- '+x for x in findings],'## PNG gallery'];sections=[]
    for v in FIGURES:
        title=v['name'][3:].replace('_',' ').title();md.append(f'### {title}\n\n{v["caption"]}\n\n![{title}]({v["name"]}.png)')
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(v["caption"])}</p><a href="{v["name"]}.png"><img loading="lazy" src="{v["name"]}.png" alt="{html.escape(title)}"></a></section>')
    md.append('## Reproduction\n\nRun .venv/Scripts/python.exe examples/build_presentation_four_mcmc20_analysis.py from the repository root. Source results are preserved. CSV tables and analysis_manifest.json record the derived data, source hashes and bootstrap seed. No inference was rerun; no PDFs were generated.')
    (OUT/'README.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Four-selector MCMC analysis</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>Four-selector MCMC analysis</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(caveats)+'</p><h2>Findings</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in findings)+'</ul>'+''.join(sections)+'</html>',encoding='utf-8')
    files=['unified_tests.csv','power_summary.csv','null_summary.csv','calibration_summary.csv','runtime_summary.csv','requested_manifest.json','source_hashes.json','verification.json','local_verification_report.json']
    manifest=dict(source=SOURCE.relative_to(ROOT).as_posix(),reference='outputs/selection_method_validation/exact_set/selected_six_with_linear_shapley_300_20260928',format='PNG only',verified_tests=len(f),source_sha256={x:hashlib.sha256((SOURCE/x).read_bytes()).hexdigest() for x in files},plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods=METHODS,figures=FIGURES,paired_bootstrap_seed=2026100901,paired_bootstrap_resamples=20000)
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (OUT/'COMPLETED').write_text(f'Created {len(FIGURES)} PNG figures from 5,000 verified tests. Source data preserved; no inference rerun.\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    f,p,c,t=load();costs=reference_plots(f,p,c,t);paired=additional_plots(f,p,t,costs);report(f,p,c,costs,paired)
    print(f'Created {len(FIGURES)} PNG figures, CSV summaries, README and gallery in {OUT}')
if __name__=='__main__':main()
