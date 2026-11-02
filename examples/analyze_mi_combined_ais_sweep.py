"""Audit the MI neighbor sweep and compare fixed-combined-AIS selectors."""
from pathlib import Path
import contextlib
import io
import json
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _read_raw
import analyze_matched_ais_methods as matched
from verify_mi_neighbors_sweep import verify_variant

BASE=matched.BASE
RUN=BASE/'mi_parameter_tuning/neighbors_12_16_20_25_30_200_trials_20260911'
OUT=RUN/'analysis'
KS=(12,16,20,25,30)
ORDER=[f'MI k={k}' for k in KS]+['SHAP','LIME 40','Spline screening']
COLORS=dict(zip(ORDER,['#b7a6cf','#967db5','#7c56a0','#613480','#3d1461','#2674a6','#d58122','#258568']))
FIGS=[]
RUN_BY_K={}

def run_for(k):
    return RUN_BY_K.get(k, RUN)


def save(fig,name,caption):
    fig.tight_layout(rect=(0,.18 if fig.legends and len(ORDER)>8 else .07,1,.94))
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);FIGS.append((name,caption))


def audit_settings():
    old=BASE/'ais_comparison/combined_shap_mi_spline/ais_combined'
    rows=[]
    for k in KS:
        with contextlib.redirect_stdout(io.StringIO()):verify_variant(run_for(k),k)
        for i in range(32):
            for folder in ['null_fresh',*[f'power_feature_0_beta_{str(b).replace(".","p")}' for b in [.3,.5,.75,1.]]]:
                suffix=Path(f'shard_{i}/mutual_information/baseline/{folder}/settings.json')
                a=json.loads((old/suffix).read_text());b=json.loads((run_for(k)/f'neighbors_{k}'/suffix).read_text())
                assert b['selector_settings']['n_neighbors']==k
                b['selector_settings']['n_neighbors']=12
                assert a==b,(k,i,folder)
                rows.append(dict(neighbors=k,shard=i,condition=folder,only_neighbor_count_changed=True))
    pd.DataFrame(rows).to_csv(OUT/'configuration_audit.csv',index=False)


def load():
    previous=pd.read_csv(BASE/'ais_comparison/analysis/combined_ais_only/combined_ais_tests.csv')
    benchmark=previous[previous.selection_method.ne('mutual_information')].copy()
    benchmark['option']=benchmark.selection_method.map({'shap':'SHAP','lime':'LIME 40','spline_screening':'Spline screening'})
    benchmark['mi_neighbors']=np.nan
    frames=[benchmark];manifest={}
    for k in KS:
        root=run_for(k)/f'neighbors_{k}';s=json.loads((root/'pooled_summary/validation_settings.json').read_text())
        manifest[str(root.relative_to(BASE))]=s
        n,t,f=_read_raw(root,'ais',s,32)
        keys=['selection_method','validation_shard','iteration','selection_event','signal_strength']
        t=t.merge(f[keys+['feature','t_obs','proposals','denominator_ess','tail_ess']],on=keys,validate='one_to_one')
        for frame,scope in ((n,'null_calibration'),(t,'power')):
            frame['scope']=scope;frame['option']=f'MI k={k}';frame['mi_neighbors']=k
            frame['source_run']=str(root.relative_to(BASE)).replace('\\','/')
            frame['ais_setting']='combined';frame['max_final_samples']=51200
            frame['pilot_iters']=10;frame['pilot_samples']=320;frame['total_proposals']=frame.proposals+3200
            frames.append(frame)
        for row in pd.read_csv(root/'pooled_summary/calibration_decisions.csv').itertuples():
            assert len(n)==row.n_iterations and n.failed.sum()==row.n_failed
            assert n.p_value.lt(row.alpha).sum()==row.n_rejected
        for row in pd.read_csv(root/'pooled_summary/power_decisions.csv').itertuples():
            g=t[t.signal_strength.eq(row.signal_strength)]
            assert len(g)==row.n_iterations==200
            assert g.target_is_signal.sum()==row.n_signal_targets
            assert np.isclose(g.successful_detection.mean(),row.marginal_detection_power)
        print('Loaded MI',k,flush=True)
    data=pd.concat(frames,ignore_index=True)
    # Same simulation/target RNG draws across options; targets may change with MI k.
    audits=[]
    for scope in ['null_calibration','power']:
        keys=['validation_shard','iteration','selection_event']+(['signal_strength'] if scope=='power' else [])
        ref=data[data.option.eq('MI k=12')&data.scope.eq(scope)]
        for option in ORDER:
            other=data[data.option.eq(option)&data.scope.eq(scope)]
            pair=ref.merge(other,on=keys,suffixes=('_ref','_other'),validate='one_to_one')
            assert len(pair)==len(ref)
            for key in ['target_seed','auxiliary_u']:np.testing.assert_array_equal(pair[key+'_ref'],pair[key+'_other'])
            audits.append(dict(option=option,scope=scope,trials=len(pair),changed_targets=int(pair.target_feature_ref.ne(pair.target_feature_other).sum())))
        control=previous[previous.selection_method.eq('mutual_information')&previous.scope.eq(scope)]
        pair=ref.merge(control,on=keys,suffixes=('_new','_old'),validate='one_to_one')
        assert len(pair)==len(ref)
        for key in ['target_feature','target_seed','auxiliary_u','t_obs','p_value','failed','proposals']:
            if key in {'auxiliary_u','t_obs','p_value'}:
                # The prior combined export has undergone additional CSV round trips.
                np.testing.assert_allclose(pair[key+'_new'],pair[key+'_old'],rtol=1e-12,atol=1e-15,equal_nan=True)
            else:
                np.testing.assert_array_equal(pair[key+'_new'],pair[key+'_old'])
        np.testing.assert_array_equal(pair.p_value_new.lt(.05),pair.p_value_old.lt(.05))
    pd.DataFrame(audits).to_csv(OUT/'matched_trial_audit.csv',index=False)
    assert len(data)==1000*len(ORDER) and not data.duplicated(['option','scope','validation_shard','iteration','selection_event','signal_strength']).any()
    data.to_csv(OUT/'unified_tests.csv',index=False)
    original=json.loads((BASE/'ais_comparison/analysis/combined_ais_only/source_manifest.json').read_text())
    manifest.update({k:v for k,v in original.items() if k in benchmark.source_run.unique()})
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'duplicate_control_check.json').write_text(json.dumps({'mi_neighbors':12,'identical_null_trials':200,'identical_power_trials':800,'old_control_included_again':False},indent=2))
    return data


def plots(p,c):
    for name,metrics in [('01_power_and_targeting',[('conditional_power','Conditional power (failures = misses)'),('marginal_power','Marginal detection power')]),
                         ('02_reliability_and_cost',[('signal_failure_rate','Signal-target failure rate'),('mean_total_proposals','Mean proposals including pilot')])]:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for option in ORDER:
            g=p[p.method.eq(option)].sort_values('beta')
            for ax,(metric,title) in zip(axes,metrics):
                ax.plot(g.beta,g[metric],marker='o',color=COLORS[option],ls='-' if option.startswith('MI') else '--',label=option)
                ax.set(title=title,xlabel='Signal strength (beta)',xticks=[.3,.5,.75,1.]);ax.grid(alpha=.2)
        if name.startswith('01'):axes[0].axhline(.8,color='gray',ls=':');axes[0].set_ylim(0,1.03)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
        fig.suptitle('MI neighborhoods and three benchmarks; identical combined AIS')
        save(fig,name,'200 matched trials per condition. Neighbor count changes the selector and therefore the population of signal targets. Proposal counts are not wall-clock timings.')
    fig,axes=plt.subplots(1,2,figsize=(12,max(6,len(ORDER)*.7)))
    for ax,metric,title in zip(axes,['conditional_power','ci_lower'],['Observed conditional power','Pointwise 95% lower bound']):
        mat=p.pivot(index='method',columns='beta',values=metric).reindex(ORDER)
        ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for i in range(len(ORDER)):
            for j in range(4):
                v=mat.iloc[i,j];ax.text(j,i,f'{v:.1%}',ha='center',va='center',color='white' if v>.65 else 'black')
        ax.set(title=title,yticks=range(len(ORDER)),yticklabels=ORDER,xticks=range(4),xticklabels=[.3,.5,.75,1.],xlabel='Signal strength (beta)')
    fig.suptitle('Power across all settings');save(fig,'03_power_heatmap','Confidence bounds are pointwise and do not adjust for repeated tuning and choosing the best setting.')
    fig,axes=plt.subplots(1,2,figsize=(12,max(6,len(ORDER)*.7)))
    for i,option in enumerate(ORDER):
        for ax,frame,metric in [(axes[0],p[p.beta.eq(.5)],'conditional_power'),(axes[1],c[c.alpha.eq(.05)],'fpr')]:
            row=frame[frame.method.eq(option)].iloc[0];v=row[metric]
            ax.errorbar(v,i,xerr=[[max(0,v-row.ci_lower)],[max(0,row.ci_upper-v)]],fmt='o',color=COLORS[option],capsize=3)
            ax.annotate(f'{v:.1%}',(v,i),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    for ax in axes:ax.set(yticks=range(len(ORDER)),yticklabels=ORDER);ax.set_ylim(len(ORDER)-.5,-.6);ax.grid(axis='x',alpha=.2)
    axes[0].axvline(.8,color='gray',ls='--');axes[0].set(title='Conditional power at beta=0.50',xlim=(.45,1))
    axes[1].axvline(.05,color='gray',ls='--');axes[1].set(title='Null false-positive rate at nominal 5%',xlim=(0,.12))
    fig.suptitle('Target-strength power and null calibration (95% Wilson intervals)');save(fig,'04_beta05_and_calibration','All null tests succeeded. Power retains failed signal tests as misses; there is no change to nominal alpha.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for i,option in enumerate(ORDER):
        g=p[p.method.eq(option)&p.beta.eq(.5)].iloc[0]
        axes[0].bar(i,g.detections/g.signal_targets,color=COLORS[option])
        axes[0].bar(i,(g.signal_targets-g.detections-g.signal_failures)/g.signal_targets,bottom=g.detections/g.signal_targets,color='#ccd4db')
        axes[0].bar(i,g.signal_failures/g.signal_targets,bottom=1-g.signal_failures/g.signal_targets,color='#b43131')
        axes[0].text(i,1.02,f'{int(g.detections)}/{int(g.signal_targets)}',ha='center',fontsize=8)
        axes[1].bar(i,g.signal_targets/200,color=COLORS[option])
    for ax in axes:ax.set(xticks=range(len(ORDER)),xticklabels=ORDER,ylim=(0,1.12));ax.tick_params(axis='x',rotation=45)
    axes[0].axhline(.8,color='gray',ls='--');axes[0].set(title='Detected / signal targets; gray = non-detection',ylabel='Fraction of signal targets')
    axes[1].set(title='How often the selected target is the signal',ylabel='Signal targets / 200 trials')
    fig.suptitle('Why conditional power alone is insufficient: beta=0.50');save(fig,'05_beta05_outcomes','Red segments indicate failed inference. MI 25 and MI 30 can have equal detection counts but different conditional power because the denominators differ.')


def main():
    OUT.mkdir(exist_ok=True)
    audit_settings();print('Verified 800 condition manifests; only MI neighbor count differs.',flush=True)
    data=load()
    summarized=data.copy();summarized['selection_method']=summarized.option
    previous=matched.OUT;matched.OUT=OUT
    p,c=matched.summarize(summarized);matched.OUT=previous
    plots(p,c)
    q=p[p.beta.eq(.5)].copy();q['point_estimate_at_least_80pct']=q.conditional_power.ge(.8);q['lower_bound_at_least_80pct']=q.ci_lower.ge(.8)
    q.to_csv(OUT/'beta05_target_check.csv',index=False)
    intro='Verified 800 low-level MI condition manifests against the previous combined-AIS MI run: only n_neighbors differs. Seeds, AIS controls, data configuration and software metadata match. The new MI-12 control exactly reproduces all 1,000 previous MI-12 trial outcomes and proposal counts; it is included once. The unified dataset contains 1,600 null and 6,400 power records across eight options, sharing simulation seeds. SHAP, LIME 40 and spline screening use combined AIS, not exact-spline inference.'
    findings='MI 30 is the strongest tested MI candidate for the beta=0.50 conditional-power objective, at 67/87=77.0%, versus MI 12 at 54/82=65.9%. None of the five MI settings meets the 80% point-estimate target. MI 25 also detects 67 signals but among 90 signal targets (74.4%); both therefore have the same marginal power, 33.5%. SHAP reaches 89.7%, LIME 40 and spline screening 85.6% at this strength. MI 30 has zero failures at beta=0.50, so the remaining gap is not unresolved AIS tests. MI 30 reaches 99.0% at beta=1.00, but its beta=0.30 power is 34.8%, below MI 12 at 40.0%. Increasing neighborhoods is not uniformly beneficial.'
    caveat='All five MI variants have zero null failures; observed null FPR at nominal 5% is 4–5%. This does not establish exact calibration. Confidence intervals are pointwise and do not account for selecting a winner on repeatedly reused seeds. Failed signal tests count as misses. Cost includes 3,200 pilot proposals and is not measured runtime. Comparisons across neighbors have matched simulation seeds, but selected features and conditional target populations can differ.'
    report=['# MI neighborhood sweep under combined AIS',intro,'## Findings',findings,caveat,'## Beta=0.50 results',matched.table(q[['method','signal_targets','detections','conditional_power','ci_lower','ci_upper','marginal_power','signal_failures','mean_total_proposals']]),'## All power results',matched.table(p[['method','beta','conditional_power','marginal_power','signal_failures']]),'## Plots']
    sections=[]
    for name,caption in FIGS:
        report.append(f'### {name[3:].replace("_"," ")}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)')
        sections.append(f'<section><h2>{html.escape(name[3:].replace("_"," "))}</h2><p>{html.escape(caption)}</p><img src="{name}.png"><p><a href="{name}.pdf">PDF</a></p></section>')
    (OUT/'analysis_report.md').write_text('\n\n'.join(report),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>MI sweep: combined AIS</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#203040}img{width:100%}section{margin:40px 0}</style><h1>MI sweep under combined AIS</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(findings)+'</p><p>'+html.escape(caveat)+'</p>'+''.join(sections)+'</html>',encoding='utf-8')
    print(q[['method','signal_targets','detections','conditional_power','ci_lower','ci_upper','mean_total_proposals']].to_string(index=False))
    print('Saved',OUT)


if __name__=='__main__':main()
