"""Combine both MI sweeps, retaining each configuration and control once."""
import json
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analyze_mi_combined_ais_sweep as sweep


def main():
    sweep.KS=(12,16,20,25,30,35,40,50)
    sweep.RUN_BY_K={k:sweep.BASE/'mi_parameter_tuning/neighbors_35_40_50_200_trials_20260912' for k in (35,40,50)}
    sweep.ORDER=[f'MI k={k}' for k in sweep.KS]+['SHAP','LIME 40','Spline screening']
    sweep.COLORS={f'MI k={k}':plt.get_cmap('viridis')(i/8) for i,k in enumerate(sweep.KS)}
    sweep.COLORS.update({'SHAP':'#2674a6','LIME 40':'#d58122','Spline screening':'#258568'})
    sweep.OUT=sweep.BASE/'mi_parameter_tuning/analysis'
    out=sweep.OUT;out.mkdir(exist_ok=True)
    sweep.audit_settings();print('Verified all 1,280 MI condition manifests.',flush=True)
    data=sweep.load()
    temp=data.copy();temp['selection_method']=temp.option
    previous=sweep.matched.OUT;sweep.matched.OUT=out
    p,c=sweep.matched.summarize(temp);sweep.matched.OUT=previous
    sweep.plots(p,c)
    p['neighbors']=p.method.str.extract(r'MI k=(\d+)').astype(float)
    c['neighbors']=c.method.str.extract(r'MI k=(\d+)').astype(float)
    q=p[p.beta.eq(.5)].copy()
    q['point_estimate_at_least_80pct']=q.conditional_power.ge(.8)
    q['lower_bound_at_least_80pct']=q.ci_lower.ge(.8)
    q.to_csv(out/'beta05_target_check.csv',index=False)
    # Parameter-axis plots reveal non-monotonicity without eight overlaid curves.
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for beta,color in zip([.3,.5,.75,1.],['#2674a6','#bf5730','#258568','#9256a0']):
        g=p[p.neighbors.notna()&p.beta.eq(beta)].sort_values('neighbors')
        for ax,metric in [(axes[0,0],'conditional_power'),(axes[0,1],'marginal_power'),(axes[1,0],'mean_total_proposals')]:
            ax.plot(g.neighbors,g[metric],'-o',color=color,label=f'beta={beta:.2f}')
    g=c[c.neighbors.notna()&c.alpha.eq(.05)].sort_values('neighbors')
    axes[1,1].errorbar(g.neighbors,g.fpr,yerr=np.maximum(0,np.vstack([g.fpr-g.ci_lower,g.ci_upper-g.fpr])),fmt='o-',capsize=3,color='#333333')
    axes[1,1].axhline(.05,color='gray',ls='--');axes[0,0].axhline(.8,color='gray',ls='--')
    for ax,title in zip(axes.flat,['Conditional power (failures = misses)','Marginal signal detection power','Mean proposals including pilot','Null FPR at alpha=0.05; pointwise 95% intervals']):
        ax.set(title=title,xlabel='MI n_neighbors',xticks=sweep.KS);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Does increasing the MI neighborhood keep helping?')
    sweep.save(fig,'06_neighbor_count_effect','Both sweeps use identical combined AIS. All FPR intervals include 5%; that is not proof of calibration. Sampling counts are not wall-clock times.')
    # Compare each option with MI-30 using trial identities, allowing selected targets to differ.
    paired=[]
    for beta in [.3,.5,.75,1.]:
        ref=data[data.option.eq('MI k=30')&data.signal_strength.eq(beta)&data.scope.eq('power')]
        for k in sweep.KS:
            g=data[data.option.eq(f'MI k={k}')&data.signal_strength.eq(beta)&data.scope.eq('power')]
            pair=ref.merge(g,on=['validation_shard','iteration','selection_event'],suffixes=('_ref','_new'),validate='one_to_one')
            assert len(pair)==200
            old=pair.successful_detection_ref.astype(bool);new=pair.successful_detection_new.astype(bool)
            paired.append(dict(neighbors=k,beta=beta,gained_signal_detections=int((~old&new).sum()),lost_signal_detections=int((old&~new).sum()),changed_targets=int(pair.target_feature_ref.ne(pair.target_feature_new).sum())))
    pd.DataFrame(paired).to_csv(out/'paired_detection_changes_vs_30.csv',index=False)
    introduction='Combined both MI sweeps into one dataset: neighbors 12, 16, 20, 25, 30, 35, 40 and 50, with existing SHAP, LIME 40 and spline-screening benchmarks under the same combined AIS. There are 11,000 configuration-specific records (2,200 null and 8,800 power); these share simulation seeds and are not independent datasets. Each condition uses 200 trials. The MI-30 control appears once. The repeated MI-12 run matches earlier control outcomes and is also counted once.'
    findings='None of the eight MI settings reaches 80% observed conditional power at beta=0.50. MI-50 is numerically highest at 64/82=78.05%, versus MI-30 at 67/87=77.01%. MI-50 detects fewer signals overall (32.0% marginal power versus 33.5% for MI-30); its slightly higher conditional power reflects the changed signal-target denominator. MI-35 and MI-40 reach 75.29% and 75.61%. SHAP remains at 89.69%, LIME 40 and spline screening at 85.57%.'
    tradeoff='Larger neighborhoods do not improve performance uniformly. At beta=0.30, MI-50 drops to 22.5% conditional and 4.5% marginal power, versus 34.8% and 8.0% for MI-30. MI-50 improves beta=0.75 to 98.0%, but MI-30 remains slightly higher at beta=1.00 (99.0% versus 98.1%). All new settings have zero inference failures at beta=0.50, so there is no unresolved-test deficit to recover at that strength.'
    calibration='Observed null FPR at nominal 5% rises to 7.0%, 6.0% and 5.5% for neighbors 35, 40 and 50, versus 5.0% for MI-30. All null tests succeed. Each pointwise 95% interval includes 5%, so these data do not establish inflation, but they also do not justify claiming that larger k leaves calibration unharmed.'
    recommendation='MI-30 remains the more balanced candidate for the beta=0.50 objective: it has more detections than any new setting, stronger weak-signal performance than MI-50, and no observed FPR increase relative to nominal 5%. MI-50 has the highest conditional point estimate at beta=0.50 but still misses 80%, with a confidence lower bound well below 80%. This sweep provides no clear basis for simply continuing to increase neighbors. Rankings are exploratory; repeated tuning on these seeds requires fresh-seed confirmation.'
    text=['# All MI neighborhoods under combined AIS',introduction,'## Main findings',findings,tradeoff,calibration,'## Interpretation',recommendation,'## Validation','Compared 1,280 low-level MI condition manifests with the earlier MI control, allowing only n_neighbors to differ. Seeds, runtime metadata and all other declared settings agree. Raw counts match pooled summaries. Trial keys are unique; target RNG draws match across options, while the selected target can change with neighbor count. Failed signal tests count as misses. Proposal costs include 3,200 pilot draws; no timing measurements are inferred.', '## Beta=0.50 results',sweep.matched.table(q[['method','signal_targets','detections','conditional_power','ci_lower','ci_upper','marginal_power','signal_failures','mean_total_proposals']]),'## Null calibration',sweep.matched.table(c[c.alpha.eq(.05)][['method','rejections','n','fpr','ci_lower','ci_upper','failures']]),'## Plots']
    sections=[]
    for name,caption in sweep.FIGS:
        text.append(f'### {name[3:].replace("_"," ")}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)')
        sections.append(f'<section><h2>{html.escape(name[3:].replace("_"," "))}</h2><p>{html.escape(caption)}</p><img src="{name}.png" alt="{html.escape(caption)}"><p><a href="{name}.pdf">PDF</a></p></section>')
    (out/'analysis_report.md').write_text('\n\n'.join(text),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All MI neighborhoods</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#203040}img{width:100%}section{margin:40px 0}</style><h1>All MI neighborhoods: combined AIS</h1>'+''.join('<p>'+html.escape(t)+'</p>' for t in [introduction,findings,tradeoff,calibration,recommendation])+''.join(sections)+'</html>',encoding='utf-8')
    print(q[['method','detections','signal_targets','conditional_power','ci_lower','marginal_power']].to_string(index=False))
    print('Saved',out)


if __name__=='__main__':main()
