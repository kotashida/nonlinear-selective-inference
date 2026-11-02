"""Inventory exact-set experiments and compare matched baseline/combined AIS."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _read_raw, _wilson_interval

BASE = Path(__file__).resolve().parents[1] / 'outputs/selection_method_validation/exact_set'
OUT = BASE / 'ais_comparison/analysis'
METHODS = {'shap': 'SHAP', 'lime': 'LIME (40 explanations)',
           'spline_screening': 'Spline screening', 'mutual_information': 'Mutual information'}
RUNS = {
    ('baseline', 'others'): 'ais_comparison/baseline_four_methods_lime10/ais_all_methods',
    ('combined', 'others'): 'ais_comparison/combined_shap_mi_spline/ais_combined',
    ('baseline', 'lime'): 'lime_ais_tuning/five_settings_200_trials/ais_baseline',
    ('combined', 'lime'): 'lime_ais_tuning/five_settings_200_trials/ais_combined',
}
AIS = {'baseline': (12800, 5, 80), 'combined': (51200, 10, 320)}


def read(p):
    return json.loads(p.read_text())


def inventory():
    rows, folders = [], []
    for top in sorted(BASE.iterdir()):
        if not top.is_dir() or top == OUT:
            continue
        files = list(top.rglob('*'))
        configs = [p for p in files if p.name == 'run_config.txt']
        pooled = [p for p in files if p.name == 'validation_settings.json' and p.parent.name == 'pooled_summary']
        folders.append(dict(folder=top.name, files=sum(p.is_file() for p in files),
                            recorded_runs=len(configs), pooled_runs=len(pooled),
                            classification='derived analysis; do not append as new trials' if top.name.startswith('unified_') else 'source experiments'))
        roots = sorted({p.parent for p in configs} | {p.parent.parent for p in pooled})
        for run in roots:
            path = run / 'pooled_summary/validation_settings.json'
            s = read(path) if path.exists() else {}
            rel = str(run.relative_to(BASE)).replace('\\', '/')
            if rel in RUNS.values():
                reason = 'included (SHAP/MI/spline only)' if rel == RUNS['baseline', 'others'] else 'included'
            elif 'explanations_40' == run.name and s.get('lime_params', {}).get('kernel_width') is None:
                reason = 'overlapping LIME baseline seeds; audit duplicates, do not append'
            else:
                reason = 'different selector/AIS/inference settings or incomplete; excluded'
            rows.append(dict(source_run=rel, methods=' '.join(s.get('methods', [])),
                             null_iterations=s.get('n_null_iters'), power_iterations=s.get('n_power_iters'),
                             inference=s.get('inference_method'), cap=s.get('max_final_samples'),
                             pilot_rounds=s.get('pilot_iters'), pilot_samples=s.get('pilot_samples'),
                             lime_parameters=json.dumps(s.get('lime_params')), completed_shards=len(list(run.glob('shard_*/validation_settings.json'))), decision=reason))
    pd.DataFrame(folders).to_csv(OUT/'folder_inventory.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT/'run_inventory.csv', index=False)
    return folders


def load():
    frames, manifest = [], {}
    for (ais, kind), rel in RUNS.items():
        run = BASE / rel
        s = read(run/'pooled_summary/validation_settings.json')
        assert s['n_null_iters'] == s['n_power_iters'] == 200
        assert s['shard_root_seeds'] == list(range(20260908, 20260940))
        assert tuple(s[k] for k in ('max_final_samples','pilot_iters','pilot_samples')) == AIS[ais]
        methods = ['lime'] if kind == 'lime' else ['shap','spline_screening','mutual_information']
        manifest[rel] = s
        n,t,f = _read_raw(run, 'ais', {**s, 'methods': methods}, 32)
        keys = ['selection_method','validation_shard','iteration','selection_event','signal_strength']
        t = t.merge(f[keys+['feature','t_obs','proposals','denominator_ess','tail_ess']], on=keys, validate='one_to_one')
        for frame,scope in ((n,'null_calibration'),(t,'power')):
            assert len(frame) == 200 * len(methods) * (4 if scope == 'power' else 1)
            frame['scope'],frame['ais_setting'],frame['source_run'] = scope,ais,rel
            frame['max_final_samples'],frame['pilot_iters'],frame['pilot_samples'] = AIS[ais]
            frame['total_proposals'] = frame.proposals + AIS[ais][1]*AIS[ais][2]
            frames.append(frame)
        # Reconcile source summaries with raw outcomes for selected methods.
        for row in pd.read_csv(run/'pooled_summary/calibration_decisions.csv').itertuples():
            if row.selection_method not in methods: continue
            g=n[n.selection_method.eq(row.selection_method)]
            assert len(g)==row.n_iterations and g.failed.sum()==row.n_failed
            assert g.p_value.lt(row.alpha).sum()==row.n_rejected
        for row in pd.read_csv(run/'pooled_summary/power_decisions.csv').itertuples():
            if row.selection_method not in methods: continue
            g=t[t.selection_method.eq(row.selection_method)&t.signal_strength.eq(row.signal_strength)]
            assert len(g)==row.n_iterations and g.target_is_signal.sum()==row.n_signal_targets
            assert np.isclose(g.successful_detection.mean(),row.marginal_detection_power)
    # Compare actual low-level settings, including selectors and package versions.
    audits=[]
    for method in METHODS:
        kind='lime' if method=='lime' else 'others'
        for i in range(32):
            for folder in ['null_fresh',*[f'power_feature_0_beta_{str(b).replace(".","p")}' for b in [.3,.5,.75,1.]]]:
                suffix=Path(f'shard_{i}/{method}/baseline/{folder}/settings.json')
                a,b=[read(BASE/RUNS[x,kind]/suffix) for x in ('baseline','combined')]
                for k in a.keys() | b.keys():
                    if k not in {'max_final_samples','pilot_iters','pilot_samples'}:
                        assert a.get(k)==b.get(k),(method,i,folder,k)
                audits.append(dict(method=method,shard=i,condition=folder,matched=True))
    pd.DataFrame(audits).to_csv(OUT/'configuration_audit.csv',index=False)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    data=pd.concat(frames,ignore_index=True)
    keys=['ais_setting','selection_method','scope','validation_shard','iteration','signal_strength','selection_event']
    assert not data.duplicated(keys).any() and len(data)==8000
    data.to_csv(OUT/'unified_tests.csv',index=False)
    return data


def paired(data):
    rows=[]
    for (method,scope,beta),g in data.groupby(['selection_method','scope','signal_strength'],dropna=False):
        a=g[g.ais_setting.eq('baseline')]; b=g[g.ais_setting.eq('combined')]
        p=a.merge(b,on=['validation_shard','iteration','selection_event'],suffixes=('_old','_new'),validate='one_to_one')
        assert len(p)==200
        for key in ['target_feature','target_seed','auxiliary_u','t_obs']:
            np.testing.assert_array_equal(p[key+'_old'],p[key+'_new'],err_msg=f'{method}/{scope}/{beta}/{key}')
        if scope=='null_calibration':
            np.testing.assert_array_equal(p.observed_selected_features_old,p.observed_selected_features_new)
        old=p.p_value_old.lt(.05)&~p.failed_old
        new=p.p_value_new.lt(.05)&~p.failed_new
        if scope=='power':
            np.testing.assert_array_equal(p.target_is_signal_old,p.target_is_signal_new)
            old &= p.target_is_signal_old.astype(bool);new &= p.target_is_signal_new.astype(bool)
        rows.append(dict(method=method,scope=scope,beta=beta,paired_trials=len(p),
                         gained_rejections=int((~old&new).sum()),lost_rejections=int((old&~new).sum()),
                         recovered_failures=int((p.failed_old&~p.failed_new).sum()),
                         new_failures=int((~p.failed_old&p.failed_new).sum())))
    result=pd.DataFrame(rows);result.to_csv(OUT/'paired_changes.csv',index=False)
    return result


def duplicate_audit(data):
    rows=[]
    for name in ['lime_parameter_tuning/explanations_200_trials','lime_parameter_tuning/initial_400_trials']:
        run=BASE/name/'explanations_40';s=read(run/'pooled_summary/validation_settings.json')
        n,t,_=_read_raw(run,'ais',s,32)
        for scope,frame in [('null_calibration',n),('power',t)]:
            ref=data[data.selection_method.eq('lime')&data.ais_setting.eq('baseline')&data.scope.eq(scope)]
            keys=['validation_shard','iteration','selection_event']+(['signal_strength'] if scope=='power' else [])
            p=ref.merge(frame,on=keys,suffixes=('_ref','_copy'),validate='one_to_one')
            assert len(p)==len(ref)
            for k in ['target_feature','target_seed','auxiliary_u','p_value','failed']:
                np.testing.assert_array_equal(p[k+'_ref'],p[k+'_copy'])
            rows.append(dict(source=str(run.relative_to(BASE)),scope=scope,duplicate_trials=len(p),
                             extra_unmatched_trials=len(frame)-len(p),decision='exclude duplicate trials; extra trials lack matched combined AIS'))
    pd.DataFrame(rows).to_csv(OUT/'duplicate_audit.csv',index=False)


def summarize(data):
    power=[];cal=[]
    for (method,ais,scope,beta),g in data.groupby(['selection_method','ais_setting','scope','signal_strength'],dropna=False):
        valid=~g.failed & g.p_value.notna()
        if scope=='power':
            sig=g[g.target_is_signal.astype(bool)]
            k=int((sig.p_value.lt(.05)&~sig.failed).sum());lo,hi=_wilson_interval(k,len(sig))
            power.append(dict(method=method,ais=ais,beta=beta,n=len(g),signal_targets=len(sig),detections=k,
                              conditional_power=k/len(sig),ci_lower=float(lo),ci_upper=float(hi),marginal_power=k/len(g),
                              failures=int(g.failed.sum()),signal_failures=int(sig.failed.sum()),signal_failure_rate=sig.failed.mean(),
                              mean_total_proposals=g.total_proposals.mean(),mean_final_proposals=g.proposals.mean()))
        else:
            for alpha in [.01,.05,.1]:
                k=int((valid&g.p_value.lt(alpha)).sum());lo,hi=_wilson_interval(k,valid.sum())
                cal.append(dict(method=method,ais=ais,alpha=alpha,n=len(g),failures=int((~valid).sum()),rejections=k,
                                fpr=k/valid.sum(),ci_lower=float(lo),ci_upper=float(hi),mean_total_proposals=g.total_proposals.mean()))
    p,c=pd.DataFrame(power),pd.DataFrame(cal)
    p.to_csv(OUT/'power_summary.csv',index=False);c.to_csv(OUT/'calibration_summary.csv',index=False)
    return p,c


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plots(p,c):
    for metric,title in [('conditional_power','Conditional power (failed signal tests count as misses)'),
                         ('mean_total_proposals','Mean proposals per power trial, including pilot'),
                         ('signal_failure_rate','Signal-target inference failure rate'),('marginal_power','Marginal signal detection power')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,(method,label) in zip(axes.flat,METHODS.items()):
            for ais,color in [('baseline','#4878a8'),('combined','#df7d35')]:
                g=p[p.method.eq(method)&p.ais.eq(ais)].sort_values('beta')
                ax.plot(g.beta,g[metric],'-o',color=color,label=ais.title())
                if metric=='conditional_power':ax.fill_between(g.beta,g.ci_lower,g.ci_upper,color=color,alpha=.12)
            ax.set(title=label,xlabel='Signal strength (beta)',xticks=[.3,.5,.75,1]);ax.grid(alpha=.2)
            if metric=='conditional_power':ax.axhline(.8,color='gray',ls='--');ax.set_ylim(0,1.03)
        axes[0,0].legend();fig.suptitle(title+ ('\nShading: pointwise 95% Wilson intervals' if metric=='conditional_power' else ''))
        fig.tight_layout(rect=(0,0,1,.94));save(fig,metric)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for j,method in enumerate(METHODS):
        for ais,color,offset in [('baseline','#4878a8',-.12),('combined','#df7d35',.12)]:
            for ax,frame,value in [(axes[0],p[p.beta.eq(1)],'conditional_power'),(axes[1],c[c.alpha.eq(.05)],'fpr')]:
                g=frame[frame.method.eq(method)&frame.ais.eq(ais)].iloc[0];v=g[value]
                ax.errorbar(v,j+offset,xerr=[[max(0,v-g.ci_lower)],[max(0,g.ci_upper-v)]],fmt='o',capsize=3,color=color,label=ais.title() if j==0 else None)
    for ax in axes:ax.set(yticks=range(4),yticklabels=list(METHODS.values()));ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].axvline(.8,color='gray',ls='--');axes[0].set(title='Conditional power at beta = 1',xlim=(.65,1.02));axes[0].legend()
    axes[1].axvline(.05,color='gray',ls='--');axes[1].set(title='Null FPR at nominal 5%',xlim=(0,.13))
    fig.suptitle('Matched 200 trials per condition; pointwise 95% intervals');fig.tight_layout();save(fig,'power_and_calibration')


def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def write_report(p,c,changes):
    report=['# Matched four-method AIS comparison','',
            'Inventoried every existing top-level folder (see folder_inventory.csv and run_inventory.csv). Combined 1,600 null and 6,400 power records: four methods, two AIS settings, 200 trials per condition. These share simulation seeds and are not 8,000 independent datasets.',
            'LIME uses 40 explanations in both arms. The original four-method pilot used 10 explanations, so its LIME records are excluded. Earlier identical LIME runs overlap the same seeds; duplicate_audit.csv verifies those overlaps and they are not counted again. Extra 400-run trials have no matching combined-AIS counterpart. Other selector sweeps, the original smaller-budget AIS pilot, exact-spline references, MCMC runs and derived analysis exports are excluded from this controlled comparison.',
            'Compared all 640 pairs of low-level condition manifests. Only max_final_samples, pilot_iters and pilot_samples differ within each method. Selector parameters, seeds, iteration counts, runtime metadata and remaining simulation/inference settings match. All 4,000 trial pairs match inference targets, auxiliary draws, target seeds and observed statistics; null selected sets also match.',
            'Baseline AIS: cap 12,800; pilot 5 x 80. Combined AIS: cap 51,200; pilot 10 x 320. Both use final batches of 800, denominator ESS 80, tail ESS 15 and early stopping. Method selectors differ by design; parameters within each selector are fixed between AIS arms.',
            'Power counts failed signal tests as misses. Marginal power divides signal detections by all trials. Calibration rates use successful null tests, with failures separately reported. Proposal costs include pilot draws but are not wall-clock times. Wilson intervals are pointwise, unadjusted for previous tuning on these seeds. Fresh-seed validation remains necessary for independent confirmation.',
            '## Findings',
            'At beta=1, combined AIS raises SHAP from 85.6% to 97.1%, LIME from 79.8% to 92.3%, and MI from 85.6% to 93.3%. Spline screening stays at 96.2%. All four combined-AIS pointwise 95% lower bounds exceed 80% at beta=1. At beta=0.75 all four lower bounds also exceed 80%.',
            'The change is not uniformly beneficial. MI at beta=0.5 loses two detections (56/82 to 54/82; 68.3% to 65.9%), with no failures in either arm. LIME at beta=0.3 loses one net detection (40.3% to 38.8%). These small paired differences do not establish a systematic degradation, but a strict no-regression requirement is not met by the observed outcomes. Spline screening has identical signal-detection decisions at every strength, although individual p-values can change.',
            'Across 800 power tests per method, SHAP failures fall from 20 to 4 with no newly failed tests. MI falls from 11 to 1 (11 recovered, one newly failed). LIME falls from 35 to 14 (24 recovered, three newly failed). Spline screening has zero failures under both settings. All null tests succeed; combined-AIS FPR at nominal 5% is 3.5% for SHAP, 4.5% for LIME and MI, and 3% for spline screening. These finite-sample checks do not prove calibration.',
            'At beta=1, combined/baseline mean proposal-count ratios are 1.91 for SHAP, 2.24 for LIME, 2.01 for MI and 2.51 for spline screening. The additional pilot work dominates the spline overhead without an observed power gain. Counts are not comparable wall-clock costs across selectors. Combined AIS is the strongest tested shared candidate for the beta=1 power goal, but it does not guarantee no degradation at every strength. The repeated use of these seeds calls for fresh-seed confirmation before treating this as a final performance guarantee.',
            '## Power at beta=1',table(p[p.beta.eq(1)][['method','ais','signal_targets','detections','conditional_power','ci_lower','ci_upper','signal_failures','mean_total_proposals']]),
            '## All signal strengths',table(p[['method','ais','beta','conditional_power','marginal_power','signal_failures']]),
            '## Calibration at alpha=0.05',table(c[c.alpha.eq(.05)]),
            '## Paired reliability',table(changes.groupby(['method','scope'])[['recovered_failures','new_failures']].sum().reset_index()),
            '## Files','unified_tests.csv is the canonical test-level dataset with source_run provenance. Do not concatenate it with previous combined exports. power_summary.csv and calibration_summary.csv provide rates and intervals. paired_changes.csv provides gains/losses and new/recovered failures. Four curve plots and power_and_calibration have PNG/PDF versions. Original experiment folders are preserved.']
    (OUT/'analysis_report.md').write_text('\n\n'.join(report),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    folders=inventory();print('Inventoried',len(folders),'top-level folders',flush=True)
    data=load();print('Verified 8000 records and 640 paired configurations',flush=True)
    changes=paired(data);duplicate_audit(data);p,c=summarize(data);plots(p,c)
    write_report(p,c,changes)
    print(p[p.beta.eq(1)].to_string(index=False));print(c[c.alpha.eq(.05)].to_string(index=False))
    print(changes.groupby(['method','scope'])[['recovered_failures','new_failures']].sum().to_string())
    print('Saved',OUT)


if __name__=='__main__':main()
