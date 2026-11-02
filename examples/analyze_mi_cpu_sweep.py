"""Replay observed MI rankings and summarize the full CPU sweep after inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from examples.run_mi_cpu_sweep import read_manifest
from si_shap.power import _generate_power_dataset
from si_shap.selection import MutualInformationSelector, target_from_selected_set


def analyze_variant(m,c):
    root=ROOT/m['output_root']/c['name']
    rows=[];nulls=[]
    selector=MutualInformationSelector(n_neighbors=c['neighbors'],estimator=c['estimator'],bandwidth=c['bandwidth'])
    for i in range(32):
        folder=root/f'shard_{i}'/'mutual_information/baseline'
        null=pd.read_csv(folder/'null_fresh/p_value_results.csv')
        null['shard']=i; nulls.append(null)
        for beta in [.3,.5,.75,1.]:
            run=folder/('power_feature_0_beta_'+str(beta).replace('.','p'))
            s=json.loads((run/'settings.json').read_text())
            targets=pd.read_csv(run/'target_results.csv')
            features=pd.read_csv(run/'feature_results.csv').set_index('iteration')
            seeds=np.random.SeedSequence(s['seed']).spawn(s['n_iters'])
            for t in targets.itertuples():
                ds,ts,_=seeds[t.iteration-1].spawn(3)
                X,y=_generate_power_dataset(np.random.default_rng(ds),s['n_samples'],s['n_features'],s['signal_features'],beta,feature_correlation=s['feature_correlation'])
                u=float(np.random.default_rng(int(ts.generate_state(1,dtype=np.uint32)[0])).random())
                result=selector.select(X,y,2)
                rank=int(np.flatnonzero(result.ranking==0)[0]+1)
                assert target_from_selected_set(result.selected_features,u)==t.target_feature
                f=features.loc[t.iteration]
                assert np.isclose(result.scores[t.target_feature],f.importance_score,rtol=1e-10,atol=1e-12)
                assert np.isclose(u,t.auxiliary_u,rtol=0,atol=1e-15)
                scores=result.scores
                greater=int((scores>scores[0]).sum());equal=int((scores==scores[0]).sum())
                rows.append(dict(configuration=c['name'],shard=i,iteration=t.iteration,beta=beta,
                    new_iteration=t.iteration>(7 if i<8 else 6),
                    data_sha256=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),auxiliary_u=u,
                    selected_features=json.dumps(result.selected_features.tolist()),ranking=json.dumps(result.ranking.tolist()),scores=json.dumps(scores.tolist()),
                    signal_selected=rank<=2,signal_rank=rank,target_feature=t.target_feature,signal_targeted=t.target_feature==0,
                    zero_cutoff=bool(np.sort(scores)[-2]==0),zero_signal=bool(scores[0]==0),
                    tie_averaged_inclusion=float(np.clip((2-greater)/equal,0,1)),
                    p_value=t.p_value,failed=not np.isfinite(t.p_value),detection=t.target_feature==0 and t.p_value<.05,
                    denominator_ess=f.denominator_ess,tail_ess=f.tail_ess,proposals=f.proposals))
    d=pd.DataFrame(rows)
    assert len(d)==4000 and d.new_iteration.sum()==3200
    d.to_csv(root/'observed_diagnostics.csv',index=False)
    n=pd.concat(nulls,ignore_index=True)
    assert len(n)==1000
    n.to_csv(root/'null_trials.csv',index=False)
    return d,n


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant');args=p.parse_args()
    m=read_manifest();out=ROOT/m['output_root']
    frames=[];cal=[]
    for c in m['configurations']:
        if args.variant and c['name']!=args.variant:continue
        root=out/c['name']
        if (root/'observed_diagnostics.csv').exists() and (root/'null_trials.csv').exists():
            d=pd.read_csv(root/'observed_diagnostics.csv');n=pd.read_csv(root/'null_trials.csv')
        else:d,n=analyze_variant(m,c)
        assert len(d)==4000 and len(n)==1000
        frames.append(d)
        good=np.isfinite(n.p_value)
        for alpha in [.01,.05,.1]:
            cal.append(dict(configuration=c['name'],alpha=alpha,n=len(n),failures=int((~good).sum()),rejections=int((n.p_value<alpha).sum()),fpr_completed=float((n.loc[good,'p_value']<alpha).mean())))
    if args.variant:return
    d=pd.concat(frames,ignore_index=True)
    assert len(d)==40000 and not d.duplicated(['configuration','shard','iteration','beta']).any()
    assert d.groupby(['shard','iteration','beta']).data_sha256.nunique().eq(1).all()
    assert d.groupby(['shard','iteration','beta']).auxiliary_u.nunique().eq(1).all()
    summary=[]
    for subset,data in [('all_1000',d),('new_800',d[d.new_iteration])]:
        for (config,beta),g in data.groupby(['configuration','beta']):
            sig=g[g.signal_targeted]
            summary.append(dict(subset=subset,configuration=config,beta=beta,n=len(g),
                signal_inclusion=g.signal_selected.mean(),targeting=g.signal_targeted.mean(),mean_signal_rank=g.signal_rank.mean(),
                tie_averaged_inclusion=g.tie_averaged_inclusion.mean(),zero_cutoff_trials=int(g.zero_cutoff.sum()),
                signal_targets=len(sig),detections=int(g.detection.sum()),conditional_power=g.detection.sum()/len(sig) if len(sig) else np.nan,
                marginal_power=g.detection.mean(),failures=int(g.failed.sum()),signal_failures=int(sig.failed.sum()),
                median_signal_den_ess=sig.denominator_ess.median(),median_signal_tail_ess=sig.tail_ess.median(),mean_proposals=g.proposals.mean()))
    pd.DataFrame(summary).to_csv(out/'power_selection_summary.csv',index=False)
    pd.DataFrame(cal).to_csv(out/'calibration_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    s=pd.DataFrame(summary)
    for config,g in s[s.subset.eq('all_1000')].groupby('configuration'):
        for ax,col in zip(axes,['targeting','conditional_power','marginal_power']):
            ax.plot(g.beta,g[col],marker='o',label=config)
    for ax,title in zip(axes,['Probability of targeting signal','Conditional power (failures retained)','Marginal detection power']):
        ax.set(title=title,xlabel='beta',ylim=(0,1.03));ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(out/'power_curves.png',dpi=180);plt.close(fig)
    print(s.to_string(index=False))


if __name__=='__main__':main()
