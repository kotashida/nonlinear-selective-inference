"""Plot verified saved results only; preserve all frozen scientific sources."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'outputs/benchmarks/publication_five_mcmc20_focused_20261010'
METHODS = ['spline_exact', 'spline_mcmc', 'shap_mcmc', 'lime_mcmc', 'mutual_information_mcmc']
LABELS = dict(zip(METHODS, ['Spline (exact)', 'Spline (MCMC)', 'Tree SHAP (MCMC)', 'LIME (MCMC)', 'MI (MCMC)']))
COLORS = dict(zip(METHODS, ['#c44e52', '#258568', '#2674a6', '#d58122', '#9256a0']))
DATASETS = ['concrete', 'heating_load', 'cooling_load', 'gas_turbine', 'housing', 'red_wine', 'white_wine', 'abalone']
NAMES = {d: d.replace('_', ' ').capitalize() for d in DATASETS}
KEYS = ['study', 'dataset', 'condition', 'trial']
FIGURES = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    # pandas' default NA vocabulary includes "null", a genuine study label here.
    return pd.read_csv(path, keep_default_na=False, na_values=[''], low_memory=False)


def cp(k, n):
    return (0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def rates(frame, alpha=.05, column='p_value'):
    rows = []
    for method in METHODS:
        g = frame[frame.method.eq(method)]
        p = pd.to_numeric(g[column], errors='coerce')
        valid = np.isfinite(p) & p.between(0, 1)
        n, k, missing = len(g), int((valid & p.lt(alpha)).sum()), int((~valid).sum())
        assert n
        rows.append(dict(method=method, attempted=n, rejected=k, failures=missing,
                         alpha=alpha, kind=column, rate=k/n, lo=cp(k,n)[0], hi=cp(k+missing,n)[1]))
    return pd.DataFrame(rows)


def audit(source):
    verification = json.loads((source/'verification.json').read_text())
    assert json.loads((source/'COMPLETED').read_text()) == verification
    assert json.loads((source/'local_verification.json').read_text())['status'] == 'passed'
    for filename, checksum in verification['output_sha256'].items():
        assert sha(source/filename) == checksum, f'Changed verified data: {filename}'
    config = json.loads((source/'requested_manifest.json').read_text())
    f = read_csv(source/'results.csv')
    assert not f.duplicated(KEYS+['method']).any()
    assert set(f.method) == set(METHODS) and set(f.study) == {'null', 'real'}
    expected = {}
    for c in config['null_conditions']:
        expected[('null','gaussian_design',c['id'])] = c['repetitions']
    for name in config['real_datasets']:
        for c in config['real_conditions']:
            if name in c.get('datasets', config['real_datasets']):
                expected[('real',name,c['id'])] = c['repetitions']
    assert len(f) == sum(expected.values())*5 == verification['tests'] == 70000
    assert len(f[f.study.eq('null')]) == 30000
    for key, g in f.groupby(KEYS[:3]):
        assert key in expected and g.groupby('method').size().eq(expected[key]).all()
        assert set(g.trial) == set(range(expected[key]))
    assert len(f.groupby(KEYS[:3])) == len(expected)
    assert f.groupby(KEYS).input_sha256.nunique().eq(1).all()
    assert not f.inference_failed.any() and f.p_value.between(0,1).all()
    assert f.loc[f.method.ne('spline_exact'),'proposals'].eq(10240).all()
    assert f.loc[f.method.eq('spline_exact'),'proposals'].eq(0).all()
    stored = read_csv(source/'rates.csv').set_index(KEYS[:3]+['method','alpha'])
    for key, g in f.groupby(KEYS[:3]):
        for alpha in config['alpha_levels']:
            for r in rates(g, alpha).itertuples():
                v = stored.loc[(*key,r.method,alpha)]
                assert (r.attempted,r.rejected,r.failures) == (v.attempted,v.rejected,v.failures)
                np.testing.assert_allclose([r.rate,r.lo,r.hi], [v.all_attempt_lower,v.missing_ci95_lower,v.missing_ci95_upper],atol=1e-12)
    a = f[f.method.eq('spline_exact')].set_index(KEYS)
    b = f[f.method.eq('spline_mcmc')].set_index(KEYS).reindex(a.index)
    assert a.selected.eq(b.selected).all() and a.target.eq(b.target).all()
    np.testing.assert_allclose(a.radial_t_obs,b.radial_t_obs,rtol=1e-12,atol=1e-12)
    return f, config, verification


def legend(fig, methods=METHODS, ncol=5):
    handles = [plt.Line2D([],[],color=COLORS[m],marker='x',lw=1,label=LABELS[m]) for m in methods]
    fig.legend(handles=handles,loc='outside lower center',ncol=ncol,frameon=False,fontsize=9)


def save(fig, out, category, name, caption):
    folder = out/category
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(folder/f'{name}.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    FIGURES.append(dict(category=category,name=name,caption=caption))
    print(f'Saved {category}/{name}',flush=True)


def panels(title, xlabel='', ylabel='', names=DATASETS):
    cols = min(4,len(names))
    fig, axes = plt.subplots(int(np.ceil(len(names)/cols)),cols,
        figsize=(4*cols,3.25*int(np.ceil(len(names)/cols))+.5),squeeze=False,layout='constrained')
    for ax,name in zip(axes.flat,names):
        ax.set(title=NAMES.get(name,name),xlabel=xlabel,ylabel=ylabel)
        ax.grid(alpha=.17)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title,fontsize=14)
    return fig,list(axes.flat)[:len(names)]


def errorpoints(ax, data, offset=0, color=None, marker='o', label=None, annotate=False):
    for i,m in enumerate(METHODS):
        r = data.set_index('method').loc[m]
        ax.errorbar(i+offset,r.rate,yerr=[[r.rate-r.lo],[r.hi-r.rate]],
            fmt=marker,color=color or COLORS[m],capsize=4,ms=5,lw=1.2,label=label if i==0 else None)
        if annotate:
            ax.annotate(f'{r.rate:.2%}',(i+offset,r.hi),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(range(5),[LABELS[m] for m in METHODS],rotation=12)


def null_plots(f, config, out, source=None):
    null = f[f['mode'].eq('gaussian_null')]
    cells = [('gaussian_design','iid_n100','Independent Gaussian; 5,000 trials/arm'),
             ('gaussian_design','correlated_n100','AR(1), rho=0.5; 1,000 trials/arm'),
             ('heating_load','gaussian_null_n100','Heating predictors; 1,000 trials/arm'),
             ('red_wine','gaussian_null_n100','Red wine predictors; 1,000 trials/arm')]
    for cond, stem, naive in [('iid_n100','type1_error_5percent',False),('iid_n100','selective_vs_naive_5percent',True)]:
        g = null[(null.study=='null') & null.condition.eq(cond)]
        fig,ax = plt.subplots(figsize=(10.5,5.5))
        fig.subplots_adjust(top=.77,bottom=.25,left=.09,right=.98)
        fig.suptitle('Selective versus naive tests of the same selected target' if naive else 'Type I error at the 5% significance level',y=.96,fontsize=16)
        fig.text(.5,.885,'n = 100, p = 20; 5,000 paired independent-Gaussian null trials per method',ha='center',fontsize=10)
        data = rates(g)
        errorpoints(ax,data,offset=-.1 if naive else 0,color='#2674a6' if naive else None,label='Selective test' if naive else None,annotate=not naive)
        if naive:
            errorpoints(ax,rates(g,column='unadjusted_p_value'),offset=.1,color='#d58122',marker='x',label='Naive test')
        ax.axhline(.05,color='#d62728',ls='--',lw=1.3,label='Nominal alpha = 0.05')
        ax.set(ylim=(0,.6 if naive else .1),ylabel='Type I error rate',xlabel='Selection method / inference engine')
        ax.set_yticks([0,.05,.1,.2,.3,.4,.5,.6] if naive else [0,.05,.1])
        ax.yaxis.set_minor_locator(MultipleLocator(.01))
        ax.grid(axis='y',alpha=.17)
        ax.legend(frameon=False,loc='upper left' if naive else 'lower right')
        fig.text(.09,.04,'Strict p < 0.05. Bars: pointwise 95% exact binomial intervals; zero failed tests.\nMCMC: 511 replicas, 20 steps; spline comparator uses exact inference.',fontsize=9)
        save(fig,out,'paper_style_fpr',stem,'Primary synthetic null; matched methods; pointwise exact 95% binomial intervals.')
        data.to_csv(out/'paper_style_fpr'/f'{stem}_data.csv',index=False)
    fig,axes = panels('Type I error across prespecified null conditions',ylabel='Type I error rate',names=[x[2] for x in cells])
    for ax,(dataset,condition,title) in zip(axes,cells):
        errorpoints(ax,rates(null[null.dataset.eq(dataset)&null.condition.eq(condition)]))
        ax.axhline(.05,color='#d62728',ls='--')
        ax.set_ylim(0,.12)
        ax.tick_params(axis='x',labelsize=8,rotation=30)
    save(fig,out,'paper_style_fpr','null_conditions_5percent','Primary iid null; the other three panels are secondary. Pointwise exact 95% intervals; n=100 throughout.')
    curve_rows = []
    fig,axes = panels('Null rejection calibration: strict p < significance level',xlabel='Significance level',ylabel='Rejection fraction',names=[x[2] for x in cells])
    for ax,(dataset,condition,title) in zip(axes,cells):
        g = null[null.dataset.eq(dataset)&null.condition.eq(condition)]
        for alpha in np.arange(.005,.1001,.005):
            z = rates(g,float(alpha));z['dataset']=dataset;z['condition']=condition;curve_rows.append(z)
        curves = pd.concat(curve_rows[-20:])
        for m in METHODS:
            q = curves[curves.method.eq(m)]
            ax.plot(q.alpha,q.rate,color=COLORS[m],lw=1.2,marker='x',ms=3,markevery=3)
        ax.plot([0,.1],[0,.1],'--',color='#d62728')
        ax.set(xlim=(0,.105),ylim=(0,.12))
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    # Use common limits across conditions and retain all plotted curve values.
    curve_upper = max(.12, float(np.ceil(pd.concat(curve_rows).rate.max()/.02)*.02))
    for ax in axes:
        ax.set_ylim(0,curve_upper)
    legend(fig)
    save(fig,out,'paper_style_fpr','selective_calibration_curve','Red dashed line: nominal rate. Descriptive curves reuse the same trials across alpha levels; they are not independent experiments. Common axes retain every plotted value.')
    pd.concat(curve_rows).to_csv(out/'paper_style_fpr/calibration_curve_data.csv',index=False)
    primary = read_csv((source or out.parent)/'rates.csv')
    primary = primary[primary.study.eq('null')&primary.condition.eq('iid_n100')&primary.alpha.eq(.05)].set_index('method').reindex(METHODS)
    fig,ax = plt.subplots(figsize=(10,4.5),layout='constrained')
    ax.scatter(range(5),primary.all_attempt_lower,c=[COLORS[m] for m in METHODS],label='Observed rejection rate')
    for i,m in enumerate(METHODS):
        ax.plot([i,i],[primary.loc[m,'all_attempt_lower'],primary.loc[m,'primary_family_simultaneous_upper']],color=COLORS[m],lw=1.3)
        ax.plot(i,primary.loc[m,'primary_family_simultaneous_upper'],'_',color=COLORS[m],ms=12)
    ax.axhline(.05,color='#d62728',ls='--',label='Nominal 5%')
    ax.axhline(.06,color='0.3',ls=':',label='Prespecified 6% upper-bound tolerance')
    ax.set(xticks=range(5),xticklabels=[LABELS[m] for m in METHODS],ylim=(0,.08),ylabel='Rejection probability',title='Primary calibration: simultaneous one-sided 95% upper bounds')
    ax.legend(frameon=False,fontsize=9)
    save(fig,out,'paper_style_fpr','primary_simultaneous_calibration','Bonferroni family: five methods, iid null, alpha=.05. Vertical segments terminate at simultaneous one-sided upper bounds; these are not two-sided intervals.')


def observed_plots(f, source, out):
    obs = f[f['mode'].eq('observed')]
    allrates = pd.concat([rates(obs[obs.dataset.eq(d)]).assign(dataset=d) for d in DATASETS])
    allrates.to_csv(out/'observed_rejection_plot_data.csv',index=False)
    fig,ax = plt.subplots(figsize=(14,5),layout='constrained')
    for j,m in enumerate(METHODS):
        q = allrates[allrates.method.eq(m)].set_index('dataset').reindex(DATASETS)
        ax.errorbar(np.arange(8)+(j-2)*.13,q.rate,yerr=[q.rate-q.lo,q.hi-q.rate],fmt='x',capsize=3,color=COLORS[m],label=LABELS[m])
    ax.set(xticks=range(8),xticklabels=[NAMES[d] for d in DATASETS],ylim=(0,1.04),ylabel='Working-model rejection fraction',title='Observed-response benchmarks (n = 100, alpha = 0.05; 500 subsamples/task)')
    ax.legend(frameon=False,ncol=3,loc='lower left')
    save(fig,out,'paper_plots','02_real_data_rejection_rates','Pointwise exact 95% binomial intervals. Observed responses have unknown noise scale; rejection is neither Type I error nor power. Uncertainty is conditional on each fixed dataset and split.')
    display = []
    for d in DATASETS:
        q = allrates[allrates.dataset.eq(d)].set_index('method')
        display.append([f"{q.loc[m,'rate']:.3f}\n[{q.loc[m,'lo']:.3f}, {q.loc[m,'hi']:.3f}]" for m in METHODS])
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');ax.axis('off')
    table=ax.table(cellText=display,rowLabels=[NAMES[d] for d in DATASETS],colLabels=[LABELS[m] for m in METHODS],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.4)
    for (row,col),cell in table.get_celld().items():
        cell.set_linewidth(.4)
        if row==0:cell.set_facecolor('#e9eef3')
    ax.set_title('Observed-response rejection fractions and pointwise 95% intervals\n500 subsamples/task; n = 100; strict p < 0.05',pad=18)
    save(fig,out,'paper_plots','01_real_data_table','Working-model rejection fractions; intervals reflect subsampling Monte Carlo uncertainty, conditional on the frozen datasets and split.')
    for zoom,stem in [(False,'03_pvalue_ecdf'),(True,'04_pvalue_ecdf_near_alpha')]:
        fig,axes=panels('Observed-response p-value distributions'+(' near alpha = 0.05' if zoom else ''),xlabel='Selective p-value',ylabel='Empirical CDF')
        for ax,d in zip(axes,DATASETS):
            for m in METHODS:
                p=np.sort(obs.loc[obs.dataset.eq(d)&obs.method.eq(m),'p_value'].to_numpy())
                ax.step(np.r_[0,p],np.r_[0,np.arange(1,len(p)+1)/len(p)],where='post',color=COLORS[m],lw=1.1)
            ax.axvline(.05,color='0.4',ls=':')
            ax.set(xlim=(0,.1 if zoom else 1),ylim=(0,1.02))
        legend(fig);save(fig,out,'validation_plots',stem,'ECDF uses p <= x; rejection uses p < .05. Observed-response distributions have no uniform-null interpretation.')
    fig,axes=panels('Observed-response distributions of -log10(p)',xlabel='-log10(selective p-value)',ylabel='Fraction of trials per bin')
    bins=np.linspace(0,100,51)
    for ax,d in zip(axes,DATASETS):
        for m in METHODS:
            p=obs.loc[obs.dataset.eq(d)&obs.method.eq(m),'p_value'].to_numpy()
            assert np.all(p>0), 'Define and disclose a zero-p plotting convention before plotting.'
            ax.hist(-np.log10(p),bins=bins,weights=np.ones(len(p))/len(p),histtype='step',color=COLORS[m])
        ax.set_xlim(0,100)
        ax.axvline(-np.log10(1/512),color='0.4',ls=':')
    legend(fig);save(fig,out,'validation_plots','05_log_pvalue_histograms','Common bins and axes; vertical line is MCMC maximum -log10(p), log10(512). Exact tests can yield much smaller p-values. No values are dropped or clipped.')
    fig,axes=panels('Naive versus selective tests on observed responses',xlabel='Naive p-value of the same target',ylabel='Selective p-value')
    for ax,d in zip(axes,DATASETS):
        for m in METHODS:
            g=obs[obs.dataset.eq(d)&obs.method.eq(m)]
            ax.scatter(g.unadjusted_p_value,g.p_value,s=4,alpha=.25,color=COLORS[m],rasterized=True)
        ax.plot([0,1],[0,1],'--',color='0.5',lw=.7);ax.axhline(.05,color='0.5',ls=':');ax.axvline(.05,color='0.5',ls=':')
        ax.set(xlim=(0,1),ylim=(0,1))
    legend(fig);save(fig,out,'validation_plots','06_naive_selective','Same selected target within each method; different selectors can test different targets.')
    pred=read_csv(source/'prediction.csv')
    for metric,title in [('rmse','RMSE'),('mae','MAE'),('r2','R-squared')]:
        fig,axes=panels(f'Fixed-holdout predictive performance: {title}',ylabel=title)
        for ax,d in zip(axes,DATASETS):
            for i,m in enumerate(METHODS):
                r=pred[pred.dataset.eq(d)&pred.method.eq(m)&pred.metric.eq(metric)].iloc[0]
                ax.errorbar(i,r['mean'],yerr=[[r['mean']-r.conditional_mc_ci95_lower],[r.conditional_mc_ci95_upper-r['mean']]],fmt='x',color=COLORS[m],capsize=3)
            if metric=='rmse':
                r=pred[pred.dataset.eq(d)&pred.metric.eq('baseline_rmse')].iloc[0]
                ax.axhline(r['mean'],ls='--',color='0.5',lw=.8)
            ax.set_xticks([])
        legend(fig);save(fig,out,'paper_plots',f'prediction_{metric}','Means and existing 2,000-resample bootstrap 95% intervals conditional on frozen datasets/splits. RMSE dashed lines show training-mean baselines. Response units are calibration-standardized within task.')
    return obs


def diagnostics(obs, f, source, out):
    for column,stem,title,scale in [('elapsed_seconds','09_runtime_distributions','Per-test elapsed time (seconds)',True),('proposals','10_sampling_effort','Recorded MCMC transitions per test',False),('mcmc_replicate_acceptance_rate','21_replica_acceptance','MCMC replica acceptance rates',False),('mcmc_moved_replicas','22_replica_movement','MCMC replicas that moved (out of 511)',False)]:
        methods=METHODS if column in ('elapsed_seconds','proposals') else METHODS[1:]
        fig,axes=panels(title,ylabel=column.replace('_',' '))
        for ax,d in zip(axes,DATASETS):
            arrays=[obs.loc[obs.dataset.eq(d)&obs.method.eq(m),column].dropna().to_numpy() for m in methods]
            box=ax.boxplot(arrays,positions=np.arange(len(methods)),widths=.6,showfliers=False,patch_artist=True,manage_ticks=False)
            for patch,m in zip(box['boxes'],methods):patch.set_facecolor(COLORS[m]);patch.set_alpha(.6)
            for i,(m,a) in enumerate(zip(methods,arrays)):
                ax.plot(i,np.median(a),'o',ms=3,color=COLORS[m])
            ax.set_xticks([])
            if scale:ax.set_yscale('log')
        legend(fig,methods);save(fig,out,'validation_plots',stem,'Boxes: median, quartiles and 1.5-IQR whiskers; outlier markers omitted. No AIS ESS or importance-weight diagnostics apply. Exact inference records zero transitions.')
    fig,axes=panels('Observed-task inference failures',ylabel='Failed / attempted')
    for ax,d in zip(axes,DATASETS):
        vals=[obs.loc[obs.dataset.eq(d)&obs.method.eq(m),'inference_failed'].mean() for m in METHODS]
        ax.bar(range(5),vals,color=[COLORS[m] for m in METHODS]);ax.set_ylim(0,1);ax.set_xticks([])
        ax.text(.5,.5,'0 / 500 for every method',ha='center',transform=ax.transAxes)
    legend(fig);save(fig,out,'validation_plots','07_failure_rates','Zero recorded inference failures. Computation completed; this does not itself establish statistical validity.')
    frequency=read_csv(source/'selection_frequency.csv')
    meta=json.loads((source/'provenance.json').read_text())['prepared_data']['datasets']
    for d in DATASETS:
        fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
        names=meta[d]['feature_names'];n=len(names)
        for j,m in enumerate(METHODS):
            q=frequency[frequency.dataset.eq(d)&frequency.method.eq(m)].set_index('feature')
            counts=q['count'].reindex(range(n),fill_value=0).to_numpy()
            intervals=np.array([cp(int(k),500) for k in counts])
            ax.errorbar(np.arange(n)+(j-2)*.13,counts/500,yerr=[counts/500-intervals[:,0],intervals[:,1]-counts/500],fmt='x',color=COLORS[m],capsize=2)
        ax.set(xticks=range(n),xticklabels=names,ylim=(0,1.05),ylabel='Feature selection frequency',title=f'{NAMES[d]}: observed-response top-two selection')
        ax.tick_params(axis='x',rotation=35)
        legend(fig);save(fig,out,'validation_plots',f'16_features_{d}','Pointwise exact 95% binomial intervals; all predictor indices included, including zero observed selections. Selection frequency is not causal importance.')
    paired=f[f.method.eq('spline_exact')][KEYS+['p_value','selected','target']].merge(f[f.method.eq('spline_mcmc')][KEYS+['p_value','selected','target']],on=KEYS,suffixes=('_exact','_mcmc'),validate='one_to_one')
    paired['absolute_difference']=abs(paired.p_value_mcmc-paired.p_value_exact)
    paired['decision_disagreement']=paired.p_value_mcmc.lt(.05)!=paired.p_value_exact.lt(.05)
    paired.to_csv(out/'spline_paired_plot_data.csv',index=False)
    po=paired[paired.study.eq('real')&paired.condition.eq('observed_n100')]
    fig,axes=panels('Paired spline p-values: exact versus MCMC rank test',xlabel='Exact selective p-value (log scale)',ylabel='MCMC rank p-value (log scale)')
    for ax,d in zip(axes,DATASETS):
        q=po[po.dataset.eq(d)]
        assert q.p_value_exact.gt(0).all()
        ax.scatter(q.p_value_exact,q.p_value_mcmc,s=5,alpha=.35,color=COLORS['spline_mcmc'],rasterized=True)
        ax.plot([1e-100,1],[1e-100,1],'--',lw=.8,color='0.5')
        ax.axhline(.05,color='0.5',ls=':');ax.axvline(.05,color='0.5',ls=':');ax.axhline(1/512,color='#d58122',ls=':',lw=.8)
        ax.set(xscale='log',yscale='log',xlim=(1e-100,1),ylim=(1e-3,1.1),xticks=[1e-90,1e-60,1e-30,1])
        ax.text(.03,.96,f'n = {len(q)}; decisions differ: {int(q.decision_disagreement.sum())}',ha='left',va='top',transform=ax.transAxes,fontsize=8)
    save(fig,out,'validation_plots','17_spline_pvalue_comparison','Same selection, target, and statistic on each trial. MCMC rank p-values are discrete randomized-test outputs, not point estimates of exact p-values; numerical differences are not sampling-error estimates. Full exact range retained; MCMC grid minimum is 1/512.')
    fig,axes=panels('Paired spline absolute p-value differences',xlabel='|p(MCMC rank) - p(exact)|',ylabel='Fraction of observed-response trials')
    for ax,d in zip(axes,DATASETS):
        a=po.loc[po.dataset.eq(d),'absolute_difference'].to_numpy()
        ax.hist(a,bins=np.linspace(0,1,31),weights=np.ones(len(a))/len(a),color=COLORS['spline_mcmc'],alpha=.7)
        ax.set_xlim(0,1)
    save(fig,out,'validation_plots','18_spline_absolute_difference','Descriptive differences between two valid test constructions; not an exact-tail estimator error or confidence-interval coverage assessment.')
    pair=read_csv(source/'paired_comparisons.csv')
    fig,axes=panels('Paired observed-response rejection differences versus spline exact',ylabel='Rejection fraction difference')
    for ax,d in zip(axes,DATASETS):
        q=pair[pair.study.eq('real')&pair.dataset.eq(d)&pair.condition.eq('observed_n100')&pair.metric.eq('rejection_difference')&pair.alpha.eq(.05)]
        for i,m in enumerate(METHODS[1:]):
            r=q[(q.left.eq(m)&q.right.eq('spline_exact'))|(q.right.eq(m)&q.left.eq('spline_exact'))].iloc[0]
            sign=1 if r.left==m else -1
            mid=sign*r.complete_pair_difference
            lo,hi=sorted([sign*r.paired_missing_ci95_lower,sign*r.paired_missing_ci95_upper])
            ax.errorbar(i,mid,yerr=[[mid-lo],[hi-mid]],fmt='x',color=COLORS[m],capsize=3)
        ax.axhline(0,color='0.5',ls='--');ax.set_xticks([])
    legend(fig,METHODS[1:]);save(fig,out,'validation_plots','19_paired_rejection_differences','Conservative paired discordance-based 95% intervals from verified summaries. Different selectors may target different hypotheses; spline exact/MCMC target identity is enforced.')
    fig,axes=panels('Observed-response rejection stability over predetermined trial blocks',xlabel='Trial block (100 trials)',ylabel='Rejection fraction')
    stability=[]
    for ax,d in zip(axes,DATASETS):
        for m in METHODS:
            g=obs[obs.dataset.eq(d)&obs.method.eq(m)]
            ys=[]
            for block in range(5):
                a=g[g.trial.ge(block*100)&g.trial.lt((block+1)*100)]
                y=float(a.p_value.lt(.05).mean());ys.append(y)
                stability.append(dict(dataset=d,method=m,block=block+1,attempted=len(a),rate=y))
            ax.plot(range(1,6),ys,marker='x',lw=1,color=COLORS[m])
        ax.set(ylim=(0,1.02),xticks=range(1,6))
    legend(fig);save(fig,out,'validation_plots','20_stability_block','Five fixed consecutive blocks of 100 independent outer trials; not within-chain MCMC convergence diagnostics.')
    pd.DataFrame(stability).to_csv(out/'stability_blocks.csv',index=False)


def signal_plots(f, out):
    signal=f[f['mode'].eq('gaussian_signal')]
    for conditional,stem in [(False,'marginal_injected_target_detection'),(True,'conditional_injected_target_detection')]:
        fig,axes=panels('Injected-feature detection'+(' conditional on targeting' if conditional else ' per attempted trial'),xlabel='Signal strength beta',ylabel='Detection probability',names=['heating_load','red_wine'])
        records=[]
        for ax,d in zip(axes,['heating_load','red_wine']):
            for j,m in enumerate(METHODS):
                for strength in [.5,1.]:
                    g=signal[signal.dataset.eq(d)&signal.method.eq(m)&signal.beta.eq(strength)]
                    targeted=g.target.eq(g.injected_feature)
                    n=int(targeted.sum()) if conditional else len(g)
                    k=int((targeted&g.p_value.lt(.05)).sum())
                    lo,hi=cp(k,n) if n else (np.nan,np.nan)
                    records.append(dict(dataset=d,method=m,beta=strength,attempted=len(g),targeted=int(targeted.sum()),denominator=n,detected=k,rate=k/n if n else np.nan,lo=lo,hi=hi))
                    if n:
                        ax.errorbar(strength+(j-2)*.025,k/n,yerr=[[k/n-lo],[hi-k/n]],fmt='x',color=COLORS[m],capsize=3)
            ax.set(ylim=(0,1.08),xticks=[.5,1.],xlim=(.35,1.15))
            if conditional:
                counts = [[next(r['denominator'] for r in records if r['dataset']==d and r['method']==m and r['beta']==strength) for strength in [.5,1.]] for m in METHODS]
                table=ax.table(cellText=counts,rowLabels=[LABELS[m] for m in METHODS],
                    colLabels=['n (beta = 0.5)','n (beta = 1)'],cellLoc='center',bbox=[.47,.08,.48,.38])
                table.auto_set_font_size(False);table.set_fontsize(8)
                for (row,col),cell in table.get_celld().items():
                    cell.set_linewidth(.35)
                    if row==0:cell.set_facecolor('#e9eef3')
                    if col==-1:cell.get_text().set_color(COLORS[METHODS[row-1]])
        legend(fig);save(fig,out,'paper_plots',stem,'Detection requires targeting the injected feature and p < .05. Exact pointwise 95% intervals; 500 outer trials per design/strength. Conditional panels tabulate qualifying target counts; no redraw until selected.')
        pd.DataFrame(records).to_csv(out/f'{stem}_data.csv',index=False)


def report(source, out, config, verification):
    notes = ('Figures use the completed focused MCMC experiment only. No selection, inference, or benchmark was rerun. '
        'Methods retain their validated configurations (511 MCMC replicas, 20 steps). '
        'Null plots assess Type I error; observed-response plots show working-model rejection and conditional-on-dataset prediction. '
        'Observed rejection fractions are not power or calibration. Pointwise intervals are not simultaneous; the dedicated primary-family plot uses the prespecified simultaneous upper bounds. '
        'Spline rank p-values need not approximate exact p-values. AIS ESS, weights, PSIS, and AIS Monte Carlo error plots are inapplicable; hub-tail binomial intervals are not exact-tail uncertainty intervals. '
        'Related dataset families are not treated as independent population replicates.')
    cards=[]
    for row in FIGURES:
        stem=f"{row['category']}/{row['name']}"
        cards.append(f'<section><h2>{html.escape(row["name"].replace("_"," "))}</h2><p>{html.escape(row["caption"])}</p><a href="{stem}.png"><img loading="lazy" src="{stem}.png" alt="{html.escape(row["caption"])}"></a><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.svg">SVG</a> · <a href="{stem}.png">PNG</a></p></section>')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Focused MCMC benchmark figures</title><style>body{font:16px system-ui;margin:32px auto;max-width:1500px;padding:0 24px;color:#20262d}img{max-width:100%;height:auto}section{border-top:1px solid #ddd;padding:20px 0}p{line-height:1.6}a{color:#185d91}</style><h1>Focused five-method MCMC benchmark</h1><p>'+html.escape(notes)+'</p>'+''.join(cards)+'</html>',encoding='utf-8')
    command=r'.\.venv\Scripts\python.exe examples/plot_publication_focused_gallery.py'
    (out/'README.md').write_text('# Focused MCMC benchmark figure gallery\n\n'+notes+'\n\n70,000 evaluations; 14,000 paired trials; no inference failures.\n\nRegenerate from repository root:\n\n```powershell\n'+command+'\n```\n\n'+ '\n'.join(f'- [{r["name"]}]({r["category"]}/{r["name"]}.pdf): {r["caption"]}' for r in FIGURES)+'\n',encoding='utf-8')
    manifest=dict(experiment=config['experiment'],input=str(source.resolve()),script_sha256=sha(__file__),
        verified_inputs=verification['output_sha256'],plots=FIGURES,plot_count=len(FIGURES),benchmark_rerun=False,
        strict_rejection='p < alpha',primary_alpha=.05,mcmc_replicas=511,mcmc_steps=20,
        omitted_inapplicable_ais_plots=['ESS','weight concentration','PSIS','AIS proposals','AIS MC error'],
        output_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='plot_manifest.json'})
    (out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT)
    args=parser.parse_args()
    f,config,verification=audit(args.input)
    out=args.input/'reports/03_full_diagnostics';out.mkdir(parents=True,exist_ok=True)
    FIGURES.clear()
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,
        'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    null_plots(f,config,out,source=args.input)
    obs=observed_plots(f,args.input,out)
    diagnostics(obs,f,args.input,out)
    signal_plots(f,out)
    report(args.input,out,config,verification)
    print(f'Finished: {len(FIGURES)} plots in PNG/PDF/SVG; gallery: {out/"index.html"}')


if __name__=='__main__':
    main()
