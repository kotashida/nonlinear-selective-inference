"""Analytic-only RBF Shapley study with independent Gaussian features.

Run from the repository root with .venv/Scripts/python.exe
benchmarks/run_rbf_shapley_analytic_independent.py. No MCMC is performed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
import importlib.metadata as metadata
import json
from pathlib import Path
import time

import run_rbf_shapley as core
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
NAME = 'rbf_shapley_analytic_independent_250_20261024'
ARMS = ['selected_shapley', 'full_shapley']
LABELS = ['Selected-Features Shapley Effect', 'All-Features Shapley Effect']
COLORS = ['#2674a6', '#d58122']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration():
    m = core.read(ROOT/'benchmarks/rbf_shapley_mcmc20_cpu32_250_20261022.json')
    for key in list(m):
        if key.startswith('mcmc_') or key in ('local_probability', 'local_scales', 'required_versions'):
            del m[key]
    m.update(experiment=NAME, output_root=f'outputs/rbf_shapley/{NAME}',
             designs=['independent_gaussian'], correlation=0.0, workers=1,
             inference='analytic_truncated_chi',
             design_scope='one fixed iid N(0,I_4) design, responses redrawn each trial')
    return m


def trials(m):
    encoded = core.encoded(m).decode()
    g = core.geometry(encoded, 0, 'additive_rbf')
    rows = []
    for ci, condition in enumerate(m['conditions']):
        mu = core.mean_response(m, 0, condition)
        for iteration in range(m['iterations']):
            noise_seed = core.seed(m, 2, 0, iteration)
            target_seed = core.seed(m, 3, 0, iteration)
            y = mu + m['sigma']*np.random.default_rng(noise_seed).normal(size=m['n'])
            yr, mur = g.Q.T@y, g.Q.T@mu
            selected = g.select(yr, m['k'])
            u = float(np.random.default_rng(target_seed).uniform())
            j = selected[int(u*len(selected))]
            mask = sum(1 << a for a in selected)
            for arm in ARMS:
                A, B = g.target(arm, mask, j)
                result = core.radial_test(g, yr, B, selected, m['sigma'])
                # Only the analytic selected tail is a reported test method.
                # The untruncated tail is retained as a numerical diagnostic.
                naive = result.pop('p_naive')
                p = result.pop('p_selective')
                projection = float(np.linalg.norm(B.T@mur))
                rows.append(dict(condition=condition['id'], iteration=iteration,
                    arm=arm, method='analytic', design='independent_gaussian',
                    noise_seed=noise_seed, target_seed=target_seed, auxiliary_u=u,
                    data_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                    selected=list(selected), target=j, rank=B.shape[1], full_rank=g.r,
                    p_value=p, untruncated_reference_p=naive, rejected=p < .05,
                    mean_contribution=max(0., float(mur@A@mur)),
                    projected_mean_norm=projection,
                    null_true=projection <= m['null_tolerance']*max(1., np.linalg.norm(mu)),
                    **result))
    return pd.DataFrame(rows), g


def validate(m, d, g):
    core.require(len(d) == m['iterations']*4, 'Wrong test count')
    core.require(not d.duplicated(['condition','iteration','arm']).any(), 'Duplicate tests')
    core.require(d.p_value.between(0,1).all() and np.isfinite(d.p_value).all(), 'Invalid tails')
    core.require((d.rejected == (d.p_value < .05)).all(), 'Decision mismatch')
    for _, pair in d.groupby(['condition','iteration']):
        core.require(set(pair.arm) == set(ARMS), 'Missing paired target')
        core.require(pair[['target','data_sha256','noise_seed','target_seed','auxiliary_u']].nunique().eq(1).all(),
                     'Unpaired responses or targets')
        core.require(pair.selected.map(tuple).nunique() == 1, 'Unpaired selection')
    # Rebuild responses and independently check saved radii, target truth,
    # and selection boundaries across every test's conditioned ray.
    encoded = core.encoded(m).decode()
    for condition in m['conditions']:
        mu = core.mean_response(m, 0, condition)
        for row in d[d.condition == condition['id']].itertuples():
            y = mu + m['sigma']*np.random.default_rng(row.noise_seed).normal(size=m['n'])
            yr, mur = g.Q.T@y, g.Q.T@mu
            selected = g.select(yr, m['k'])
            core.require(list(selected) == row.selected, 'Reconstructed selection mismatch')
            A, B = g.target(row.arm, sum(1 << j for j in selected), row.target)
            projected = B@(B.T@yr)
            radius = np.linalg.norm(projected)/m['sigma']
            core.require(np.isclose(radius, row.radius), 'Reconstructed radius mismatch')
            truth = np.linalg.norm(B.T@mur) <= m['null_tolerance']*max(1., np.linalg.norm(mu))
            core.require(truth == row.null_true, 'Incorrect null truth')
            z, direction = yr-projected, projected/(m['sigma']*radius)
            intervals = core.selection_intervals(g, z, direction, selected, m['sigma'])
            for r in radius*np.array([.1,.5,1.,2.,10.]):
                inside = any(lo < r < hi for lo,hi in intervals)
                actual = g.select(z+m['sigma']*r*direction, m['k']) == selected
                core.require(inside == actual, 'Interval/selector disagreement')
    full = d[d.arm == 'full_shapley']
    core.require(full['rank'].eq(g.r).all(), 'Unexpected full-space rank')
    core.require(np.allclose(full.p_value, full.untruncated_reference_p, atol=1e-10),
                 'Full-space analytic reference disagreement')


def summaries(m, d):
    rows = []
    for (condition, arm, truth), group in d.groupby(['condition','arm','null_true']):
        for alpha in m['alpha_levels']:
            k, n = int((group.p_value < alpha).sum()), len(group)
            ci = core.binomial(k,n)
            lo, hi = ci['ci_lower'], ci['ci_upper']
            rows.append(dict(condition=condition, arm=arm, method='analytic', null_true=bool(truth),
                alpha=alpha, n=n, rejections=k, rate=k/n, ci_lower=lo, ci_upper=hi))
    return pd.DataFrame(rows)


def plot(m, d, summary, output):
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    figures = []
    def save(fig, name, caption):
        fig.tight_layout(rect=(0,.02,1,.94))
        for ext in ['png','pdf','svg']:
            fig.savefig(output/f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
        figures.append(dict(name=name, caption=caption))
    def percent(ax):
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2); ax.set_axisbelow(True)

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax, condition in zip(axes, ['global_null','quadratic_signal']):
        s = summary[(summary.condition == condition) & (summary.alpha == .05)].set_index('arm').loc[ARMS]
        ax.bar(range(2), s.rate, color=COLORS, width=.6)
        ax.errorbar(range(2), s.rate, yerr=[s.rate-s.ci_lower,s.ci_upper-s.rate], fmt='none',color='#203040',capsize=4)
        for x,r in enumerate(s.itertuples()):
            ax.text(x,r.ci_upper+.015,f'{r.rate:.1%}\n{r.rejections}/{r.n}',ha='center',fontsize=9)
        if condition == 'global_null': ax.axhline(.05,ls='--',color='gray')
        ax.set(title='Global null' if condition == 'global_null' else 'Quadratic signal (RMS 0.5)',
               xticks=range(2),xticklabels=[label.replace(" Shapley", "\nShapley") for label in LABELS],ylim=(0,.25 if condition == 'global_null' else 1.15),
               ylabel='Rejection fraction'); percent(ax)
    fig.suptitle('Independent Gaussian features: analytic selective tests')
    save(fig,'01_rejection_rates','Strict p < 0.05; pointwise 95% exact binomial intervals. Both targets share each response and tested feature. Signal rejection compares different Shapley hypotheses.')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for j,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
        s=summary[(summary.condition=='global_null') & (summary.arm==arm)].sort_values('alpha')
        axes[0].errorbar(s.alpha+(j-.5)*.002,s.rate,yerr=[s.rate-s.ci_lower,s.ci_upper-s.rate],fmt='o',capsize=3,color=color,label=label)
        p=np.sort(d[(d.condition=='global_null') & (d.arm==arm)].p_value)
        axes[1].step(np.r_[0,p,1],np.r_[0,np.arange(1,len(p)+1)/len(p),1],where='post',color=color)
    axes[0].plot([0,.12],[0,.12],'--',color='gray');axes[0].set(title='Null rejection rates (250 trials)',xlabel='Nominal alpha (points offset)',ylabel='Rejection fraction',xticks=m['alpha_levels'],ylim=(0,.22));axes[0].legend(fontsize=9)
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(title='Null p-value empirical distribution',xlabel='p-value',ylabel='Fraction at or below p',xlim=(0,1),ylim=(0,1))
    for ax in axes: percent(ax)
    fig.suptitle('Null calibration under independent Gaussian features')
    save(fig,'02_null_calibration','Analytic selected tails; pointwise exact binomial intervals. One fixed independently generated design. A 250-trial cell does not establish precise conditional calibration.')

    fig,axes=plt.subplots(1,2,figsize=(13,5));outcomes=[]
    for ax,arm,label in zip(axes,ARMS,LABELS):
        group=d[(d.condition=='quadratic_signal') & (d.arm==arm)]
        subsets=[group[group.target==0],group]
        rates=[g.rejected.mean() for g in subsets]
        ax.bar(range(2),rates,color='#258568',label='Detected',width=.6)
        ax.bar(range(2),1-np.array(rates),bottom=rates,color='#bcc8d1',label='Resolved, not detected',width=.6)
        for x,(g,rate) in enumerate(zip(subsets,rates)):
            ax.text(x,1.025,f'n={len(g)}',ha='center');ax.text(x,rate/2,f'{rate:.1%}',ha='center',color='white')
            outcomes.append(dict(arm=arm,stratum=['target_0','all_signal_tests'][x],n=len(g),detected=int(g.rejected.sum()),true_nulls=int(g.null_true.sum())))
        ax.set(title=label,xticks=range(2),xticklabels=['Targeting feature 0','All signal tests'],ylim=(0,1.13));percent(ax)
    axes[1].legend(fontsize=8,loc='lower right');fig.suptitle('Signal outcomes: generating feature and Shapley hypotheses')
    save(fig,'03_signal_outcomes','Analytic p < 0.05. Feature independence in the population does not imply exact orthogonality of finite-sample RBF blocks. Null truth is evaluated from the actual mean and target space for every test.')
    pd.DataFrame(outcomes).to_csv(output/'signal_outcomes.csv',index=False)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,condition in zip(axes,['global_null','quadratic_signal']):
        g=d[(d.condition==condition)&(d.arm==ARMS[0])]
        rates=[(g.target==j).mean() for j in range(m['p'])]
        ax.bar(range(m['p']),rates,color=['#2674a6']+['#bcc8d1']*3)
        for j,r in enumerate(rates): ax.text(j,r+.015,f'{r:.1%}',ha='center')
        ax.axhline(.5,color='gray',ls='--');ax.set(title=condition.replace('_',' ').title(),xlabel='Tested feature',ylabel='Fraction of trials',xticks=range(m['p']),ylim=(0,.65));percent(ax)
    fig.suptitle('Which features were tested?')
    save(fig,'04_target_frequencies','Each selected pair supplies one uniformly randomized target; paired arms share that target. The dashed line marks the population targeting ceiling; finite samples may exceed it.')
    sections=[]
    intro='Analytic-only RBF Shapley inference with independent N(0, I_4) Gaussian features. n=100, k=2; 250 trials per condition. One fixed X, Gaussian response noise with known sigma=1. Signal: centered feature-0 squared, RMS 0.5. No MCMC is run.'
    md=['# Analytic RBF Shapley: independent features',intro,'Population independence does not enforce sample orthogonality. All-Features Shapley Effect tests span all 12 candidate RBF directions and need not isolate the generating feature. See ../README.md for numerical results.']
    for item in figures:
        title=item['name'][3:].replace('_',' ').title()
        md.append(f'## {title}\n\n{item["caption"]}\n\n![{title}]({item["name"]}.png)')
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(item["caption"])}</p><img src="{item["name"]}.png" alt="{html.escape(title)}"></section>')
    (output/'README.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analytic independent RBF study</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}</style><h1>Analytic RBF Shapley: independent features</h1><p>'+html.escape(intro)+'</p>'+''.join(sections)+'</html>',encoding='utf-8')
    return figures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/f'outputs/rbf_shapley/{NAME}')
    parser.add_argument('--plot-only',action='store_true',help='Regenerate figures from verified saved results')
    args=parser.parse_args();out=args.output.resolve();m=configuration()
    if args.plot_only:
        prov=core.read(out/'provenance.json')
        for name,sha in prov['files_sha256'].items():
            core.require(digest(out/name)==sha, f'Artifact checksum mismatch: {name}')
        core.require(core.read(out/'requested_manifest.json')==m, 'Manifest mismatch')
        d=pd.read_csv(out/'results.csv')
        d['selected']=d.selected.map(ast.literal_eval)
        g=core.geometry(core.encoded(m).decode(),0,'additive_rbf')
        validate(m,d,g)
        summary=summaries(m,d)
        figures=plot(m,d,summary,out/'analysis')
        readme=out/'README.md'
        content=readme.read_text(encoding='utf-8')
        content=content.replace('Full-candidate tested spaces','All-Features Shapley Effect tested spaces').replace('selected-set spaces rank','Selected-Features Shapley Effect spaces rank')
        content=content.replace('| selected_shapley |','| Selected-Features Shapley Effect |').replace('| full_shapley |','| All-Features Shapley Effect |')
        readme.write_text(content,encoding='utf-8')
        prov.update(figures=figures,plot_script_sha256=digest(Path(__file__)),
            files_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*')
                          if p.is_file() and p.name not in ('provenance.json','COMPLETED')})
        core.atomic(out/'provenance.json',prov)
        print(f'Regenerated {len(figures)} figures from saved analytic results: {out}')
        return
    core.require(not out.exists(), 'Use a new output folder; existing results are never overwritten')
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        d,g=trials(m);validate(m,d,g)
    out.mkdir(parents=True)
    X,_=core.design(core.encoded(m).decode(),0)
    pd.DataFrame(X,columns=[f'x{j}' for j in range(m['p'])]).to_csv(out/'design.csv',index=False)
    core.atomic(out/'requested_manifest.json',m)
    d.to_csv(out/'results.csv',index=False)
    summary=summaries(m,d);summary.to_csv(out/'calibration_power_summary.csv',index=False)
    figures=plot(m,d,summary,out/'analysis')
    lines=['# Analytic-only RBF Shapley: independent Gaussian features',
        'Same seed and standard-normal draws as the correlated study, with the AR(1) transformation removed. One fixed iid N(0,I_4) design; 250 responses per condition, 500 paired datasets, 1,000 analytic tests. Selection uses exhaustive full-candidate RSS Shapley scores, top k=2, then one uniformly randomized selected target.',
        '| Condition | Target | Rejection at 5% | 95% exact interval | True nulls / tests |',
        '|---|---|---:|---:|---:|']
    for r in summary[summary.alpha==.05].itertuples():
        lines.append(f'| {r.condition} | {r.arm} | {r.rate:.1%} ({r.rejections}/{r.n}) | {r.ci_lower:.1%}–{r.ci_upper:.1%} | {r.n if r.null_true else 0}/{r.n} |')
    lines += ['', 'Independence is a population property; the saved finite design and RBF blocks are not forced to be orthogonal. Target truth is computed from the actual noiseless mean. All-Features Shapley Effect tested spaces have rank 12 and Selected-Features Shapley Effect spaces rank 6; rejection does not establish an individual or causal feature effect.',
        '[Plot gallery](analysis/index.html). Figures follow the presentation_four_selectors_mcmc20_250_20261008 reference style.',
        'Reproduce in a new folder: `.venv/Scripts/python.exe benchmarks/run_rbf_shapley_analytic_independent.py --output outputs/rbf_shapley/NEW_FOLDER`. Only analytic selected p-values are reported as a testing method. The untruncated chi tail is stored solely as a numerical diagnostic. No MCMC was performed.']
    (out/'README.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    core.atomic(out/'verification.json',dict(status='verified',tests=len(d),datasets=len(d)//2,
        checks=['complete coverage','paired arms','finite tails','reconstructed selection, radii and null truth',
                'selection intervals versus selector on five radii per test','full-space analytic tail identity']))
    core.atomic(out/'provenance.json',dict(manifest_sha256=core.sha(m),
        sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'benchmarks/run_rbf_shapley.py',ROOT/'benchmarks/rbf_network.py']},
        versions={p:metadata.version(p) for p in ['numpy','scipy','matplotlib','pandas']},
        elapsed_seconds=time.perf_counter()-start,figures=figures,
        files_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}))
    core.atomic(out/'COMPLETED',dict(status='complete',tests=len(d),datasets=len(d)//2,method='analytic'))
    print(summary[summary.alpha==.05].to_string(index=False))
    print(f'Created {out}')


if __name__=='__main__':
    main()
