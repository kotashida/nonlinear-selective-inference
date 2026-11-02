"""Minimal manuscript figures/tables from verified results; no inference runs."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import plot_publication_focused_gallery as g
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

METHODS = ['shap_mcmc','lime_mcmc','spline_mcmc','spline_exact','mutual_information_mcmc']
LABELS = dict(zip(METHODS,['SHAP (MCMC)','LIME (MCMC)','Spline (MCMC)','Spline (exact)','MI (MCMC)']))
SHORT = ['SHAP','LIME','Spline','Exact','MI']
MARKERS = dict(zip(METHODS,['o','s','^','x','D']))
STYLES = dict(zip(METHODS,['-','--','-.',':','-']))
CAPTIONS = {}


def style():
    plt.rcParams.update({'font.family':'serif',
        'font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
        'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.55,'axes.spines.top':True,'axes.spines.right':True,
        'axes.grid':False,'lines.linewidth':.8,'lines.markersize':3.3,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.5,
        'ytick.major.width':.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white'})


def save(fig,out,name,caption):
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'{name}.{ext}',dpi=400,bbox_inches='tight',pad_inches=.035)
    CAPTIONS[name]=caption
    plt.close(fig)
    print('Saved',name,flush=True)


def common_legend(fig):
    handles=[Line2D([],[],color=g.COLORS[m],marker=MARKERS[m],linestyle=STYLES[m],
                    markerfacecolor='white' if MARKERS[m]!='x' else g.COLORS[m],
                    markeredgewidth=.7,label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),
               ncol=5,frameon=False,handlelength=1.8,columnspacing=1.6)


def null_axis(ax,data):
    q=g.rates(data).set_index('method')
    for i,m in enumerate(METHODS):
        r=q.loc[m]
        ax.errorbar(i,r.rate,yerr=[[r.rate-r.lo],[r.hi-r.rate]],fmt=MARKERS[m],
            color=g.COLORS[m],capsize=2,elinewidth=.65,markeredgewidth=.7,
            markerfacecolor='white' if MARKERS[m]!='x' else g.COLORS[m])
    ax.axhline(.05,color='#d62728',linestyle='--',linewidth=.7,zorder=0)
    ax.set(xlim=(-.4,4.4),ylim=(0,.10),xticks=range(5),xticklabels=SHORT,
           yticks=[0,.05,.1],xlabel='Method',ylabel='Type I error rate')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    assert q.hi.max()<.1, 'Widen the Type I error axis; never clip an interval.'


def power_data(frame):
    rows=[]
    signal=frame[frame['mode'].eq('gaussian_signal')]
    for dataset in ['heating_load','red_wine']:
        for method in METHODS:
            for strength in [.5,1.]:
                q=signal[signal.dataset.eq(dataset)&signal.method.eq(method)&signal.beta.eq(strength)]
                targeted=q.target.eq(q.injected_feature)
                n,k=int(targeted.sum()),int((targeted&q.p_value.lt(.05)).sum())
                lo,hi=g.cp(k,n)
                ml,mh=g.cp(k,len(q))
                rows.append(dict(dataset=dataset,method=method,beta=strength,
                    attempts=len(q),targeted=n,rejected=k,power=k/n,lo=lo,hi=hi,
                    marginal=k/len(q),marginal_lo=ml,marginal_hi=mh))
    return pd.DataFrame(rows)


def power_axis(ax,rows,marginal=False):
    key,lo,hi=('marginal','marginal_lo','marginal_hi') if marginal else ('power','lo','hi')
    for m in METHODS:
        q=rows[rows.method.eq(m)].sort_values('beta')
        ax.plot(q.beta,q[key],color=g.COLORS[m],linestyle=STYLES[m],
                marker=MARKERS[m],markeredgewidth=.7,
                markerfacecolor='white' if MARKERS[m]!='x' else g.COLORS[m],zorder=3)
        ax.errorbar(q.beta,q[key],yerr=[q[key]-q[lo],q[hi]-q[key]],fmt='none',
                    ecolor=g.COLORS[m],elinewidth=.55,capsize=1.5,alpha=.65,zorder=2)
    ax.set(xlim=(.45,1.05),ylim=(0,1.035),xticks=[.5,1.],yticks=[0,.5,1.],
           xlabel=r'Signal amplitude $\beta$',ylabel='Detection probability' if marginal else 'Conditional power')


def figures(frame,power,out):
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.35))
    fig.subplots_adjust(left=.075,right=.985,bottom=.28,top=.88,wspace=.31)
    for ax,condition,title in zip(axes,['iid_n100','correlated_n100'],
            ['(a) Independent Gaussian','(b) Correlated Gaussian']):
        null_axis(ax,frame[frame.study.eq('null')&frame.condition.eq(condition)])
        ax.set_title(title,pad=5)
    common_legend(fig)
    save(fig,out,'figure_1_type1_error',
        'Type I error at alpha=0.05 on synthetic Gaussian designs (n=100, p=20). '
        '(a) Independent predictors: 5,000 paired trials per method. '
        '(b) AR(1) predictors with rho=0.5: 1,000 paired trials per method. '
        'Markers show strict p<0.05 rejection fractions; bars are pointwise 95% '
        'Clopper-Pearson intervals. The red dashed line marks alpha=0.05. '
        'SHAP, LIME, Spline, and MI use the validated MCMC procedure; Exact uses '
        'the identical spline selector with exact inference. No inference failures occurred. '
        'Sample size is fixed, so the horizontal axis is method rather than sample size.')
    fig,axes=plt.subplots(2,2,figsize=(7.05,4.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.16,top=.94,hspace=.67,wspace=.31)
    for i,dataset in enumerate(['heating_load','red_wine']):
        data=frame[frame.dataset.eq(dataset)&frame['mode'].eq('gaussian_null')]
        null_axis(axes[i,0],data)
        power_axis(axes[i,1],power[power.dataset.eq(dataset)])
        axes[i,0].set_title(f'({chr(97+2*i)}) {g.NAMES[dataset]}: null',pad=5)
        axes[i,1].set_title(f'({chr(98+2*i)}) {g.NAMES[dataset]}: signal',pad=5)
    common_legend(fig)
    save(fig,out,'figure_2_type1_and_power',
        'Type I error (left) and conditional injected-target power (right), using '
        'the left/right structure of Shiraishi et al. (2025), Figure 3. '
        'Rows correspond to heating-load and red-wine predictor designs. '
        'All panels use n=100 and alpha=0.05. Null panels use 1,000 paired trials '
        'per design with fresh independent N(0,I) noise. Signal panels use 500 '
        'paired outer trials per design and amplitude beta in {0.5,1.0}. '
        'Power is P(p<0.05 | the tested target is the injected feature). '
        'Qualifying denominators, rejections, and exact pointwise 95% intervals '
        'are reported in Table S1; bars show those intervals. Lines connect only '
        'the two evaluated amplitudes, without fitting or interpolation claims. '
        'The signal is the protocol\'s nonlinear mean beta*h(x), not the reference '
        'paper\'s linear-coefficient experiment. Conditioning differs across '
        'selectors; spline exact/MCMC share their selected set and randomized target. '
        'The left-axis range is 0-0.10 to make deviations from 0.05 readable.')
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.45))
    fig.subplots_adjust(left=.075,right=.985,bottom=.29,top=.88,wspace=.31)
    for ax,dataset,title in zip(axes,['heating_load','red_wine'],['(a) Heating load','(b) Red wine']):
        power_axis(ax,power[power.dataset.eq(dataset)],marginal=True)
        ax.set_title(title,pad=5)
    common_legend(fig)
    save(fig,out,'figure_s1_marginal_detection',
        'Marginal injected-target detection P(target is the injected feature and '
        'p<0.05), using all 500 outer trials per design/amplitude as the denominator. '
        'Bars are pointwise exact 95% binomial intervals. Read alongside Figure 2 '
        'to distinguish targeting probability from conditional power. '
        'This is injected-feature detection, not rejection of all possible nonnull '
        'projections; other targets can also have nonzero projection means.')


def latex_table(out,name,headers,rows,caption,label):
    caption=re.sub(r'(?<!\\)%',lambda _:r'\%',caption)
    text='\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n'
    text+='\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text+='\\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n\\toprule\n'
    text+=' & '.join(headers)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+' \\\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    (out/f'{name}.tex').write_text(text,encoding='utf-8')


def table_figure(out,name,headers,rows,caption,widths=None):
    fig,ax=plt.subplots(figsize=(7.05,.42+.29*(len(rows)+1)))
    ax.axis('off')
    table=ax.table(cellText=rows,colLabels=headers,loc='center',cellLoc='right',
                   colLoc='right',colWidths=widths)
    table.auto_set_font_size(False);table.set_fontsize(7.5);table.scale(1,1.35)
    for (row,col),cell in table.get_celld().items():
        cell.set_facecolor('white');cell.set_linewidth(0)
        cell.PAD=.045
        if col==0:cell.get_text().set_ha('left')
        if row==0:
            cell.visible_edges='TB';cell.set_linewidth(.6)
        elif row==len(rows):
            cell.visible_edges='B';cell.set_linewidth(.6)
    save(fig,out,name,caption)


def tables(frame,power,out):
    observed=frame[frame['mode'].eq('observed')]
    datasets=g.DATASETS
    headers=['Method','Concrete','Heating','Cooling','Gas','Housing','Red wine','White wine','Abalone']
    rows=[];interval_rows=[]
    for method in METHODS:
        row=[LABELS[method]]
        for d in datasets:
            q=g.rates(observed[observed.dataset.eq(d)]).set_index('method').loc[method]
            row.append(f'{q.rate:.3f}')
            interval_rows.append(dict(dataset=d,method=method,attempts=int(q.attempted),
                rejections=int(q.rejected),rate=q.rate,lo=q.lo,hi=q.hi))
        rows.append(row)
    caption=('Observed-response working-model rejection fractions (n=100, '
        'alpha=0.05; 500 subsamples per task). Entries are point estimates; '
        'pointwise exact 95% Monte Carlo intervals are given in Table S2. '
        'The noise scale and response truth are unknown: these fractions are '
        'neither Type I error estimates nor power estimates. Uncertainty is '
        'conditional on the frozen source data and splits. Related tasks are '
        'not independent dataset families. Gas denotes gas turbine.')
    table_figure(out,'table_1_real_data',headers,rows,caption,[.205]+[.795/8]*8)
    latex_table(out,'table_1_real_data',headers,rows,caption,'tab:real-data')
    pd.DataFrame(interval_rows).to_csv(out/'table_s2_real_data_intervals.csv',index=False)
    latex_table(out,'table_s2_real_data_intervals',['Dataset','Method','Rejections / attempts','Rate','95\\% interval'],
        [[g.NAMES[r['dataset']],LABELS[r['method']],f"{r['rejections']} / {r['attempts']}",f"{r['rate']:.3f}",f"[{r['lo']:.3f}, {r['hi']:.3f}]"] for r in interval_rows],
        'Observed-response rejection fractions and pointwise exact 95\\% binomial intervals. These are conditional-on-dataset Monte Carlo intervals, not population intervals or evidence of calibration.','tab:real-data-intervals')
    power.to_csv(out/'table_s1_power_counts.csv',index=False)
    latex_table(out,'table_s1_power_counts',['Design','Method','$\\beta$','Targeted / attempts','Rejections','Power','95\\% interval'],
        [[g.NAMES[r.dataset],LABELS[r.method],f'{r.beta:.1f}',f'{r.targeted} / {r.attempts}',str(r.rejected),f'{r.power:.3f}',f'[{r.lo:.3f}, {r.hi:.3f}]'] for r in power.itertuples()],
        'Conditional injected-target power and its qualifying denominators. Intervals are pointwise exact 95\\% binomial intervals; no redraw until selected is used.','tab:power-counts')
    rows=[];records=[]
    for method in METHODS:
        row=[LABELS[method]]
        for dataset,condition in [('gaussian_design','iid_n100'),('heating_load','observed_n100'),('red_wine','observed_n100')]:
            q=frame[frame.dataset.eq(dataset)&frame.condition.eq(condition)&frame.method.eq(method)].cpu_seconds
            low,median,high=q.quantile([.25,.5,.75])
            row.append(f'{median:.2f} [{low:.2f}, {high:.2f}]')
            records.append(dict(dataset=dataset,condition=condition,method=method,
                n=len(q),median_cpu_seconds=median,q25_cpu_seconds=low,q75_cpu_seconds=high))
        a=frame[frame.method.eq(method)]
        row.append(f'{int(a.inference_failed.sum())}/{len(a)}')
        rows.append(row)
    caption=('Per-test CPU time in seconds: median [25th, 75th percentile], '
        'for the independent Gaussian null and two observed-response tasks. '
        'All conditions use n=100; predictor dimensions differ between columns. '
        'Times include feature selection and inference, and prediction on observed '
        'tasks. Failures are inference failures / all attempted evaluations for '
        'that method across the complete study. Workers used one numerical thread '
        'and one random-forest job, with at most 32 concurrent workers. '
        'MCMC used 511 replicas and 20 steps throughout; the spline exact '
        'comparator has identical selection parameters.')
    headers=['Method','Gaussian null','Heating load','Red wine','Failures']
    table_figure(out,'table_2_computation',headers,rows,caption,[.19,.23,.23,.23,.12])
    latex_table(out,'table_2_computation',headers,rows,caption,'tab:computation')
    pd.DataFrame(records).to_csv(out/'table_2_computation.csv',index=False)


def report(out,source,verification):
    title='Minimal manuscript figures and tables'
    lead=('Compact serif typography, thin lines, small markers, no background grid, '
          'shared legends, and booktabs-style tables follow the reference paper\'s '
          'presentation. All 70,000 verified evaluations are unchanged. '
          'Type I error uses method categories because only n=100 was evaluated. '
          'The two signal amplitudes support two observed points, not a fitted power curve.')
    cards=[];md=['# '+title,'',lead,'','PDF and SVG are vector exports; PNG is 400 dpi. '
        'LaTeX tables require `\\usepackage{booktabs}`. Copy captions from captions.md. '
        'Table S2 has 40 rows; its editable TeX source can be split across pages for the supplement.','',
        'Regenerate from the repository root:','',r'`.\.venv\Scripts\python.exe examples\plot_publication_manuscript.py`','']
    for name,caption in CAPTIONS.items():
        md.extend(['## '+name,'',caption,''])
        cards.append(f'<section><h2>{html.escape(name.replace("_"," "))}</h2><img src="{name}.png" alt="{html.escape(caption)}"><p>{html.escape(caption)}</p><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a> · <a href="{name}.png">PNG</a>'+ (f' · <a href="{name}.tex">LaTeX</a>' if (out/f'{name}.tex').exists() else '')+'</p></section>')
    (out/'captions.md').write_text('\n'.join(md),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px Georgia,serif;max-width:1100px;margin:32px auto;padding:0 24px;line-height:1.6;color:#222}img{max-width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:30px}h1,h2{font-weight:normal}a{color:#185d91}</style><h1>'+title+'</h1><p>'+html.escape(lead)+'</p>'+''.join(cards)+'<p>Supplement: <a href="table_s1_power_counts.tex">Table S1: power counts and intervals</a> · <a href="table_s2_real_data_intervals.tex">Table S2: real-data intervals</a> · <a href="captions.md">All captions and regeneration command</a></p></html>',encoding='utf-8')
    manifest=dict(input=str(source.resolve()),verified_inputs=verification['output_sha256'],
        script_sha256=g.sha(__file__),audit_script_sha256=g.sha(g.__file__),
        reference='references/papers/2025_Shiraishi_SelectiveInferenceFeatureSelectionPipelines.pdf, Figure 3 and Table 1',
        reference_sha256=g.sha(g.ROOT/'references/papers/2025_Shiraishi_SelectiveInferenceFeatureSelectionPipelines.pdf'),
        reference_adaptations=['Method categories replace unevaluated sample-size axis',
            'Two observed nonlinear signal amplitudes, n=100',
            'Exact 95% pointwise intervals retained',
            'Conditional injected-target power; qualifying counts in Table S1',
            'Real-response results called working-model rejection, not power'],
        captions=CAPTIONS,benchmark_rerun=False,algorithm_settings_changed=False,
        output_sha256={p.name:g.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def independent_only(frame,source,verification):
    """A separate manuscript packet containing only the completed iid null."""
    out=source/'reports/01_independent_gaussian';out.mkdir(parents=True,exist_ok=True)
    data=frame[frame.study.eq('null')&frame.dataset.eq('gaussian_design')&frame.condition.eq('iid_n100')]
    assert len(data)==25000 and data.groupby('method').size().eq(5000).all()
    assert data['mode'].eq('gaussian_null').all() and data.n.eq(100).all()
    style();CAPTIONS.clear()
    fig,ax=plt.subplots(figsize=(3.45,2.6))
    fig.subplots_adjust(left=.18,right=.97,bottom=.35,top=.96)
    null_axis(ax,data)
    ax.set_xlabel('Selection method')
    handles=[Line2D([],[],color=g.COLORS[m],marker=MARKERS[m],linestyle='none',
        markerfacecolor='white' if MARKERS[m]!='x' else g.COLORS[m],
        markeredgewidth=.7,label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),
        ncol=2,frameon=False,handletextpad=.5,columnspacing=1.5,fontsize=7)
    caption=('Type I error under the independent Gaussian null: '
        'X_ij are iid N(0,1), y is independent N(0,I), n=100 and p=20. '
        'Each method uses the same 5,000 paired trials. Points are rejection '
        'fractions for strict p<0.05; bars are pointwise exact 95% binomial '
        'intervals. The red dashed line marks alpha=0.05. '
        'Spline (exact) uses the same selector as Spline (MCMC). '
        'Only n=100 was evaluated; no sample-size trend is implied.')
    save(fig,out,'figure_1_type1_error',caption)
    rates=g.rates(data).set_index('method')
    stored=g.read_csv(source/'rates.csv')
    primary=stored[stored.study.eq('null')&stored.condition.eq('iid_n100')&stored.alpha.eq(.05)].set_index('method')
    rows=[];records=[]
    for m in METHODS:
        r=rates.loc[m]
        time=data.loc[data.method.eq(m),'cpu_seconds']
        q25,median,q75=time.quantile([.25,.5,.75])
        upper=primary.loc[m,'primary_family_simultaneous_upper']
        rows.append([LABELS[m],f'{int(r.rejected)}/5000',f'{r.rate:.4f}',
            f'[{r.lo:.4f}, {r.hi:.4f}]',f'{upper:.4f}',
            f'{median:.2f} [{q25:.2f}, {q75:.2f}]'])
        records.append(dict(method=m,attempts=5000,rejected=int(r.rejected),
            type1_error=r.rate,ci95_lower=r.lo,ci95_upper=r.hi,
            simultaneous_upper=upper,median_cpu_seconds=median,
            q25_cpu_seconds=q25,q75_cpu_seconds=q75,failures=int(r.failures)))
    caption=('Independent-Gaussian null results for the five methods '
        '(n=100, p=20, alpha=0.05; 5,000 paired trials per method). '
        'Intervals are pointwise exact 95% binomial intervals. '
        'The final probability column gives simultaneous one-sided 95% upper '
        'bounds across the five methods (Bonferroni); the prespecified tolerance '
        'is 0.06. CPU time is per-test seconds, median [25th,75th percentile]. '
        'All methods completed with zero inference failures; MCMC retains '
        '511 replicas and 20 steps and all selector settings are unchanged.')
    headers=['Method','Reject / trials','Type I error','95% CI','Simult. upper','CPU time (s)']
    table_figure(out,'table_1_null_results',headers,rows,caption,[.19,.14,.12,.2,.13,.22])
    latex_table(out,'table_1_null_results',[x.replace('%',r'\%') for x in headers],
        rows,caption,'tab:independent-gaussian-null')
    pd.DataFrame(records).to_csv(out/'table_1_null_results.csv',index=False)
    note=('This packet uses only the 25,000 evaluations from the completed '
        'independent-Gaussian null condition. Correlated-Gaussian and real-predictor '
        'results remain in the original galleries and are excluded here. '
        'No independent-Gaussian signal condition was run, so this packet contains '
        'no power figure. The existing heating-load/red-wine power estimates '
        'cannot be relabeled as independent-Gaussian power. Additional simulations '
        'would be needed for that comparison; none were started. '
        'All data and algorithm settings remain unchanged.')
    command=r'.\.venv\Scripts\python.exe examples\plot_publication_manuscript.py --independent-only'
    md=['# Independent-Gaussian manuscript set','',note,'',
        'Vector PDF/SVG and 400-dpi PNG. The table also has CSV and LaTeX '
        'sources; LaTeX requires `\\usepackage{booktabs}`.','',
        'Regenerate:','',f'```powershell\n{command}\n```','']
    cards=[]
    for name,caption in CAPTIONS.items():
        md.extend(['## '+name,'',caption,''])
        cards.append(f'<section><h2>{html.escape(name.replace("_"," "))}</h2>'
            f'<img src="{name}.png" alt="{html.escape(caption)}">'
            f'<p>{html.escape(caption)}</p><p><a href="{name}.pdf">PDF</a> | '
            f'<a href="{name}.svg">SVG</a> | <a href="{name}.png">PNG</a></p></section>')
    (out/'captions.md').write_text('\n'.join(md),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Independent-Gaussian manuscript set</title><style>'
        'body{font:16px Georgia,serif;max-width:1100px;margin:32px auto;padding:0 24px;line-height:1.6}'
        'img{max-width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:24px}'
        'h1,h2{font-weight:normal}</style><h1>Independent-Gaussian manuscript set</h1><p>'
        +html.escape(note)+'</p>'+''.join(cards)+
        '<p><a href="table_1_null_results.tex">LaTeX table</a> | '
        '<a href="table_1_null_results.csv">CSV table</a> | '
        '<a href="captions.md">Captions and regeneration command</a></p></html>',encoding='utf-8')
    manifest=dict(condition='iid_n100',included_evaluations=25000,paired_trials=5000,
        n=100,p=20,predictor_covariance='identity',response='independent N(0,I)',
        independent_gaussian_power_available=False,benchmark_rerun=False,
        algorithm_settings_changed=False,verified_inputs=verification['output_sha256'],
        script_sha256=g.sha(__file__),audit_script_sha256=g.sha(g.__file__),
        captions=CAPTIONS,
        output_sha256={p.name:g.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Independent-Gaussian manuscript set:',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=g.DEFAULT)
    parser.add_argument('--independent-only',action='store_true',help='Export only the completed iid Gaussian null condition.')
    args=parser.parse_args()
    frame,config,verification=g.audit(args.input)
    if args.independent_only:
        independent_only(frame,args.input,verification)
        return
    out=args.input/'reports/02_other_designs';out.mkdir(parents=True,exist_ok=True)
    style();CAPTIONS.clear()
    power=power_data(frame)
    # Independently check power denominators against the frozen diagnostics.
    diag=g.read_csv(args.input/'diagnostics.csv')
    for row in power.itertuples():
        condition='gaussian_signal_b05_n100' if row.beta==.5 else 'gaussian_signal_b1_n100'
        r=diag[diag.dataset.eq(row.dataset)&diag.condition.eq(condition)&diag.method.eq(row.method)&diag.metric.eq('injected_target_detection')].iloc[0]
        assert row.targeted==r.targeted and row.rejected==r.detections
    figures(frame,power,out)
    tables(frame,power,out)
    report(out,args.input,verification)
    print('Manuscript set:',out)


if __name__=='__main__':
    main()
