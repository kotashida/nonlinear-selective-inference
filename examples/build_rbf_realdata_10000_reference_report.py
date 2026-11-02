"""Paper-style Table 1 and presentation-style benchmark plots from saved tests."""
from __future__ import annotations
import argparse
import html
import io
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import run_rbf_shapley_realdata_10000 as study
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import pymupdf

ORDER=['concrete','heating_load','cooling_load','gas_turbine','housing','red_wine','white_wine','abalone']
NAMES={'concrete':'Concrete','heating_load':'Heating load','cooling_load':'Cooling load',
       'gas_turbine':'Gas turbine','housing':'Housing','red_wine':'Red wine',
       'white_wine':'White wine','abalone':'Abalone'}
ARMS=['selected_shapley','full_shapley']
COLORS=['#2674a6','#d58122']
PAPER=ROOT/'references/papers/2025_Shiraishi_SelectiveInferenceFeatureSelectionPipelines.pdf'
SLIDES=next((ROOT/'presentation/261020_si_rd').glob('261020-*.pdf'))



def rate_text(value):
    text=f'{value:.3f}'
    return text[1:] if text.startswith('0.') else text


def load(source):
    m=study.core.read(source/'requested_manifest.json');done=study.core.read(source/'COMPLETED')
    study.core.require(study.core.sha(m)==done['manifest_sha256'] and done['status']=='verified','Unverified results')
    for name,sha in done['export_sha256'].items():
        study.core.require(Path(name).name==name and study.digest(source/name)==sha,'Export checksum mismatch')
    d=pd.read_csv(source/'results.csv',usecols=['dataset','n','repetition','arm','status','p_value','null_true'],
                  dtype={'null_true':'boolean'})
    s=pd.read_csv(source/'summary.csv')
    study.core.require(len(d)==500000 and m['repetitions']==10000 and m['sample_sizes']==[100,150,200],'Unexpected benchmark design')
    nd=d[d.dataset==study.NULL_DATASET].copy()
    d=d[d.dataset!=study.NULL_DATASET].copy()
    study.core.require(len(d)==480000 and len(nd)==20000,'Real/null allocation mismatch')
    study.core.require(not d.duplicated(['dataset','n','repetition','arm']).any(),'Duplicate tests')
    for (name,n,arm),g in d.groupby(['dataset','n','arm']):
        study.core.require(set(g.repetition)==set(range(10000)),'Incomplete repetition coverage')
        for alpha in m['alpha_levels']:
            r=s[(s.dataset==name)&(s.n==n)&(s.arm==arm)&(s.alpha==alpha)].iloc[0]
            ok=g[g.status=='ok'];k=int((ok.p_value<alpha).sum());ci=study.core.binomial(k,len(ok))
            study.core.require(r.rejections==k and r.resolved==len(ok) and r.failures==10000-len(ok),'Summary count mismatch')
            study.core.require(np.allclose([r.resolved_rejection_rate,r.resolved_ci_lower,r.resolved_ci_upper],
                                           [ci['rate'],ci['ci_lower'],ci['ci_upper']],rtol=1e-12,atol=1e-14),'Summary rate/interval mismatch')
    study.core.require(d.status.eq('ok').all() and nd.status.eq('ok').all(),
                       'This publication report expects zero unresolved tests; revise denominator labels for failures')
    null_summary=pd.read_csv(source/'type1_error_summary.csv')
    for arm,g in nd.groupby('arm'):
        study.core.require(len(g)==10000 and g.null_true.notna().all() and g.null_true.all(),'Invalid synthetic null truth')
        study.core.require(set(g.repetition)==set(range(10000)),'Incomplete synthetic null coverage')
        for alpha in m['alpha_levels']:
            r=null_summary[(null_summary.arm==arm)&(null_summary.alpha==alpha)].iloc[0]
            ci=study.core.binomial(int((g.p_value<alpha).sum()),len(g))
            study.core.require(r.rejections==ci['rejections'] and r.n==10000 and r.failures==0,'Null summary count mismatch')
            study.core.require(np.allclose([r.rate,r.ci_lower,r.ci_upper],
                                           [ci['rate'],ci['ci_lower'],ci['ci_upper']],rtol=1e-12,atol=1e-14),'Null summary interval mismatch')
    return m,d,s[s.alpha==.05].copy(),done,nd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/benchmarks'/study.NAME)
    args=parser.parse_args();source=args.input.resolve();m,d,s,done,nd=load(source)
    out=source/'analysis/paper_presentation';out.mkdir(parents=True,exist_ok=True)
    figures=[];pdf_assets={}
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':11,
        'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    def row(name,n,arm): return s[(s.dataset==name)&(s.n==n)&(s.arm==arm)].iloc[0]
    def save(fig,name,title,caption):
        for ext in ('png','svg'): fig.savefig(out/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
        buffer=io.BytesIO();fig.savefig(buffer,format='pdf',bbox_inches='tight',facecolor='white')
        pdf_assets[name]=buffer.getvalue()
        plt.close(fig);figures.append(dict(name=name,title=title,caption=caption))

    # Table 1: three sample-size rows, eight dataset columns, slash-paired rates.
    fig,ax=plt.subplots(figsize=(15,3.2));ax.set_axis_off()
    xs=np.r_[.035,np.linspace(.14,.94,8)]
    ax.text(.005,.99,'Real-data rejection rates',
            fontsize=13,va='top',transform=ax.transAxes)
    ax.text(.005,.87,'Each cell: Selected-Features Shapley Effect / All-Features Shapley Effect.',
            fontsize=9.5,va='top',transform=ax.transAxes)
    for y in [.72,.55,.12]: ax.plot([0,1],[y,y],color='black',lw=1 if y!=.55 else .65,transform=ax.transAxes)
    for x,label in zip(xs,['n']+[NAMES[v] for v in ORDER]):
        ax.text(x,.635,label,ha='center',va='center',fontsize=11,transform=ax.transAxes)
    compact=[]
    for n,y in zip([100,150,200],[.46,.33,.20]):
        ax.text(xs[0],y,str(n),ha='center',va='center',transform=ax.transAxes)
        record={'n':n}
        for j,name in enumerate(ORDER):
            a,b=[row(name,n,arm).resolved_rejection_rate for arm in ARMS]
            aa,bb=rate_text(a),rate_text(b)
            record[name]=f'{aa}/{bb}'
            left=r'\mathbf{'+aa+'}' if a>=b else aa
            right=r'\mathbf{'+bb+'}' if b>=a else bb
            ax.text(xs[j+1],y,'$'+left+r'\,/\,'+right+'$',ha='center',va='center',fontsize=11.5,transform=ax.transAxes)
        compact.append(record)
    save(fig,'01_table1','Real-data rejection rates',
         'Rows show n=100,150,200. Each cell lists Selected-Features / All-Features rates. Larger rates are bold, with both bold in ties. The effects have different null hypotheses; bold does not establish superior variable identification. All cells contain 10,000 resolved tests.')
    pd.DataFrame(compact).to_csv(out/'table1_compact.csv',index=False)
    s.to_csv(out/'table1_detailed.csv',index=False)
    latex=[r'\begin{tabular}{r'+'c'*8+'}',r'\hline',
           'n & '+' & '.join(NAMES[v] for v in ORDER)+r' \\',r'\hline']
    for n in [100,150,200]:
        cells=[]
        for name in ORDER:
            a,b=[row(name,n,arm).resolved_rejection_rate for arm in ARMS]
            aa,bb=rate_text(a),rate_text(b)
            if a>=b: aa=r'\textbf{'+aa+'}'
            if b>=a: bb=r'\textbf{'+bb+'}'
            cells.append(aa+'/'+bb)
        latex.append(str(n)+' & '+' & '.join(cells)+r' \\')
    latex.extend([r'\hline',r'\end{tabular}',
        '% Cells: Selected-Features Shapley Effect / All-Features Shapley Effect.',
        '% Analytic p < 0.05; 10000 resolved tests per cell; real-data null truth unknown.'])
    (out/'table1.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')

    for i,n in enumerate([100,150,200],2):
        fig,ax=plt.subplots(figsize=(14,5),layout='constrained')
        for j,(arm,color) in enumerate(zip(ARMS,COLORS)):
            rates=[row(name,n,arm).resolved_rejection_rate for name in ORDER]
            ax.plot(np.arange(8)+(j-.5)*.14,rates,linestyle='none',marker='x',markersize=7,
                    markeredgewidth=1.5,color=color,label=study.LABELS[arm])
        ax.set(xticks=range(8),xticklabels=[NAMES[v] for v in ORDER],ylim=(0,1.04),
            ylabel='Working-model rejection fraction',
            title=f'Observed-response benchmarks (n = {n}, alpha = 0.05; 10,000 subsamples/task)')
        ax.legend(frameon=False,ncol=2,loc='lower left',fontsize=10)
        save(fig,f'{i:02}_observed_response_n{n}',f'Observed-response benchmarks: n = {n}',
             'Presentation-style offset x markers, serif typography, dataset order and 0-1 vertical scale. As in the reference slide, confidence intervals are omitted here; exact 95% intervals and counts are in the detailed table. These are working-model selective rejection fractions, with sigma=1 as a working scale.')

    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    for ax,name in zip(axes.flat,ORDER):
        for arm,color in zip(ARMS,COLORS):
            g=s[(s.dataset==name)&(s.arm==arm)].sort_values('n')
            ax.plot(g.n,g.resolved_rejection_rate,'o-',color=color,label=study.LABELS[arm],lw=1.6,ms=4)
            ax.fill_between(g.n,g.resolved_ci_lower,g.resolved_ci_upper,color=color,alpha=.12)
        ax.set(title=NAMES[name],xticks=[100,150,200],ylim=(0,1.04),xlabel='Sample size (n)')
        ax.grid(alpha=.17)
    for ax in axes[:,0]: ax.set_ylabel('Selective rejection rate')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01))
    fig.suptitle('Selective rejection rates as sample size increases',fontsize=16)
    fig.tight_layout(rect=(0,.065,1,.94))
    save(fig,'05_sample_size_curves','Sample-size trends with confidence intervals',
         'Same data as Table 1; shading gives pointwise 95% exact binomial intervals. Samples are nested within repetition, and the two effects share the actual target at each n. Intervals are not simultaneous. Several cells are at the 100% ceiling already at n=100; larger n therefore need not produce a visible increase.')

    # Real-data null truth is unknown. Use the separately saved synthetic null,
    # explicitly retaining its n=100,p=4 design instead of relabeling real data.
    null_done=done
    null_manifest=m['synthetic_null']
    study.core.require(null_manifest['n']==100 and null_manifest['p']==4 and m['k']==2,'Unexpected null design')
    null_rates=[]
    for arm in ARMS:
        g=nd[nd.arm==arm]
        study.core.require(len(g)==10000 and g.null_true.all(),'Invalid global-null stratum')
        null_rates.append(dict(arm=arm,label=study.LABELS[arm],alpha=.05,
                               **study.core.binomial(int((g.p_value<.05).sum()),len(g))))
    nr=pd.DataFrame(null_rates);nr.to_csv(out/'type1_error_rates.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained')
    for j,r in enumerate(nr.itertuples()):
        ax.errorbar(j,r.rate,yerr=[[r.rate-r.ci_lower],[r.ci_upper-r.rate]],fmt='x',markersize=9,
                    markeredgewidth=1.8,capsize=5,color=COLORS[j])
    ax.axhline(.05,color='#888888',ls='--',lw=.9)
    ax.set(xticks=[0,1],xticklabels=[study.LABELS[a].replace(' Shapley','\nShapley') for a in ARMS],
           xlim=(-.6,1.6),ylim=(0,.10),ylabel='Type I error rate',
           title='Type I error rate')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig,'06_type1_error_rate','Type I error rate: synthetic global null',
         'Separate synthetic validation: fresh independent N(0,1) features, Y=N(0,I), n=100, p=4, k=2, known sigma=1. '
         'Strict analytic p < 0.05; markers show 10,000 trials per effect with pointwise 95% exact binomial intervals. '
         'This is not a Type I error estimate from the real datasets, and does not cover n=150,200 or their 7-11 feature designs.')

    # A print-ready landscape report: table, three slide-style plots, trends, detail.
    report=out/'benchmark_reference_report.pdf';W,H=841.89,595.28
    pdf=pymupdf.open();page=None
    def paragraph(text,x,y,width):
        rect=pymupdf.Rect(x,y,x+width,H-40)
        spare,scale=page.insert_htmlbox(rect,text,css='* {font-family: serif; font-size: 10pt; line-height: 1.3; color: #263443;} body {margin:0;}',scale_low=1)
        study.core.require(spare>=0,'Report paragraph overflow')
        return y+(rect.height-spare)
    def header(title,number,synthetic=False):
        nonlocal page
        page=pdf.new_page(width=W,height=H)
        page.insert_text((34,34),title,fontname='tibo',fontsize=18,color=(.125,.19,.25))
        page.draw_line((34,44),(W-34,44),color=(.835,.506,.133),width=1)
        footer=('Synthetic global null | n = 100, p = 4 | alpha = 0.05 | 10,000 trials per effect' if synthetic else
                'Analytic RBF Shapley | n = 100, 150, 200 | alpha = 0.05 | 10,000 trials per cell')
        page.insert_text((34,H-18),footer,fontname='tiro',fontsize=9)
        page.insert_text((W-42,H-18),str(number),fontname='tiro',fontsize=9)
    def image_fit(path,x,y,width,height):
        vector=pymupdf.open('pdf',pdf_assets[path.stem])
        page.show_pdf_page(pymupdf.Rect(x,y,x+width,y+height),vector,0,keep_proportion=True)
        vector.close()
    for number,item in enumerate(figures,1):
        title='Real-data rejection rates' if number==1 else item['title']
        header(title,number,synthetic=item['name']=='06_type1_error_rate')
        if number==1:
            image_fit(out/(item['name']+'.svg'),30,60,W-60,180)
            y=265
            y=paragraph('<b>Design.</b> Eight real-data tasks; 240,000 nested row subsamples and 480,000 real-data analytic tests, plus 20,000 synthetic-null tests. '
                'At each n the effects share rows, response, selected pair and randomized target. '
                'Only the feature universe defining the tested Shapley effect differs.',38,y,W-76)+16
            y=paragraph('<b>Observed pattern.</b> Heating load and cooling load are at 100% for both effects at every n; gas turbine is nearly saturated. '
                'Selected-Features rejection increases with sample size on Concrete, Housing, Red wine, White wine and Abalone. '
                'At n=200, All-Features is 100% on seven tasks and 99.92% on White wine.',38,y,W-76)+16
            y=paragraph('<b>Interpretation.</b> Real-data null truth is unknown. These are selective rejection rates under the '
                'working Gaussian model with sigma=1, rather than measured power against known false nulls. '
                'All-Features may test the full RBF candidate space; higher rates do not establish a unique feature effect.',38,y,W-76)+16
            paragraph('<b>Reference formats.</b> Shiraishi et al. (2025), Table 1 (p. 8), motivates the slash-paired table. '
                'The October 20 presentation, final slide (PDF p. 12), motivates the offset x-marker figures. '
                'The present methods and denominators differ from those references.',38,y,W-76)
        else:
            image_fit(out/(item['name']+'.svg'),30,60,W-60,H-178)
            paragraph(item['caption'],38,H-97,W-76)
    header('Detailed results: counts and pointwise 95% intervals',len(figures)+1)
    table=['<table><thead><tr><th>Dataset</th><th>n</th><th>Selected-Features Shapley Effect</th><th>All-Features Shapley Effect</th></tr></thead><tbody>']
    for index,name in enumerate(ORDER):
        for n in [100,150,200]:
            entries=[NAMES[name],str(n)]
            for arm in ARMS:
                r=row(name,n,arm)
                entries.append(f'{r.resolved_rejection_rate:.4f} [{r.resolved_ci_lower:.4f}, {r.resolved_ci_upper:.4f}] &nbsp; {int(r.rejections)}/{int(r.resolved)}')
            table.append('<tr style="background:'+('#f4f6f8' if index%2==0 else 'white')+'">'+''.join('<td>'+e+'</td>' for e in entries)+'</tr>')
    table.append('</tbody></table>')
    spare,scale=page.insert_htmlbox(pymupdf.Rect(36,62,W-36,H-95),''.join(table),css='* {font-family:serif;font-size:9pt;} body {margin:0;} table {width:100%;border-collapse:collapse;border-top:1px solid black;border-bottom:1px solid black;} th {text-align:left;border-bottom:0.5px solid black;padding:5px;} td {padding:3px 5px;}',scale_low=1)
    study.core.require(spare>=0,'Detailed result table overflow')
    paragraph('Rates use analytic p &lt; 0.05. Brackets give pointwise 95% Clopper-Pearson intervals. '
        'All intended tests resolved: 10,000 per cell, zero failures. A rate of 1.000 has uncertainty and is not a claim of population power equal to one.',38,H-80,W-76)
    pdf.set_metadata(dict(title='Analytic RBF Shapley real-data benchmark',author='RBF Shapley benchmark analysis'))
    pdf.save(report,garbage=4,deflate=True);pdf.close()

    intro=('500,000 verified analytic tests (480,000 real-data and 20,000 synthetic-null);  8 datasets; n=100,150,200; 10,000 repetitions per effect and cell. '
           'No failures. Display order matches the benchmark slide. Effects test different hypotheses; '
           'real-data null truth is unknown and sigma=1 is a working scale.')
    md=['# Paper and presentation-style benchmark results',intro,
        '[Printable seven-page report](benchmark_reference_report.pdf)',
        '[Compact Table 1 CSV](table1_compact.csv) | [Detailed CSV](table1_detailed.csv) | [LaTeX table](table1.tex)']
    sections=[]
    for item in figures:
        md.append(f'## {item["title"]}\n\n{item["caption"]}\n\n![{item["title"]}]({item["name"]}.png)')
        sections.append(f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["caption"])}</p><a href="{item["name"]}.svg"><img src="{item["name"]}.png" alt="{html.escape(item["title"])}"></a></section>')
    md.append('## Reproduction\n\n`.venv/Scripts/python.exe examples/build_rbf_realdata_10000_reference_report.py`')
    (out/'README.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real-data benchmark: paper and presentation formats</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>Real-data RBF Shapley benchmark</h1><p>'+html.escape(intro)+'</p><p><a href="benchmark_reference_report.pdf">Printable seven-page report</a> | <a href="table1_compact.csv">Table 1 CSV</a> | <a href="table1_detailed.csv">Detailed CSV</a> | <a href="table1.tex">LaTeX table</a></p>'+''.join(sections)+'</html>',encoding='utf-8')
    study.core.atomic(out/'analysis_provenance.json',dict(validated_tests=len(d)+len(nd),failures=0,source_exports=done['export_sha256'],
        references={str(p.relative_to(ROOT)):study.digest(p) for p in [PAPER,SLIDES]},
        reference_pages={'paper':8,'presentation':12},script_sha256=study.digest(Path(__file__)),figures=figures,
        synthetic_null_source=str(source.relative_to(ROOT)),synthetic_null_exports=null_done['export_sha256'],
        files_sha256={p.name:study.digest(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_provenance.json'}))
    print(f'Created {len(figures)} figures, compact/detailed/LaTeX tables and seven-page PDF report: {out}')


if __name__=='__main__': main()
