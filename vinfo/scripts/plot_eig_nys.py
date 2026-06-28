# -*- coding: utf-8 -*-
"""Part A: data-selection accuracy (ranking-based) curves.
inv-mode (full eNTK) TOP accuracy vs selection %, for the 4 comparison settings.
Reads data_selection predictions txt (git-tracked). Portable paths.
Output: <vinfo>/report_figs/acc_s{1..4}.png
"""
import os, glob, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

VINFO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE  = os.path.join(VINFO, 'freeshap_res', 'data_selection')
OUT   = os.path.join(VINFO, 'report_figs'); os.makedirs(OUT, exist_ok=True)

DATASETS = [('sst2',5000), ('mnli',5000), ('ag_news',5000), ('mr',5000),
            ('qqp',5000), ('rte',2490), ('mrpc',3668)]
RANKS = [1, 5, 20]
LAMS  = ['1e-01','1e-02','1e-03','1e-04','1e-05','1e-06']

def lam_disp(l): return l.replace('e-0','e-')
def find_one(pat):
    g=sorted(glob.glob(pat)); return g[0] if g else None
def parse_inv_top(path):
    txt=open(path).read()
    m=re.findall(r'inv mode lambda=[^\n]*\ntop:\s*\n\[([^\]]*)\]', txt)
    if not m: return None
    return np.array([int(x) for x in m[-1].split(',')])/100.0

def eig_txt(ds,num,rank,lam):
    return find_one(f'{BASE}/{ds}/eigen/predictions/bert_seed2024_num{num}_val*_eig{rank}.0_eiglam{lam}_invlam1e-06_*tmc500_predictions.txt')
def nys_txt(ds,num,rank,lam):
    return find_one(f'{BASE}/{ds}/nystrom/predictions/bert_seed2024_num{num}_val*_nys{rank}.0_nyslam{lam}_invlam1e-06_*tmc500_predictions.txt')

def eig_method(lam): return (f'eigen λ={lam_disp(lam)}', (lambda l:(lambda ds,n,r: eig_txt(ds,n,r,l)))(lam))
def nys_method(lam): return (f'nys λ={lam_disp(lam)}',   (lambda l:(lambda ds,n,r: nys_txt(ds,n,r,l)))(lam))

def colors_for(labels):
    palette=['#000000','#d62728','#ff7f0e','#2ca02c','#1f77b4','#9467bd','#8c564b','#e377c2','#17becf',
             '#bcbd22','#7f7f7f','#aec7e8']
    return {lab:palette[i%len(palette)] for i,lab in enumerate(labels)}

SETTINGS = [
    ('s1','1) eigen λ=1e-2 vs nystrom λ=1e-1..1e-6', [eig_method('1e-02')]+[nys_method(l) for l in LAMS]),
    ('s2','2) eigen internal λ=1e-1..1e-6',          [eig_method(l) for l in LAMS]),
    ('s3','3) nystrom internal λ=1e-1..1e-6',        [nys_method(l) for l in LAMS]),
    ('s4','4) matched: eigen λ vs nystrom λ',        [m for l in reversed(LAMS) for m in (eig_method(l),nys_method(l))]),
]

def make_acc(tag, title, methods):
    labels=[l for l,_ in methods]; col=colors_for(labels)
    nrows,ncols=len(DATASETS),len(RANKS)
    fig,axes=plt.subplots(nrows,ncols,figsize=(6.0*ncols,3.2*nrows),squeeze=False); hands={}
    for i,(ds,num) in enumerate(DATASETS):
        for j,rank in enumerate(RANKS):
            ax=axes[i][j]; plotted=False
            for lab,fn in methods:
                p=fn(ds,num,rank)
                if not p: continue
                y=parse_inv_top(p)
                if y is None: continue
                lw=2.4 if lab.startswith('eigen') else 1.4
                ln,=ax.plot(np.arange(1,len(y)+1),y,color=col[lab],lw=lw,alpha=0.9,
                            zorder=10 if lab.startswith('eigen') else 5,label=lab)
                hands.setdefault(lab,ln); plotted=True
            ax.set_title(f'{ds} (num{num}) — rank {rank}%',fontsize=10.5)
            ax.set_xlabel('selection %',fontsize=8.5); ax.set_ylabel('val acc (%)',fontsize=8.5)
            ax.tick_params(labelsize=8); ax.grid(True,alpha=0.3)
            if not plotted:
                ax.text(0.5,0.5,'(no data yet)',ha='center',va='center',transform=ax.transAxes,color='gray',fontsize=12)
    H=[hands[k] for k in labels if k in hands]; L=[k for k in labels if k in hands]
    if H: fig.legend(H,L,loc='upper center',ncol=min(8,len(L)),fontsize=9,frameon=True,bbox_to_anchor=(0.5,0.998))
    fig.suptitle(f'[{title}]  data-selection TOP accuracy (full eNTK) vs selection %',y=1.012,fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.975])
    out=f'{OUT}/acc_{tag}.png'; fig.savefig(out,dpi=130,bbox_inches='tight'); plt.close(fig); print('saved:',out)

if __name__=='__main__':
    for tag,title,methods in SETTINGS:
        make_acc(tag,title,methods)
    print('\nACC FIGURES IN:',OUT)
