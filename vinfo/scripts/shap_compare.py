# -*- coding: utf-8 -*-
"""Compare per-point Shapley VALUES: inv (reference) vs eigen / nystrom.

Generalized over a lambda sweep (1e-1 .. 1e-6) in 4 comparison modes; each mode
reuses the same 4 analyses (corr/Δ heatmap, abs-diff bars, error-dist, ranking
heatmap). Mode 1 is a matched-λ scatter grid.

  mode1 matched : per-λ 2x3 scatter grid, eigen-λ vs nys-λ (vs inv)
  mode2         : eigen λ=1e-2  vs  nys λ=1e-1..1e-6   (= original report)
  mode3         : eigen internal λ=1e-1..1e-6
  mode4         : nystrom internal λ=1e-1..1e-6

Paths derive from this file's location -> runs on any server.
Missing result files -> that cell is left blank automatically.
Figures saved to <vinfo>/report_figs/<tag>_<kind>.png
"""
import os, glob, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

VINFO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../vinfo
SH    = os.path.join(VINFO, 'freeshap_res', 'shapley')
OUT   = os.path.join(VINFO, 'report_figs'); os.makedirs(OUT, exist_ok=True)

DS    = [('sst2',5000,872), ('mnli',5000,1000), ('ag_news',5000,1000), ('mr',5000,1000),
         ('qqp',5000,1000), ('rte',2490,277), ('mrpc',3668,408)]
RANKS = [1, 5, 20]
LAMS  = ['1e-01','1e-02','1e-03','1e-04','1e-05','1e-06']   # big -> small
TAIL  = 'signFalse_earlystopTrue_tmc500.pkl'
MODEL = 'bert'    # ntk_prompt -> bert-base-uncased

def disp(l): return l.replace('e-0','e-')
def one(pat):
    g = sorted(glob.glob(pat)); return g[0] if g else None
def inv_path(ds,N,V):       return one(f'{SH}/{ds}/inv/{MODEL}_seed2024_num{N}_val{V}_lam1e-06_{TAIL}')
def eig_path(ds,N,V,r,lam): return one(f'{SH}/{ds}/eigen/{MODEL}_seed2024_num{N}_val{V}_eig{r}.0_eiglam{lam}_invlam1e-06_cholesky_float32_{TAIL}')
def nys_path(ds,N,V,r,lam): return one(f'{SH}/{ds}/nystrom/{MODEL}_seed2024_num{N}_val{V}_nys{r}.0_nyslam{lam}_invlam1e-06_cholesky_float32_{TAIL}')

def load_sv(path):
    d = pickle.load(open(path,'rb')); dv = np.array(d['dv_result']); si = np.array(d['sampled_idx'])
    return dv[:,1,:].sum(axis=1), si
def aligned(svA,siA,svB,siB):
    dA=dict(zip(siA.tolist(),svA)); dB=dict(zip(siB.tolist(),svB))
    common=sorted(set(siA.tolist())&set(siB.tolist()))
    return np.array([dA[i] for i in common]), np.array([dB[i] for i in common])
def _rank(x): r=np.empty(len(x)); r[np.argsort(x)]=np.arange(len(x)); return r
def spearman(a,b): return np.corrcoef(_rank(a),_rank(b))[0,1] if len(a)>1 else np.nan
def topk_ov(a,b,k=5):
    n=len(a); kk=max(1,int(n*k/100))
    return len(set(np.argsort(a)[::-1][:kk].tolist())&set(np.argsort(b)[::-1][:kk].tolist()))/kk

def eig_method(lam): return (f'eigen λ={disp(lam)}', (lambda l: (lambda ds,N,V,r: eig_path(ds,N,V,r,l)))(lam))
def nys_method(lam): return (f'nys λ={disp(lam)}',   (lambda l: (lambda ds,N,V,r: nys_path(ds,N,V,r,l)))(lam))

def colors_for(labels):
    palette=['#000000','#d62728','#ff7f0e','#2ca02c','#1f77b4','#9467bd','#8c564b','#e377c2','#17becf']
    return {lab:palette[i%len(palette)] for i,lab in enumerate(labels)}

def compute(methods):
    diffs={}; stats={}; rankstats={}
    for ds,N,V in DS:
        ip=inv_path(ds,N,V)
        if not ip: continue
        isv,isi=load_sv(ip); istd=isv.std() or 1.0
        for r in RANKS:
            diffs[(ds,r)]={}
            for label,fn in methods:
                p=fn(ds,N,V,r)
                if not p: continue
                asv,asi=load_sv(p); a,b=aligned(asv,asi,isv,isi)
                if len(a)==0: continue
                d=a-b
                diffs[(ds,r)][label]=d
                co=np.corrcoef(a,b)[0,1] if len(a)>1 else np.nan
                stats[(ds,r,label)]=(np.abs(d).mean(), np.abs(d).max(), co, a.std()/istd)
                rankstats[(ds,r,label)]=(spearman(a,b), topk_ov(a,b,5))
    return diffs,stats,rankstats

def _rows():
    rows=[(ds,r) for ds,_,_ in DS for r in RANKS]
    return rows, [f'{ds} r{r}%' for ds,r in rows]

def fig_heatmap(methods, stats, outpath, suptitle):
    labels=[l for l,_ in methods]; rows,rlab=_rows()
    C=np.full((len(rows),len(labels)),np.nan); M=np.full((len(rows),len(labels)),np.nan)
    for ri,(ds,r) in enumerate(rows):
        for ci,m in enumerate(labels):
            if (ds,r,m) in stats: me,mx,co,sr=stats[(ds,r,m)]; C[ri,ci]=co; M[ri,ci]=me
    fig,axs=plt.subplots(1,2,figsize=(max(8,1.1*len(labels)+5),11))
    c1=LinearSegmentedColormap.from_list('rg',['#b2182b','#f7f7c0','#1a9850']); c1.set_bad('white')
    im0=axs[0].imshow(np.ma.masked_invalid(C),cmap=c1,vmin=0,vmax=1,aspect='auto')
    axs[0].set_title('correlation( approx_sv , inv_sv )\n(1=identical, ~0=unrelated)',fontsize=11)
    c2=LinearSegmentedColormap.from_list('gr',['#1a9850','#f7f7c0','#b2182b']); c2.set_bad('white')
    im1=axs[1].imshow(np.ma.masked_invalid(M),cmap=c2,vmin=0,vmax=0.5,aspect='auto')
    axs[1].set_title('mean | approx_sv − inv_sv |\n(color capped 0.5; number=true)',fontsize=11)
    for ax,Mat,im in [(axs[0],C,im0),(axs[1],M,im1)]:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels([m.replace('λ=','\nλ=') for m in labels],fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rlab,fontsize=7.5)
        for ri in range(len(rows)):
            for ci in range(len(labels)):
                v=Mat[ri,ci]
                if not np.isnan(v): ax.text(ci,ri,f'{v:.2f}',ha='center',va='center',fontsize=6.2)
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.suptitle(suptitle,fontsize=12); fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outpath,dpi=130,bbox_inches='tight'); plt.close(fig); print('saved:',outpath)

def fig_ranking(methods, rankstats, outpath, suptitle):
    labels=[l for l,_ in methods]; rows,rlab=_rows()
    S=np.full((len(rows),len(labels)),np.nan); T=np.full((len(rows),len(labels)),np.nan)
    for ri,(ds,r) in enumerate(rows):
        for ci,m in enumerate(labels):
            if (ds,r,m) in rankstats: sp,ov=rankstats[(ds,r,m)]; S[ri,ci]=sp; T[ri,ci]=ov
    fig,axR=plt.subplots(1,2,figsize=(max(8,1.1*len(labels)+5),11))
    cg=LinearSegmentedColormap.from_list('rg',['#b2182b','#f7f7c0','#1a9850']); cg.set_bad('white')
    imS=axR[0].imshow(np.ma.masked_invalid(S),cmap=cg,vmin=0,vmax=1,aspect='auto')
    axR[0].set_title('Spearman rank corr ( approx , inv )\n(1=same order, ~0=unrelated)',fontsize=11)
    imT=axR[1].imshow(np.ma.masked_invalid(T),cmap=cg,vmin=0,vmax=1,aspect='auto')
    axR[1].set_title('top-5% overlap with inv\n(1=identical, ~0.05=random)',fontsize=11)
    for ax,Mat,im in [(axR[0],S,imS),(axR[1],T,imT)]:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels([m.replace('λ=','\nλ=') for m in labels],fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rlab,fontsize=7.5)
        for ri in range(len(rows)):
            for ci in range(len(labels)):
                v=Mat[ri,ci]
                if not np.isnan(v): ax.text(ci,ri,f'{v:.2f}',ha='center',va='center',fontsize=6.2)
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.suptitle(suptitle,fontsize=12); fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outpath,dpi=130,bbox_inches='tight'); plt.close(fig); print('saved:',outpath)

def fig_absdiff(methods, stats, outpath, suptitle):
    labels=[l for l,_ in methods]; col=colors_for(labels); nr,nc=len(DS),len(RANKS)
    fig,ax3=plt.subplots(nr,nc,figsize=(6.0*nc,3.0*nr),squeeze=False)
    for i,(ds,N,V) in enumerate(DS):
        for j,r in enumerate(RANKS):
            ax=ax3[i][j]; ms=[m for m in labels if (ds,r,m) in stats]
            if not ms: ax.text(0.5,0.5,'(no data)',ha='center',va='center',transform=ax.transAxes,color='gray')
            else:
                xs=np.arange(len(ms)); means=[stats[(ds,r,m)][0] for m in ms]; maxs=[stats[(ds,r,m)][1] for m in ms]
                ax.bar(xs,means,color=[col[m] for m in ms],alpha=0.85,zorder=2)
                ax.scatter(xs,maxs,color='k',marker='_',s=130,zorder=5)
                for x,mx in zip(xs,maxs): ax.plot([x,x],[1e-3,mx],color='k',lw=0.5,ls=':',zorder=1)
                ax.set_yscale('log'); ax.set_ylim(0.03,30); ax.axhline(1.0,color='crimson',ls='--',lw=0.7)
                ax.set_xticks(xs); ax.set_xticklabels([m.replace('eigen ','eig ') for m in ms],rotation=45,ha='right',fontsize=7)
            ax.set_title(f'{ds} (num{N}) — rank {r}%',fontsize=10)
            ax.set_ylabel('|approx−inv| (log)',fontsize=8); ax.tick_params(labelsize=7.5)
    fig.suptitle(suptitle,y=1.012,fontsize=12); fig.tight_layout(rect=[0,0,1,0.985])
    fig.savefig(outpath,dpi=130,bbox_inches='tight'); plt.close(fig); print('saved:',outpath)

def fig_errordist(methods, diffs, outpath, suptitle):
    labels=[l for l,_ in methods]; col=colors_for(labels); XLIM=1.0; nr,nc=len(DS),len(RANKS)
    fig,ax2=plt.subplots(nr,nc,figsize=(6.0*nc,3.0*nr),squeeze=False); hands={}; bins=np.linspace(-XLIM,XLIM,61)
    for i,(ds,N,V) in enumerate(DS):
        for j,r in enumerate(RANKS):
            ax=ax2[i][j]; res=diffs.get((ds,r),{})
            if not res: ax.text(0.5,0.5,'(no data)',ha='center',va='center',transform=ax.transAxes,color='gray')
            clipped=0; tot=0
            for m in labels:
                if m in res:
                    d=res[m]; clipped+=int((np.abs(d)>XLIM).sum()); tot+=d.size
                    lw=2.0 if m.startswith('eigen') else 1.3
                    ax.hist(np.clip(d,-XLIM,XLIM),bins=bins,histtype='step',color=col[m],lw=lw,density=True)
                    hands.setdefault(m,plt.Line2D([0],[0],color=col[m],lw=lw))
            ax.axvline(0,color='gray',ls=':',lw=0.8); ax.set_title(f'{ds} (num{N}) — rank {r}%',fontsize=10)
            if tot: ax.text(0.02,0.97,f'{100*clipped/tot:.0f}% |err|>{XLIM}',transform=ax.transAxes,va='top',fontsize=6.5,color='dimgray')
            ax.set_xlabel('error (approx − inv)',fontsize=8); ax.set_ylabel('density',fontsize=8); ax.tick_params(labelsize=7.5)
    H=[hands[k] for k in labels if k in hands]; L=[k for k in labels if k in hands]
    if H: fig.legend(H,L,loc='upper center',ncol=min(7,len(L)),fontsize=9,bbox_to_anchor=(0.5,0.999))
    fig.suptitle(suptitle,y=1.012,fontsize=12); fig.tight_layout(rect=[0,0,1,0.978])
    fig.savefig(outpath,dpi=130,bbox_inches='tight'); plt.close(fig); print('saved:',outpath)

def fig_matched_scatter(outpath):
    fig,axs=plt.subplots(2,3,figsize=(15,9.5)); LIM=3.0
    for idx,lam in enumerate(LAMS):
        ax=axs[idx//3][idx%3]; any_pt=False
        for method,color,nm in [(eig_method(lam),'#1a1a1a','eigen'),(nys_method(lam),'#d62728','nys')]:
            label,fn=method; xs=[]; ys=[]
            for ds,N,V in DS:
                ip=inv_path(ds,N,V)
                if not ip: continue
                isv,isi=load_sv(ip)
                for r in RANKS:
                    p=fn(ds,N,V,r)
                    if not p: continue
                    asv,asi=load_sv(p); a,b=aligned(asv,asi,isv,isi)
                    xs.extend(b.tolist()); ys.extend(a.tolist())
            if xs:
                any_pt=True
                ax.scatter(np.clip(xs,-LIM,LIM),np.clip(ys,-LIM,LIM),s=4,alpha=0.25,color=color,label=nm)
        ax.plot([-LIM,LIM],[-LIM,LIM],color='gray',ls='--',lw=0.8)
        ax.set_xlim(-LIM,LIM); ax.set_ylim(-LIM,LIM); ax.set_title(f'λ = {disp(lam)}',fontsize=11)
        ax.set_xlabel('inv sv',fontsize=8); ax.set_ylabel('approx sv',fontsize=8); ax.tick_params(labelsize=7)
        if not any_pt: ax.text(0.5,0.5,'(no data)',ha='center',va='center',transform=ax.transAxes,color='gray')
        else: ax.legend(fontsize=8,markerscale=2,loc='upper left')
    fig.suptitle('Matched-λ: eigen λ=X vs nys λ=X  (on y=x = faithful to inv; pooled over ds×rank, clipped ±3)',fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(outpath,dpi=130,bbox_inches='tight'); plt.close(fig); print('saved:',outpath)

def fig_errordist_matched():
    """세팅4(matched) 오차분포: 데이터셋마다 (행=λ, 열=rank), 각 칸에 eig-λ vs nys-λ 히스토그램.
    파일: errordist_s4_{ds}.png"""
    XLIM=1.0; bins=np.linspace(-XLIM,XLIM,61); nr,nc=len(LAMS),len(RANKS)
    for ds,N,V in DS:
        ip=inv_path(ds,N,V); isv,isi=(load_sv(ip) if ip else (None,None))
        fig,ax=plt.subplots(nr,nc,figsize=(5.2*nc,2.5*nr),squeeze=False)
        for i,lam in enumerate(LAMS):
            for j,r in enumerate(RANKS):
                a=ax[i][j]; plotted=False
                if isv is not None:
                    for kind,pth,color in [('eigen',eig_path(ds,N,V,r,lam),'#1a1a1a'),
                                           ('nys',  nys_path(ds,N,V,r,lam),'#d62728')]:
                        if not pth: continue
                        asv,asi=load_sv(pth); x,b=aligned(asv,asi,isv,isi)
                        if len(x)==0: continue
                        d=x-b
                        a.hist(np.clip(d,-XLIM,XLIM),bins=bins,histtype='step',color=color,
                               lw=2.0 if kind=='eigen' else 1.5,density=True,label=f'{kind} λ={disp(lam)}')
                        plotted=True
                a.axvline(0,color='gray',ls=':',lw=0.8)
                a.set_title(f'λ={disp(lam)} — rank {r}%',fontsize=9)
                a.set_xlabel('error (approx-inv)',fontsize=7); a.tick_params(labelsize=6.5)
                if plotted: a.legend(fontsize=7)
                else: a.text(0.5,0.5,'(no data)',ha='center',va='center',transform=a.transAxes,color='gray')
        fig.suptitle(f'[matched err-dist] {ds} (num{N}) — eigen vs nystrom at same lambda (row=lambda, col=rank; clip +-1)',fontsize=12)
        fig.tight_layout(rect=[0,0,1,0.99])
        out=f'{OUT}/errordist_s4_{ds}.png'; fig.savefig(out,dpi=130,bbox_inches='tight'); plt.close(fig); print('saved:',out)

# 4 comparison settings (shared concept with the acc-curve script plot_eig_nys.py)
SETTINGS = [
    ('s1', '1) eigen lam=1e-2 vs nystrom lam=1e-1..1e-6',
        [eig_method('1e-02')] + [nys_method(l) for l in LAMS]),
    ('s2', '2) eigen internal lam=1e-1..1e-6',
        [eig_method(l) for l in LAMS]),
    ('s3', '3) nystrom internal lam=1e-1..1e-6',
        [nys_method(l) for l in LAMS]),
    ('s4', '4) matched: eigen lam vs nystrom lam',
        [m for l in reversed(LAMS) for m in (eig_method(l), nys_method(l))]),
]

if __name__=='__main__':
    for tag,title,methods in SETTINGS:
        diffs,stats,rankstats=compute(methods)
        fig_heatmap (methods, stats,     f'{OUT}/heatmap_{tag}.png',  f'[{title}] value fidelity to inv')
        fig_absdiff (methods, stats,     f'{OUT}/absdiff_{tag}.png',  f'[{title}] |approx-inv|: bar=mean, tick=max (log y)')
        if tag!='s4':   # s4 오차분포는 데이터셋별 matched로 따로 생성
            fig_errordist(methods, diffs, f'{OUT}/errordist_{tag}.png',f'[{title}] per-point error (approx-inv), clipped +-1')
        fig_ranking (methods, rankstats, f'{OUT}/ranking_{tag}.png',  f'[{title}] ranking preservation (Spearman / top-5%)')
    fig_errordist_matched()   # errordist_s4_{ds}.png (행=λ, 열=rank, eig vs nys)
    print('\nALL FIGURES IN:', OUT)
