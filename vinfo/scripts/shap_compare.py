# -*- coding: utf-8 -*-
"""Compare per-point Shapley VALUES: inv (reference) vs eigen / nystrom."""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SH = '/home/donghwan/freeshap/vinfo/freeshap_res/shapley'
DS = [('sst2',5000,872), ('mnli',5000,1000), ('ag_news',5000,1000), ('mr',5000,1000),
      ('qqp',5000,1000), ('rte',2490,277), ('mrpc',3668,408)]
RANKS = [1, 5, 20]
NYS_LAMS = ['1e-02','1e-03','1e-04','1e-05','1e-06']
TAIL = 'signFalse_earlystopTrue_tmc500.pkl'
def disp(l): return l.replace('e-0','e-')
METHODS = ['eigen λ=1e-2'] + [f'nys λ={disp(l)}' for l in NYS_LAMS]

def one(pat):
    g = sorted(glob.glob(pat)); return g[0] if g else None
def load_sv(path):
    d = pickle.load(open(path,'rb')); dv = np.array(d['dv_result']); si = np.array(d['sampled_idx'])
    return dv[:,1,:].sum(axis=1), si
def aligned(svA, siA, svB, siB):
    dA=dict(zip(siA.tolist(),svA)); dB=dict(zip(siB.tolist(),svB))
    common=sorted(set(siA.tolist())&set(siB.tolist()))
    return np.array([dA[i] for i in common]), np.array([dB[i] for i in common])
def _rank(x): r=np.empty(len(x)); r[np.argsort(x)]=np.arange(len(x)); return r
def spearman(a,b): return np.corrcoef(_rank(a),_rank(b))[0,1]
def topk_ov(a,b,k=5):
    n=len(a); kk=max(1,int(n*k/100))
    return len(set(np.argsort(a)[::-1][:kk].tolist())&set(np.argsort(b)[::-1][:kk].tolist()))/kk
def inv_path(ds,N,V): return one(f'{SH}/{ds}/inv/bert_seed2024_num{N}_val{V}_lam1e-06_{TAIL}')
def eig_path(ds,N,V,r): return one(f'{SH}/{ds}/eigen/bert_seed2024_num{N}_val{V}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{TAIL}')
def nys_path(ds,N,V,r,lam): return one(f'{SH}/{ds}/nystrom/bert_seed2024_num{N}_val{V}_nys{r}.0_nyslam{lam}_invlam1e-06_cholesky_float32_{TAIL}')

# diffs[(ds,r)][method] = approx_sv - inv_sv ; stats[(ds,r,method)] = (mean|d|, max|d|, corr, stdratio)
diffs={}; stats={}; rankstats={}   # rankstats[(ds,r,m)] = (spearman, top5_overlap)
for ds,N,V in DS:
    ip=inv_path(ds,N,V)
    if not ip: continue
    isv,isi=load_sv(ip); istd=isv.std()
    for r in RANKS:
        diffs[(ds,r)]={}
        cand=[('eigen λ=1e-2', eig_path(ds,N,V,r))]+[(f'nys λ={disp(l)}', nys_path(ds,N,V,r,l)) for l in NYS_LAMS]
        for m,p in cand:
            if not p: continue
            asv,asi=load_sv(p); a,b=aligned(asv,asi,isv,isi); d=a-b
            diffs[(ds,r)][m]=d
            stats[(ds,r,m)]=(np.abs(d).mean(), np.abs(d).max(), np.corrcoef(a,b)[0,1], a.std()/istd)
            rankstats[(ds,r,m)]=(spearman(a,b), topk_ov(a,b,5))

# ---------- printed table ----------
print(f"\n{'dataset':8s}{'rk':>4s} {'method':13s} {'mean|d|':>8s} {'max|d|':>8s} {'corr':>6s} {'std/inv':>7s}  flag")
print('-'*64)
for ds,N,V in DS:
    for r in RANKS:
        for m in METHODS:
            k=(ds,r,m)
            if k not in stats: continue
            me,mx,co,sr=stats[k]
            flag='EXPLODE' if (mx>3 or sr>2.5) else ''
            print(f"{ds:8s}{r:>3d}% {m:13s} {me:8.3f} {mx:8.2f} {co:6.2f} {sr:7.2f}  {flag}")

# ================= Figure 1: heatmaps (corr & mean|d|) =================
rows=[(ds,r) for ds,_,_ in DS for r in RANKS]
rlab=[f'{ds} r{r}%' for ds,r in rows]
C=np.full((len(rows),len(METHODS)),np.nan); M=np.full((len(rows),len(METHODS)),np.nan)
for ri,(ds,r) in enumerate(rows):
    for ci,m in enumerate(METHODS):
        if (ds,r,m) in stats:
            me,mx,co,sr=stats[(ds,r,m)]; C[ri,ci]=co; M[ri,ci]=me

fig,axs=plt.subplots(1,2,figsize=(13,11))
# corr heatmap (1 good -> green, 0 bad -> red)
cmap1=LinearSegmentedColormap.from_list('rg',['#b2182b','#f7f7c0','#1a9850'])
im0=axs[0].imshow(np.nan_to_num(C,nan=-0.05),cmap=cmap1,vmin=0,vmax=1,aspect='auto')
axs[0].set_title('correlation( approx_sv , inv_sv )\n(1=identical, ~0=unrelated)',fontsize=11)
# mean|d| heatmap (low good -> green); color cap at 0.5, annotate true
cmap2=LinearSegmentedColormap.from_list('gr',['#1a9850','#f7f7c0','#b2182b'])
im1=axs[1].imshow(np.nan_to_num(M,nan=99),cmap=cmap2,vmin=0,vmax=0.5,aspect='auto')
axs[1].set_title('mean | approx_sv − inv_sv |\n(color capped at 0.5; number=true)',fontsize=11)
for ax,Mat,im in [(axs[0],C,im0),(axs[1],M,im1)]:
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([m.replace('λ=','\nλ=') for m in METHODS],fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rlab,fontsize=7.5)
    for ri in range(len(rows)):
        for ci in range(len(METHODS)):
            v=Mat[ri,ci]
            if not np.isnan(v):
                ax.text(ci,ri,f'{v:.2f}',ha='center',va='center',fontsize=6.2,
                        color='black')
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
fig.suptitle('Shapley VALUE fidelity to inv (full eNTK):  eigen stays faithful, nystrom explodes at mid-rank / small λ',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97])
fp1='/home/donghwan/freeshap/vinfo/shap_value_fidelity_heatmap.png'
fig.savefig(fp1,dpi=130,bbox_inches='tight'); fig.savefig(fp1.replace('.png','.pdf'),bbox_inches='tight')
print('\nsaved:',fp1)

# ================= Figure 2: error distribution (x clipped) =================
eig_color='black'; nys_colors={'1e-02':'#d62728','1e-03':'#ff7f0e','1e-04':'#2ca02c','1e-05':'#1f77b4','1e-06':'#9467bd'}
color_of={'eigen λ=1e-2':eig_color, **{f'nys λ={disp(l)}':nys_colors[l] for l in NYS_LAMS}}
XLIM=1.0
nr,nc=len(DS),len(RANKS)
fig2,ax2=plt.subplots(nr,nc,figsize=(6.0*nc,3.0*nr),squeeze=False); hands={}
bins=np.linspace(-XLIM,XLIM,61)
for i,(ds,N,V) in enumerate(DS):
    for j,r in enumerate(RANKS):
        ax=ax2[i][j]; res=diffs.get((ds,r),{})
        if not res: ax.text(0.5,0.5,'(no data)',ha='center',va='center',transform=ax.transAxes,color='gray')
        clipped_any=0; tot=0
        for m in METHODS:
            if m in res:
                d=res[m]; clipped_any+=int((np.abs(d)>XLIM).sum()); tot+=d.size
                lw=2.0 if m.startswith('eigen') else 1.3
                ax.hist(np.clip(d,-XLIM,XLIM),bins=bins,histtype='step',color=color_of[m],lw=lw,density=True)
                hands.setdefault(m,plt.Line2D([0],[0],color=color_of[m],lw=lw))
        ax.axvline(0,color='gray',ls=':',lw=0.8)
        ax.set_title(f'{ds} (num{N}) — rank {r}%',fontsize=10)
        if tot: ax.text(0.02,0.97,f'{100*clipped_any/tot:.0f}% |err|>{XLIM}\n(clipped→edges)',transform=ax.transAxes,va='top',fontsize=6.5,color='dimgray')
        ax.set_xlabel('Shapley value error (approx − inv)',fontsize=8); ax.set_ylabel('density',fontsize=8); ax.tick_params(labelsize=7.5)
H=[hands[k] for k in METHODS if k in hands]; L=[k for k in METHODS if k in hands]
fig2.legend(H,L,loc='upper center',ncol=6,fontsize=10,bbox_to_anchor=(0.5,0.999))
fig2.suptitle(f'Per-point Shapley value error vs inv  (x clipped to ±{XLIM}; exploded mass piles at edges)',y=1.012,fontsize=12)
fig2.tight_layout(rect=[0,0,1,0.978])
fp2='/home/donghwan/freeshap/vinfo/shap_value_error_dist.png'
fig2.savefig(fp2,dpi=130,bbox_inches='tight'); fig2.savefig(fp2.replace('.png','.pdf'),bbox_inches='tight')
print('saved:',fp2)

# ================= Figure 3: mean|d| bars (log y) + max|d| tick =================
fig3,ax3=plt.subplots(nr,nc,figsize=(6.0*nc,3.0*nr),squeeze=False)
for i,(ds,N,V) in enumerate(DS):
    for j,r in enumerate(RANKS):
        ax=ax3[i][j]; ms=[m for m in METHODS if (ds,r,m) in stats]
        if not ms:
            ax.text(0.5,0.5,'(no data)',ha='center',va='center',transform=ax.transAxes,color='gray')
        else:
            xs=np.arange(len(ms))
            means=[stats[(ds,r,m)][0] for m in ms]; maxs=[stats[(ds,r,m)][1] for m in ms]
            ax.bar(xs,means,color=[color_of[m] for m in ms],alpha=0.85,zorder=2)
            ax.scatter(xs,maxs,color='k',marker='_',s=130,zorder=5)
            for x,m in zip(xs,maxs): ax.plot([x,x],[1e-3,m],color='k',lw=0.5,ls=':',zorder=1)
            ax.set_yscale('log'); ax.set_ylim(0.03,30)
            ax.axhline(1.0,color='crimson',ls='--',lw=0.7)  # ~정상 sv 스케일
            ax.set_xticks(xs); ax.set_xticklabels([m.replace('eigen λ=1e-2','eigen').replace('nys λ=','nys ') for m in ms],rotation=45,ha='right',fontsize=7)
        ax.set_title(f'{ds} (num{N}) — rank {r}%',fontsize=10)
        ax.set_ylabel('|approx−inv|  (log)',fontsize=8); ax.tick_params(labelsize=7.5)
fig3.suptitle('Per-point |Shapley value − inv|:  bar=mean, tick=max  (log y; red line ~ normal sv scale)',y=1.012,fontsize=12)
fig3.tight_layout(rect=[0,0,1,0.985])
fp3='/home/donghwan/freeshap/vinfo/shap_value_absdiff_summary.png'
fig3.savefig(fp3,dpi=130,bbox_inches='tight'); fig3.savefig(fp3.replace('.png','.pdf'),bbox_inches='tight')
print('saved:',fp3)

# ================= Figure 4: RANKING heatmaps (Spearman & top-5% overlap) =================
S=np.full((len(rows),len(METHODS)),np.nan); T=np.full((len(rows),len(METHODS)),np.nan)
for ri,(ds,r) in enumerate(rows):
    for ci,m in enumerate(METHODS):
        if (ds,r,m) in rankstats:
            sp,ov=rankstats[(ds,r,m)]; S[ri,ci]=sp; T[ri,ci]=ov
figR,axR=plt.subplots(1,2,figsize=(13,11))
cmapG=LinearSegmentedColormap.from_list('rg',['#b2182b','#f7f7c0','#1a9850'])
imS=axR[0].imshow(np.nan_to_num(S,nan=-0.05),cmap=cmapG,vmin=0,vmax=1,aspect='auto')
axR[0].set_title('Spearman rank correlation ( approx , inv )\n(1 = same order, ~0 = unrelated)',fontsize=11)
imT=axR[1].imshow(np.nan_to_num(T,nan=-0.05),cmap=cmapG,vmin=0,vmax=1,aspect='auto')
axR[1].set_title('top-5% set overlap with inv\n(1 = identical, ~0.05 = random)',fontsize=11)
for ax,Mat,im in [(axR[0],S,imS),(axR[1],T,imT)]:
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([m.replace('λ=','\nλ=') for m in METHODS],fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rlab,fontsize=7.5)
    for ri in range(len(rows)):
        for ci in range(len(METHODS)):
            v=Mat[ri,ci]
            if not np.isnan(v): ax.text(ci,ri,f'{v:.2f}',ha='center',va='center',fontsize=6.2)
    figR.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
figR.suptitle('Shapley RANKING preservation vs inv:  eigen preserves order; nystrom order collapses where values explode',fontsize=12)
figR.tight_layout(rect=[0,0,1,0.97])
fpR='/home/donghwan/freeshap/vinfo/shap_value_ranking_heatmap.png'
figR.savefig(fpR,dpi=130,bbox_inches='tight'); figR.savefig(fpR.replace('.png','.pdf'),bbox_inches='tight')
print('saved:',fpR)
