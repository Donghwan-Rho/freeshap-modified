# -*- coding: utf-8 -*-
"""kernel_vs_sv_report.pdf — Kernel approx 은 잘 되는데 SV 는 왜 망가지나?

세 축:
  (a) K approximation 자체는 monotone in rank/d 인가?     ← Yes for both 예상
  (b) SV L2/Spearman error 는 어떻게?                     ← eigen 은 monotone, nys 는 non-monotone 예상
  (c) 그리고 conditioning tracking:  κ(W_d)  vs  d          ← nys 의 d↑ 시 발산 확인

Data:
  NTK  : freeshap_res/ntk/{ds}/bert_seed2024_num{N}_val{V}_signFalse.pkl
  SV   : freeshap_res/shapley/{ds}/{inv,eigen,nystrom}/*.pkl
  Ref  : inv SV (λ=1e-6) as ground truth
"""
import os, glob, pickle, time, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

# 한글 폰트
import matplotlib.font_manager as fm
_fp = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
if os.path.exists(_fp):
    fm.fontManager.addfont(_fp)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ========== 경로 ==========
VINFO = "/extdata1/donghwan/freeshap/vinfo"
NTK   = f"{VINFO}/freeshap_res/ntk"
SH    = f"{VINFO}/freeshap_res/shapley"
OUT   = f"{VINFO}/kernel_vs_sv.pdf"

DS = [('sst2',5000,872), ('mnli',5000,1000), ('ag_news',5000,1000),
      ('mr',5000,1000), ('qqp',5000,1000), ('rte',2490,277), ('mrpc',3668,408)]
RANK_PCTS_SV     = [1, 5, 10, 15, 20, 25, 30]   # SV pkl 이 존재하는 rank
RANK_PCTS_KERNEL = [1, 5, 10, 15, 20, 25, 30]   # kernel-only monotone 체크
LAMS = ['1e-01','1e-02','1e-03','1e-04','1e-05','1e-06']
NYS_LANDMARK_SEED = 1234              # freeshap 하드코딩

# ========== 유틸 ==========
def load_ntk_K(ds, N, V):
    """Return K_trtr (n_train, n_train) as np.float64 symmetric."""
    p = f"{NTK}/{ds}/bert_seed2024_num{N}_val{V}_signFalse.pkl"
    if not os.path.exists(p): return None
    import torch
    with open(p,'rb') as f: d = pickle.load(f)
    K = d['ntk']
    if K.ndim == 3: K = K[0]
    K = K.to("cpu", dtype=torch.float64).numpy()   # (N+V, N)
    K = K[:N, :N]
    K = 0.5*(K + K.T)
    return K

def load_sv(path):
    if not os.path.exists(path): return None, None
    d = pickle.load(open(path,'rb'))
    dv = np.array(d['dv_result'])
    si = np.array(d['sampled_idx'])
    return dv[:,1,:].sum(axis=1), si

def align(svA,siA,svB,siB):
    dA=dict(zip(siA.tolist(),svA)); dB=dict(zip(siB.tolist(),svB))
    common=sorted(set(siA.tolist())&set(siB.tolist()))
    return (np.array([dA[i] for i in common]), np.array([dB[i] for i in common]))

def _rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
def spearman(a,b):
    return np.corrcoef(_rank(a),_rank(b))[0,1] if len(a)>1 else np.nan

TAIL = 'signFalse_earlystopTrue_tmc500.pkl'
def inv_path(ds,N,V):
    return f"{SH}/{ds}/inv/bert_seed2024_num{N}_val{V}_lam1e-06_{TAIL}"
def eig_path(ds,N,V,r,lam):
    return f"{SH}/{ds}/eigen/bert_seed2024_num{N}_val{V}_eig{r}.0_eiglam{lam}_invlam1e-06_cholesky_float32_{TAIL}"
def nys_path(ds,N,V,r,lam):
    return f"{SH}/{ds}/nystrom/bert_seed2024_num{N}_val{V}_nys{r}.0_nyslam{lam}_invlam1e-06_cholesky_float32_{TAIL}"

# ========== 커널 근사 ==========
def eigen_approx(evals, evecs, r):
    """K_eigen = U_r Λ_r U_r^T"""
    U = evecs[:, -r:]
    L = evals[-r:]
    return (U * L) @ U.T

def nystrom_approx(K, d, seed=NYS_LANDMARK_SEED, jitter=1e-8):
    """K_nys = C W^{-1} C^T using landmark seed 1234 (freeshap default)."""
    n = K.shape[0]
    rng = np.random.RandomState(seed)
    S = np.sort(rng.choice(n, size=d, replace=False))
    W = K[np.ix_(S, S)]
    C = K[:, S]
    W_reg = W + jitter * np.eye(d)
    try:
        L = np.linalg.cholesky(W_reg)
    except np.linalg.LinAlgError:
        # fallback: pseudoinverse
        Winv = np.linalg.pinv(W_reg)
        return C @ Winv @ C.T, np.linalg.cond(W_reg)
    Winv_CT = np.linalg.solve(L.T, np.linalg.solve(L, C.T))
    K_nys = C @ Winv_CT
    return K_nys, np.linalg.cond(W_reg)

def frob_rel(A, B):
    return np.linalg.norm(A - B, 'fro') / np.linalg.norm(A, 'fro')

# ========== 메인 계산 ==========
def compute_all():
    """
    Returns records[(ds, method, rank_pct, lam)] = dict with:
      kernel_err (Frobenius rel), sv_err_L2, sv_spearman, sv_topk_ov,
      cond_number (nys 만), N
    """
    records = {}
    for ds, N, V in DS:
        t0 = time.time()
        print(f"\n=== {ds} (n={N}) ===")
        K = load_ntk_K(ds, N, V)
        if K is None: print("  NTK missing, skip"); continue

        # Full eigendecomp once (used for both eigen approx and reference)
        evals, evecs = np.linalg.eigh(K)   # ascending
        print(f"  eigh: {time.time()-t0:.1f}s, min λ={evals[0]:.3e}, max λ={evals[-1]:.3e}")

        # -------- kernel error (both eigen and nys, all rank_pct) --------
        # Kernel-only monotone 체크.  Eigen conditioning at reg λ=1e-2 (freeshap default).
        LAM_REG = 1e-2
        for pct in RANK_PCTS_KERNEL:
            r = max(1, int(N * pct / 100))
            # eigen approx
            top_evals = evals[-r:]
            K_e = eigen_approx(evals, evecs, r)
            e_e = frob_rel(K, K_e)
            cond_eig = (top_evals[-1] + LAM_REG) / (top_evals[0] + LAM_REG)
            # nys approx
            K_n, cond_nys = nystrom_approx(K, r)
            e_n = frob_rel(K, K_n)
            records[(ds, 'eigen', pct, 'kernel_only')] = dict(kernel_err=e_e, cond=cond_eig, r=r)
            records[(ds, 'nys',   pct, 'kernel_only')] = dict(kernel_err=e_n, cond=cond_nys, r=r)
            if pct in [1, 5, 20, 30]:
                print(f"  pct={pct:>2d}%: eig_err={e_e:.4f} nys_err={e_n:.4f} | "
                      f"cond_eig(Λ+λ)={cond_eig:.2e} cond_nys(W)={cond_nys:.2e}")

        # -------- SV 실측 (rank_pct ∈ [1,5,20] 만) --------
        isv, isi = load_sv(inv_path(ds, N, V))
        if isv is None: print("  inv SV missing"); continue

        for pct in RANK_PCTS_SV:
            for lam in LAMS:
                r = max(1, int(N * pct / 100))
                for method, path_fn in [('eigen', eig_path), ('nys', nys_path)]:
                    p = path_fn(ds, N, V, pct, lam)
                    asv, asi = load_sv(p)
                    if asv is None: continue
                    a, b = align(asv, asi, isv, isi)
                    if len(a) == 0: continue
                    sv_l2   = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
                    sv_sp   = spearman(a, b)
                    n_top = max(1, int(len(a)*0.05))
                    top_a = set(np.argsort(a)[-n_top:].tolist())
                    top_b = set(np.argsort(b)[-n_top:].tolist())
                    sv_top5 = len(top_a & top_b) / n_top
                    bot_a = set(np.argsort(a)[:n_top].tolist())
                    bot_b = set(np.argsort(b)[:n_top].tolist())
                    sv_bot5 = len(bot_a & bot_b) / n_top

                    # 그리고 kernel err 도 다시 (동일 rank_pct)
                    if method == 'eigen':
                        ke = frob_rel(K, eigen_approx(evals, evecs, r))
                        cn = None
                    else:
                        K_n, cn = nystrom_approx(K, r)
                        ke = frob_rel(K, K_n)

                    key = (ds, method, pct, lam)
                    records[key] = dict(kernel_err=ke, cond=cn,
                                        sv_err_L2=sv_l2, sv_spearman=sv_sp,
                                        sv_top5_ov=sv_top5, sv_bot5_ov=sv_bot5,
                                        r=r)
        print(f"  {ds} done in {time.time()-t0:.1f}s")
    return records

# ========== 시각화 ==========
def fig_scatter_kernel_vs_sv(recs, ax_metric='spearman'):
    """X = kernel Frob err, Y = 1 - spearman (or SV L2 err).  points colored by dataset, marker by method."""
    fig, ax = plt.subplots(figsize=(10, 7))
    color_map = plt.cm.tab10(np.linspace(0,1,len(DS)))
    for i,(ds,N,V) in enumerate(DS):
        for method, marker in [('eigen','o'), ('nys','^')]:
            xs, ys = [], []
            for pct in RANK_PCTS_SV:
                for lam in LAMS:
                    key = (ds, method, pct, lam)
                    if key not in recs: continue
                    r = recs[key]
                    if 'sv_spearman' not in r: continue
                    xs.append(r['kernel_err'])
                    if ax_metric == 'spearman':
                        ys.append(1 - r['sv_spearman'])
                    else:
                        ys.append(r['sv_err_L2'])
            if xs:
                ax.scatter(xs, ys, marker=marker, s=45, alpha=0.7,
                           edgecolor=color_map[i], facecolor=color_map[i] if marker=='o' else 'none',
                           linewidth=1.2, label=f'{ds}-{method}' if pct==1 else None)
    # 각주
    ax.set_xlabel('Kernel approx error  ‖K − K_approx‖_F / ‖K‖_F', fontsize=11)
    ax.set_ylabel('SV ranking error  1 − Spearman ρ(SV, SV_full)' if ax_metric=='spearman' else 'SV L2 rel error', fontsize=11)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    # 범례 커스터마이즈
    handles = [plt.Line2D([0],[0], marker='o', color='k', linestyle='', label='eigen (filled)'),
               plt.Line2D([0],[0], marker='^', color='k', linestyle='', markerfacecolor='none', label='nys (hollow)')]
    for i,(ds,_,_) in enumerate(DS):
        handles.append(plt.Line2D([0],[0], marker='s', color=color_map[i], linestyle='', label=ds))
    ax.legend(handles=handles, fontsize=8, loc='upper left', ncol=2)
    ax.set_title(f'Kernel approx 은 잘 되는데 SV 는 어떻게? \n'
                 f'(각 점 = (dataset × rank ∈ [1%, 5%, 20%] × λ 6개))\n'
                 f'이론: 왼쪽 아래 = 좋음. eigen 이 nys 보다 왼쪽 아래에 몰려야 함.',
                 fontsize=11)
    return fig

def fig_monotone_kernel(recs):
    """Line plot: kernel err vs rank_pct.  eigen, nys 각 dataset 별."""
    nrows = 2; ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 8), sharex=True)
    axs = axs.flatten()
    for i,(ds,N,V) in enumerate(DS):
        ax = axs[i]
        for method, color, style in [('eigen','C0','-o'), ('nys','C3','-^')]:
            xs, ys = [], []
            for pct in RANK_PCTS_KERNEL:
                key = (ds, method, pct, 'kernel_only')
                if key in recs:
                    xs.append(pct); ys.append(recs[key]['kernel_err'])
            ax.plot(xs, ys, style, color=color, lw=1.5, label=method, markersize=6)
        ax.set_xlabel('rank / d  (% of n_train)', fontsize=9)
        ax.set_ylabel('‖K − K_approx‖_F / ‖K‖_F', fontsize=9)
        ax.set_yscale('log')
        ax.set_title(f'{ds} (n={N})', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=8)
    # 남는 subplot 지우기
    for j in range(len(DS), nrows*ncols):
        axs[j].axis('off')
    fig.suptitle('(a) Kernel approximation error vs rank/d — 둘 다 monotone 감소인가?',
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig

def fig_monotone_sv(recs):
    """Line plot: 1 - Spearman vs rank_pct at fixed λ=1e-2. eigen vs nys per dataset."""
    LAM_FIX = '1e-02'
    nrows = 2; ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 8), sharex=True)
    axs = axs.flatten()
    for i,(ds,N,V) in enumerate(DS):
        ax = axs[i]
        for method, color, style in [('eigen','C0','-o'), ('nys','C3','-^')]:
            xs, ys = [], []
            for pct in RANK_PCTS_SV:
                key = (ds, method, pct, LAM_FIX)
                if key in recs and 'sv_spearman' in recs[key]:
                    xs.append(pct); ys.append(1 - recs[key]['sv_spearman'])
            if xs:
                ax.plot(xs, ys, style, color=color, lw=1.5, label=method, markersize=8)
        ax.set_xlabel('rank / d  (% of n_train)', fontsize=9)
        ax.set_ylabel('1 − Spearman ρ(SV, SV_full)', fontsize=9)
        ax.set_yscale('log')
        ax.set_title(f'{ds} (n={N})', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=8)
    for j in range(len(DS), nrows*ncols):
        axs[j].axis('off')
    fig.suptitle(f'(b) SV ranking error vs rank/d at λ={LAM_FIX} — eigen 은 monotone, nys 는?',
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig

def fig_conditioning(recs):
    """κ vs rank/d for both eigen and nys, so gap is directly comparable."""
    nrows = 2; ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 8), sharex=True)
    axs = axs.flatten()
    for i,(ds,N,V) in enumerate(DS):
        ax = axs[i]
        for method, color, style, label in [
                ('eigen','C0','-o','eigen: κ(Λ_r+λI), λ=1e-2'),
                ('nys',  'C3','-^','nys:   κ(W_d + 1e-8·I)')]:
            xs, ys = [], []
            for pct in RANK_PCTS_KERNEL:
                key = (ds, method, pct, 'kernel_only')
                if key in recs and recs[key].get('cond') is not None:
                    xs.append(pct); ys.append(recs[key]['cond'])
            if xs:
                ax.plot(xs, ys, style, color=color, lw=1.5, markersize=6, label=label)
        ax.set_xlabel('rank / d  (% of n_train)', fontsize=9)
        ax.set_ylabel('condition number', fontsize=9)
        ax.set_yscale('log')
        ax.set_title(f'{ds} (n={N})', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
    for j in range(len(DS), nrows*ncols):
        axs[j].axis('off')
    fig.suptitle('(c) Conditioning of the matrix that gets inverted\n'
                 'eigen: (Λ_r + λI)  — λ가 명시적 protection.  nys: (W_d + 1e-8·I)  — jitter만.\n'
                 'κ 커질수록 SV 계산 시 오차 폭발.',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def fig_amplification_bar(recs):
    """K_err 는 비슷한데 SV_err 은 얼마나 배 증폭되는가?  Ratio bar per dataset × pct."""
    LAM_FIX = '1e-02'
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ai, pct in enumerate([1, 5]):
        ax = axs[ai]
        labels, e_kerr, e_sverr, n_kerr, n_sverr = [], [], [], [], []
        for ds, N, V in DS:
            k_e = recs.get((ds,'eigen',pct,LAM_FIX),{}).get('kernel_err', np.nan)
            k_n = recs.get((ds,'nys',pct,LAM_FIX),{}).get('kernel_err', np.nan)
            s_e = recs.get((ds,'eigen',pct,LAM_FIX),{}).get('sv_err_L2', np.nan)
            s_n = recs.get((ds,'nys',pct,LAM_FIX),{}).get('sv_err_L2', np.nan)
            labels.append(ds)
            e_kerr.append(k_e); n_kerr.append(k_n)
            e_sverr.append(s_e); n_sverr.append(s_n)
        x = np.arange(len(labels))
        w = 0.2
        ax.bar(x - 1.5*w, e_kerr, w, label='eigen kernel', color='C0', alpha=0.5)
        ax.bar(x - 0.5*w, e_sverr, w, label='eigen SV', color='C0', alpha=1.0)
        ax.bar(x + 0.5*w, n_kerr, w, label='nys kernel', color='C3', alpha=0.5)
        ax.bar(x + 1.5*w, n_sverr, w, label='nys SV', color='C3', alpha=1.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
        ax.set_yscale('log')
        ax.set_ylabel('rel error  (kernel Frob / SV L2)', fontsize=10)
        ax.set_title(f'rank/d = {pct}%, λ = {LAM_FIX}', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle('Kernel error 는 비슷한데 SV error 는 왜 이렇게 다른가?',
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ========== 실행 ==========
if __name__ == '__main__':
    print("[compute] start")
    recs = compute_all()
    print(f"\n[compute] {len(recs)} records")

    with PdfPages(OUT) as pdf:
        for f in [fig_scatter_kernel_vs_sv(recs, 'spearman'),
                  fig_amplification_bar(recs),
                  fig_monotone_kernel(recs),
                  fig_monotone_sv(recs),
                  fig_conditioning(recs)]:
            pdf.savefig(f, bbox_inches='tight')
            plt.close(f)

    # 저장: 표 요약
    print("\n[summary] eigen vs nys at rank_pct=1%, λ=1e-2:")
    print(f"{'ds':<10s}  {'k_e':>8s} {'k_n':>8s}  {'sv_e':>8s} {'sv_n':>8s}  {'spr_e':>7s} {'spr_n':>7s}  {'cond_n':>10s}")
    for ds,N,V in DS:
        r_e = recs.get((ds,'eigen',1,'1e-02'), {})
        r_n = recs.get((ds,'nys',1,'1e-02'), {})
        if r_e and r_n:
            print(f"{ds:<10s}  {r_e.get('kernel_err',np.nan):8.4f} {r_n.get('kernel_err',np.nan):8.4f}  "
                  f"{r_e.get('sv_err_L2',np.nan):8.4f} {r_n.get('sv_err_L2',np.nan):8.4f}  "
                  f"{r_e.get('sv_spearman',np.nan):7.3f} {r_n.get('sv_spearman',np.nan):7.3f}  "
                  f"{r_n.get('cond',np.nan):10.2e}")

    print(f"\n[write] {OUT}")
