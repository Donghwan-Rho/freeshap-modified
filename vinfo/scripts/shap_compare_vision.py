# -*- coding: utf-8 -*-
"""VISION (cifar10) per-point Shapley VALUE comparison: inv (full eNTK) vs eigen / nystrom.

NLP report swept lambda; the vision experiment (n01_vision.sh) instead sweeps the
rank/d at a FIXED lambda=1e-2:
    eigen rank r in {1,5,10,15,20,25,30}%,  nystrom d in {1,5,10,15,20,25,30}%.
So here the "methods" axis = the rank/d sweep, and rows = seeds (cifar10 is the
only dataset). inv = full eNTK reference (per seed).

Reads shapley pkl (server-local; gitignored). Missing files -> blank cell.
Figures -> <vinfo>/report_figs_vision/<kind>.png
"""
import os, glob, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

VINFO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH    = os.path.join(VINFO, 'freeshap_res', 'shapley', 'cifar10')
OUT   = os.path.join(VINFO, 'report_figs_vision'); os.makedirs(OUT, exist_ok=True)

SEEDS = [2024, 2025, 2026]           # 2026 not run yet -> blanks
RANKS = [1, 5, 10, 15, 20, 25, 30]   # eigen rank % == nystrom d %
NUM   = 5000; VAL = 1000
TAIL  = 'signFalse_earlystopTrue_tmc500.pkl'

def one(pat):
    g = sorted(glob.glob(pat)); return g[0] if g else None
def inv_path(s):    return one(f'{SH}/inv/resnet_seed{s}_num{NUM}_val{VAL}_lam1e-02_{TAIL}')
def eig_path(s, r): return one(f'{SH}/eigen/resnet_seed{s}_num{NUM}_val{VAL}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{TAIL}')
def nys_path(s, r): return one(f'{SH}/nystrom/resnet_seed{s}_num{NUM}_val{VAL}_nys{r}.0_nyslam1e-02_invlam1e-06_cholesky_float32_{TAIL}')

def load_sv(path):
    d = pickle.load(open(path, 'rb')); dv = np.array(d['dv_result']); si = np.array(d['sampled_idx'])
    return dv[:, 1, :].sum(axis=1), si
def aligned(svA, siA, svB, siB):
    dA = dict(zip(siA.tolist(), svA)); dB = dict(zip(siB.tolist(), svB))
    common = sorted(set(siA.tolist()) & set(siB.tolist()))
    return np.array([dA[i] for i in common]), np.array([dB[i] for i in common])
def _rank(x): r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
def spearman(a, b): return np.corrcoef(_rank(a), _rank(b))[0, 1] if len(a) > 1 else np.nan
def topk_ov(a, b, k=5):
    n = len(a); kk = max(1, int(n * k / 100))
    return len(set(np.argsort(a)[::-1][:kk].tolist()) & set(np.argsort(b)[::-1][:kk].tolist())) / kk

# method columns: eigen r1..r30 then nystrom d1..d30
def eig_label(r): return f'eig {r}%'
def nys_label(r): return f'nys {r}%'
METHODS = [(eig_label(r), (lambda rr: (lambda s: eig_path(s, rr)))(r)) for r in RANKS] \
        + [(nys_label(r), (lambda rr: (lambda s: nys_path(s, rr)))(r)) for r in RANKS]
EIG_LABELS = [eig_label(r) for r in RANKS]
NYS_LABELS = [nys_label(r) for r in RANKS]

def compute():
    diffs = {}; stats = {}; rankstats = {}
    for s in SEEDS:
        ip = inv_path(s)
        if not ip: continue
        isv, isi = load_sv(ip); istd = isv.std() or 1.0
        for label, fn in METHODS:
            p = fn(s)
            if not p: continue
            asv, asi = load_sv(p); a, b = aligned(asv, asi, isv, isi)
            if len(a) == 0: continue
            d = a - b
            diffs[(s, label)] = d
            co = np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan
            stats[(s, label)] = (np.abs(d).mean(), np.abs(d).max(), co, a.std() / istd)
            rankstats[(s, label)] = (spearman(a, b), topk_ov(a, b, 5))
    return diffs, stats, rankstats

def rank_color(r, base):  # base 'eig'/'nys' -> shade by rank
    frac = 0.25 + 0.75 * (RANKS.index(r) / (len(RANKS) - 1))
    if base == 'eig': return (0.0, 0.0, frac)          # blue gradient
    return (frac, 0.0, 0.0)                              # red gradient
COL = {**{eig_label(r): rank_color(r, 'eig') for r in RANKS},
       **{nys_label(r): rank_color(r, 'nys') for r in RANKS}}

def seed_rows():
    present = [s for s in SEEDS]
    return present

def fig_heatmap(stats, outpath):
    labels = [l for l, _ in METHODS]; rows = SEEDS
    C = np.full((len(rows), len(labels)), np.nan); M = np.full((len(rows), len(labels)), np.nan)
    for ri, s in enumerate(rows):
        for ci, m in enumerate(labels):
            if (s, m) in stats:
                me, mx, co, sr = stats[(s, m)]; C[ri, ci] = co; M[ri, ci] = me
    fig, axs = plt.subplots(2, 1, figsize=(0.7 * len(labels) + 3, 6.5))
    c1 = LinearSegmentedColormap.from_list('rg', ['#b2182b', '#f7f7c0', '#1a9850']); c1.set_bad('white')
    im0 = axs[0].imshow(np.ma.masked_invalid(C), cmap=c1, vmin=0, vmax=1, aspect='auto')
    axs[0].set_title('correlation( approx_sv , inv_sv )   (1=identical, ~0=unrelated)', fontsize=10)
    c2 = LinearSegmentedColormap.from_list('gr', ['#1a9850', '#f7f7c0', '#b2182b']); c2.set_bad('white')
    im1 = axs[1].imshow(np.ma.masked_invalid(M), cmap=c2, vmin=0, vmax=0.3, aspect='auto')
    axs[1].set_title('mean | approx_sv - inv_sv |   (color capped 0.3; number=true)', fontsize=10)
    for ax, Mat, im in [(axs[0], C, im0), (axs[1], M, im1)]:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'seed {s}' for s in rows], fontsize=8)
        ax.axvline(len(RANKS) - 0.5, color='k', lw=1.2)
        for ri in range(len(rows)):
            for ci in range(len(labels)):
                v = Mat[ri, ci]
                if not np.isnan(v): ax.text(ci, ri, f'{v:.2f}', ha='center', va='center', fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle('(1) value fidelity to inv  |  left group = eigen rank sweep, right = nystrom d sweep', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(outpath, dpi=130, bbox_inches='tight'); plt.close(fig); print('saved:', outpath)

def fig_ranking(rankstats, outpath):
    labels = [l for l, _ in METHODS]; rows = SEEDS
    S = np.full((len(rows), len(labels)), np.nan); T = np.full((len(rows), len(labels)), np.nan)
    for ri, s in enumerate(rows):
        for ci, m in enumerate(labels):
            if (s, m) in rankstats: sp, ov = rankstats[(s, m)]; S[ri, ci] = sp; T[ri, ci] = ov
    fig, axR = plt.subplots(2, 1, figsize=(0.7 * len(labels) + 3, 6.5))
    cg = LinearSegmentedColormap.from_list('rg', ['#b2182b', '#f7f7c0', '#1a9850']); cg.set_bad('white')
    imS = axR[0].imshow(np.ma.masked_invalid(S), cmap=cg, vmin=0, vmax=1, aspect='auto')
    axR[0].set_title('Spearman rank corr ( approx , inv )   (1=same order, ~0=unrelated)', fontsize=10)
    imT = axR[1].imshow(np.ma.masked_invalid(T), cmap=cg, vmin=0, vmax=1, aspect='auto')
    axR[1].set_title('top-5% overlap with inv   (1=identical, ~0.05=random)', fontsize=10)
    for ax, Mat, im in [(axR[0], S, imS), (axR[1], T, imT)]:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'seed {s}' for s in rows], fontsize=8)
        ax.axvline(len(RANKS) - 0.5, color='k', lw=1.2)
        for ri in range(len(rows)):
            for ci in range(len(labels)):
                v = Mat[ri, ci]
                if not np.isnan(v): ax.text(ci, ri, f'{v:.2f}', ha='center', va='center', fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle('(4) ranking preservation (Spearman / top-5% overlap)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(outpath, dpi=130, bbox_inches='tight'); plt.close(fig); print('saved:', outpath)

def fig_absdiff(stats, outpath):
    nr, nc = len(SEEDS), 2
    fig, ax3 = plt.subplots(nr, nc, figsize=(7.5 * nc, 3.0 * nr), squeeze=False)
    for i, s in enumerate(SEEDS):
        for j, (fam, labs) in enumerate([('eigen', EIG_LABELS), ('nystrom', NYS_LABELS)]):
            ax = ax3[i][j]; ms = [m for m in labs if (s, m) in stats]
            if not ms:
                ax.text(0.5, 0.5, '(no data)', ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                xs = np.arange(len(ms)); means = [stats[(s, m)][0] for m in ms]; maxs = [stats[(s, m)][1] for m in ms]
                ax.bar(xs, means, color=[COL[m] for m in ms], alpha=0.85, zorder=2)
                ax.scatter(xs, maxs, color='k', marker='_', s=130, zorder=5)
                for x, mx in zip(xs, maxs): ax.plot([x, x], [1e-3, mx], color='k', lw=0.5, ls=':', zorder=1)
                ax.set_yscale('log'); ax.set_ylim(0.005, 30); ax.axhline(1.0, color='crimson', ls='--', lw=0.7)
                ax.set_xticks(xs); ax.set_xticklabels(ms, rotation=45, ha='right', fontsize=7)
            ax.set_title(f'seed {s} — {fam}', fontsize=10)
            ax.set_ylabel('|approx-inv| (log)', fontsize=8); ax.tick_params(labelsize=7.5)
    fig.suptitle('(2) |approx-inv|: bar=mean, tick=max (log y; red line ~ normal sv scale)', y=1.005, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(outpath, dpi=130, bbox_inches='tight'); plt.close(fig); print('saved:', outpath)

def fig_errordist(diffs, outpath):
    XLIM = 1.0; bins = np.linspace(-XLIM, XLIM, 61); nr, nc = len(SEEDS), 2
    fig, ax2 = plt.subplots(nr, nc, figsize=(7.5 * nc, 3.0 * nr), squeeze=False)
    for i, s in enumerate(SEEDS):
        for j, (fam, labs) in enumerate([('eigen', EIG_LABELS), ('nystrom', NYS_LABELS)]):
            ax = ax2[i][j]; present = [m for m in labs if (s, m) in diffs]
            if not present:
                ax.text(0.5, 0.5, '(no data)', ha='center', va='center', transform=ax.transAxes, color='gray')
            clipped = 0; tot = 0
            for m in present:
                d = diffs[(s, m)]; clipped += int((np.abs(d) > XLIM).sum()); tot += d.size
                ax.hist(np.clip(d, -XLIM, XLIM), bins=bins, histtype='step', color=COL[m], lw=1.4, density=True, label=m)
            ax.axvline(0, color='gray', ls=':', lw=0.8); ax.set_title(f'seed {s} — {fam}', fontsize=10)
            if tot: ax.text(0.02, 0.97, f'{100 * clipped / tot:.0f}% |err|>{XLIM}', transform=ax.transAxes, va='top', fontsize=6.5, color='dimgray')
            if present: ax.legend(fontsize=6.5, ncol=2)
            ax.set_xlabel('error (approx - inv)', fontsize=8); ax.set_ylabel('density', fontsize=8); ax.tick_params(labelsize=7.5)
    fig.suptitle('(3) per-point error (approx - inv), clipped +-1', y=1.005, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(outpath, dpi=130, bbox_inches='tight'); plt.close(fig); print('saved:', outpath)

if __name__ == '__main__':
    diffs, stats, rankstats = compute()
    fig_heatmap(stats, f'{OUT}/heatmap.png')
    fig_absdiff(stats, f'{OUT}/absdiff.png')
    fig_errordist(diffs, f'{OUT}/errordist.png')
    fig_ranking(rankstats, f'{OUT}/ranking.png')
    print('\nVALUE-FIDELITY FIGURES IN:', OUT)
