# -*- coding: utf-8 -*-
"""VISION Part A: data-selection accuracy (ranking-based) curves.

Each 'top' curve = val accuracy (x100) as you add training data top-first by a
Shapley ranking (x = selection %). An eigen/nystrom prediction txt holds TWO
'top' blocks that share the SAME approx (eigen/nys-r) ranking but differ in the
predictor used to MEASURE accuracy:
  - 'inv mode'    -> approx ranking, evaluated by full eNTK (inv) predictor  << USE THIS
  - '<kind> mode' -> approx ranking, evaluated by the low-rank predictor,
                     which artificially collapses near selection = rank d% (an artifact).
The standalone inv/ file's 'inv' block = the inv (oracle) ranking, full-eNTK eval.

So the correct data-selection accuracy of a method = its 'inv mode' block:
approximate the Shapley ranking with eigen/nystrom, but predict acc with full eNTK.
Per seed we plot the oracle (black, inv ranking) vs each approx ranking (color),
ALL evaluated by full eNTK. glob allows any num. Missing files -> '(no data yet)'.
Reads git-tracked txt. Output -> <vinfo>/report_figs_vision/acc_*.png
"""
import os, glob, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

VINFO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE  = os.path.join(VINFO, 'freeshap_res', 'data_selection', 'cifar10')
OUT   = os.path.join(VINFO, 'report_figs_vision'); os.makedirs(OUT, exist_ok=True)

SEEDS = [2024, 2025, 2026]
RANKS = [1, 5, 10, 15, 20, 25, 30]

def one(pat):
    g = sorted(glob.glob(pat)); return g[0] if g else None
def parse_modes(path):
    """-> {mode_name: np.array(top)/100.0} for every '<x> mode ... top: [..]' block."""
    if not path or not os.path.exists(path): return {}
    txt = open(path).read(); out = {}
    for m in re.finditer(r'(\w+) mode lambda=[^\n]*\ntop:\s*\n\[([^\]]*)\]', txt):
        out[m.group(1)] = np.array([int(x) for x in m.group(2).split(',')]) / 100.0
    return out

def eig_txt(s, r): return one(f'{BASE}/eigen/predictions/resnet_seed{s}_num*_val*_eig{r}.0_eiglam1e-02_invlam1e-06_*tmc500_predictions.txt')
def nys_txt(s, r): return one(f'{BASE}/nystrom/predictions/resnet_seed{s}_num*_val*_nys{r}.0_nyslam1e-02_invlam1e-06_*tmc500_predictions.txt')
def inv_txt(s):    return one(f'{BASE}/inv/predictions/resnet_seed{s}_num*_val*_lam1e-02_*tmc500_predictions.txt')

def rank_color(i, base):
    frac = 0.25 + 0.75 * (i / (len(RANKS) - 1))
    return (0.0, 0.0, frac) if base == 'eig' else (frac, 0.0, 0.0)

def make_sweep(kind, txt_fn, cap):
    """rows = seeds; each panel: inv reference (black) + approx curve per rank."""
    nr = len(SEEDS)
    fig, axes = plt.subplots(nr, 1, figsize=(8.5, 3.2 * nr), squeeze=False)
    for i, s in enumerate(SEEDS):
        ax = axes[i][0]; plotted = False
        # inv reference: prefer standalone inv file, else inv-mode inside any approx file
        ref = parse_modes(inv_txt(s)).get('inv')
        if ref is None:
            for r in RANKS:
                mm = parse_modes(txt_fn(s, r))
                if 'inv' in mm: ref = mm['inv']; break
        if ref is not None:
            ax.plot(np.arange(1, len(ref) + 1), ref, color='k', lw=2.6, label='inv (full eNTK)', zorder=20)
            plotted = True
        for j, r in enumerate(RANKS):
            mm = parse_modes(txt_fn(s, r))
            # IMPORTANT: use the 'inv' block = (eigen/nys-r Shapley ranking) evaluated by
            # full eNTK(inv) predictor. The method's own '<kind> mode' block instead evaluates
            # with the low-rank predictor, which artificially craters near selection = rank d%.
            y = mm.get('inv')
            if y is None: continue
            ax.plot(np.arange(1, len(y) + 1), y, color=rank_color(j, 'eig' if kind == 'eigen' else 'nys'),
                    lw=1.5, alpha=0.9, label=f'{kind[:3]} {r}%'); plotted = True
        ax.set_title(f'seed {s}', fontsize=11)
        ax.set_xlabel('selection %', fontsize=9); ax.set_ylabel('val acc (%)', fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)
        if plotted: ax.legend(fontsize=7.5, ncol=2, loc='lower right')
        else: ax.text(0.5, 0.5, '(no data yet)', ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=13)
    fig.suptitle(f'[Part A] {cap}  —  curve higher = better ranking (closer to/above inv)', y=1.005, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f'{OUT}/acc_{kind}.png'; fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig); print('saved:', out)

def make_matched():
    """rows = seeds, cols = ranks; each panel: inv vs eigen-r vs nystrom-r."""
    nr, nc = len(SEEDS), len(RANKS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.7 * nr), squeeze=False)
    for i, s in enumerate(SEEDS):
        ref = parse_modes(inv_txt(s)).get('inv')
        for j, r in enumerate(RANKS):
            ax = axes[i][j]; plotted = False
            if ref is not None:
                ax.plot(np.arange(1, len(ref) + 1), ref, color='k', lw=2.0, label='inv'); plotted = True
            for kind, fn, color in [('eigen', eig_txt, '#1f4fd8'), ('nystrom', nys_txt, '#d62728')]:
                # 'inv' block = approx ranking evaluated by full eNTK(inv) predictor (see make_sweep note)
                y = parse_modes(fn(s, r)).get('inv')
                if y is None: continue
                ax.plot(np.arange(1, len(y) + 1), y, color=color, lw=1.4, label=kind[:3]); plotted = True
            ax.set_title(f'seed {s} — {r}%', fontsize=8.5)
            ax.tick_params(labelsize=6.5); ax.grid(True, alpha=0.3)
            if plotted:
                if i == 0 and j == 0: ax.legend(fontsize=6.5, loc='lower right')
            else:
                ax.text(0.5, 0.5, '(no data)', ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=9)
    fig.suptitle('[Part A — matched] inv vs eigen-r vs nystrom-r at same rank/d (row=seed, col=rank/d %)', y=1.003, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = f'{OUT}/acc_matched.png'; fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig); print('saved:', out)

if __name__ == '__main__':
    make_sweep('eigen', eig_txt, 'eigen rank sweep vs inv')
    make_sweep('nystrom', nys_txt, 'nystrom d sweep vs inv')
    make_matched()
    print('\nACC FIGURES IN:', OUT)
