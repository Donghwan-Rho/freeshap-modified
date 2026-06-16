# -*- coding: utf-8 -*-
import glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/donghwan/freeshap/vinfo/freeshap_res/data_selection'
DATASETS = [('sst2',5000), ('mnli',5000), ('ag_news',5000), ('mr',5000),
            ('qqp',5000), ('rte',2490), ('mrpc',3668)]
RANKS = [1, 5, 20]   # rank 20%: nystrom fixed at lam=1e-2 (only that file exists)
NYS_LAMS = ['1e-02','1e-03','1e-04','1e-05','1e-06']

def parse_inv_top(path):
    txt = open(path).read()
    m = re.findall(r'inv mode lambda=[^\n]*\ntop:\s*\n\[([^\]]*)\]', txt)
    if not m:
        return None
    arr = [int(x) for x in m[-1].split(',')]
    return np.array(arr) / 100.0   # -> percent

def find_one(pattern):
    g = sorted(glob.glob(pattern))
    return g[0] if g else None

def eigen_path(ds, num, rank):
    return find_one(f'{BASE}/{ds}/eigen/predictions/'
                    f'bert_seed2024_num{num}_val*_eig{rank}.0_eiglam1e-02_invlam1e-06_*tmc500_predictions.txt')

def nys_path(ds, num, rank, lam):
    return find_one(f'{BASE}/{ds}/nystrom/predictions/'
                    f'bert_seed2024_num{num}_val*_nys{rank}.0_nyslam{lam}_invlam1e-06_*tmc500_predictions.txt')

eig_color = 'black'
nys_colors = {'1e-02':'#d62728','1e-03':'#ff7f0e','1e-04':'#2ca02c',
              '1e-05':'#1f77b4','1e-06':'#9467bd'}
def lam_disp(l): return l.replace('e-0','e-')

nrows, ncols = len(DATASETS), len(RANKS)
fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 3.2*nrows), squeeze=False)
handles = {}

for i,(ds,num) in enumerate(DATASETS):
    for j,rank in enumerate(RANKS):
        ax = axes[i][j]
        plotted = False
        # eigen (full eNTK eval of eigen-TMC ranking)
        ep = eigen_path(ds, num, rank)
        if ep:
            y = parse_inv_top(ep)
            if y is not None:
                x = np.arange(1, len(y)+1)
                ln, = ax.plot(x, y, color=eig_color, lw=2.4, zorder=10, label='eigen λ=1e-2')
                handles.setdefault('eigen λ=1e-2', ln); plotted = True
        # nystrom (full eNTK eval of nystrom-TMC ranking), 5 lambdas
        for lam in NYS_LAMS:
            np_ = nys_path(ds, num, rank, lam)
            if np_:
                y = parse_inv_top(np_)
                if y is not None:
                    x = np.arange(1, len(y)+1)
                    lbl = f'nys λ={lam_disp(lam)}'
                    ln, = ax.plot(x, y, color=nys_colors[lam], lw=1.4, alpha=0.9, label=lbl)
                    handles.setdefault(lbl, ln); plotted = True
        ax.set_title(f'{ds}  (num{num}) — rank {rank}%', fontsize=10.5)
        ax.set_xlabel('selection %', fontsize=8.5)
        ax.set_ylabel('val acc (%)', fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, '(no data yet)', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=12)

# shared legend in fixed order
order = ['eigen λ=1e-2'] + [f'nys λ={lam_disp(l)}' for l in NYS_LAMS]
H = [handles[k] for k in order if k in handles]
L = [k for k in order if k in handles]
fig.legend(H, L, loc='upper center', ncol=6, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, 0.998))
fig.suptitle('eigen vs nystrom — inv-mode (full eNTK) TOP accuracy vs selection %  '
             '(curves differ only in TMC-Shapley ranking source)',
             y=1.012, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.975])
out_png = '/home/donghwan/freeshap/vinfo/eigen_vs_nystrom_acc_comparison.png'
out_pdf = '/home/donghwan/freeshap/vinfo/eigen_vs_nystrom_acc_comparison.pdf'
fig.savefig(out_png, dpi=130, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print('saved:', out_png)
print('saved:', out_pdf)

# quick coverage report
print('\n== 커버리지 ==')
for ds,num in DATASETS:
    for rank in RANKS:
        e = 'E' if eigen_path(ds,num,rank) else '.'
        ns = ''.join('N' if nys_path(ds,num,rank,l) else '.' for l in NYS_LAMS)
        print(f'  {ds:8s} r{rank}%: eigen[{e}] nys[{ns}]  (N順 {",".join(lam_disp(l) for l in NYS_LAMS)})')
