"""SST-2 pos70: eigen mode comparison.  LR vs A1 of the actual Shapley
selection within the r-rank eigen subspace."""
from datasets import load_dataset
import numpy as np
import json
import glob

ROOT = "./freeshap_res"
SST2_TRAIN_LABELS = None

def labels():
    global SST2_TRAIN_LABELS
    if SST2_TRAIN_LABELS is None:
        ds = load_dataset("sst2")
        SST2_TRAIN_LABELS = np.array(ds["train"]["label"])
    return SST2_TRAIN_LABELS

def load_idx(p):
    return np.array([int(l.strip()) for l in open(p) if l.strip()])

L = labels()

base = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/pos70"

# Per-rank eigen sidecar (LR + A1) and indices.  Filter to r in {1, 5, 10, 30}.
RANKS = [1, 5, 10, 30]

print(f"{'sel':>4} {'k':>5} | "
      + " ".join(f"r={r}%LR_pos% r={r}%A1_pos%" for r in RANKS))

for sel in [1, 2, 5, 10, 15, 20, 30]:
    k = int(2000 * sel / 100)
    parts = [f"{sel:>3}% {k:>5d} |"]
    for r in RANKS:
        lr_idx_p = glob.glob(f"{base}/lrfshap/eigen/indices/*eig{float(r):.1f}_*lrfshap*_indices.txt")
        a1_idx_p = glob.glob(f"{base}/a1/eigen/indices/*eig{float(r):.1f}_*a1*_indices.txt")
        if not lr_idx_p or not a1_idx_p:
            parts.append("       N/A       N/A")
            continue
        lr_idx = load_idx(lr_idx_p[0])[:2000]
        a1_idx = load_idx(a1_idx_p[0])[:2000]
        lr_top_pos = (L[lr_idx[:k]] == 1).mean() * 100
        a1_top_pos = (L[a1_idx[:k]] == 1).mean() * 100
        parts.append(f"  {lr_top_pos:>6.1f}    {a1_top_pos:>6.1f} ")
    print(" ".join(parts))

print()
print("===== accuracy (eigen-mode Shapley + eigen prediction) =====")
print(f"{'sel':>4} | " + " ".join(f"r={r}%LR_acc r={r}%A1_acc Δ" for r in RANKS))
for sel in [1, 2, 5, 10, 15, 20, 30]:
    parts = [f"{sel:>3}% |"]
    for r in RANKS:
        lr_side = glob.glob(f"{base}/lrfshap/eigen/sidecar/*eig{float(r):.1f}_*lrfshap*.json")
        a1_side = glob.glob(f"{base}/a1/eigen/sidecar/*eig{float(r):.1f}_*a1*.json")
        if not lr_side or not a1_side:
            parts.append("        N/A        N/A       N/A")
            continue
        lrs = json.load(open(lr_side[0]))
        a1s = json.load(open(a1_side[0]))
        i = lrs["num_train_selected_list"].index(sel)
        lr_a = lrs["top_results_eigen"][i] / 100
        a1_a = a1s["top_results_eigen"][i] / 100
        parts.append(f"  {lr_a:>6.2f}   {a1_a:>6.2f} {a1_a-lr_a:+6.2f}")
    print(" ".join(parts))

print()
print("===== accuracy (eigen-mode Shapley + INV prediction) =====")
print(f"{'sel':>4} | " + " ".join(f"r={r}%LR_acc r={r}%A1_acc Δ" for r in RANKS))
for sel in [1, 2, 5, 10, 15, 20, 30]:
    parts = [f"{sel:>3}% |"]
    for r in RANKS:
        lr_side = glob.glob(f"{base}/lrfshap/eigen/sidecar/*eig{float(r):.1f}_*lrfshap*.json")
        a1_side = glob.glob(f"{base}/a1/eigen/sidecar/*eig{float(r):.1f}_*a1*.json")
        if not lr_side or not a1_side:
            parts.append("        N/A        N/A       N/A")
            continue
        lrs = json.load(open(lr_side[0]))
        a1s = json.load(open(a1_side[0]))
        i = lrs["num_train_selected_list"].index(sel)
        lr_a = lrs["top_results_inv"][i] / 100
        a1_a = a1s["top_results_inv"][i] / 100
        parts.append(f"  {lr_a:>6.2f}   {a1_a:>6.2f} {a1_a-lr_a:+6.2f}")
    print(" ".join(parts))
