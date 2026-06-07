"""SST-2 pos70: head-to-head class composition + accuracy comparison between
LRFShap and A1 (LRFShap = top-r by lambda, A1 = top-r by supervised score).
INV mode Shapley + sel data selection.
"""
from datasets import load_dataset
import numpy as np
import json

ROOT = "./freeshap_res"

# Indices files (Shapley value sorted high -> low)
lr_idx_p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/"
            f"pos70/lrfshap/inv/indices/"
            f"bert_seed2026_num2000_val872_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt")
a1_idx_p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/"
            f"pos70/a1/inv/indices/"
            f"bert_seed2026_num2000_val872_lam1e-06_a1_signFalse_earlystopTrue_tmc500_indices.txt")

# Sidecar (accuracy)
lr_side_p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/"
             f"pos70/lrfshap/inv/sidecar/"
             f"bert_seed2026_num2000_val872_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500.json")
a1_side_p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/"
             f"pos70/a1/inv/sidecar/"
             f"bert_seed2026_num2000_val872_lam1e-06_a1_signFalse_earlystopTrue_tmc500.json")

def load_idx(p):
    return np.array([int(l.strip()) for l in open(p) if l.strip()])

ds = load_dataset("sst2")
labels = np.array(ds["train"]["label"])

lr_idx = load_idx(lr_idx_p)[:2000]
a1_idx = load_idx(a1_idx_p)[:2000]

lr_pool = labels[lr_idx]
a1_pool = labels[a1_idx]
# Both are pos70 -> pool should be ~70% positive
print(f"LR pool: pos={int((lr_pool==1).sum())}, neg={int((lr_pool==0).sum())}")
print(f"A1 pool: pos={int((a1_pool==1).sum())}, neg={int((a1_pool==0).sum())}")
print()

lr_side = json.load(open(lr_side_p))
a1_side = json.load(open(a1_side_p))
sels = lr_side["num_train_selected_list"]
lr_top = lr_side["top_results_inv"]
a1_top = a1_side["top_results_inv"]
lr_rand = lr_side["random_results_inv"]
a1_rand = a1_side["random_results_inv"]
val_neg_frac = lr_side["acc_at_f0"] / 10000  # = P(val label == 0)
print(f"val class-0 (negative) frac = {val_neg_frac:.4f} (predict-all-neg trivial = {val_neg_frac*100:.2f}%)")
print(f"val class-1 (positive) frac = {1-val_neg_frac:.4f} (predict-all-pos trivial = {(1-val_neg_frac)*100:.2f}%)")
print()
print("=" * 110)
print("SST-2 pos70 INV mode — head-to-head")
print("=" * 110)
hdr = (f"{'sel':>4} {'k':>5} | {'LR pos%':>8} {'A1 pos%':>8} {'LR neg%':>8} {'A1 neg%':>8}"
       f" | {'LR top_acc':>10} {'A1 top_acc':>10} {'Δ(A1-LR)':>10}"
       f" | {'LR rand':>8} {'A1 rand':>8}")
print(hdr)
print("-" * len(hdr))
for sel in [1, 2, 5, 10, 15, 20, 30, 50, 100]:
    k = int(2000 * sel / 100)
    lr_pos = (lr_pool[:k] == 1).sum()
    a1_pos = (a1_pool[:k] == 1).sum()
    lr_neg = k - lr_pos
    a1_neg = k - a1_pos
    i = sels.index(sel) if sel in sels else None
    if i is None:
        lr_a = a1_a = lr_r = a1_r = float('nan')
    else:
        lr_a = lr_top[i] / 100
        a1_a = a1_top[i] / 100
        lr_r = lr_rand[i] / 100
        a1_r = a1_rand[i] / 100
    print(f"{sel:>3}% {k:>5d} | "
          f"{lr_pos/k*100:>7.1f}  {a1_pos/k*100:>7.1f}  {lr_neg/k*100:>7.1f}  {a1_neg/k*100:>7.1f}"
          f" | {lr_a:>9.2f}  {a1_a:>9.2f}  {a1_a-lr_a:>+9.2f}"
          f" | {lr_r:>7.2f}  {a1_r:>7.2f}")
