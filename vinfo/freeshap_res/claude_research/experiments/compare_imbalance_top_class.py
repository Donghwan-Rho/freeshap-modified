"""Compare class composition of Shapley top-r between baseline (natural dist)
and imbalance pos70 enforced on SST-2.  Both seed=2026, n=2000."""
from datasets import load_dataset
import numpy as np
import os

ds = load_dataset("sst2")
labels = np.array(ds["train"]["label"])
print(f"SST-2 train: n={len(labels)}, pos_frac={labels.mean():.4f}")
print()

ROOT = "./freeshap_res"
base_p = (f"{ROOT}/data_selection/sst2/inv/indices/"
          f"bert_seed2026_num2000_val872_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt")
imb_p  = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/pos70/lrfshap/inv/"
          f"indices/bert_seed2026_num2000_val872_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt")

def load_idx(p):
    return np.array([int(l.strip()) for l in open(p) if l.strip()])

bi = load_idx(base_p)
ii = load_idx(imb_p)
b_pool = labels[bi].mean()
i_pool = labels[ii].mean()
print(f"baseline indices file:  {base_p}")
print(f"  n={len(bi)}, train pool pos_frac={b_pool:.4f} (natural)")
print(f"imbalance indices file: {imb_p}")
print(f"  n={len(ii)}, train pool pos_frac={i_pool:.4f} (enforced 70/30)")
print()

print(f"{'sel':>5} {'k':>6}  baseline_top70_frac  imbalance_top70_frac  pool_baseline  pool_imbalance")
for sel in [1, 2, 5, 10, 15, 20, 30, 50, 100]:
    k = int(2000 * sel / 100)
    bp = labels[bi[:k]].mean()
    ip = labels[ii[:k]].mean()
    # Relative to pool: if Shapley selection were class-blind, top-k frac should equal pool frac.
    # Excess majority = (top-k pos_frac) - (pool pos_frac).
    excess_b = bp - b_pool
    excess_i = ip - i_pool
    print(f"{sel:>4}%  {k:>5d}  {bp:>17.4f}    {ip:>18.4f}    {b_pool:>11.4f}    {i_pool:>13.4f}    "
          f"excess(b)={excess_b:+.4f}  excess(i)={excess_i:+.4f}")
