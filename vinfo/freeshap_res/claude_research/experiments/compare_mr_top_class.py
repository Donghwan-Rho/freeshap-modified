"""MR: Shapley top-r class composition.
  (1) MR baseline (natural)        seed=2026, n=2000  (LRFShap inv from upstream)
  (2) MR imbalance pos=0.7         seed=2026, n=2000  (LRFShap inv from ours)
"""
from datasets import load_dataset
import numpy as np

ROOT = "./freeshap_res"

base_p = f"{ROOT}/data_selection/mr/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt"
imb_p  = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/mr/pos70/lrfshap/inv/indices/"
          f"bert_seed2026_num2000_val1066_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt")

def load_idx(p):
    return np.array([int(l.strip()) for l in open(p) if l.strip()])

ds = load_dataset("rotten_tomatoes")
labels = np.array(ds["train"]["label"])
print(f"MR train: n={len(labels)}, pos_frac={labels.mean():.4f}")
print()

for tag, path in [("baseline (natural)", base_p), ("imb pos70 (LRFShap)", imb_p)]:
    idx_all = load_idx(path)[:2000]
    lab = labels[idx_all]
    n_pos = int((lab == 1).sum())
    n_neg = int((lab == 0).sum())
    maj_lab = 1 if n_pos > n_neg else 0
    maj_name = "positive (1)" if maj_lab == 1 else "negative (0)"
    print(f">>> {tag} — pool: pos={n_pos} ({n_pos/2000:.3f}), neg={n_neg} ({n_neg/2000:.3f}) | maj={maj_name}")
    print(f"     {'sel':>4} {'k':>5}  {'pos':>5} {'neg':>5}  {'pos%':>6} {'neg%':>6}  {'maj-in-top%':>11}")
    for sel in [1, 2, 5, 10, 15, 20, 30, 50, 100]:
        k = int(2000 * sel / 100)
        top = lab[:k]
        tp = int((top == 1).sum())
        tn = int((top == 0).sum())
        maj_in_top = (top == maj_lab).mean()
        print(f"     {sel:>3}%  {k:>4d}  {tp:>4d}  {tn:>4d}  {tp/k*100:>5.1f}  {tn/k*100:>5.1f}  "
              f"{maj_in_top*100:>10.1f}")
    print()
