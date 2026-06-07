"""Compare top-r class composition: baseline (natural) vs new imbalanced setting.
For new datasets MNLI / QQP (and later AG News, RTE, MRPC as they finish)."""
from datasets import load_dataset
import numpy as np

ROOT = "./freeshap_res"

def load_idx(p, n_keep=2000):
    return np.array([int(l.strip()) for l in open(p) if l.strip()])[:n_keep]

# ---------- MNLI ----------
print("\n" + "=" * 78)
print("MNLI — natural vs cls60_20_20 (LRFShap INV, sorted by Shapley high→low)")
print("=" * 78)
mnli_labels = np.array(load_dataset("glue", "mnli")["train"]["label"])

base_p = f"{ROOT}/data_selection/mnli/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt"
imb_p = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/mnli/cls60_20_20/lrfshap/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt"

base_idx = load_idx(base_p, 2000)
imb_idx = load_idx(imb_p, 2000)

base_pool = mnli_labels[base_idx]
imb_pool = mnli_labels[imb_idx]

print(f"\nPool composition (n=2000 trained set):")
print(f"  baseline: class0={int((base_pool==0).sum())}, class1={int((base_pool==1).sum())}, class2={int((base_pool==2).sum())}")
print(f"  cls60_20_20: class0={int((imb_pool==0).sum())}, class1={int((imb_pool==1).sum())}, class2={int((imb_pool==2).sum())}")

print(f"\n{'sel':>4} {'k':>5} | {'BASELINE (자연 33/33/33)':>30} | {'cls60_20_20 (60/20/20)':>30}")
print(f"{'':>4} {'':>5} | {'class0':>8} {'class1':>8} {'class2':>8} | {'class0':>8} {'class1':>8} {'class2':>8}")
for sel in [1, 2, 5, 10, 15, 20]:
    k = int(2000 * sel / 100)
    bt = base_pool[:k]; it = imb_pool[:k]
    b0 = (bt == 0).sum() / k * 100
    b1 = (bt == 1).sum() / k * 100
    b2 = (bt == 2).sum() / k * 100
    i0 = (it == 0).sum() / k * 100
    i1 = (it == 1).sum() / k * 100
    i2 = (it == 2).sum() / k * 100
    print(f"{sel:>3}% {k:>5d} | {b0:>7.1f}% {b1:>7.1f}% {b2:>7.1f}% | {i0:>7.1f}% {i1:>7.1f}% {i2:>7.1f}%")

# ---------- QQP ----------
print("\n" + "=" * 78)
print("QQP — natural vs pos50 (LRFShap INV)")
print("=" * 78)
qqp_labels = np.array(load_dataset("glue", "qqp")["train"]["label"])

base_p = f"{ROOT}/data_selection/qqp/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt"
imb_p = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/qqp/pos50/lrfshap/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt"

base_idx = load_idx(base_p, 2000)
imb_idx = load_idx(imb_p, 2000)
base_pool = qqp_labels[base_idx]
imb_pool = qqp_labels[imb_idx]

print(f"\nPool composition (n=2000):")
print(f"  baseline: class0={int((base_pool==0).sum())} ({(base_pool==0).mean():.3f}), class1={int((base_pool==1).sum())} ({(base_pool==1).mean():.3f})")
print(f"  pos50:    class0={int((imb_pool==0).sum())} ({(imb_pool==0).mean():.3f}), class1={int((imb_pool==1).sum())} ({(imb_pool==1).mean():.3f})")

print(f"\n{'sel':>4} {'k':>5} | {'BASELINE (자연 63/37, class0 maj)':>34} | {'pos50 (강제 50/50)':>26}")
print(f"{'':>4} {'':>5} | {'class0':>8} {'class1':>8} | {'class0':>8} {'class1':>8}")
for sel in [1, 2, 5, 10, 15, 20]:
    k = int(2000 * sel / 100)
    bt = base_pool[:k]; it = imb_pool[:k]
    b0 = (bt == 0).sum() / k * 100
    b1 = (bt == 1).sum() / k * 100
    i0 = (it == 0).sum() / k * 100
    i1 = (it == 1).sum() / k * 100
    print(f"{sel:>3}% {k:>5d} | {b0:>7.1f}% {b1:>7.1f}% | {i0:>7.1f}% {i1:>7.1f}%")
