"""QQP 의 3-point 비교 — 50/50 (baseline) vs 63/37 (자연) vs 90/10 (deep imb).
pos10 진행 중이므로 일단 50/50 + 63/37 표시. pos10 끝나면 자동 추가."""
from datasets import load_dataset
import numpy as np
import os

ROOT = "./freeshap_res"

def load_idx(p, n_keep=2000):
    return np.array([int(l.strip()) for l in open(p) if l.strip()])[:n_keep]

qqp_labels = np.array(load_dataset("glue", "qqp")["train"]["label"])

points = []
# 50/50 (강제 balanced) — true controlled baseline
p1 = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/qqp/pos50/lrfshap/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt"
if os.path.exists(p1):
    points.append(("50/50 (강제 balanced, true baseline)", load_idx(p1)))

# 63/37 (자연)
p2 = f"{ROOT}/data_selection/qqp/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt"
if os.path.exists(p2):
    points.append(("63/37 (자연, mild imbalance)", load_idx(p2)))

# 90/10 (강제 deep)
p3 = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/qqp/pos10/lrfshap/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt"
if os.path.exists(p3):
    points.append(("90/10 (강제 deep imbalance)", load_idx(p3)))
else:
    points.append(("90/10 (강제 deep imbalance)", None))

print(f"\nQQP — top-r 안의 class composition (LRFShap INV mode)")
print(f"  val_majority natural: class0=62%, class1=38%")
print(f"\n{'point':<40} pool: class0 / class1")
for title, idx in points:
    if idx is None:
        print(f"  {title:<40}  (pending)")
        continue
    pool = qqp_labels[idx]
    n0 = int((pool == 0).sum()); n1 = int((pool == 1).sum())
    print(f"  {title:<40}  pool: {n0:>4d} / {n1:>4d}  ({n0/(n0+n1)*100:.1f}% / {n1/(n0+n1)*100:.1f}%)")

print(f"\n{'sel':>4} {'k':>5}", end="")
for title, _ in points:
    print(f" | {title[:24]:>24}", end="")
print()
print(f"{'':>4} {'':>5}", end="")
for _ in points:
    print(f" | {'class0':>11} {'class1':>11}", end="")
print()

for sel in [1, 2, 5, 10, 15, 20, 30]:
    k = int(2000 * sel / 100)
    print(f"{sel:>3}% {k:>5d}", end="")
    for title, idx in points:
        if idx is None:
            print(f" | {'pending':>24}", end="")
            continue
        top = qqp_labels[idx[:k]]
        n0 = (top == 0).sum() / k * 100
        n1 = (top == 1).sum() / k * 100
        print(f" | {n0:>10.1f}% {n1:>10.1f}%", end="")
    print()
