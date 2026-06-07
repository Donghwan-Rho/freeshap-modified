"""Quick check of newly completed settings."""
import json, os
ROOT = "./freeshap_res/claude_research/data_selection_test/imbalance/data_selection"

def load(ds, rtag, method, val, n, r):
    p = f"{ROOT}/{ds}/{rtag}/{method}/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json"
    return json.load(open(p))["top_results_inv"] if os.path.exists(p) else None

print("\n=== RTE pos50 (강제 balanced, n=1300, val=277) — LR vs A1 @ eigen r=10% ===")
lr = load("rte", "pos50", "lrfshap", 277, 1300, 10)
a1 = load("rte", "pos50", "a1",      277, 1300, 10)
print(f"  sel  LR     A1     Δ(A1-LR)")
for s in [1, 2, 5, 10, 15, 20]:
    print(f"  {s:>3}%  {lr[s-1]/100:5.2f}  {a1[s-1]/100:5.2f}  {(a1[s-1]-lr[s-1])/100:+5.2f}")

print("\n=== MRPC pos70 (강제 70/30, n=2000, val=408) — LR vs A1 @ eigen r=10% ===")
lr = load("mrpc", "pos70", "lrfshap", 408, 2000, 10)
a1 = load("mrpc", "pos70", "a1",      408, 2000, 10)
print(f"  sel  LR     A1     Δ(A1-LR)")
for s in [1, 2, 5, 10, 15, 20]:
    print(f"  {s:>3}%  {lr[s-1]/100:5.2f}  {a1[s-1]/100:5.2f}  {(a1[s-1]-lr[s-1])/100:+5.2f}")

print("\n=== QQP pos30 (강제 70/30, label0 maj 70%, n=2000, val=1000) — LR only (A1 진행 중) @ eigen r=10% ===")
lr = load("qqp", "pos30", "lrfshap", 1000, 2000, 10)
print(f"  sel  LR")
for s in [1, 2, 5, 10, 15, 20]:
    print(f"  {s:>3}%  {lr[s-1]/100:5.2f}")
