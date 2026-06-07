"""Quick text-only summary of MRPC n=2300 valbal pos50."""
import json, os
ROOT = "./freeshap_res/claude_research/data_selection_test/imbalance/data_selection/mrpc/pos50"

def load(method, mode, r):
    if mode == "inv":
        p = f"{ROOT}/{method}/inv/sidecar/bert_seed2026_num2300_valbal258_lam1e-06_{method}_signFalse_earlystopTrue_tmc500.json"
    else:
        p = f"{ROOT}/{method}/eigen/sidecar/bert_seed2026_num2300_valbal258_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json"
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return d["top_results_inv"]   # INV-mode prediction column

inv_lr = load("lrfshap", "inv", None)
print(f"\n=== MRPC n=2300 valbal pos50  (50/50, val=258 balanced, label1 maj direction) ===\n")
print(f"INV mode (full kernel, LR == A1):")
print(f"  sel% : " + "  ".join(f"{s:>3}%" for s in [1,2,3,4,5,10,20]))
print(f"  acc  : " + "  ".join(f"{inv_lr[s-1]/100:>4.1f}" for s in [1,2,3,4,5,10,20]))

print(f"\n각 rank 의 A1 > LR 비교 (sel% × rank, eigen + INV-prediction)")
header = f"  {'rank':>5}  " + "  ".join(f"sel{s:>2}%" for s in [1,2,3,4,5,10,20])
print(header)
for r in [1, 5, 10, 15, 20, 25, 30]:
    lr = load("lrfshap", "eigen", r)
    a1 = load("a1", "eigen", r)
    if lr is None or a1 is None:
        print(f"  r={r:>3}%  " + "  ".join(" · " for _ in [1,2,3,4,5,10,20]))
        continue
    row = [f"r={r:>3}%"]
    for s in [1, 2, 3, 4, 5, 10, 20]:
        if a1[s-1] == lr[s-1]: tok = " - "
        elif a1[s-1] > lr[s-1]: tok = " O "
        else: tok = " X "
        row.append(tok)
    print(f"  {row[0]}  " + "  ".join(row[1:]))

print(f"\n각 (sel, rank) 의 LR / A1 actual acc:")
print(f"  {'rank':>5}  " + "  ".join(f"sel{s:>2}% LR/A1" for s in [1,5,10,20]))
for r in [1, 5, 10, 15, 20, 25, 30]:
    lr = load("lrfshap", "eigen", r)
    a1 = load("a1", "eigen", r)
    if lr is None or a1 is None: continue
    parts = [f"r={r:>3}%"]
    for s in [1, 5, 10, 20]:
        parts.append(f"{lr[s-1]/100:5.2f}/{a1[s-1]/100:5.2f}")
    print(f"  {parts[0]}  " + "    ".join(parts[1:]))
