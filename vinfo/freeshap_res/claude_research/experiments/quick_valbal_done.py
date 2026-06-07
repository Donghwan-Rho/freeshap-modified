"""Quick summary of all currently completed valbal sub-runs.
Reads sidecars under data_selection_test/imbalance/data_selection.
Prints INV baseline accuracy + per-rank LR/A1 actual + O/X/- table.
"""
import json, os, sys
ROOT = "./freeshap_res/claude_research/data_selection_test/imbalance/data_selection"
SELS = [1, 2, 3, 4, 5, 10, 20]
RANKS = [1, 5, 10, 15, 20, 25, 30]

def load(ds, ratio, method, mode, r, num, val):
    if mode == "inv":
        p = f"{ROOT}/{ds}/{ratio}/{method}/inv/sidecar/bert_seed2026_num{num}_valbal{val}_lam1e-06_{method}_signFalse_earlystopTrue_tmc500.json"
    else:
        p = f"{ROOT}/{ds}/{ratio}/{method}/eigen/sidecar/bert_seed2026_num{num}_valbal{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["top_results_inv"]

def report(title, ds, ratio, num, val):
    print(f"\n===== {title} =====\n")
    inv_lr = load(ds, ratio, "lrfshap", "inv", None, num, val)
    if inv_lr is None:
        print(f"  (INV baseline not yet available)")
        return
    print(f"INV (full kernel, LR == A1):")
    print(f"  sel% : " + "  ".join(f"{s:>3}%" for s in SELS))
    print(f"  acc  : " + "  ".join(f"{inv_lr[s-1]/100:>4.1f}" for s in SELS))

    print(f"\nO/X/- of A1 vs LR  (eigen r% × sel%):")
    print(f"  {'rank':>5}  " + "  ".join(f"sel{s:>2}%" for s in SELS))
    any_done = False
    for r in RANKS:
        lr = load(ds, ratio, "lrfshap", "eigen", r, num, val)
        a1 = load(ds, ratio, "a1",      "eigen", r, num, val)
        if lr is None and a1 is None:
            print(f"  r={r:>3}%  " + "  ".join(" .. " for _ in SELS) + "  (both missing)")
            continue
        if lr is None or a1 is None:
            who = "lr only" if a1 is None else "a1 only"
            print(f"  r={r:>3}%  " + "  ".join(" ?? " for _ in SELS) + f"  ({who})")
            continue
        any_done = True
        row = [f"r={r:>3}%"]
        for s in SELS:
            if a1[s-1] == lr[s-1]: tok = " -  "
            elif a1[s-1] > lr[s-1]: tok = " O  "
            else: tok = " X  "
            row.append(tok)
        print(f"  {row[0]}  " + "".join(row[1:]))

    if any_done:
        print(f"\nLR / A1 actual acc (sel × rank):")
        print(f"  {'rank':>5}  " + "    ".join(f"sel{s:>2}% LR/A1" for s in [1, 5, 10, 20]))
        for r in RANKS:
            lr = load(ds, ratio, "lrfshap", "eigen", r, num, val)
            a1 = load(ds, ratio, "a1",      "eigen", r, num, val)
            if lr is None or a1 is None:
                continue
            parts = [f"r={r:>3}%"]
            for s in [1, 5, 10, 20]:
                parts.append(f"{lr[s-1]/100:5.2f}/{a1[s-1]/100:5.2f}")
            print(f"  {parts[0]}  " + "    ".join(parts[1:]))

report("MRPC n=2300 valbal pos50  (50/50, val=258 balanced)",            "mrpc", "pos50", 2300, 258)
report("MRPC n=2300 valbal pos70  (70/30, val=258 balanced)",            "mrpc", "pos70", 2300, 258)
report("MRPC n=2300 valbal pos90  (90/10, val=258 balanced)  [partial]", "mrpc", "pos90", 2300, 258)
report("SST-2 n=5000 valbal pos50 (50/50, val=856 balanced)",            "sst2", "pos50", 5000, 856)
report("SST-2 n=5000 valbal pos70 (70/30, val=856 balanced)  [partial]", "sst2", "pos70", 5000, 856)
report("AG News n=5000 valbal cls25 (balanced 4-class, val=1000)",       "ag_news", "cls25_25_25_25", 5000, 1000)
