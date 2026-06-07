"""Top vs random baseline comparison for SST-2 and MR, in 50/50 and 70/30 settings.

50/50 = baseline natural distribution (existing FreeShap + LRFShap + A1 results).
70/30 = our imbalance pos70 enforcement.

For each (dataset, setting, method, approximate, sel%), we look at:
  top_acc (Shapley top-r selection)
  random_acc (uniform random subset of same size)
  Δ = top - random
"""
import json
import glob
import re
import numpy as np

ROOT = "./freeshap_res"

def load_sidecar(p):
    return json.load(open(p))

def parse_pred(path, mode="inv"):
    """Extract top + random list from predictions.txt for the requested mode."""
    txt = open(path).read()
    # Match "<mode> mode lambda" then top: [..] then random: [..]
    m = re.search(rf"{mode} mode lambda[^\n]*\ntop:\s*\n\[([^\]]*)\]\s*\nrandom:\s*\n\[([^\]]*)\]", txt, re.DOTALL)
    if not m:
        return None, None
    top = [int(x.strip()) for x in m.group(1).split(",")]
    rnd = [int(x.strip()) for x in m.group(2).split(",")]
    return top, rnd

# ----- SETTING / METHOD selectors -----

def baseline_inv(ds, val):
    """Existing LRFShap inv (= A1 inv since monkey-patch only affects eigen)."""
    p = (f"{ROOT}/data_selection/{ds}/inv/predictions/"
         f"bert_seed2026_num2000_val{val}_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt")
    return parse_pred(p, mode="inv")

def baseline_eigen(ds, val, r, method):
    """Existing LRFShap/A1 eigen at rank r%. method='lrfshap' uses upstream, 'a1' uses claude_research label-aware."""
    import os
    if method == "lrfshap":
        # Upstream task_data_selection.py — two filename conventions exist.
        cands = [
            f"{ROOT}/data_selection/{ds}/eigen/predictions/"
            f"bert_seed2026_num2000_val{val}_eig{r}_lam1e-02_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt",
            f"{ROOT}/data_selection/{ds}/eigen/predictions/"
            f"bert_seed2026_num2000_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt",
        ]
    else:
        cands = [
            f"{ROOT}/claude_research/data_selection_test/data_selection/{ds}/eigen/predictions/"
            f"bert_seed2026_num2000_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_label_signFalse_earlystopTrue_tmc500_predictions.txt",
        ]
    for p in cands:
        if os.path.exists(p):
            return parse_pred(p, mode="eigen")
    return None, None

def imb_inv(ds, val, method):
    """pos70 inv (lrfshap == a1)."""
    p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/pos70/{method}/inv/sidecar/"
         f"bert_seed2026_num2000_val{val}_lam1e-06_{method}_signFalse_earlystopTrue_tmc500.json")
    d = load_sidecar(p)
    return d["top_results_inv"], d["random_results_inv"]

def imb_eigen(ds, val, r, method):
    """pos70 eigen at rank r%."""
    import os
    p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/pos70/{method}/eigen/sidecar/"
         f"bert_seed2026_num2000_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json")
    if not os.path.exists(p):
        return None, None
    d = load_sidecar(p)
    return d["top_results_eigen"], d["random_results_eigen"]

# ----- helpers -----
def at(list_, sel):
    """list_ is length-100 sorted by sel% 1..100; pick index sel-1."""
    if list_ is None: return None
    return list_[sel-1] / 100  # int *100 -> percent

# ----- analysis -----
DATASETS = [("sst2", 872), ("mr", 1000)]
DATASETS_IMB = [("sst2", 872), ("mr", 1066)]
SEL_LIST = [1, 2, 5, 10, 15, 20, 30, 50, 100]
RANKS = [1, 5, 10, 30]

for (ds, val_b), (_, val_i) in zip(DATASETS, DATASETS_IMB):
    print(f"\n{'='*100}\n  Dataset: {ds.upper()}  (baseline val={val_b}, imb pos70 val={val_i})\n{'='*100}")

    # INV mode
    bt_inv, br_inv = baseline_inv(ds, val_b)
    it_inv, ir_inv = imb_inv(ds, val_i, "lrfshap")  # a1 inv is identical
    print(f"\n  -- INV mode (single method; LRFShap inv == A1 inv) --")
    print(f"  {'sel':>4} | {'50/50_top':>10} {'50/50_rand':>11} {'Δ':>6} | "
          f"{'70/30_top':>10} {'70/30_rand':>11} {'Δ':>6}")
    for s in SEL_LIST:
        bt = at(bt_inv, s); br = at(br_inv, s)
        it = at(it_inv, s); ir = at(ir_inv, s)
        print(f"  {s:>3}% | {bt:>9.2f} {br:>10.2f} {bt-br:>+5.2f} | "
              f"{it:>9.2f} {ir:>10.2f} {it-ir:>+5.2f}")

    # EIGEN mode
    for r in RANKS:
        print(f"\n  -- EIGEN r={r}% (LRFShap vs A1) --")
        bt_l, br_l = baseline_eigen(ds, val_b, r, "lrfshap")
        bt_a, br_a = baseline_eigen(ds, val_b, r, "a1")
        it_l, ir_l = imb_eigen(ds, val_i, r, "lrfshap")
        it_a, ir_a = imb_eigen(ds, val_i, r, "a1")
        print(f"  {'sel':>4} | {'50/50_LR_top':>11} {'50/50_A1_top':>11} {'50/50_rand':>11} | "
              f"{'70/30_LR_top':>11} {'70/30_A1_top':>11} {'70/30_rand':>11}")
        for s in SEL_LIST:
            bl = at(bt_l, s); ba = at(bt_a, s); brs = at(br_l, s)
            il = at(it_l, s); ia = at(it_a, s); irs = at(ir_l, s) if ir_l is not None else None
            def f(x): return f"{x:>9.2f}" if x is not None else "    n/a  "
            print(f"  {s:>3}% | {f(bl):>11} {f(ba):>11} {f(brs):>11} | "
                  f"{f(il):>11} {f(ia):>11} {f(irs):>11}")
