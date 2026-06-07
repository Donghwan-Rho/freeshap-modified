"""Comprehensive summary — 50/50 baseline + 70/30 imb + 90/10 imb across SST-2/MR.

For each (dataset, setting, sel%):
  random  — uniform baseline
  INV     — full kernel Shapley (LR ≡ A1 in inv mode)
  EIGEN r=10% LR  — top-r by lambda Shapley + r=10% eigen prediction
  EIGEN r=10% A1  — top-r by supervised Shapley + r=10% eigen prediction
"""
import json, re, os

ROOT = "./freeshap_res"

def parse_pred(path, mode="inv"):
    if not os.path.exists(path):
        return None, None
    txt = open(path).read()
    m = re.search(rf"{mode} mode lambda[^\n]*\ntop:\s*\n\[([^\]]*)\]\s*\nrandom:\s*\n\[([^\]]*)\]", txt, re.DOTALL)
    if not m:
        return None, None
    top = [int(x.strip()) for x in m.group(1).split(",")]
    rnd = [int(x.strip()) for x in m.group(2).split(",")]
    return top, rnd

def load_sidecar(p):
    if not os.path.exists(p): return None
    return json.load(open(p))

def at(L, sel):
    return None if L is None else L[sel-1] / 100

def baseline_inv(ds, val):
    p = f"{ROOT}/data_selection/{ds}/inv/predictions/bert_seed2026_num2000_val{val}_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt"
    return parse_pred(p, "inv")

def baseline_eig_lr(ds, val, r):
    p = f"{ROOT}/data_selection/{ds}/eigen/predictions/bert_seed2026_num2000_val{val}_eig{r}_lam1e-02_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt"
    return parse_pred(p, "eigen")

def baseline_eig_a1(ds, val, r):
    p = f"{ROOT}/claude_research/data_selection_test/data_selection/{ds}/eigen/predictions/bert_seed2026_num2000_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_label_signFalse_earlystopTrue_tmc500_predictions.txt"
    return parse_pred(p, "eigen")

def imb_inv(ds, val, ratio, method):
    p = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/pos{ratio}/{method}/inv/sidecar/bert_seed2026_num2000_val{val}_lam1e-06_{method}_signFalse_earlystopTrue_tmc500.json"
    d = load_sidecar(p)
    return (d["top_results_inv"], d["random_results_inv"]) if d else (None, None)

def imb_eig(ds, val, r, ratio, method):
    p = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/pos{ratio}/{method}/eigen/sidecar/bert_seed2026_num2000_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json"
    d = load_sidecar(p)
    return (d["top_results_eigen"], d["random_results_eigen"]) if d else (None, None)

SEL_LIST = [1, 2, 5, 10, 15, 20, 30, 50, 100]
R_FOCUS = 10  # eigen rank %

datasets = [("sst2", 872, 872), ("mr", 1000, 1066)]

for ds, val_b, val_i in datasets:
    print(f"\n{'#'*92}\n#  Dataset: {ds.upper()}\n{'#'*92}")

    # Load all
    bt_inv,  br_inv  = baseline_inv(ds, val_b)
    bt_lr,   br_lr   = baseline_eig_lr(ds, val_b, R_FOCUS)
    bt_a1,   br_a1   = baseline_eig_a1(ds, val_b, R_FOCUS)

    for ratio in [70, 90]:
        # inv (LR==A1)
        globals()[f"it_inv_{ratio}"], globals()[f"ir_inv_{ratio}"] = imb_inv(ds, val_i, ratio, "lrfshap")
        # eigen LR
        globals()[f"it_lr_{ratio}"], globals()[f"ir_lr_{ratio}"] = imb_eig(ds, val_i, R_FOCUS, ratio, "lrfshap")
        # eigen A1
        globals()[f"it_a1_{ratio}"], globals()[f"ir_a1_{ratio}"] = imb_eig(ds, val_i, R_FOCUS, ratio, "a1")

    # ===== TABLE 1: INV mode =====
    print(f"\n  TABLE 1 — INV mode (full kernel; LRFShap ≡ A1)")
    print(f"  {'sel':>4} | {'50/50_rand':>10} {'50/50_top':>10} | {'70/30_rand':>10} {'70/30_top':>10} | {'90/10_rand':>10} {'90/10_top':>10}")
    for s in SEL_LIST:
        br = at(br_inv, s); bt = at(bt_inv, s)
        ir70 = at(globals()['ir_inv_70'], s); it70 = at(globals()['it_inv_70'], s)
        ir90 = at(globals()['ir_inv_90'], s); it90 = at(globals()['it_inv_90'], s)
        def f(x): return f"{x:>9.2f}" if x is not None else "    n/a  "
        print(f"  {s:>3}% | {f(br)} {f(bt)} | {f(ir70)} {f(it70)} | {f(ir90)} {f(it90)}")

    # ===== TABLE 2: EIGEN r=10% LR =====
    print(f"\n  TABLE 2 — EIGEN r={R_FOCUS}%  LRFShap (top-r by lambda)")
    print(f"  {'sel':>4} | {'50/50_rand':>10} {'50/50_top':>10} | {'70/30_rand':>10} {'70/30_top':>10} | {'90/10_rand':>10} {'90/10_top':>10}")
    for s in SEL_LIST:
        br = at(br_lr, s); bt = at(bt_lr, s)
        ir70 = at(globals()['ir_lr_70'], s); it70 = at(globals()['it_lr_70'], s)
        ir90 = at(globals()['ir_lr_90'], s); it90 = at(globals()['it_lr_90'], s)
        def f(x): return f"{x:>9.2f}" if x is not None else "    n/a  "
        print(f"  {s:>3}% | {f(br)} {f(bt)} | {f(ir70)} {f(it70)} | {f(ir90)} {f(it90)}")

    # ===== TABLE 3: EIGEN r=10% A1 =====
    print(f"\n  TABLE 3 — EIGEN r={R_FOCUS}%  A1 (top-r by supervised score)")
    print(f"  {'sel':>4} | {'50/50_rand':>10} {'50/50_top':>10} | {'70/30_rand':>10} {'70/30_top':>10} | {'90/10_rand':>10} {'90/10_top':>10}")
    for s in SEL_LIST:
        br = at(br_a1, s); bt = at(bt_a1, s)
        ir70 = at(globals()['ir_a1_70'], s); it70 = at(globals()['it_a1_70'], s)
        ir90 = at(globals()['ir_a1_90'], s); it90 = at(globals()['it_a1_90'], s)
        def f(x): return f"{x:>9.2f}" if x is not None else "    n/a  "
        print(f"  {s:>3}% | {f(br)} {f(bt)} | {f(ir70)} {f(it70)} | {f(ir90)} {f(it90)}")
