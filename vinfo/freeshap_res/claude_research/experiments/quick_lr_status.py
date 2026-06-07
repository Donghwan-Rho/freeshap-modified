"""Quick LR-only status — MNLI cls60_20_20 + QQP pos50."""
import json, os

ROOT = "./freeshap_res/claude_research/data_selection_test/imbalance/data_selection"

def load_inv(ds, rtag, method, val):
    p = f"{ROOT}/{ds}/{rtag}/{method}/inv/sidecar/bert_seed2026_num2000_val{val}_lam1e-06_{method}_signFalse_earlystopTrue_tmc500.json"
    return json.load(open(p)) if os.path.exists(p) else None

def load_eig(ds, rtag, method, val, r):
    p = f"{ROOT}/{ds}/{rtag}/{method}/eigen/sidecar/bert_seed2026_num2000_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json"
    return json.load(open(p)) if os.path.exists(p) else None

def emit(title, ds, rtag, val):
    d_inv = load_inv(ds, rtag, "lrfshap", val)
    d_eig = load_eig(ds, rtag, "lrfshap", val, 10)
    if d_inv is None or d_eig is None:
        print(f"\n### {title}: data missing")
        return
    print(f"\n### {title}")
    print(f"  val_majority (class 0 frac) = {d_inv['acc_at_f0']/10000:.4f}")
    print(f"  sel | INV rand  INV top  Δ | r=10 rand  r=10 top  Δ")
    for sel in [1, 2, 5, 10, 15, 20, 30, 50, 100]:
        i = sel - 1
        ir = d_inv["random_results_inv"][i] / 100
        it = d_inv["top_results_inv"][i] / 100
        er = d_eig["random_results_eigen"][i] / 100
        et = d_eig["top_results_eigen"][i] / 100
        print(f"  {sel:>3}% | {ir:>7.2f}   {it:>5.2f}  {it-ir:+5.2f} | {er:>7.2f}    {et:>5.2f}  {et-er:+5.2f}")

emit("MNLI cls60_20_20 (LR only, A1 진행 중)", "mnli", "cls60_20_20", 1000)
emit("QQP pos50 (LR only, A1 진행 중)", "qqp", "pos50", 1000)
