"""Per-dataset summary at eigen r=10% (fixed rank).
For each dataset we emit:
  Table A — top-r Shapley 의 majority class % (LR vs A1) by (setting × sel%)
  Table B — accuracy (random / INV / LR / A1) by (setting × sel%)
"""
from datasets import load_dataset
import json, os, re, numpy as np

ROOT = "./freeshap_res"
R_FIXED = 10  # eigen rank %

def load_idx(p, n_keep=None):
    if not os.path.exists(p): return None
    a = np.array([int(l.strip()) for l in open(p) if l.strip()])
    return a[:n_keep] if n_keep else a

def load_sidecar(p):
    return json.load(open(p)) if os.path.exists(p) else None

def parse_pred(path, mode="inv"):
    if not os.path.exists(path): return None, None
    txt = open(path).read()
    m = re.search(rf"{mode} mode lambda[^\n]*\ntop:\s*\n\[([^\]]*)\]\s*\nrandom:\s*\n\[([^\]]*)\]", txt, re.DOTALL)
    if not m: return None, None
    return [int(x) for x in m.group(1).split(",")], [int(x) for x in m.group(2).split(",")]

def at(L, sel):
    return None if L is None else L[sel-1] / 100

def majority_pct(label_arr):
    """Return % of the most common class in this slice."""
    if len(label_arr) == 0: return None
    _, counts = np.unique(label_arr, return_counts=True)
    return counts.max() / len(label_arr) * 100

DATASETS = [
    ("sst2", "sst2", None, 872, 2000, [
        ("baseline 56/44 자연", "baseline"),
        ("pos70", ("imb", "pos70")),
        ("pos90", ("imb", "pos90")),
    ]),
    ("mr", "rotten_tomatoes", None, 1066, 2000, [
        ("pos70", ("imb", "pos70")),
        ("pos90", ("imb", "pos90")),
    ]),
    ("mnli", "glue", "mnli", 1000, 2000, [
        ("baseline 33/33/33 자연", "baseline"),
        ("cls60_20_20 (60/20/20)", ("imb", "cls60_20_20")),
        ("cls90_05_05 (90/5/5)", ("imb", "cls90_05_05")),
    ]),
    ("qqp", "glue", "qqp", 1000, 2000, [
        ("baseline 63/37 자연", "baseline"),
        ("pos50 (강제 50/50)", ("imb", "pos50")),
        ("pos10 (label0 maj 90%)", ("imb", "pos10")),
    ]),
    ("rte", "glue", "rte", 277, 1300, [
        ("pos30 (label0 maj 70%)", ("imb", "pos30")),
        ("pos10 (label0 maj 90%)", ("imb", "pos10")),
    ]),
    ("mrpc", "glue", "mrpc", 408, 2000, [
        ("baseline 자연 32/68 (label1 maj)", "baseline"),
        ("pos50 (강제 balanced)", ("imb", "pos50")),
        ("pos90 (label1 maj 90%)", ("imb", "pos90")),
    ]),
    ("ag_news", "ag_news", None, 1000, 2000, [
        ("baseline 25/25/25/25 자연", "baseline"),
        ("cls55_15_15_15", ("imb", "cls55_15_15_15")),
        ("cls85_05_05_05", ("imb", "cls85_05_05_05")),
    ]),
]

def baseline_lr_idx(ds, val, n):
    return load_idx(f"{ROOT}/data_selection/{ds}/inv/indices/bert_seed2026_num{n}_val{val}_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt", n)
def baseline_a1_idx(ds, val, n):
    return load_idx(f"{ROOT}/claude_research/data_selection_test/data_selection/{ds}/eigen/indices/bert_seed2026_num{n}_val{val}_eig{R_FIXED}.0_eiglam1e-02_invlam1e-06_cholesky_float32_label_signFalse_earlystopTrue_tmc500_indices.txt", n)
def baseline_lr_eig_pred(ds, val, n):
    return parse_pred(f"{ROOT}/data_selection/{ds}/eigen/predictions/bert_seed2026_num{n}_val{val}_eig{R_FIXED}_lam1e-02_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt", "eigen")
def baseline_a1_eig_side(ds, val, n):
    return load_sidecar(f"{ROOT}/claude_research/data_selection_test/data_selection/{ds}/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{R_FIXED}.0_eiglam1e-02_invlam1e-06_cholesky_float32_label_signFalse_earlystopTrue_tmc500.json")
def baseline_inv_pred(ds, val, n):
    return parse_pred(f"{ROOT}/data_selection/{ds}/inv/predictions/bert_seed2026_num{n}_val{val}_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt", "inv")

def imb_idx(ds, rtag, method, val, n):
    return load_idx(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/{method}/eigen/indices/bert_seed2026_num{n}_val{val}_eig{R_FIXED}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500_indices.txt", n)
def imb_eig_side(ds, rtag, method, val, n):
    return load_sidecar(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/{method}/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{R_FIXED}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json")
def imb_inv_side(ds, rtag, method, val, n):
    return load_sidecar(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/{method}/inv/sidecar/bert_seed2026_num{n}_val{val}_lam1e-06_{method}_signFalse_earlystopTrue_tmc500.json")

SEL_LIST = [1, 2, 5, 10, 20]

def ds_train_labels(hf_name, hf_config):
    if hf_config:
        return np.array(load_dataset(hf_name, hf_config)["train"]["label"])
    return np.array(load_dataset(hf_name)["train"]["label"])

for ds_name, hf_name, hf_config, val, n_train, settings in DATASETS:
    print("\n" + "="*100)
    print(f"  Dataset: {ds_name.upper()}  (n_train={n_train}, val={val}, eigen rank fixed at {R_FIXED}%)")
    print("="*100)
    full_train_labels = ds_train_labels(hf_name, hf_config)

    # Gather all data per setting
    settings_data = []
    for setting_label, source in settings:
        if source == "baseline":
            idx_lr  = baseline_lr_idx(ds_name, val, n_train)         # INV idx == LR eigen idx for natural baseline w/o monkey-patch? Actually upstream LR idx is INV. For *eigen* LR baseline indices we'd need eigen sidecar. Use INV indices as approximation (LR top by Shapley value sort).
            idx_a1  = baseline_a1_idx(ds_name, val, n_train)
            lr_eig  = baseline_lr_eig_pred(ds_name, val, n_train)    # (top, rand)
            a1_side = baseline_a1_eig_side(ds_name, val, n_train)
            inv_pr  = baseline_inv_pred(ds_name, val, n_train)
        else:
            rtag = source[1]
            idx_lr  = imb_idx(ds_name, rtag, "lrfshap", val, n_train)
            idx_a1  = imb_idx(ds_name, rtag, "a1",      val, n_train)
            lr_side = imb_eig_side(ds_name, rtag, "lrfshap", val, n_train)
            a1_side = imb_eig_side(ds_name, rtag, "a1",      val, n_train)
            lr_eig  = (lr_side["top_results_eigen"], lr_side["random_results_eigen"]) if lr_side else None
            inv_s   = imb_inv_side(ds_name, rtag, "lrfshap", val, n_train)
            inv_pr  = (inv_s["top_results_inv"], inv_s["random_results_inv"]) if inv_s else None
        settings_data.append((setting_label, idx_lr, idx_a1, lr_eig, a1_side, inv_pr))

    # ---- Table A: top-r 의 majority class % ----
    print(f"\n[A] Top Shapley sample 의 majority class % (eigen r={R_FIXED}%; LR vs A1)")
    print(f"    pool 비율 dataset별 표시:")
    for label, idx_lr, idx_a1, *_ in settings_data:
        idx_show = idx_lr if idx_lr is not None else idx_a1
        if idx_show is None:
            print(f"      {label:<30}: indices missing"); continue
        pool = full_train_labels[idx_show]
        u, c = np.unique(pool, return_counts=True)
        print(f"      {label:<30}: pool {dict(zip(u.tolist(), c.tolist()))}")

    header = f"    {'setting':<30}  " + "  ".join(f"sel{s:>2}%(LR/A1)" for s in SEL_LIST)
    print()
    print(header)
    for label, idx_lr, idx_a1, *_ in settings_data:
        cells = []
        for s in SEL_LIST:
            k = int(n_train * s / 100)
            lr_v = majority_pct(full_train_labels[idx_lr[:k]]) if idx_lr is not None else None
            a1_v = majority_pct(full_train_labels[idx_a1[:k]]) if idx_a1 is not None else None
            def f(x): return f"{x:5.1f}" if x is not None else " n/a "
            cells.append(f"  {f(lr_v)}/{f(a1_v)}")
        print(f"    {label:<30}  " + " ".join(cells))

    # ---- Table B: accuracy ----
    print(f"\n[B] Selection accuracy (eigen r={R_FIXED}%) — random / INV-top / LR-top / A1-top")
    header = f"    {'setting':<30}  " + "  ".join(f"sel{s:>2}%(R/INV/LR/A1)" for s in SEL_LIST)
    print(header)
    for label, idx_lr, idx_a1, lr_eig, a1_side, inv_pr in settings_data:
        cells = []
        for s in SEL_LIST:
            rnd  = at(lr_eig[1], s) if lr_eig else (at(a1_side["random_results_eigen"], s) if a1_side else None)
            inv  = at(inv_pr[0], s) if inv_pr else None
            lr_t = at(lr_eig[0], s) if lr_eig else None
            a1_t = at(a1_side["top_results_eigen"], s) if a1_side else None
            def f(x): return f"{x:5.2f}" if x is not None else " n/a "
            cells.append(f"  {f(rnd)}/{f(inv)}/{f(lr_t)}/{f(a1_t)}")
        print(f"    {label:<30}  " + " ".join(cells))
