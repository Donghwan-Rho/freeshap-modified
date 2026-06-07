"""Unified summary across all datasets — per-dataset section showing:
  (a) pool composition for each setting
  (b) top 1% / 5% class composition (LRFShap inv mode)
  (c) accuracy at sel 1% / 5% / 10% (random / LRFShap / A1, eigen r=10%)
"""
from datasets import load_dataset
import json, os, numpy as np

ROOT = "./freeshap_res"

def load_idx(p, n_keep=None):
    if not os.path.exists(p): return None
    idx = np.array([int(l.strip()) for l in open(p) if l.strip()])
    return idx[:n_keep] if n_keep else idx

def load_sidecar(p):
    return json.load(open(p)) if os.path.exists(p) else None

def at(L, sel):
    return None if L is None else L[sel-1] / 100

# ----- dataset registry -----
# baseline (자연) uses 사용자 기존 폴더; imbalance uses our claude_research path.
def baseline_lr_inv_idx(ds, val, n):
    return load_idx(f"{ROOT}/data_selection/{ds}/inv/indices/bert_seed2026_num{n}_val{val}_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt", n)

def baseline_lr_inv_pred(ds, val, n):
    p = f"{ROOT}/data_selection/{ds}/inv/predictions/bert_seed2026_num{n}_val{val}_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt"
    if not os.path.exists(p): return None
    import re
    txt = open(p).read()
    m = re.search(r"inv mode lambda[^\n]*\ntop:\s*\n\[([^\]]*)\]\s*\nrandom:\s*\n\[([^\]]*)\]", txt, re.DOTALL)
    if not m: return None
    return ([int(x) for x in m.group(1).split(",")], [int(x) for x in m.group(2).split(",")])

def baseline_a1_eig_side(ds, val, n, r):
    p = f"{ROOT}/claude_research/data_selection_test/data_selection/{ds}/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_label_signFalse_earlystopTrue_tmc500.json"
    return load_sidecar(p)

def imb_idx(ds, rtag, method, val, n):
    return load_idx(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/{method}/inv/indices/bert_seed2026_num{n}_val{val}_lam1e-06_{method}_signFalse_earlystopTrue_tmc500_indices.txt", n)

def imb_inv_side(ds, rtag, method, val, n):
    return load_sidecar(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/{method}/inv/sidecar/bert_seed2026_num{n}_val{val}_lam1e-06_{method}_signFalse_earlystopTrue_tmc500.json")

def imb_eig_side(ds, rtag, method, val, n, r):
    return load_sidecar(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/{method}/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_{method}_signFalse_earlystopTrue_tmc500.json")

DATASETS = [
    # (name, hf_name, hf_config, val, n_train, n_classes, settings)
    # settings: list of (label, source) — source = 'baseline' (자연) or ('imb', rtag)
    ("sst2", "sst2", None, 872, 2000, 2, [
        ("baseline 자연 56/44 (label1 maj)", "baseline"),
        ("pos70 (label1 70%)", ("imb", "pos70")),
        ("pos90 (label1 90%)", ("imb", "pos90")),
    ]),
    ("mr", "rotten_tomatoes", None, 1066, 2000, 2, [
        ("baseline 자연 52/48 (label1 maj)", "baseline"),
        ("pos70 (label1 70%)", ("imb", "pos70")),
        ("pos90 (label1 90%)", ("imb", "pos90")),
    ]),
    ("mnli", "glue", "mnli", 1000, 2000, 3, [
        ("baseline 자연 33/33/33", "baseline"),
        ("cls60_20_20 (label0 60% maj)", ("imb", "cls60_20_20")),
    ]),
    ("qqp", "glue", "qqp", 1000, 2000, 2, [
        ("baseline 자연 63/37 (label0 maj)", "baseline"),
        ("pos50 (강제 balanced)", ("imb", "pos50")),
    ]),
    ("rte", "glue", "rte", 277, 1300, 2, [
        # baseline 은 RTE n=2490 또는 n=2000 setup 의 자연 분포; rte n=1300 의 새 imbalance 만 표시
        ("pos30 (label1 30%, label0 maj 70%)", ("imb", "pos30")),
    ]),
]
MRPC_VAL_N = (408, 2000)  # placeholder, will add when ready
AG_NEWS_VAL_N = (1000, 2000)  # placeholder

def ds_labels(ds_name, hf_name, hf_config):
    if hf_config:
        ds = load_dataset(hf_name, hf_config)
        if ds_name == "mnli":
            return np.array(ds["train"]["label"]), np.array(ds["validation_matched"]["label"])
    else:
        ds = load_dataset(hf_name)
    return np.array(ds["train"]["label"]), np.array(ds["validation"]["label"])

def fmt_class_pct(labels_top, n_classes):
    return [f"{(labels_top == c).sum() / len(labels_top) * 100:.1f}%" for c in range(n_classes)]

for ds_name, hf_name, hf_config, val, n_train, n_classes, settings in DATASETS:
    print("\n" + "=" * 90)
    print(f"  Dataset: {ds_name.upper()}  (n_train={n_train}, val={val}, classes={n_classes})")
    print("=" * 90)

    full_train_labels, val_labels = ds_labels(ds_name, hf_name, hf_config)
    val_majority_frac = max((val_labels == c).mean() for c in range(n_classes))
    val_majority_class = int(np.argmax([np.sum(val_labels == c) for c in range(n_classes)]))
    print(f"  val majority: class {val_majority_class} (frac={val_majority_frac:.3f})")

    rows = []
    for setting_label, source in settings:
        if source == "baseline":
            idx_lr = baseline_lr_inv_idx(ds_name, val, n_train)
            pred = baseline_lr_inv_pred(ds_name, val, n_train)
            d_a1 = baseline_a1_eig_side(ds_name, val, n_train, 10)
            # baseline LR eigen r=10% — sidecar path
            d_lr = None
            p_lr_pred = f"{ROOT}/data_selection/{ds_name}/eigen/predictions/bert_seed2026_num{n_train}_val{val}_eig10_lam1e-02_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt"
        else:
            tag = source[1]
            idx_lr = imb_idx(ds_name, tag, "lrfshap", val, n_train)
            inv_s = imb_inv_side(ds_name, tag, "lrfshap", val, n_train)
            pred = (inv_s["top_results_inv"], inv_s["random_results_inv"]) if inv_s else None
            d_a1 = imb_eig_side(ds_name, tag, "a1", val, n_train, 10)
            d_lr = imb_eig_side(ds_name, tag, "lrfshap", val, n_train, 10)
        rows.append((setting_label, idx_lr, pred, d_lr, d_a1))

    # --- pool composition ---
    print(f"\n  Pool composition (n={n_train}, LRFShap pool):")
    for label, idx, _, _, _ in rows:
        if idx is None:
            print(f"    {label}: (indices missing)"); continue
        pool = full_train_labels[idx]
        counts = [int((pool == c).sum()) for c in range(n_classes)]
        print(f"    {label:<45} {counts}")

    # --- top 1% / 5% class composition (LRFShap INV) ---
    print(f"\n  Top-r class composition (LRFShap, INV mode, k=20 for 1% / k=100 for 5%):")
    head = f"    {'setting':<45} "
    for sel_pct in [1, 5]:
        head += f"sel{sel_pct}% {'/'.join(f'c{c}' for c in range(n_classes)):<{4*n_classes+1}}  "
    print(head)
    for label, idx, _, _, _ in rows:
        if idx is None:
            print(f"    {label:<45} (no data)"); continue
        line = f"    {label:<45} "
        pool = full_train_labels[idx]
        for sel_pct in [1, 5]:
            k = int(n_train * sel_pct / 100)
            top_lab = pool[:k]
            pcts = fmt_class_pct(top_lab, n_classes)
            line += f"  {' '.join(pcts):<{6*n_classes}}  "
        print(line)

    # --- selection accuracy at sel 1, 5, 10 (eigen r=10%) ---
    print(f"\n  Selection accuracy (eigen r=10%) at sel 1% / 5% / 10%  [LR_top / A1_top / random]:")
    print(f"    {'setting':<45}    {'sel1%':>22}      {'sel5%':>22}      {'sel10%':>22}")
    for label, idx, pred, d_lr, d_a1 in rows:
        if d_lr is None and d_a1 is None:
            print(f"    {label:<45}    (eigen sidecars missing)")
            continue
        line = f"    {label:<45}"
        for sel in [1, 5, 10]:
            lr_t = at(d_lr["top_results_eigen"], sel) if d_lr else None
            a1_t = at(d_a1["top_results_eigen"], sel) if d_a1 else None
            rnd  = at(d_lr["random_results_eigen"], sel) if d_lr else (at(d_a1["random_results_eigen"], sel) if d_a1 else None)
            def f(x): return f"{x:5.2f}" if x is not None else " n/a "
            line += f"    {f(lr_t)} / {f(a1_t)} / {f(rnd)}"
        print(line)
