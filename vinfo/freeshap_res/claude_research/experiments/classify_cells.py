"""Cell-level classification: A1 vs LR at sel 1% (eigen r=10%).
Threshold: |A1 - LR| <= 3pp → ≈, A1 - LR > +3pp → ≫, < -3pp → <"""
import json, os, re

ROOT = "./freeshap_res"
R = 10
THR = 3.0  # pp

def parse_pred(path, mode="eigen"):
    """Return raw int list (×100), same unit as sidecar JSON."""
    if not os.path.exists(path): return None, None
    txt = open(path).read()
    m = re.search(rf"{mode} mode lambda[^\n]*\ntop:\s*\n\[([^\]]*)\]\s*\nrandom:\s*\n\[([^\]]*)\]", txt, re.DOTALL)
    if not m: return None, None
    return [int(x) for x in m.group(1).split(",")], [int(x) for x in m.group(2).split(",")]

def load_sidecar(p):
    return json.load(open(p)) if os.path.exists(p) else None

def at(L, sel):
    return None if L is None else L[sel-1] / 100.0

def baseline_lr(ds, val, n):
    cands = [
        f"{ROOT}/data_selection/{ds}/eigen/predictions/bert_seed2026_num{n}_val{val}_eig{R}_lam1e-02_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt",
        f"{ROOT}/data_selection/{ds}/eigen/predictions/bert_seed2026_num{n}_val{val}_eig{R}.0_eiglam1e-02_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt",
    ]
    for c in cands:
        t, _ = parse_pred(c, "eigen")
        if t is not None: return t
    return None

def baseline_a1(ds, val, n):
    p = f"{ROOT}/claude_research/data_selection_test/data_selection/{ds}/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{R}.0_eiglam1e-02_invlam1e-06_cholesky_float32_label_signFalse_earlystopTrue_tmc500.json"
    d = load_sidecar(p)
    return d["top_results_eigen"] if d else None

def imb_lr(ds, rtag, val, n):
    p = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/lrfshap/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{R}.0_eiglam1e-02_invlam1e-06_cholesky_float32_lrfshap_signFalse_earlystopTrue_tmc500.json"
    d = load_sidecar(p)
    return d["top_results_eigen"] if d else None

def imb_a1(ds, rtag, val, n):
    p = f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/{ds}/{rtag}/a1/eigen/sidecar/bert_seed2026_num{n}_val{val}_eig{R}.0_eiglam1e-02_invlam1e-06_cholesky_float32_a1_signFalse_earlystopTrue_tmc500.json"
    d = load_sidecar(p)
    return d["top_results_eigen"] if d else None

# (dataset, val, n_train, setting_label, src_lr, src_a1, imbalance_type)
# imbalance_type: 'balanced' / 'natural-imb' / 'mild-forced' / 'extreme-forced'
CELLS = [
    ("sst2", 872, 2000, "baseline 56/44 (자연)",      lambda: baseline_lr("sst2", 872, 2000),      lambda: baseline_a1("sst2", 872, 2000),      "natural-balanced"),
    ("sst2", 872, 2000, "pos70 (강제 70/30)",         lambda: imb_lr("sst2","pos70",872,2000),     lambda: imb_a1("sst2","pos70",872,2000),     "mild-forced"),
    ("sst2", 872, 2000, "pos90 (강제 90/10)",         lambda: imb_lr("sst2","pos90",872,2000),     lambda: imb_a1("sst2","pos90",872,2000),     "extreme-forced"),
    ("mr",  1000, 2000, "baseline 50/50 (자연)",      lambda: baseline_lr("mr",  1000, 2000),      lambda: baseline_a1("mr",  1000, 2000),      "natural-balanced"),
    ("mr",  1066, 2000, "pos70 (강제 70/30)",         lambda: imb_lr("mr","pos70",1066,2000),      lambda: imb_a1("mr","pos70",1066,2000),      "mild-forced"),
    ("mr",  1066, 2000, "pos90 (강제 90/10)",         lambda: imb_lr("mr","pos90",1066,2000),      lambda: imb_a1("mr","pos90",1066,2000),      "extreme-forced"),
    ("mnli",1000, 2000, "baseline 33/33/33 (자연)",   lambda: baseline_lr("mnli",1000, 2000),      lambda: baseline_a1("mnli",1000, 2000),      "natural-balanced"),
    ("mnli",1000, 2000, "cls60_20_20 (강제 60/20/20)",lambda: imb_lr("mnli","cls60_20_20",1000,2000),lambda: imb_a1("mnli","cls60_20_20",1000,2000),"mild-forced"),
    ("mnli",1000, 2000, "cls90_05_05 (강제 90/5/5)",  lambda: imb_lr("mnli","cls90_05_05",1000,2000),lambda: imb_a1("mnli","cls90_05_05",1000,2000),"extreme-forced"),
    ("qqp", 1000, 2000, "baseline 63/37 (자연 mild-imb)",lambda: baseline_lr("qqp",1000,2000),     lambda: baseline_a1("qqp",1000,2000),        "natural-imb"),
    ("qqp", 1000, 2000, "pos50 (강제 50/50)",         lambda: imb_lr("qqp","pos50",1000,2000),     lambda: imb_a1("qqp","pos50",1000,2000),     "forced-balanced"),
    ("qqp", 1000, 2000, "pos10 (강제 90/10)",         lambda: imb_lr("qqp","pos10",1000,2000),     lambda: imb_a1("qqp","pos10",1000,2000),     "extreme-forced"),
    ("rte", 277,  2000, "baseline 50/50 (자연)",      lambda: baseline_lr("rte",277,2000),         lambda: baseline_a1("rte",277,2000),         "natural-balanced"),
    ("rte", 277,  1300, "pos30 (강제 70/30, n=1300)", lambda: imb_lr("rte","pos30",277,1300),      lambda: imb_a1("rte","pos30",277,1300),      "mild-forced"),
    ("rte", 277,  1300, "pos10 (강제 90/10, n=1300)", lambda: imb_lr("rte","pos10",277,1300),      lambda: imb_a1("rte","pos10",277,1300),      "extreme-forced"),
    ("mrpc",408,  2000, "baseline 32/68 (자연 mid-imb)",lambda: baseline_lr("mrpc",408,2000),      lambda: baseline_a1("mrpc",408,2000),        "natural-imb"),
    ("mrpc",408,  2000, "pos50 (강제 50/50)",         lambda: imb_lr("mrpc","pos50",408,2000),     lambda: imb_a1("mrpc","pos50",408,2000),     "forced-balanced"),
    ("mrpc",408,  2000, "pos90 (강제 90/10)",         lambda: imb_lr("mrpc","pos90",408,2000),     lambda: imb_a1("mrpc","pos90",408,2000),     "extreme-forced"),
    ("ag_news",1000,2000,"baseline 25/25/25/25 (자연)",lambda: baseline_lr("ag_news",1000,2000),   lambda: baseline_a1("ag_news",1000,2000),    "natural-balanced"),
    ("ag_news",1000,2000,"cls55_15_15_15 (mild)",      lambda: imb_lr("ag_news","cls55_15_15_15",1000,2000),lambda: imb_a1("ag_news","cls55_15_15_15",1000,2000),"mild-forced"),
    ("ag_news",1000,2000,"cls85_05_05_05 (extreme)",   lambda: imb_lr("ag_news","cls85_05_05_05",1000,2000),lambda: imb_a1("ag_news","cls85_05_05_05",1000,2000),"extreme-forced"),
]

def classify(delta):
    if delta is None: return "n/a"
    if delta >= THR: return "A1≫LR"
    if delta <= -THR: return "A1<LR"
    return "A1≈LR"

print(f"\n{'dataset':<10} {'setting':<32} {'type':<18}  sel1%(LR/A1/Δ)        sel5%(LR/A1/Δ)         class@1%  class@5%")
print("-" * 130)
rows = []
for ds, val, n, label, lr_fn, a1_fn, itype in CELLS:
    lr = lr_fn(); a1 = a1_fn()
    lr1 = at(lr, 1); a11 = at(a1, 1)
    lr5 = at(lr, 5); a15 = at(a1, 5)
    d1 = None if (lr1 is None or a11 is None) else a11 - lr1
    d5 = None if (lr5 is None or a15 is None) else a15 - lr5
    c1 = classify(d1); c5 = classify(d5)
    def f(x): return f"{x:6.2f}" if x is not None else " n/a "
    def fd(x): return f"{x:+6.2f}" if x is not None else " n/a "
    print(f"{ds:<10} {label:<32} {itype:<18}  {f(lr1)}/{f(a11)}/{fd(d1)}    {f(lr5)}/{f(a15)}/{fd(d5)}    {c1:<8} {c5:<8}")
    rows.append((ds, label, itype, c1, c5))

# Aggregate counts
from collections import Counter
print("\n=== 비율 카운트 (sel 1% 기준) ===")
c1_counts = Counter(r[3] for r in rows)
total = sum(c1_counts.values())
for k, v in sorted(c1_counts.items()):
    print(f"  {k}: {v}/{total} ({v/total*100:.1f}%)")

print("\n=== 비율 카운트 (sel 5% 기준) ===")
c5_counts = Counter(r[4] for r in rows)
for k, v in sorted(c5_counts.items()):
    print(f"  {k}: {v}/{total} ({v/total*100:.1f}%)")

print("\n=== type 별 sel 1% 분류 ===")
from collections import defaultdict
by_type = defaultdict(lambda: Counter())
for r in rows:
    by_type[r[2]][r[3]] += 1
for t, c in sorted(by_type.items()):
    t_total = sum(c.values())
    parts = ", ".join(f"{k}: {v}/{t_total}" for k, v in sorted(c.items()))
    print(f"  {t:<18} → {parts}")
