"""lrfshap_vs_a1.pdf — per-setting (2x4 fig + table) comparison
with random baseline.

A1-centric marker scheme (matches make_lrfshap_vs_a1_valbal_pdf.py):
    !   A1 <= random            (A1 itself harmful)
    O   A1 > LR  (and > random) (A1 strictly better than LR)
    X   A1 < LR  (A1 still > random)
    -   A1 == LR (A1 still > random)
    †   trailing flag: LR <= random (LR collapsed — regime A1 is designed for)
    *   missing data

Two data sources are handled here:
  * baseline (n=2000, natural val) -> predictions.txt files (top + random)
  * imb (forced ratio)            -> sidecar JSON (top_results_inv,
                                                    random_results_inv)
"""
import json, os, re, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = "./freeshap_res"
SEL_RANGE = list(range(1, 21))     # 1..20% on figure
RANKS = [1, 5, 10, 15, 20, 25, 30]
TABLE_SELS = [1, 2, 3, 4, 5, 10, 20]
OUT = "./freeshap_res/claude_research/reports/lrfshap_vs_a1.pdf"


# ---------- prediction.txt parser (top + random) ----------
def parse_pred(path, mode):
    """Parse a {mode} mode block from a predictions.txt file.

    Returns (top_list, random_list). Either element may be None if the
    file is missing or the regex fails.
    """
    if not os.path.exists(path):
        return None, None
    try:
        txt = open(path).read()
    except Exception as e:
        print(f"[warn] cannot read {path}: {e}", file=sys.stderr)
        return None, None
    m = re.search(
        rf"{mode} mode lambda[^\n]*\ntop:\s*\n\[([^\]]*)\]\s*\nrandom:\s*\n\[([^\]]*)\]",
        txt, re.DOTALL)
    if not m:
        return None, None
    try:
        top = [int(x) for x in m.group(1).split(",")]
        rnd = [int(x) for x in m.group(2).split(",")]
    except Exception as e:
        print(f"[warn] parse failure {path}: {e}", file=sys.stderr)
        return None, None
    return top, rnd


def load_sidecar(p):
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        print(f"[warn] cannot read {p}: {e}", file=sys.stderr)
        return None


# ---------- per-source data loaders ----------
def fs_inv(stg):
    """Full-Shapley INV. Returns (top, random)."""
    if stg['type'] == 'baseline':
        return parse_pred(
            f"{ROOT}/data_selection/{stg['ds']}/inv/predictions/"
            f"bert_seed2026_num{stg['n']}_val{stg['val']}_lam1e-06_signFalse_"
            f"earlystopTrue_tmc500_predictions.txt",
            "inv")
    p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/"
         f"{stg['ds']}/{stg['rtag']}/lrfshap/inv/sidecar/"
         f"bert_seed2026_num{stg['n']}_val{stg['val']}_lam1e-06_lrfshap_"
         f"signFalse_earlystopTrue_tmc500.json")
    d = load_sidecar(p)
    if d is None:
        return None, None
    return d.get("top_results_inv"), d.get("random_results_inv")


def lr_eig_inv(stg, r):
    """LRFShap eigen at rank r%. Returns (top_inv, random_inv)."""
    if stg['type'] == 'baseline':
        cands = [
            f"{ROOT}/data_selection/{stg['ds']}/eigen/predictions/"
            f"bert_seed2026_num{stg['n']}_val{stg['val']}_eig{r}_"
            f"lam1e-02_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt",
            f"{ROOT}/data_selection/{stg['ds']}/eigen/predictions/"
            f"bert_seed2026_num{stg['n']}_val{stg['val']}_eig{r}.0_"
            f"eiglam1e-02_invlam1e-06_cholesky_float32_signFalse_"
            f"earlystopTrue_tmc500_predictions.txt",
        ]
        for c in cands:
            if os.path.exists(c):
                return parse_pred(c, "inv")
        return None, None
    p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/"
         f"{stg['ds']}/{stg['rtag']}/lrfshap/eigen/sidecar/"
         f"bert_seed2026_num{stg['n']}_val{stg['val']}_eig{r}.0_eiglam1e-02_"
         f"invlam1e-06_cholesky_float32_lrfshap_signFalse_earlystopTrue_tmc500.json")
    d = load_sidecar(p)
    if d is None:
        return None, None
    return d.get("top_results_inv"), d.get("random_results_inv")


def a1_eig_inv(stg, r):
    """A1 eigen at rank r%. Returns (top_inv, random_inv).

    Baseline A1 lives under the `claude_research/data_selection_test/data_selection/`
    tree (label_*) — its sidecar likewise carries random_results_inv.
    """
    if stg['type'] == 'baseline':
        p = (f"{ROOT}/claude_research/data_selection_test/data_selection/"
             f"{stg['ds']}/eigen/sidecar/"
             f"bert_seed2026_num{stg['n']}_val{stg['val']}_eig{r}.0_eiglam1e-02_"
             f"invlam1e-06_cholesky_float32_label_signFalse_earlystopTrue_tmc500.json")
        d = load_sidecar(p)
        if d is None:
            return None, None
        return d.get("top_results_inv"), d.get("random_results_inv")
    p = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/"
         f"{stg['ds']}/{stg['rtag']}/a1/eigen/sidecar/"
         f"bert_seed2026_num{stg['n']}_val{stg['val']}_eig{r}.0_eiglam1e-02_"
         f"invlam1e-06_cholesky_float32_a1_signFalse_earlystopTrue_tmc500.json")
    d = load_sidecar(p)
    if d is None:
        return None, None
    return d.get("top_results_inv"), d.get("random_results_inv")


# ---------- settings ----------
SETTINGS = [
    {"name":"sst2_baseline",      "ds":"sst2",   "val":872,  "n":2000, "type":"baseline", "label":"SST-2 baseline (natural ~56/44, label1 majority)"},
    {"name":"sst2_70_30",         "ds":"sst2",   "val":872,  "n":2000, "type":"imb", "rtag":"pos70", "label":"SST-2 forced 70/30 (label1 majority)"},
    {"name":"sst2_90_10",         "ds":"sst2",   "val":872,  "n":2000, "type":"imb", "rtag":"pos90", "label":"SST-2 forced 90/10 (label1 majority)"},

    {"name":"mr_baseline",        "ds":"mr",     "val":1000, "n":2000, "type":"baseline", "label":"MR baseline (natural ~50/50)"},
    {"name":"mr_70_30",           "ds":"mr",     "val":1066, "n":2000, "type":"imb", "rtag":"pos70", "label":"MR forced 70/30 (label1 majority)"},
    {"name":"mr_90_10",           "ds":"mr",     "val":1066, "n":2000, "type":"imb", "rtag":"pos90", "label":"MR forced 90/10 (label1 majority)"},

    {"name":"mnli_baseline",      "ds":"mnli",   "val":1000, "n":2000, "type":"baseline", "label":"MNLI baseline (natural ~33/33/33)"},
    {"name":"mnli_60_20_20",      "ds":"mnli",   "val":1000, "n":2000, "type":"imb", "rtag":"cls60_20_20", "label":"MNLI forced 60/20/20 (label0 majority)"},
    {"name":"mnli_90_5_5",        "ds":"mnli",   "val":1000, "n":2000, "type":"imb", "rtag":"cls90_05_05", "label":"MNLI forced 90/5/5 (label0 majority)"},

    {"name":"qqp_baseline",       "ds":"qqp",    "val":1000, "n":2000, "type":"baseline", "label":"QQP baseline (natural ~63/37, label0 majority)"},
    {"name":"qqp_50_50",          "ds":"qqp",    "val":1000, "n":2000, "type":"imb", "rtag":"pos50", "label":"QQP forced 50/50"},
    {"name":"qqp_90_10",          "ds":"qqp",    "val":1000, "n":2000, "type":"imb", "rtag":"pos10", "label":"QQP forced 90/10 (label0 majority)"},

    {"name":"rte_baseline",       "ds":"rte",    "val":277,  "n":2000, "type":"baseline", "label":"RTE baseline (natural ~50/50, label0 majority 50.2%)"},
    {"name":"rte_70_30_n1300",    "ds":"rte",    "val":277,  "n":1300, "type":"imb", "rtag":"pos30", "label":"RTE forced 70/30 (n=1300, label0 majority)"},
    {"name":"rte_90_10_n1300",    "ds":"rte",    "val":277,  "n":1300, "type":"imb", "rtag":"pos10", "label":"RTE forced 90/10 (n=1300, label0 majority)"},

    {"name":"mrpc_baseline",      "ds":"mrpc",   "val":408,  "n":2000, "type":"baseline", "label":"MRPC baseline (natural ~32/68, label1 majority)"},
    {"name":"mrpc_50_50",         "ds":"mrpc",   "val":408,  "n":2000, "type":"imb", "rtag":"pos50", "label":"MRPC forced 50/50"},
    {"name":"mrpc_90_10",         "ds":"mrpc",   "val":408,  "n":2000, "type":"imb", "rtag":"pos90", "label":"MRPC forced 90/10 (label1 majority)"},

    {"name":"ag_news_baseline",      "ds":"ag_news","val":1000, "n":2000, "type":"baseline", "label":"AG News baseline (natural ~25/25/25/25)"},
    {"name":"ag_news_55_15_15_15",   "ds":"ag_news","val":1000, "n":2000, "type":"imb", "rtag":"cls55_15_15_15", "label":"AG News forced 55/15/15/15 (label0 majority)"},
    {"name":"ag_news_85_5_5_5",      "ds":"ag_news","val":1000, "n":2000, "type":"imb", "rtag":"cls85_05_05_05", "label":"AG News forced 85/5/5/5 (label0 majority)"},
]


# ---------- table cell marker logic (A1-centric, valbal version) ----------
def cell_marker(lr_v, a1_v, rnd_v):
    """Return (text, facecolor) for one (rank,sel%) cell.

    A1-centric:
        !  A1 <= random       (A1 harmful)
        O  A1 > LR and A1 > random
        X  A1 < LR (A1 still > random)
        -  A1 == LR (A1 still > random)
    Trailing † appended whenever LR <= random  (LR collapsed).
    """
    if lr_v is None or a1_v is None or rnd_v is None:
        return "*", "#ffffff"
    if a1_v <= rnd_v:
        base, color = "!", "#ffd6d6"
    elif a1_v > lr_v:
        base, color = "O", "#d6f5d6"
    elif a1_v < lr_v:
        base, color = "X", "#fff2cc"
    else:
        base, color = "-", "#ffffff"
    if lr_v <= rnd_v:
        base = base + "†"
    return base, color


# ---------- per-setting page ----------
def make_page(stg, pdf, summary, sanity):
    fs_top, fs_rnd = fs_inv(stg)

    # sanity tracking for the requesting prompt
    src = stg['type']
    sanity[src]["fs_total"] += 1
    if fs_top is None:
        sanity[src]["fs_missing_top"] += 1
    if fs_rnd is None:
        sanity[src]["fs_missing_rnd"] += 1

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1], hspace=0.55, wspace=0.35)

    # Pick a robust random baseline curve for the figure. Prefer FS-INV random,
    # else average of all available rank-wise random_results_inv.
    rand_for_fig = fs_rnd
    if rand_for_fig is None:
        rand_curves = []
        for r in RANKS:
            _, ra = lr_eig_inv(stg, r)
            if ra is not None:
                rand_curves.append(ra)
            _, ra = a1_eig_inv(stg, r)
            if ra is not None:
                rand_curves.append(ra)
        if rand_curves:
            arr = np.array(rand_curves, dtype=float)
            rand_for_fig = arr.mean(axis=0).tolist()

    # ---- per-rank panels ----
    for i, r in enumerate(RANKS):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        lr_top, lr_rnd = lr_eig_inv(stg, r)
        a1_top, a1_rnd = a1_eig_inv(stg, r)

        # sanity tracking per source
        sanity[src]["lr_total"] += 1
        if lr_top is None:
            sanity[src]["lr_missing_top"] += 1
        if lr_rnd is None:
            sanity[src]["lr_missing_rnd"] += 1
        sanity[src]["a1_total"] += 1
        if a1_top is None:
            sanity[src]["a1_missing_top"] += 1
        if a1_rnd is None:
            sanity[src]["a1_missing_rnd"] += 1

        if fs_top is not None:
            ax.plot(SEL_RANGE, [fs_top[s-1]/100 for s in SEL_RANGE],
                    color='red', label='FreeShap (INV)', linewidth=1.8)
        if lr_top is not None:
            ax.plot(SEL_RANGE, [lr_top[s-1]/100 for s in SEL_RANGE],
                    color='blue', label=f'LRFShap r={r}%', marker='o',
                    markersize=3.5, linewidth=1.3)
        if a1_top is not None:
            ax.plot(SEL_RANGE, [a1_top[s-1]/100 for s in SEL_RANGE],
                    color='green', label=f'A1 r={r}%', marker='s',
                    markersize=3.5, linewidth=1.3)
        if rand_for_fig is not None:
            ax.plot(SEL_RANGE, [rand_for_fig[s-1]/100 for s in SEL_RANGE],
                    color='gray', label='random', linestyle='--', linewidth=1.2)
        ax.set_title(f"eigen r = {r}%", fontsize=10)
        ax.set_xlabel("selected %", fontsize=8)
        ax.set_ylabel("val accuracy (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.0, loc='lower right')
        ax.grid(True, alpha=0.3)

    # 8th panel — legend explainer
    ax_blank = fig.add_subplot(gs[1, 3])
    ax_blank.axis('off')
    legend_text = (
        "Markers in lower table (A1-centric):\n"
        "  O : A1 > LR  (A1 also beats random)\n"
        "  X : A1 < LR  (A1 still beats random)\n"
        "  - : A1 = LR  (A1 still beats random)\n"
        "  ! : A1 <= random  (A1 itself harmful)\n"
        "  † : LR <= random  (LR collapsed -\n"
        "      regime A1 is designed for)\n"
        "  * : missing data\n"
        "\nRandom baseline shown in figure is the\n"
        "INV-mode random curve from the FS-INV\n"
        "source (predictions.txt or sidecar)."
    )
    ax_blank.text(0.0, 0.95, legend_text, fontsize=8.5, va='top', ha='left',
                  family='monospace')

    # ---- table ----
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis('off')

    if fs_top is None:
        ax_t.text(0.5, 0.5, "(INV baseline data missing - table unavailable)",
                  ha='center', va='center', fontsize=10)
        summary[(stg['ds'], stg.get('rtag', 'baseline'))] = {
            "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
            "lr_harm": 0, "a1_harm": 0, "n_cells": 0}
    else:
        col_labels = [""] + [f"sel{s}%" for s in TABLE_SELS]
        rows = []
        cell_colors = []

        # Row 0: INV (acc + harm vs FS-INV random)
        inv_row = ["INV (acc)"]
        inv_colors = ["#dddddd"]
        for s in TABLE_SELS:
            val = fs_top[s-1] / 100.0
            text = f"{val:.2f}"
            bg = "#ffffff"
            if fs_rnd is not None and fs_top[s-1] <= fs_rnd[s-1]:
                text += "\n[!]"
                bg = "#ffd6d6"
            inv_row.append(text)
            inv_colors.append(bg)
        rows.append(inv_row)
        cell_colors.append(inv_colors)

        per_setting_stats = {"a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
                             "lr_harm": 0, "a1_harm": 0, "n_cells": 0}
        for r in RANKS:
            lr_top, lr_rnd = lr_eig_inv(stg, r)
            a1_top, a1_rnd = a1_eig_inv(stg, r)
            base_rnd = lr_rnd if lr_rnd is not None else fs_rnd

            row = [f"r={r}%"]
            colors = ["#eeeeee"]
            for s in TABLE_SELS:
                lr_v = lr_top[s-1] if lr_top is not None else None
                a1_v = a1_top[s-1] if a1_top is not None else None
                rnd_v = base_rnd[s-1] if base_rnd is not None else None
                marker, bg = cell_marker(lr_v, a1_v, rnd_v)
                row.append(marker)
                colors.append(bg)
                base = marker[0] if marker else "*"
                if base in ("O", "X", "-", "!"):
                    per_setting_stats["n_cells"] += 1
                if base == "O":
                    per_setting_stats["a1_gt_lr"] += 1
                elif base == "X":
                    per_setting_stats["a1_lt_lr"] += 1
                elif base == "-":
                    per_setting_stats["a1_eq_lr"] += 1
                if base == "!":
                    per_setting_stats["a1_harm"] += 1
                if marker.endswith("†"):
                    per_setting_stats["lr_harm"] += 1
            rows.append(row)
            cell_colors.append(colors)

        tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc='upper center',
                         cellLoc='center', colWidths=[0.12] + [0.10]*len(TABLE_SELS),
                         cellColours=cell_colors)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.55)

        for (i, j), cell in tbl.get_celld().items():
            if i == 0:
                cell.set_facecolor("#dddddd")
                cell.set_text_props(weight='bold')
            elif j == 0:
                cell.set_text_props(weight='bold')

        summary[(stg['ds'], stg.get('rtag', 'baseline'))] = per_setting_stats

    fig.suptitle(stg['label'] + f"   |   n_train={stg['n']}, val={stg['val']}",
                 fontsize=13, y=0.995)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- summary page ----------
def make_summary_page(pdf, summary):
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    fig.suptitle("Summary: A1 vs LRFShap vs random  (per-cell breakdown across rank x sel%)",
                 fontsize=14, y=0.97)

    headers = ["dataset / ratio", "cells", "A1 > LR", "A1 < LR", "A1 = LR",
               "LR <= rnd (harm)", "A1 <= rnd (harm)",
               "A1 win %", "LR-safe %", "A1-safe %"]
    rows = []
    tot = {"n_cells": 0, "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
           "lr_harm": 0, "a1_harm": 0}

    for (ds, rtag), st in summary.items():
        nc = st["n_cells"]
        a1w = st["a1_gt_lr"]
        a1l = st["a1_lt_lr"]
        a1e = st["a1_eq_lr"]
        lrh = st["lr_harm"]
        a1h = st["a1_harm"]
        a1w_pct = f"{100.0*a1w/nc:.1f}" if nc else "-"
        lr_safe = f"{100.0*(nc-lrh)/nc:.1f}" if nc else "-"
        a1_safe = f"{100.0*(nc-a1h)/nc:.1f}" if nc else "-"
        rows.append([f"{ds}/{rtag}", str(nc), str(a1w), str(a1l), str(a1e),
                     str(lrh), str(a1h), a1w_pct, lr_safe, a1_safe])
        for k in tot:
            tot[k] += st[k]

    nc = tot["n_cells"]
    if nc > 0:
        rows.append([f"ALL ({len(summary)} settings)", str(nc),
                     str(tot["a1_gt_lr"]), str(tot["a1_lt_lr"]),
                     str(tot["a1_eq_lr"]),
                     str(tot["lr_harm"]), str(tot["a1_harm"]),
                     f"{100.0*tot['a1_gt_lr']/nc:.1f}",
                     f"{100.0*(nc-tot['lr_harm'])/nc:.1f}",
                     f"{100.0*(nc-tot['a1_harm'])/nc:.1f}"])

    tbl = ax.table(cellText=rows, colLabels=headers, loc='upper center',
                   cellLoc='center', colWidths=[0.20] + [0.08]*9)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.45)

    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#cccccc")
            cell.set_text_props(weight='bold')
        elif i == len(rows):
            cell.set_facecolor("#ffe0a0")
            cell.set_text_props(weight='bold')
        elif j == 0:
            cell.set_text_props(weight='bold')

    note = (
        "Notes:\n"
        "  - n_cells = 7 ranks x 7 sel% = 49 per setting (when all data present).\n"
        "  - 'A1 win %' = fraction of cells where A1 strictly beats LR (within the non-harm O/X/= subset).\n"
        "  - 'LR-safe %' = fraction of cells where LR > random (NOT harmful for LR).\n"
        "  - 'A1-safe %' = fraction of cells where A1 > random (NOT harmful for A1).\n"
        "  - Cells marked '!' (A1 harmful) are counted toward n_cells but contribute\n"
        "    to a1_harm only; the O/X/= counts use the A1-centric scheme.\n"
        "  - Per-cell random baseline = LR-eigen sidecar's random_results_inv at the\n"
        "    same rank, with fallback to FS-INV random when missing."
    )
    ax.text(0.02, 0.20, note, fontsize=9.5, va='top', ha='left', family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- main ----------
def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    summary = {}
    sanity = {
        "baseline": {"fs_total": 0, "fs_missing_top": 0, "fs_missing_rnd": 0,
                     "lr_total": 0, "lr_missing_top": 0, "lr_missing_rnd": 0,
                     "a1_total": 0, "a1_missing_top": 0, "a1_missing_rnd": 0},
        "imb":      {"fs_total": 0, "fs_missing_top": 0, "fs_missing_rnd": 0,
                     "lr_total": 0, "lr_missing_top": 0, "lr_missing_rnd": 0,
                     "a1_total": 0, "a1_missing_top": 0, "a1_missing_rnd": 0},
    }
    n_pages = 0
    with PdfPages(OUT) as pdf:
        for stg in SETTINGS:
            print(f"[plot] {stg['name']}")
            try:
                make_page(stg, pdf, summary, sanity)
                n_pages += 1
            except Exception as e:
                print(f"[error] skip {stg['name']}: {e}", file=sys.stderr)
        make_summary_page(pdf, summary)
        n_pages += 1

    # Aggregate cell-level harm/collapse stats from per-setting summary.
    cells_total = sum(s["n_cells"] for s in summary.values())
    a1_harm_total = sum(s["a1_harm"] for s in summary.values())
    lr_harm_total = sum(s["lr_harm"] for s in summary.values())

    print(f"[done] saved -> {OUT}")
    print(f"[summary] pages          = {n_pages}")
    print(f"[summary] settings       = {len(SETTINGS)}")
    print(f"[summary] cells total    = {cells_total}")
    if cells_total:
        print(f"[summary] '!'  cells (A1 harmful)     = {a1_harm_total} "
              f"({100.0*a1_harm_total/cells_total:.1f}%)")
        print(f"[summary] '†'  cells (LR collapsed)   = {lr_harm_total} "
              f"({100.0*lr_harm_total/cells_total:.1f}%)")
    # sanity per source
    for src, s in sanity.items():
        print(f"[sanity:{src}] FS  pages={s['fs_total']}  "
              f"missing_top={s['fs_missing_top']}  missing_rnd={s['fs_missing_rnd']}")
        if s['lr_total']:
            print(f"[sanity:{src}] LR  panels={s['lr_total']}  "
                  f"missing_top={s['lr_missing_top']} "
                  f"({100.0*s['lr_missing_top']/s['lr_total']:.1f}%)  "
                  f"missing_rnd={s['lr_missing_rnd']} "
                  f"({100.0*s['lr_missing_rnd']/s['lr_total']:.1f}%)")
            print(f"[sanity:{src}] A1  panels={s['a1_total']}  "
                  f"missing_top={s['a1_missing_top']} "
                  f"({100.0*s['a1_missing_top']/s['a1_total']:.1f}%)  "
                  f"missing_rnd={s['a1_missing_rnd']} "
                  f"({100.0*s['a1_missing_rnd']/s['a1_total']:.1f}%)")


if __name__ == "__main__":
    main()
