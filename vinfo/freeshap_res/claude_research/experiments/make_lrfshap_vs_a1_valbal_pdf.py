"""lrfshap_vs_a1_valbal.pdf — per-setting (2x4 fig + table) comparison
with random baseline. Forked from make_lrfshap_vs_a1_pdf.py.

Key changes vs the original:
1. Only valbal (balanced-val) settings. ratios + n + val taken from the
   completed imbalance/data_selection runs.
2. Adds a random baseline curve and uses it to flag "harmful" cells
   ([!]) in the per-rank table.
3. Per-cell marker scheme:
       O    A1 strictly better than LR (both beat random)
       X    A1 not better than LR (both beat random)
       !    at least one of {A1, LR} <= random (harmful)
       -    A1 == LR exactly
       *    missing data
"""
import json, os, re, sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/data_selection_test/imbalance/data_selection"
OUT  = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/reports/lrfshap_vs_a1_valbal.pdf"

SEL_RANGE  = list(range(1, 21))           # 1..20 % on figure
RANKS      = [1, 5, 10, 15, 20, 25, 30]
TABLE_SELS = [1, 2, 3, 4, 5, 10, 20]


# ---------- ratio list ----------
# (ds, rtag, n_train, val, pretty_label)
SETTINGS = [
    # MRPC (binary)
    ("mrpc",   "pos50",         2300, 258,  "MRPC forced 50/50 (n=2300, valbal=258)"),
    ("mrpc",   "pos70",         2300, 258,  "MRPC forced 70/30 label1-majority (n=2300, valbal=258)"),
    ("mrpc",   "pos90",         2300, 258,  "MRPC forced 90/10 label1-majority (n=2300, valbal=258)"),
    # SST-2 (binary)
    ("sst2",   "pos50",         5000, 856,  "SST-2 forced 50/50 (n=5000, valbal=856)"),
    ("sst2",   "pos70",         5000, 856,  "SST-2 forced 70/30 label1-majority (n=5000, valbal=856)"),
    ("sst2",   "pos90",         5000, 856,  "SST-2 forced 90/10 label1-majority (n=5000, valbal=856)"),
    # MR (binary)
    ("mr",     "pos50",         4500, 1000, "MR forced 50/50 (n=4500, valbal=1000)"),
    ("mr",     "pos70",         4500, 1000, "MR forced 70/30 label1-majority (n=4500, valbal=1000)"),
    ("mr",     "pos90",         4500, 1000, "MR forced 90/10 label1-majority (n=4500, valbal=1000)"),
    # QQP (binary; label0 is the natural majority direction, so pos50 = 50/50,
    # pos30 = 70/30 label0-maj, pos10 = 90/10 label0-maj)
    ("qqp",    "pos50",         5000, 1000, "QQP forced 50/50 (n=5000, valbal=1000)"),
    ("qqp",    "pos30",         5000, 1000, "QQP forced 70/30 label0-majority (n=5000, valbal=1000)"),
    ("qqp",    "pos10",         5000, 1000, "QQP forced 90/10 label0-majority (n=5000, valbal=1000)"),
    # RTE (binary; n=1300 because 90/10 binds on the smaller pool, valbal=262)
    ("rte",    "pos50",         1300, 262,  "RTE forced 50/50 (n=1300, valbal=262)"),
    ("rte",    "pos70",         1300, 262,  "RTE forced 70/30 label1-majority (n=1300, valbal=262)"),
    ("rte",    "pos90",         1300, 262,  "RTE forced 90/10 label1-majority (n=1300, valbal=262)"),
    # AG News (4-class)
    ("ag_news","cls25_25_25_25",5000, 1000, "AG News balanced 25/25/25/25 (n=5000, valbal=1000)"),
    ("ag_news","cls55_15_15_15",5000, 1000, "AG News forced 55/15/15/15 label0-majority (n=5000, valbal=1000)"),
    ("ag_news","cls85_05_05_05",5000, 1000, "AG News forced 85/5/5/5 label0-majority (n=5000, valbal=1000)"),
    # MNLI (3-class)
    ("mnli",   "cls33_33_33",   5000, 1000, "MNLI balanced 33/33/33 (n=5000, valbal=1000)"),
    ("mnli",   "cls60_20_20",   5000, 1000, "MNLI forced 60/20/20 label0-majority (n=5000, valbal=1000)"),
    ("mnli",   "cls90_05_05",   5000, 1000, "MNLI forced 90/5/5 label0-majority (n=5000, valbal=1000)"),
]


# ---------- sidecar loaders ----------
def _load(p):
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        print(f"[warn] cannot read {p}: {e}", file=sys.stderr)
        return None


def fs_inv(ds, rtag, n, v):
    """Full-Shapley INV (lrfshap inv sidecar). Returns (top, random) or (None, None)."""
    p = (f"{BASE}/{ds}/{rtag}/lrfshap/inv/sidecar/"
         f"bert_seed2026_num{n}_valbal{v}_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500.json")
    d = _load(p)
    if d is None:
        return None, None
    return d.get("top_results_inv"), d.get("random_results_inv")


def lr_eig(ds, rtag, n, v, r):
    """LRFShap eigen sidecar at rank r. Returns (top_inv, random_inv) or (None,None)."""
    p = (f"{BASE}/{ds}/{rtag}/lrfshap/eigen/sidecar/"
         f"bert_seed2026_num{n}_valbal{v}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_"
         f"lrfshap_signFalse_earlystopTrue_tmc500.json")
    d = _load(p)
    if d is None:
        return None, None
    return d.get("top_results_inv"), d.get("random_results_inv")


def a1_eig(ds, rtag, n, v, r):
    """A1 eigen sidecar at rank r. Returns (top_inv, random_inv) or (None,None)."""
    p = (f"{BASE}/{ds}/{rtag}/a1/eigen/sidecar/"
         f"bert_seed2026_num{n}_valbal{v}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_"
         f"a1_signFalse_earlystopTrue_tmc500.json")
    d = _load(p)
    if d is None:
        return None, None
    return d.get("top_results_inv"), d.get("random_results_inv")


# ---------- table cell marker logic ----------
# A1-centric scheme:
#   ! : A1 itself is harmful (A1 <= random)  → A1 's contribution is meaningless
#   O : A1 beats LR AND A1 beats random
#   X : A1 does not beat LR but A1 still beats random
#   - : A1 == LR (and A1 > random)
# A trailing dagger (†) is appended whenever LR <= random — this flags that LR
# was collapsed at that cell, which is precisely the regime A1 is meant to
# rescue.  The LR-vs-random status no longer toggles the harm flag.
def cell_marker(lr_v, a1_v, rnd_v):
    """Return (text, facecolor) for one (rank,sel%) cell."""
    if lr_v is None or a1_v is None or rnd_v is None:
        return "*", "#ffffff"
    # Primary marker is A1-centric.
    if a1_v <= rnd_v:
        base, color = "!", "#ffd6d6"   # red: A1 harmful (not better than random)
    elif a1_v > lr_v:
        base, color = "O", "#d6f5d6"   # green: A1 > LR and A1 > random
    elif a1_v < lr_v:
        base, color = "X", "#fff2cc"   # yellow: LR > A1 (A1 still > random)
    else:
        base, color = "-", "#ffffff"
    # Secondary indicator: LR collapsed (LR <= random).
    if lr_v is not None and rnd_v is not None and lr_v <= rnd_v:
        base = base + "†"
    return base, color


# ---------- per-setting page ----------
def make_page(stg, pdf, summary):
    ds, rtag, n, v, label = stg
    fs_top, fs_rnd = fs_inv(ds, rtag, n, v)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1], hspace=0.55, wspace=0.35)

    # Pick a robust random baseline curve for the figure. Prefer FS (INV) random,
    # else average of all available rank-wise random_results_inv.
    rand_for_fig = fs_rnd
    if rand_for_fig is None:
        rand_curves = []
        for r in RANKS:
            _, ra = lr_eig(ds, rtag, n, v, r)
            if ra is not None:
                rand_curves.append(ra)
            _, ra = a1_eig(ds, rtag, n, v, r)
            if ra is not None:
                rand_curves.append(ra)
        if rand_curves:
            arr = np.array(rand_curves, dtype=float)
            rand_for_fig = arr.mean(axis=0).tolist()

    # ---- per-rank panels ----
    for i, r in enumerate(RANKS):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        lr_top, _ = lr_eig(ds, rtag, n, v, r)
        a1_top, _ = a1_eig(ds, rtag, n, v, r)
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

    ax_blank = fig.add_subplot(gs[1, 3])
    ax_blank.axis('off')
    # legend explainer in the blank panel
    legend_text = (
        "Markers in lower table (A1-centric):\n"
        "  O : A1 > LR  (A1 also beats random)\n"
        "  X : A1 < LR  (A1 still beats random)\n"
        "  - : A1 = LR  (A1 still beats random)\n"
        "  ! : A1 ≤ random  (A1 itself harmful)\n"
        "  † : LR ≤ random  (LR collapsed —\n"
        "      regime A1 is designed for)\n"
        "  * : missing data\n"
        "\nRandom baseline shown in figure is the\n"
        "INV-mode random curve from the lrfshap\n"
        "sidecar (random_results_inv)."
    )
    ax_blank.text(0.0, 0.95, legend_text, fontsize=8.5, va='top', ha='left',
                  family='monospace')

    # ---- table ----
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis('off')

    if fs_top is None:
        ax_t.text(0.5, 0.5, "(INV baseline data missing — table unavailable)",
                  ha='center', va='center', fontsize=10)
    else:
        col_labels = [""] + [f"sel{s}%" for s in TABLE_SELS]

        # Per-cell baseline accuracy for harm flagging.
        # For each rank we use the rank-specific random curve from the lrfshap
        # eigen sidecar (random_results_inv at that rank). If missing, fall
        # back to FS-INV random.
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

        # Per-rank rows
        per_setting_stats = {"a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
                             "lr_harm": 0, "a1_harm": 0, "n_cells": 0,
                             # by_rank[r] = same stat dict per rank R
                             "by_rank": {r: {"a1_gt_lr": 0, "a1_lt_lr": 0,
                                             "a1_eq_lr": 0, "lr_harm": 0,
                                             "a1_harm": 0, "n_cells": 0}
                                         for r in RANKS},
                             # by_rank_sel[(r, s)] = per-cell indicators (raw 0/1)
                             # so the cross-setting (group × rank × sel) XX/YY
                             # tables on the appended summary pages can sum them.
                             "by_rank_sel": {(r, s): {"valid": 0,
                                                       "a1_gt_lr": 0,
                                                       "a1_gt_inv": 0,
                                                       "a1_gt_rnd": 0,
                                                       "a1_gt_all": 0}
                                              for r in RANKS for s in TABLE_SELS}}
        for r in RANKS:
            lr_top, lr_rnd = lr_eig(ds, rtag, n, v, r)
            a1_top, a1_rnd = a1_eig(ds, rtag, n, v, r)

            # Use the INV (FS) random as the per-cell baseline for ALL ranks.
            # Rationale: figure curve also shows fs_rnd (INV random) as the random
            # reference, and `random` means "Shapley-agnostic uniform-random
            # selection of sel% indices + INV-mode ridge predict" — a single
            # well-defined baseline.  Per-rank random_results_inv would otherwise
            # differ across sidecars due to np.random state progression even
            # though all are uniform random samples.
            base_rnd = fs_rnd

            row = [f"r={r}%"]
            colors = ["#eeeeee"]
            for s in TABLE_SELS:
                lr_v = lr_top[s-1] if lr_top is not None else None
                a1_v = a1_top[s-1] if a1_top is not None else None
                rnd_v = base_rnd[s-1] if base_rnd is not None else None
                marker, bg = cell_marker(lr_v, a1_v, rnd_v)
                row.append(marker)
                colors.append(bg)
                # accumulate stats — base marker is the first character,
                # the optional dagger trails as second char.
                base = marker[0] if marker else "*"
                rk_st = per_setting_stats["by_rank"][r]
                if base in ("O", "X", "-", "!"):
                    per_setting_stats["n_cells"] += 1
                    rk_st["n_cells"] += 1
                if base == "O":
                    per_setting_stats["a1_gt_lr"] += 1
                    rk_st["a1_gt_lr"] += 1
                elif base == "X":
                    per_setting_stats["a1_lt_lr"] += 1
                    rk_st["a1_lt_lr"] += 1
                elif base == "-":
                    per_setting_stats["a1_eq_lr"] += 1
                    rk_st["a1_eq_lr"] += 1
                if base == "!":
                    per_setting_stats["a1_harm"] += 1
                    rk_st["a1_harm"] += 1
                if marker.endswith("†"):
                    per_setting_stats["lr_harm"] += 1
                    rk_st["lr_harm"] += 1

                # Per-cell raw indicators for the group × rank × sel breakdown
                # tables on the appended summary pages.  Valid cell requires
                # LR, A1, INV (= fs_top), and random (= fs_rnd) all present.
                inv_v = fs_top[s-1] if fs_top is not None else None
                cell_st = per_setting_stats["by_rank_sel"][(r, s)]
                if (lr_v is not None and a1_v is not None
                        and inv_v is not None and rnd_v is not None):
                    cell_st["valid"] = 1
                    cell_st["a1_gt_lr"]  = int(a1_v > lr_v)
                    cell_st["a1_gt_inv"] = int(a1_v > inv_v)
                    cell_st["a1_gt_rnd"] = int(a1_v > rnd_v)
                    cell_st["a1_gt_all"] = int(a1_v > max(lr_v, inv_v, rnd_v))
            rows.append(row)
            cell_colors.append(colors)

        tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc='upper center',
                         cellLoc='center', colWidths=[0.12] + [0.10]*len(TABLE_SELS),
                         cellColours=cell_colors)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.55)

        # bold first column + header
        for (i, j), cell in tbl.get_celld().items():
            if i == 0:
                cell.set_facecolor("#dddddd")
                cell.set_text_props(weight='bold')
            elif j == 0:
                cell.set_text_props(weight='bold')

        summary[(ds, rtag)] = per_setting_stats

    fig.suptitle(label + f"   |   n_train={n}, valbal={v}",
                 fontsize=13, y=0.995)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- imbalance group mapping ----------
# balanced / mild / extreme — defined so class-count is irrelevant.
IMB_GROUP = {
    # binary (label1 / label0 majority, both labelled by majority share)
    ("mrpc","pos50"): "balanced", ("mrpc","pos70"): "mild", ("mrpc","pos90"): "extreme",
    ("sst2","pos50"): "balanced", ("sst2","pos70"): "mild", ("sst2","pos90"): "extreme",
    ("mr","pos50"):   "balanced", ("mr","pos70"):   "mild", ("mr","pos90"):   "extreme",
    ("rte","pos50"):  "balanced", ("rte","pos70"):  "mild", ("rte","pos90"):  "extreme",
    # QQP: pos_ratio = label-1 fraction; label0 is the natural majority so
    # pos50 = 50/50, pos30 = 70/30 (label0 70%), pos10 = 90/10 (label0 90%).
    ("qqp","pos50"):  "balanced", ("qqp","pos30"):  "mild", ("qqp","pos10"):  "extreme",
    # 4-class
    ("ag_news","cls25_25_25_25"): "balanced",
    ("ag_news","cls55_15_15_15"): "mild",
    ("ag_news","cls85_05_05_05"): "extreme",
    # 3-class
    ("mnli","cls33_33_33"): "balanced",
    ("mnli","cls60_20_20"): "mild",
    ("mnli","cls90_05_05"): "extreme",
}


# ---------- summary page ----------
def _stat_row(label, st):
    """Helper: render one stats dict as a table row."""
    nc = st["n_cells"]
    a1w = st["a1_gt_lr"]
    a1l = st["a1_lt_lr"]
    a1e = st["a1_eq_lr"]
    lrh = st["lr_harm"]
    a1h = st["a1_harm"]
    a1w_pct = f"{100.0*a1w/nc:.1f}" if nc else "-"
    lr_safe = f"{100.0*(nc-lrh)/nc:.1f}" if nc else "-"
    a1_safe = f"{100.0*(nc-a1h)/nc:.1f}" if nc else "-"
    return [label, str(nc), str(a1w), str(a1l), str(a1e),
            str(lrh), str(a1h), a1w_pct, lr_safe, a1_safe]


def make_summary_page(pdf, summary):
    # Two tables on one page: (top) dataset/ratio breakdown,
    # (bottom) rank-aggregate across all settings.
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 0.35], hspace=0.20)
    ax_top = fig.add_subplot(gs[0, 0]); ax_top.axis('off')
    ax_bot = fig.add_subplot(gs[1, 0]); ax_bot.axis('off')
    ax_note = fig.add_subplot(gs[2, 0]); ax_note.axis('off')

    fig.suptitle("Summary: A1 vs LRFShap vs random  (per-cell breakdown)",
                 fontsize=15, y=0.99)

    headers = ["group", "cells", "A1 > LR", "A1 < LR", "A1 = LR",
               "LR ≤ rnd (harm)", "A1 ≤ rnd (harm)",
               "A1 win %", "LR-safe %", "A1-safe %"]

    # ---- TOP: per-setting (dataset/ratio) breakdown ----
    rows = []
    tot = {"n_cells": 0, "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
           "lr_harm": 0, "a1_harm": 0}
    for (ds, rtag), st in summary.items():
        rows.append(_stat_row(f"{ds}/{rtag}", st))
        for k in tot:
            tot[k] += st[k]
    n_settings = len(summary)
    rows.append(_stat_row(f"ALL ({n_settings} settings)", tot))

    tbl = ax_top.table(cellText=rows, colLabels=headers, loc='upper center',
                       cellLoc='center', colWidths=[0.20] + [0.083]*9)
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.8); tbl.scale(1.0, 1.45)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#cccccc"); cell.set_text_props(weight='bold')
        elif i == len(rows):
            cell.set_facecolor("#ffe0a0"); cell.set_text_props(weight='bold')
        elif j == 0:
            cell.set_text_props(weight='bold')
    ax_top.set_title("Breakdown by setting (rows = dataset/ratio; columns aggregate over rank × sel%)",
                     fontsize=11, loc='left', pad=12)

    # ---- BOTTOM: per-rank aggregate across all settings ----
    rank_rows = []
    rank_total = {"n_cells": 0, "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
                  "lr_harm": 0, "a1_harm": 0}
    for r in RANKS:
        agg = {"n_cells": 0, "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
               "lr_harm": 0, "a1_harm": 0}
        for st in summary.values():
            br = st.get("by_rank", {}).get(r)
            if br is None:
                continue
            for k in agg:
                agg[k] += br[k]
        rank_rows.append(_stat_row(f"rank r={r}%", agg))
        for k in rank_total:
            rank_total[k] += agg[k]
    rank_rows.append(_stat_row(f"ALL ranks ({n_settings} settings)", rank_total))

    tbl2 = ax_bot.table(cellText=rank_rows, colLabels=headers, loc='upper center',
                        cellLoc='center', colWidths=[0.20] + [0.083]*9)
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(9.3); tbl2.scale(1.0, 1.5)
    for (i, j), cell in tbl2.get_celld().items():
        if i == 0:
            cell.set_facecolor("#cccccc"); cell.set_text_props(weight='bold')
        elif i == len(rank_rows):
            cell.set_facecolor("#ffe0a0"); cell.set_text_props(weight='bold')
        elif j == 0:
            cell.set_text_props(weight='bold')
    ax_bot.set_title("Breakdown by rank (rows = eigen rank r%; aggregated across all settings × sel%)",
                     fontsize=11, loc='left', pad=12)

    # ---- Notes ----
    note = (
        "Notes:\n"
        "  - n_cells = 7 ranks × 7 sel% = 49 per setting (when all data present);\n"
        "    per-rank rows aggregate sel% × all settings.\n"
        "  - 'A1 win %' = fraction of cells where A1 strictly beats LR (and beats random — '!' cells excluded).\n"
        "  - 'LR-safe %' = fraction of cells where LR > random.\n"
        "  - 'A1-safe %' = fraction of cells where A1 > random.\n"
        "  - Per-cell random baseline = INV (FS) random (random_results_inv from the lrfshap inv sidecar).\n"
        "    Same baseline used for the figure curves on each setting page."
    )
    ax_note.text(0.02, 0.95, note, fontsize=9.5, va='top', ha='left', family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- group × rank page ----------
def make_group_rank_page(pdf, summary):
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.35], hspace=0.32)
    ax_bal = fig.add_subplot(gs[0, 0]); ax_bal.axis('off')
    ax_mld = fig.add_subplot(gs[1, 0]); ax_mld.axis('off')
    ax_ext = fig.add_subplot(gs[2, 0]); ax_ext.axis('off')
    ax_note = fig.add_subplot(gs[3, 0]); ax_note.axis('off')

    fig.suptitle("Per-rank breakdown grouped by imbalance level "
                 "(balanced / mild / extreme)",
                 fontsize=15, y=0.99)

    headers = ["group", "cells", "A1 > LR", "A1 < LR", "A1 = LR",
               "LR ≤ rnd (harm)", "A1 ≤ rnd (harm)",
               "A1 win %", "LR-safe %", "A1-safe %"]

    for group_name, ax_g in [("balanced", ax_bal),
                              ("mild",     ax_mld),
                              ("extreme",  ax_ext)]:
        group_settings = [k for k in summary
                          if IMB_GROUP.get(k) == group_name]
        n_g = len(group_settings)
        member_labels = ", ".join(f"{ds}/{rtag}" for (ds, rtag) in group_settings)

        rows = []
        gtotal = {"n_cells": 0, "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
                  "lr_harm": 0, "a1_harm": 0}
        for r in RANKS:
            agg = {"n_cells": 0, "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
                   "lr_harm": 0, "a1_harm": 0}
            for k_setting in group_settings:
                br = summary[k_setting].get("by_rank", {}).get(r)
                if br is None:
                    continue
                for k in agg:
                    agg[k] += br[k]
            rows.append(_stat_row(f"r={r}%", agg))
            for k in gtotal:
                gtotal[k] += agg[k]
        rows.append(_stat_row(f"ALL ranks ({n_g} settings)", gtotal))

        tbl = ax_g.table(cellText=rows, colLabels=headers, loc='upper center',
                         cellLoc='center', colWidths=[0.20] + [0.083]*9)
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.3); tbl.scale(1.0, 1.45)
        for (i, j), cell in tbl.get_celld().items():
            if i == 0:
                cell.set_facecolor("#cccccc"); cell.set_text_props(weight='bold')
            elif i == len(rows):
                cell.set_facecolor("#ffe0a0"); cell.set_text_props(weight='bold')
            elif j == 0:
                cell.set_text_props(weight='bold')
        ax_g.set_title(f"{group_name.upper()} imbalance — settings: {member_labels}",
                       fontsize=10.5, loc='left', pad=10)

    note = (
        "Group definitions (class-count agnostic):\n"
        "  - BALANCED: pos50 (binary majority 50/50), cls25 (AG News 4-class even),\n"
        "              cls33 (MNLI 3-class even).\n"
        "  - MILD:     pos70 (binary 70/30), qqp pos30 (label0 70%), cls55 (AG News\n"
        "              label0 55%), cls60 (MNLI label0 60%).\n"
        "  - EXTREME:  pos90 (binary 90/10), qqp pos10 (label0 90%), cls85 (AG News\n"
        "              label0 85%), cls90 (MNLI label0 90%).\n"
        "\nEach row aggregates 7 sel% × all settings in that group at the given rank.\n"
        "Random baseline = INV (FS) random (random_results_inv from the lrfshap inv sidecar)."
    )
    ax_note.text(0.02, 0.95, note, fontsize=9.5, va='top', ha='left', family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- indicator tables (group × rank × sel) ----------
INDICATORS = [
    ("A1 > LR",                       "a1_gt_lr"),
    ("A1 > INV (full kernel)",        "a1_gt_inv"),
    ("A1 > random",                   "a1_gt_rnd"),
    ("A1 > max(LR, INV, random)",     "a1_gt_all"),
]


def make_indicator_page(pdf, summary, ind_label, ind_key):
    """One page per indicator: rows = (group × rank) + ALL; cols = sel%.
    Cell value = "XX/YY"  where XX = #settings in that group satisfying the
    indicator at this (rank, sel), YY = #valid settings in that group at
    this (rank, sel) (drops partial/missing cells).
    """
    fig = plt.figure(figsize=(15, 14))
    ax = fig.add_subplot(1, 1, 1); ax.axis('off')
    fig.suptitle(f"Indicator: {ind_label}   "
                 "(cell = #settings satisfying / #valid settings)",
                 fontsize=14, y=0.985)

    headers = ["group / rank"] + [f"sel{s}%" for s in TABLE_SELS] + ["ALL sel%"]

    rows = []
    row_colors = []
    grp_palette = {"balanced": "#e6f0ff", "mild": "#fff5d6", "extreme": "#ffdede"}

    for group_name in ("balanced", "mild", "extreme"):
        group_settings = [k for k in summary
                          if IMB_GROUP.get(k) == group_name]
        n_grp = len(group_settings)

        # per-rank rows
        grp_total_num = {s: 0 for s in TABLE_SELS}
        grp_total_den = {s: 0 for s in TABLE_SELS}
        for r in RANKS:
            num_by_sel = {s: 0 for s in TABLE_SELS}
            den_by_sel = {s: 0 for s in TABLE_SELS}
            for k_setting in group_settings:
                brs = summary[k_setting].get("by_rank_sel", {})
                for s in TABLE_SELS:
                    cell = brs.get((r, s), {})
                    if cell.get("valid", 0):
                        den_by_sel[s] += 1
                        num_by_sel[s] += cell.get(ind_key, 0)
            # row cells
            cells = [f"r={r}%"] + [
                (f"{num_by_sel[s]}/{den_by_sel[s]}"
                 if den_by_sel[s] > 0 else "—")
                for s in TABLE_SELS
            ]
            # ALL-sel column
            num_all = sum(num_by_sel[s] for s in TABLE_SELS)
            den_all = sum(den_by_sel[s] for s in TABLE_SELS)
            cells.append(f"{num_all}/{den_all}" if den_all > 0 else "—")
            rows.append(cells)
            row_colors.append([grp_palette[group_name]] * len(cells))
            # accumulate group totals
            for s in TABLE_SELS:
                grp_total_num[s] += num_by_sel[s]
                grp_total_den[s] += den_by_sel[s]

        # group ALL row
        cells = [f"{group_name.upper()} ALL (n={n_grp})"] + [
            (f"{grp_total_num[s]}/{grp_total_den[s]}"
             if grp_total_den[s] > 0 else "—")
            for s in TABLE_SELS
        ]
        num_all = sum(grp_total_num[s] for s in TABLE_SELS)
        den_all = sum(grp_total_den[s] for s in TABLE_SELS)
        cells.append(f"{num_all}/{den_all}" if den_all > 0 else "—")
        rows.append(cells)
        # darken the ALL row of the group
        deeper = {"balanced": "#b8d4f0", "mild": "#f0dca0", "extreme": "#f0b8b8"}
        row_colors.append([deeper[group_name]] * len(cells))

    tbl = ax.table(cellText=rows, colLabels=headers, loc='upper center',
                   cellLoc='center',
                   colWidths=[0.16] + [0.085]*(len(TABLE_SELS) + 1),
                   cellColours=row_colors)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.0); tbl.scale(1.0, 1.32)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#cccccc"); cell.set_text_props(weight='bold')
        elif j == 0:
            cell.set_text_props(weight='bold')

    note = (
        "Reading guide:\n"
        f"  - Each cell is '<numerator>/<denominator>' where the numerator counts how\n"
        f"    many settings (out of the group's 7) satisfy `{ind_label}` at that\n"
        f"    (rank, sel%).  Denominator drops any settings whose sidecar is missing\n"
        f"    at that cell (so a partial run reduces denominator, never inflates it).\n"
        f"  - 'ALL sel%' column sums numerators and denominators across the 7 sel%\n"
        f"    columns (so each row 's ALL = sum of its 7 numerators / sum of its 7\n"
        f"    denominators; not an average of percentages).\n"
        f"  - The bolded '<GROUP> ALL (n=7)' row sums over the group's 7 settings\n"
        f"    and the 7 ranks for each sel%."
    )
    ax.text(0.02, 0.05, note, fontsize=9, va='top', ha='left', family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- top-Shapley class composition tables ----------
# Each cell summarises the most common label in the top-k indices (k = round(n * sel/100))
# of the Shapley-sorted indices.txt file.  Random row encodes the intended majority
# direction / proportion that uniform-random selection from the imbalanced pool
# would yield (constant across sel% × rank).
_DS_LOADERS = {
    "sst2":    lambda: __import__("datasets").load_dataset("sst2"),
    "mr":      lambda: __import__("datasets").load_dataset("rotten_tomatoes"),
    "qqp":     lambda: __import__("datasets").load_dataset("glue", "qqp"),
    "mnli":    lambda: __import__("datasets").load_dataset("glue", "mnli"),
    "ag_news": lambda: __import__("datasets").load_dataset("ag_news"),
    "mrpc":    lambda: __import__("datasets").load_dataset("glue", "mrpc"),
    "rte":     lambda: __import__("datasets").load_dataset("glue", "rte"),
}
_LABEL_CACHE = {}


def _train_labels(ds):
    """Return np.array of the full HF train-split labels for `ds` (cached)."""
    if ds not in _LABEL_CACHE:
        d = _DS_LOADERS[ds]()
        _LABEL_CACHE[ds] = np.asarray(d["train"]["label"])
    return _LABEL_CACHE[ds]


def _load_indices(path):
    """Load indices.txt -> np.array of ints (top-k Shapley-sorted), or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            arr = [int(line.strip()) for line in f if line.strip()]
        if not arr:
            return None
        return np.asarray(arr, dtype=int)
    except Exception as e:
        print(f"[warn] cannot read {path}: {e}", file=sys.stderr)
        return None


def _inv_indices_path(ds, rtag, n, v):
    return (f"{BASE}/{ds}/{rtag}/lrfshap/inv/indices/"
            f"bert_seed2026_num{n}_valbal{v}_lam1e-06_lrfshap_signFalse_"
            f"earlystopTrue_tmc500_indices.txt")


def _lr_eig_indices_path(ds, rtag, n, v, r):
    return (f"{BASE}/{ds}/{rtag}/lrfshap/eigen/indices/"
            f"bert_seed2026_num{n}_valbal{v}_eig{r}.0_eiglam1e-02_invlam1e-06_"
            f"cholesky_float32_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt")


def _a1_eig_indices_path(ds, rtag, n, v, r):
    return (f"{BASE}/{ds}/{rtag}/a1/eigen/indices/"
            f"bert_seed2026_num{n}_valbal{v}_eig{r}.0_eiglam1e-02_invlam1e-06_"
            f"cholesky_float32_a1_signFalse_earlystopTrue_tmc500_indices.txt")


# Intended majority direction (label, frac%) per setting for the random row.
# binary mrpc/sst2/mr/rte: label1 majority @ pos50/70/90 -> (1, 50/70/90)
# binary qqp: label0 majority (pos_ratio = label1 frac) @ pos50/30/10 -> (0, 50/70/90)
# 4-class ag_news: label0 majority
# 3-class mnli: label0 majority
RANDOM_MAJ = {
    ("mrpc",   "pos50"):         (1, 50),
    ("mrpc",   "pos70"):         (1, 70),
    ("mrpc",   "pos90"):         (1, 90),
    ("sst2",   "pos50"):         (1, 50),
    ("sst2",   "pos70"):         (1, 70),
    ("sst2",   "pos90"):         (1, 90),
    ("mr",     "pos50"):         (1, 50),
    ("mr",     "pos70"):         (1, 70),
    ("mr",     "pos90"):         (1, 90),
    ("rte",    "pos50"):         (1, 50),
    ("rte",    "pos70"):         (1, 70),
    ("rte",    "pos90"):         (1, 90),
    ("qqp",    "pos50"):         (0, 50),
    ("qqp",    "pos30"):         (0, 70),
    ("qqp",    "pos10"):         (0, 90),
    ("ag_news","cls25_25_25_25"): (0, 25),
    ("ag_news","cls55_15_15_15"): (0, 55),
    ("ag_news","cls85_05_05_05"): (0, 85),
    ("mnli",   "cls33_33_33"):    (0, 33),
    ("mnli",   "cls60_20_20"):    (0, 60),
    ("mnli",   "cls90_05_05"):    (0, 90),
}


def _top_class_str(indices_arr, labels_arr, k):
    """Return 'L{c}={frac}' for top-k slice, or '—' on bad input."""
    info = _top_class_info(indices_arr, labels_arr, k)
    if info is None:
        return "—"
    top_class, frac = info
    return f"L{top_class}={frac}"


def _top_class_info(indices_arr, labels_arr, k):
    """Return (top_class:int, frac:int 0-100) for top-k slice, or None on bad input."""
    if indices_arr is None or labels_arr is None:
        return None
    if k <= 0 or k > len(indices_arr):
        return None
    sl = indices_arr[:k]
    # Guard against any stray out-of-range index.
    if (sl < 0).any() or (sl >= len(labels_arr)).any():
        return None
    sub_labels = labels_arr[sl]
    if len(sub_labels) == 0:
        return None
    top_class, top_count = Counter(sub_labels.tolist()).most_common(1)[0]
    frac = int(round(100.0 * top_count / k))
    return int(top_class), frac


def make_top_class_page(stg, pdf, missing_log):
    """One page = one setting's top-Shapley class composition table + 4 heatmaps.

    Top: table with rows
      - "INV / random"
      - "LR / A1" group header
      - r=1%, r=5%, r=10%, r=15%, r=20%, r=25%, r=30%  (LR/A1 per rank)
    Cols: sel 1%, sel 2%, sel 3%, sel 4%, sel 5%, sel 10%, sel 20%.

    Bottom: 4 heatmaps of the top-class FRACTION (label ignored, only %):
      - random (1×7, constant per setting)
      - INV    (1×7)
      - LR     (7×7  ranks × sel)
      - A1     (7×7  ranks × sel)
    """
    ds, rtag, n, v, label = stg
    labels_arr = _train_labels(ds)

    # Pre-load all indices arrays for this setting.
    inv_arr = _load_indices(_inv_indices_path(ds, rtag, n, v))
    if inv_arr is None:
        missing_log.append(f"{ds}/{rtag} INV indices missing")
    lr_arrs = {}
    a1_arrs = {}
    for r in RANKS:
        p_lr = _lr_eig_indices_path(ds, rtag, n, v, r)
        p_a1 = _a1_eig_indices_path(ds, rtag, n, v, r)
        a_lr = _load_indices(p_lr)
        a_a1 = _load_indices(p_a1)
        if a_lr is None:
            missing_log.append(f"{ds}/{rtag} LR r={r}% indices missing")
        if a_a1 is None:
            missing_log.append(f"{ds}/{rtag} A1 r={r}% indices missing")
        lr_arrs[r] = a_lr
        a1_arrs[r] = a_a1

    # Random row entries (constant across sel/rank).
    rnd_lbl, rnd_frac = RANDOM_MAJ.get((ds, rtag), (None, None))
    rnd_cell = f"L{rnd_lbl}={rnd_frac}" if rnd_lbl is not None else "—"

    sel_pcts = TABLE_SELS  # [1,2,3,4,5,10,20]
    col_labels = [""] + [f"sel {s}%" for s in sel_pcts]

    rows = []
    cell_colors = []

    # ---- Collect frac values while building the table; reuse for heatmaps ----
    inv_fracs = []   # length len(sel_pcts)
    lr_fracs  = np.full((len(RANKS), len(sel_pcts)), np.nan, dtype=float)
    a1_fracs  = np.full((len(RANKS), len(sel_pcts)), np.nan, dtype=float)

    # Row 1: INV / random
    inv_row = ["INV / random"]
    inv_colors = ["#dddddd"]
    for s in sel_pcts:
        k = int(round(n * s / 100.0))
        inv_info = _top_class_info(inv_arr, labels_arr, k)
        inv_part = (f"L{inv_info[0]}={inv_info[1]}" if inv_info is not None else "—")
        inv_fracs.append(float(inv_info[1]) if inv_info is not None else np.nan)
        cell = f"{inv_part} / {rnd_cell}"
        inv_row.append(cell)
        inv_colors.append("#ffffff")
    rows.append(inv_row)
    cell_colors.append(inv_colors)

    # Row 2: group header for LR / A1 (non-data row, just a label spanning).
    grp_row = ["LR / A1"] + [""] * len(sel_pcts)
    grp_colors = ["#cfcfcf"] + ["#eeeeee"] * len(sel_pcts)
    rows.append(grp_row)
    cell_colors.append(grp_colors)

    # Rank rows
    for ri, r in enumerate(RANKS):
        rrow = [f"r={r}%"]
        rcolors = ["#eeeeee"]
        for si, s in enumerate(sel_pcts):
            k = int(round(n * s / 100.0))
            lr_info = _top_class_info(lr_arrs.get(r), labels_arr, k)
            a1_info = _top_class_info(a1_arrs.get(r), labels_arr, k)
            lr_part = (f"L{lr_info[0]}={lr_info[1]}" if lr_info is not None else "—")
            a1_part = (f"L{a1_info[0]}={a1_info[1]}" if a1_info is not None else "—")
            if lr_info is not None:
                lr_fracs[ri, si] = float(lr_info[1])
            if a1_info is not None:
                a1_fracs[ri, si] = float(a1_info[1])
            cell = f"{lr_part} / {a1_part}"
            rrow.append(cell)
            rcolors.append("#ffffff")
        rows.append(rrow)
        cell_colors.append(rcolors)

    # Pool ratio for the random row of heatmaps (frac %).
    pool_frac = float(rnd_frac) if rnd_frac is not None else np.nan

    # ---- Render: figure with table on top, heatmaps below ----
    fig = plt.figure(figsize=(13, 16))
    gs = fig.add_gridspec(4, 2,
                          height_ratios=[5, 0.8, 0.8, 5],
                          hspace=0.40, wspace=0.18)

    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis('off')

    ax_rnd  = fig.add_subplot(gs[1, :])
    ax_inv  = fig.add_subplot(gs[2, :])
    ax_lr   = fig.add_subplot(gs[3, 0])
    ax_a1   = fig.add_subplot(gs[3, 1])

    fig.suptitle(f"{ds} {rtag} valbal — top-Shapley class composition",
                 fontsize=14, y=0.995)
    # subtitle: setting label
    ax_tbl.text(0.5, 1.02, label, ha='center', va='bottom',
                fontsize=10.5, transform=ax_tbl.transAxes, color="#333333")

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       loc='upper center', cellLoc='center',
                       colWidths=[0.16] + [0.105] * len(sel_pcts),
                       cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)

    # bold header row + first column; style the LR/A1 group header row.
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#cccccc")
            cell.set_text_props(weight='bold')
        elif j == 0:
            cell.set_text_props(weight='bold')
        # i=2 in the rendered table is the LR/A1 group header row
        # (row 0 = column header, row 1 = INV/random, row 2 = LR/A1 group).
        if i == 2:
            cell.set_text_props(weight='bold', style='italic')

    # ---- Heatmaps ----
    # Custom YlOrRd-like cmap with a STRONGER dark-red high-end (#67000d).
    # The default matplotlib `YlOrRd` peaks at #800026, which several PDF
    # readers render as orange-red rather than the dark red the table is
    # meant to convey.  This explicit gradient keeps yellow→orange→red but
    # pushes the top of the scale to a clearly-saturated dark red.
    import matplotlib.colors as _mc
    cmap = _mc.LinearSegmentedColormap.from_list(
        "vivid_yorrd",
        ["#ffffff", "#ffeda0", "#feb24c", "#f03b20", "#bd0026", "#67000d"],
    )
    cmap.set_bad(color="lightgray")
    vmin, vmax = 0.0, 100.0

    sel_tick_labels = [f"sel{s}%" for s in sel_pcts]
    rank_tick_labels = [f"r={r}%" for r in RANKS]

    # Absolute color scale: hard-clamp to [0, 100] with a FRESH Normalize
    # object per imshow call.  Sharing a Normalize across multiple imshow()
    # calls is brittle — matplotlib's autoscale machinery can mutate the
    # object's vmin/vmax on the first call and propagate to later calls,
    # which is exactly what made some pages render 100 as orange instead of
    # the cmap's true max (dark red).  clip=True also ensures any spurious
    # >100 float (e.g. 100.0000001 from rounding) still maps to the top of
    # the cmap rather than overflow.
    import matplotlib.colors as _mc

    # Use vector rectangles instead of imshow.  matplotlib's imshow embeds a
    # raster image into the PDF; PDF readers (especially when zooming) then
    # resample that raster, which can shift colors of dense uniform regions
    # (e.g. an entire row of "100"s rendering as orange-red instead of the
    # cmap's true max).  Drawing each cell as a vector Rectangle keeps the
    # colors exactly as cmap(norm(value)) regardless of reader / zoom.
    from matplotlib.patches import Rectangle

    def _draw_heatmap(ax, data2d, *, title, yticklabels):
        """Vector-rectangle heatmap. data2d: (nrows, ncols) float (NaN OK).
        Returns a dummy ScalarMappable so caller can build a shared colorbar."""
        norm = _mc.Normalize(vmin=0.0, vmax=100.0, clip=True)
        nrows, ncols = data2d.shape
        for i in range(nrows):
            for j in range(ncols):
                val = data2d[i, j]
                if np.isnan(val):
                    facecolor = "lightgray"
                    txt = "—"
                    tcolor = "#666666"
                else:
                    facecolor = cmap(norm(val))
                    txt = f"{int(round(val))}"
                    tcolor = "white" if val >= 65 else "black"
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                       facecolor=facecolor, edgecolor='none'))
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=tcolor)
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)   # invert y so row 0 is on top
        ax.set_xticks(range(ncols))
        ax.set_xticklabels(sel_tick_labels, fontsize=8)
        if yticklabels is not None:
            ax.set_yticks(range(nrows))
            ax.set_yticklabels(yticklabels, fontsize=8)
        else:
            ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_aspect('auto')
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Return ScalarMappable for the (shared) colorbar.
        import matplotlib.cm as _cm_local
        sm = _cm_local.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        return sm

    # random heatmap (1×7, constant)
    rnd_data = np.full((1, len(sel_pcts)), pool_frac, dtype=float)
    _draw_heatmap(
        ax_rnd, rnd_data,
        title=(f"random  (pool ref ≈ {rnd_frac}%, label L{rnd_lbl})"
               if rnd_lbl is not None
               else "random  (pool ref unavailable)"),
        yticklabels=["random"],
    )

    # INV heatmap (1×7)
    inv_data = np.array([inv_fracs], dtype=float)
    _draw_heatmap(
        ax_inv, inv_data,
        title="INV (full kernel)",
        yticklabels=["INV"],
    )

    # LR heatmap (7×7)
    _draw_heatmap(
        ax_lr, lr_fracs,
        title="LRFShap eigen",
        yticklabels=rank_tick_labels,
    )

    # A1 heatmap (7×7) — keep the AxesImage so we can drive a shared colorbar
    last_im = _draw_heatmap(
        ax_a1, a1_fracs,
        title="A1 eigen",
        yticklabels=rank_tick_labels,
    )

    # One figure-level colorbar (range 0–100, shared by all 4 heatmaps).
    # Uses its own fresh Normalize too — identical to each heatmap's norm
    # so the colorbar legend matches what's plotted in every cell.
    import matplotlib.cm as _cm2
    sm = _cm2.ScalarMappable(norm=_mc.Normalize(vmin=0.0, vmax=100.0, clip=True),
                              cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.06, 0.015, 0.42])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("dominant class fraction (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    note = (
        "[cmap v4 — vector rectangles, vivid YlOrRd, hard-clipped 0-100]\n"
        "Heatmap shows the fraction (%) of the dominant class in the top-(sel%) Shapley selection.\n"
        "Label identity is intentionally ignored — only the *concentration* is shown.\n"
        "random row = pool ratio (constant). 100 → dark red (#67000d). shared figure-level colorbar."
    )
    fig.text(0.07, 0.015, note, fontsize=9, va='bottom', ha='left',
             family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- Shapley wall-clock + speedup table ----------
# Source: parse the experiment logs.  Each sub-run prints a banner like
#   "[<ds-tag>-valbal] pos=X.X method={lrfshap|a1} {inv baseline|eigen r=R%}"
#   "[<ds-tag>-valbal] pos=X.X INV baseline (method-independent)"
#   "[<ds-tag>-valbal] <ratio_tag> method={lrfshap|a1} eigen r=R%"
#   "[<ds-tag>-valbal] <ratio_tag> INV baseline (method-independent)"
# and immediately afterwards a tqdm progress block that ultimately reaches
#   "[TMC iterations]: 100%|...| 500/500 [H:MM:SS<00:00, ...s/it]"
# (the same `[TMC iterations]:` line is repeated several times across
# carriage-return updates; we keep the final 500/500 elapsed slice).

LOGDIR = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/experiments"

# Banner regex: capture optional `method=X`, optional `eigen r=R%` or
# `inv baseline` / `INV baseline (method-independent)`.
# Two flavours:
#   (a) pos-style binary:    "pos=0.5 method=lrfshap eigen r=10%"
#                             "pos=0.5 method=lrfshap inv baseline"
#                             "pos=0.5 INV baseline (method-independent)"
#   (b) cls-style multi:      "cls60_20_20 method=lrfshap eigen r=10%"
#                             "cls60_20_20 INV baseline (method-independent)"
#
# Banner prefix `[<dstag>-valbal]` already locks the dataset; we also
# capture the raw banner content for `kind` classification.
_BANNER_RE = re.compile(
    r"^\[(?P<dstag>[a-z0-9_-]+-valbal)\]\s+(?P<body>.*?)\s*$"
)

# Inside body, identify the run kind.
# kind tuple is (ratio_tag, role, rank)  where
#   role  ∈ {"INV","LR","A1"}
#   rank  = int rank percent for LR/A1, None for INV
_BODY_INV_POS_PERMETHOD = re.compile(
    r"^pos=(?P<pos>[0-9.]+)\s+method=(?P<m>lrfshap|a1)\s+inv\s+baseline\b"
)
_BODY_INV_POS_INDEP = re.compile(
    r"^pos=(?P<pos>[0-9.]+)\s+INV\s+baseline\s+\(method-independent\)"
)
_BODY_EIG_POS = re.compile(
    r"^pos=(?P<pos>[0-9.]+)\s+method=(?P<m>lrfshap|a1)\s+eigen\s+r=(?P<r>\d+)%"
)
_BODY_INV_CLS_PERMETHOD = re.compile(
    r"^(?P<tag>cls[0-9_]+)\s+method=(?P<m>lrfshap|a1)\s+inv\s+baseline\b"
)
_BODY_INV_CLS_INDEP = re.compile(
    r"^(?P<tag>cls[0-9_]+)\s+INV\s+baseline\s+\(method-independent\)"
)
_BODY_EIG_CLS = re.compile(
    r"^(?P<tag>cls[0-9_]+)\s+method=(?P<m>lrfshap|a1)\s+eigen\s+r=(?P<r>\d+)%"
)

# 500/500 elapsed regex — match the LAST occurrence on any line.
_TMC_RE = re.compile(
    r"500/500\s*\[(?P<elapsed>\d+(?::\d{1,2}){1,2})<"
)

# ds-tag (banner prefix) -> directive ds-name used by SOURCE / SETTINGS
_BANNER_DS = {
    "mrpc-n2300-valbal":  "mrpc",
    "sst2-n5000-valbal":  "sst2",
    "mr-n4500-valbal":    "mr",
    "qqp-n5000-valbal":   "qqp",
    "rte-n1300-valbal":   "rte",
    "agnews-n5000-valbal": "ag_news",
    "mnli-n5000-valbal":  "mnli",
}

# pos-float -> rtag (binary only).
# mrpc/sst2/mr/rte: pos50→pos50, pos70→pos70, pos90→pos90
# qqp: pos50→pos50, pos30→pos30, pos10→pos10  (label0 majority direction)
def _pos_to_rtag(pos_str):
    # accept "0.5", "0.7", "0.9", "0.3", "0.1"
    try:
        f = float(pos_str)
    except ValueError:
        return None
    pct = int(round(f * 100))
    return f"pos{pct}"


def _parse_banner(line):
    """Parse one log line; return (ds, rtag, role, rank) or None.

    role ∈ {"INV","LR","A1"};   rank = int|None.
    """
    m = _BANNER_RE.match(line)
    if not m:
        return None
    dstag = m.group("dstag")
    body  = m.group("body")
    ds    = _BANNER_DS.get(dstag)
    if ds is None:
        return None

    # eigen pos
    mm = _BODY_EIG_POS.match(body)
    if mm:
        rtag = _pos_to_rtag(mm.group("pos"))
        role = "LR" if mm.group("m") == "lrfshap" else "A1"
        return (ds, rtag, role, int(mm.group("r")))
    # inv pos (method-independent)
    mm = _BODY_INV_POS_INDEP.match(body)
    if mm:
        return (ds, _pos_to_rtag(mm.group("pos")), "INV", None)
    # inv pos (per-method) — treat both lrfshap-INV and a1-INV as same INV
    mm = _BODY_INV_POS_PERMETHOD.match(body)
    if mm:
        return (ds, _pos_to_rtag(mm.group("pos")), "INV", None)
    # eigen cls
    mm = _BODY_EIG_CLS.match(body)
    if mm:
        role = "LR" if mm.group("m") == "lrfshap" else "A1"
        return (ds, mm.group("tag"), role, int(mm.group("r")))
    # inv cls (independent)
    mm = _BODY_INV_CLS_INDEP.match(body)
    if mm:
        return (ds, mm.group("tag"), "INV", None)
    # inv cls (per-method)
    mm = _BODY_INV_CLS_PERMETHOD.match(body)
    if mm:
        return (ds, mm.group("tag"), "INV", None)
    return None


def _elapsed_to_seconds(s):
    """'1:23:45' or '12:34' → seconds (int)."""
    parts = s.split(":")
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h = 0; m, sec = parts
    else:
        return None
    return h * 3600 + m * 60 + sec


def _seconds_to_human(sec):
    """Format seconds → '6.5h' if ≥ 1h, else 'XX.X min' (always 1 decimal)."""
    if sec is None:
        return "—"
    if sec >= 3600:
        return f"{sec/3600.0:.1f} h"
    return f"{sec/60.0:.1f} min"


def parse_log_for_times(log_path):
    """Walk log, attach each banner to its immediately-following 500/500 line.

    Returns: dict {(ds, rtag, role, rank): seconds}.
    If multiple TMC lines follow a banner before the next banner, we keep the
    LAST 500/500 elapsed on those lines (matches the final tqdm flush).
    If a banner has no completed TMC line before the next banner, it is dropped.
    """
    out = {}
    if not os.path.exists(log_path):
        return out
    current_key = None
    last_elapsed = None
    with open(log_path, errors="replace") as f:
        for raw in f:
            # split on CR to handle tqdm multi-update single-line slabs
            for chunk in raw.split("\r"):
                key = _parse_banner(chunk)
                if key is not None:
                    # flush previous
                    if current_key is not None and last_elapsed is not None:
                        out[current_key] = last_elapsed
                    current_key = key
                    last_elapsed = None
                    continue
                if current_key is None:
                    continue
                # search 500/500 in this chunk; keep last
                for tm in _TMC_RE.finditer(chunk):
                    secs = _elapsed_to_seconds(tm.group("elapsed"))
                    if secs is not None:
                        last_elapsed = secs
    # flush trailing
    if current_key is not None and last_elapsed is not None:
        out[current_key] = last_elapsed
    return out


# (ds, rtag) -> (list_of_log_filenames, node_label, gpu_label).
# "CROSS_NODE" entries are intentionally skipped (multi-node assembly).
TIMING_SOURCE = {
    ("mrpc", "pos50"): (["run_imbalance_mrpc_n2300_valbal.log"],
                        "node01", "RTX 2080 Ti"),
    ("mrpc", "pos70"): (["run_imbalance_mrpc_n2300_valbal.log",
                         "run_imbalance_mrpc_n2300_valbal_resume.log"],
                        "node01", "RTX 2080 Ti"),
    ("mrpc", "pos90"): (["run_imbalance_mrpc_n2300_valbal_resume.log"],
                        "node01", "RTX 2080 Ti"),
    ("sst2", "pos50"): (["run_imbalance_sst2_n5000_valbal.log"],
                        "node04", "RTX 3090"),
    ("sst2", "pos70"): (["run_imbalance_sst2_n5000_valbal.log"],
                        "node04", "RTX 3090"),
    ("sst2", "pos90"): (["run_imbalance_sst2_n5000_valbal.log"],
                        "node04", "RTX 3090"),
    ("mr",   "pos50"): (["run_imbalance_mr_qqp_valbal.log"],
                        "node05", "RTX A6000"),
    ("mr",   "pos70"): (["run_imbalance_mr_qqp_valbal.log"],
                        "node05", "RTX A6000"),
    ("mr",   "pos90"): (["run_imbalance_mr_qqp_valbal.log"],
                        "node05", "RTX A6000"),
    ("qqp",  "pos50"): (["run_imbalance_mr_qqp_valbal.log"],
                        "node05", "RTX A6000"),
    # qqp/pos30, pos10 use "auto" — see _resolve_auto() below.
    ("qqp",  "pos30"): ("auto", None, None),
    ("qqp",  "pos10"): ("auto", None, None),
    ("rte",  "pos50"): (["run_imbalance_rte_n1300_valbal.log"],
                        "node04", "RTX 3090"),
    ("rte",  "pos70"): (["run_imbalance_rte_n1300_valbal.log"],
                        "node04", "RTX 3090"),
    ("rte",  "pos90"): (["run_imbalance_rte_n1300_valbal.log"],
                        "node04", "RTX 3090"),
    ("ag_news", "cls25_25_25_25"): (["run_imbalance_agnews_n5000_valbal.log"],
                                    "node03", "RTX 3090"),
    ("ag_news", "cls55_15_15_15"): (["run_imbalance_agnews_n5000_valbal.log"],
                                    "node03", "RTX 3090"),
    ("ag_news", "cls85_05_05_05"): (["run_imbalance_agnews_n5000_valbal.log"],
                                    "node03", "RTX 3090"),
    ("mnli", "cls33_33_33"): ("CROSS_NODE", None, None),
    ("mnli", "cls60_20_20"): (["run_imbalance_mnli_n5000_valbal_h100.log"],
                              "narnia", "H100 NVL"),
    ("mnli", "cls90_05_05"): ("CROSS_NODE", None, None),
}


# Ranks we'll display in the timing table (matches the rest of the PDF).
TIMING_RANKS = RANKS   # [1, 5, 10, 15, 20, 25, 30]


def _has_full_coverage(times, ds, rtag):
    """True iff INV + (LR @ all RANKS) + (A1 @ all RANKS) are all in `times`."""
    if (ds, rtag, "INV", None) not in times:
        return False
    for r in RANKS:
        if (ds, rtag, "LR", r) not in times:
            return False
        if (ds, rtag, "A1", r) not in times:
            return False
    return True


def _resolve_auto_qqp(ds, rtag, missing):
    """For qqp/pos30 and qqp/pos10: try node05 first, fall back to node-specific
    logs.  Returns (times_dict_filtered_to_this_ratio, node, gpu) or
    (None, None, None) when nothing covers."""
    assert ds == "qqp"
    candidates = []
    if rtag == "pos30":
        candidates = [
            (["run_imbalance_mr_qqp_valbal.log"],            "node05", "RTX A6000"),
            (["run_imbalance_qqp_pos30_n5000_valbal.log"],    "node01", "RTX 2080 Ti"),
        ]
    elif rtag == "pos10":
        candidates = [
            (["run_imbalance_mr_qqp_valbal.log"],            "node05", "RTX A6000"),
            (["run_imbalance_qqp_pos10_n5000_valbal.log"],    "node04", "RTX 3090"),
        ]
    best = None  # (n_present, times, node, gpu, logs)
    for logs, node, gpu in candidates:
        combined = {}
        for lg in logs:
            combined.update(parse_log_for_times(os.path.join(LOGDIR, lg)))
        if _has_full_coverage(combined, ds, rtag):
            return combined, node, gpu, logs
        # count how many of the needed (1 + 2*|RANKS|) keys are present
        need = [(ds, rtag, "INV", None)]
        for r in RANKS:
            need.append((ds, rtag, "LR", r))
            need.append((ds, rtag, "A1", r))
        n_present = sum(1 for k in need if k in combined)
        miss = [k for k in need if k not in combined]
        miss_str = ", ".join(("INV" if k[2]=="INV" else f"{k[2]} r={k[3]}%")
                             for k in miss)
        missing.append(f"[auto] {ds}/{rtag} on {node} missing: {miss_str}")
        if best is None or n_present > best[0]:
            best = (n_present, combined, node, gpu, logs)
    # No full coverage anywhere — return best partial.
    if best is not None and best[0] > 0:
        _, combined, node, gpu, logs = best
        missing.append(f"[auto] {ds}/{rtag}: no full coverage; using best partial = "
                       f"{node} ({best[0]} of {1 + 2*len(RANKS)} cells)")
        return combined, node, gpu, logs
    return None, None, None, None


def collect_timings():
    """Build a dict of (ds, rtag) -> {role,rank: secs, "_node":..., "_gpu":...,
    "_status": ... }.  Logs all missing pieces to a list returned alongside."""
    out = {}
    missing = []
    for (ds, rtag), entry in TIMING_SOURCE.items():
        logs, node, gpu = entry
        if logs == "CROSS_NODE":
            out[(ds, rtag)] = {"_status": "cross_node",
                               "_node": "multi", "_gpu": "—"}
            continue
        if logs == "auto":
            times, node, gpu, used_logs = _resolve_auto_qqp(ds, rtag, missing)
            if times is None:
                out[(ds, rtag)] = {"_status": "incomplete",
                                   "_node": "—", "_gpu": "—",
                                   "_logs": []}
                continue
        else:
            times = {}
            for lg in logs:
                times.update(parse_log_for_times(os.path.join(LOGDIR, lg)))
            used_logs = logs
        # filter to this (ds, rtag) and record per-role
        status = "ok" if _has_full_coverage(times, ds, rtag) else "partial"
        entry_out = {"_status": status,
                     "_node": node, "_gpu": gpu,
                     "_logs": used_logs}
        inv_key = (ds, rtag, "INV", None)
        entry_out["INV"] = times.get(inv_key)
        if entry_out["INV"] is None:
            missing.append(f"{ds}/{rtag}: INV time missing")
        for r in TIMING_RANKS:
            for role in ("LR", "A1"):
                k = (ds, rtag, role, r)
                v = times.get(k)
                entry_out[(role, r)] = v
                if v is None:
                    missing.append(f"{ds}/{rtag}: {role} r={r}% time missing")
        out[(ds, rtag)] = entry_out
    return out, missing


# ---------- timing table page ----------
def make_timing_page(pdf, timings):
    """Render one wide landscape page with:
       row = (ds/rtag with node + GPU),
       columns = INV, LR r∈RANKS, A1 r∈RANKS,
       cell text = 'time/speedup' (e.g. '12.3 min/38.4x').
    Cross-node rows are kept but every cell shown as '—'.
    """
    headers = [
        "setting (node, GPU)",
        "INV",
    ] + [f"LR r={r}%" for r in TIMING_RANKS] + [f"A1 r={r}%" for r in TIMING_RANKS]

    rows = []
    cell_colors = []
    # Use the SETTINGS order from the main script so the timing table follows
    # the same dataset order as the rest of the PDF.
    for stg in SETTINGS:
        ds, rtag, n, v, _label = stg
        entry = timings.get((ds, rtag))
        if entry is None:
            continue
        status = entry["_status"]
        node = entry.get("_node", "—")
        gpu  = entry.get("_gpu", "—")
        setting_label = f"{ds}/{rtag}\n({node}, {gpu})"
        if status == "cross_node":
            setting_label += "\n(cross-node, skipped)"
            row = [setting_label] + ["—"] * (1 + 2 * len(TIMING_RANKS))
            colors = ["#eeeeee"] + ["#f3f3f3"] * (1 + 2 * len(TIMING_RANKS))
            rows.append(row); cell_colors.append(colors)
            continue
        if status == "incomplete":
            setting_label += "\n(incomplete, skipped)"
            row = [setting_label] + ["—"] * (1 + 2 * len(TIMING_RANKS))
            colors = ["#eeeeee"] + ["#f3f3f3"] * (1 + 2 * len(TIMING_RANKS))
            rows.append(row); cell_colors.append(colors)
            continue
        if status == "partial":
            setting_label += "\n(partial — some cells —)"
        # status == "ok" or "partial"
        inv_sec = entry.get("INV")
        row = [setting_label]
        colors = ["#eeeeee"]
        # INV cell
        if inv_sec is None:
            row.append("—"); colors.append("#f3f3f3")
        else:
            row.append(f"{_seconds_to_human(inv_sec)}\n1.0x"); colors.append("#dddddd")
        # LR cells
        for r in TIMING_RANKS:
            s = entry.get(("LR", r))
            if s is None or inv_sec is None:
                row.append("—"); colors.append("#f3f3f3")
            else:
                spd = inv_sec / s if s > 0 else float("inf")
                row.append(f"{_seconds_to_human(s)}\n{spd:.1f}x")
                colors.append("#ffffff")
        # A1 cells
        for r in TIMING_RANKS:
            s = entry.get(("A1", r))
            if s is None or inv_sec is None:
                row.append("—"); colors.append("#f3f3f3")
            else:
                spd = inv_sec / s if s > 0 else float("inf")
                row.append(f"{_seconds_to_human(s)}\n{spd:.1f}x")
                colors.append("#ffffff")
        rows.append(row); cell_colors.append(colors)

    # Layout: landscape, wide.  16 columns (1 setting + 15 method).
    fig = plt.figure(figsize=(22, 14))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.suptitle(
        "Shapley wall-clock time and speedup (= INV time / method time)",
        fontsize=14, y=0.995,
    )

    ncols = len(headers)
    # column widths: setting gets ~14%, INV ~6%, the rest split evenly.
    setting_w = 0.13
    inv_w     = 0.055
    rem       = 1.0 - setting_w - inv_w
    other_w   = rem / (ncols - 2)
    col_widths = [setting_w, inv_w] + [other_w] * (ncols - 2)

    tbl = ax.table(cellText=rows, colLabels=headers, loc='upper center',
                   cellLoc='center', colWidths=col_widths,
                   cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.6)
    tbl.scale(1.0, 1.6)

    # Style header row + setting column.
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#cccccc")
            cell.set_text_props(weight='bold', fontsize=8)
        elif j == 0:
            cell.set_text_props(weight='bold', fontsize=7.2)

    note = (
        "Cell format: 'wall-clock time / speedup' (speedup = INV time / method time at same setting).\n"
        "INV column is the FreeShap baseline (full kernel inverse); its speedup is 1.0x by definition.\n"
        "Time is the elapsed portion of the final tqdm '500/500 [HH:MM:SS<00:00, ...]' line\n"
        "captured for each banner-tagged sub-run in the experiment logs.\n"
        "Settings tagged (cross-node, skipped) span multiple machines (mnli/cls33_33_33 split across\n"
        "node01+node03; mnli/cls90_05_05 split across 3 nodes); per-row speedups would mix hardware\n"
        "and are therefore omitted.\n"
        "Hardware labels reflect the node where each ratio was actually run; speedups are NOT\n"
        "comparable across rows because different GPUs were used."
    )
    fig.text(0.04, 0.04, note, fontsize=9.5, va='bottom', ha='left',
             family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- main ----------
def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    summary = {}
    n_pages = 0
    n_harm_cells = 0
    with PdfPages(OUT) as pdf:
        for stg in SETTINGS:
            print(f"[plot] {stg[0]}/{stg[1]}  n={stg[2]}  valbal={stg[3]}")
            make_page(stg, pdf, summary)
            n_pages += 1
        make_summary_page(pdf, summary)
        n_pages += 1
        make_group_rank_page(pdf, summary)
        n_pages += 1
        for ind_label, ind_key in INDICATORS:
            make_indicator_page(pdf, summary, ind_label, ind_key)
            n_pages += 1

        # 21 top-Shapley class composition tables (one page per setting).
        missing_log = []
        for stg in SETTINGS:
            print(f"[top-class] {stg[0]}/{stg[1]}  n={stg[2]}  valbal={stg[3]}")
            make_top_class_page(stg, pdf, missing_log)
            n_pages += 1

        # Final timing + speedup table page.
        print("[timing] parsing experiment logs ...")
        timings, timing_missing = collect_timings()
        make_timing_page(pdf, timings)
        n_pages += 1
        if timing_missing:
            print(f"[timing] partial/missing entries ({len(timing_missing)}):")
            for m in timing_missing:
                print(f"  - {m}")
        else:
            print("[timing] all per-cell entries resolved")
        # Summarise selection: which node/gpu was used per ratio.
        n_full     = sum(1 for v in timings.values() if v.get("_status") == "ok")
        n_partial  = sum(1 for v in timings.values() if v.get("_status") == "partial")
        n_cross    = sum(1 for v in timings.values() if v.get("_status") == "cross_node")
        n_incompl  = sum(1 for v in timings.values() if v.get("_status") == "incomplete")
        print(f"[timing] full rows: {n_full}, partial rows: {n_partial}, "
              f"cross-node skipped: {n_cross}, incomplete skipped: {n_incompl}")
        for (ds, rtag), v in timings.items():
            st = v.get("_status")
            if st in ("ok", "partial"):
                used = ", ".join(v.get("_logs", [])) or "—"
                tag  = "FULL" if st == "ok" else "PARTIAL"
                print(f"  - [{tag}] {ds}/{rtag} -> {v['_node']} ({v['_gpu']}) "
                      f"[logs: {used}]")
            else:
                print(f"  - [SKIP/{st}] {ds}/{rtag}")

    if missing_log:
        print(f"[top-class] missing files ({len(missing_log)}):")
        for m in missing_log:
            print(f"  - {m}")
    else:
        print("[top-class] all indices files present")

    for k, v in summary.items():
        # harm cell = any cell flagged '!'  -> n_cells * 2 (lr_harm + a1_harm overlap allowed)
        # we count cells that triggered '!' which is (lr_harm + a1_harm) at most; but the
        # marker '!' is set if EITHER side is harmful, so cells_with_!_marker <= n_cells.
        # Simpler: re-derive via 'O','X','-' counts.
        non_harm = v["a1_gt_lr"] + v["a1_lt_lr"] + v["a1_eq_lr"]
        n_harm_cells += (v["n_cells"] - non_harm)

    print(f"[done] saved -> {OUT}")
    print(f"[summary] pages = {n_pages}")
    print(f"[summary] ratios processed = {len(SETTINGS)}")
    print(f"[summary] cells with '!' (harm) marker = {n_harm_cells} / {sum(s['n_cells'] for s in summary.values())}")


if __name__ == "__main__":
    main()
