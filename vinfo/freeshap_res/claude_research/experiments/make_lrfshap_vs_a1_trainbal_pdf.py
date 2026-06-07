"""lrfshap_vs_a1_trainbal.pdf — per-setting (rank-panel fig + table) comparison
with random baseline.  **Balanced-accuracy edition.**

Forked from the naive-accuracy version.  Train side is balanced (50/50 pos),
val side varies across {balanced, mild, extreme} imbalance.  Group
classification therefore reflects the VAL imbalance level (column 5 of each
SETTINGS tuple).

Key changes vs the naive-acc version:
1. SETTINGS expanded to 10 entries (adds mrpc balanced + mild + extreme and
   keeps qqp valbal).  Some mrpc cells are PARTIAL (a few LR ranks missing) —
   the loader gracefully returns None for those cells.
2. Primary metric is **balanced accuracy**.  All cell markers, summary stats,
   group/rank tables, and indicator (numerator/denominator) tables now reduce
   over `top_results_inv_balanced` / `random_results_inv_balanced`.  Missing
   `_balanced` keys (notably qqp valbal1000) propagate as None → "—" / "*"
   markers.
3. Per-setting figure now plots BOTH naive (dotted) AND balanced (solid) acc
   curves on the same rank-r panel, so the visual comparison is preserved
   while the marker/table logic is balanced-only.
4. RANDOM_MAJ extended to include mrpc entries (label1 ≈ 68% natural).
"""
import json, os, re, sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/data_selection_test/imbalance/data_selection"
OUT  = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/reports/lrfshap_vs_a1_trainbal.pdf"

SEL_RANGE  = list(range(1, 21))           # 1..20 % on figure
RANKS      = [1, 5, 10, 15, 20, 25, 30]
TABLE_SELS = [1, 2, 3, 4, 5, 10, 20]


# ---------- ratio list ----------
# (ds, train_rtag, n_train, val_tag, val_label, pretty_label)
# val_tag examples: "valbal1000", "valimb500_pos70", "valimb500_pos90".
# val_label ∈ {"balanced", "mild", "extreme"} — used directly for IMB_GROUP.
SETTINGS = [
    # MR: train 50/50, val 3 levels
    ("mr",   "pos50", 4500, "valbal1000",       "balanced",
     "MR  train 50/50 + val balanced 50/50 (n=4500, val=1000)"),
    ("mr",   "pos50", 4500, "valimb500_pos70",  "mild",
     "MR  train 50/50 + val 70/30 label1-maj (n=4500, val=500)"),
    ("mr",   "pos50", 4500, "valimb500_pos90",  "extreme",
     "MR  train 50/50 + val 90/10 label1-maj (n=4500, val=500)"),
    # SST-2: train 50/50, val 3 levels
    ("sst2", "pos50", 5000, "valbal856",        "balanced",
     "SST-2 train 50/50 + val balanced 50/50 (n=5000, val=856)"),
    ("sst2", "pos50", 5000, "valimb400_pos70",  "mild",
     "SST-2 train 50/50 + val 70/30 label1-maj (n=5000, val=400)"),
    ("sst2", "pos50", 5000, "valimb400_pos90",  "extreme",
     "SST-2 train 50/50 + val 90/10 label1-maj (n=5000, val=400)"),
    # MRPC: train 50/50, val 3 levels (some partial coverage)
    ("mrpc", "pos50", 2300, "valbal258",        "balanced",
     "MRPC train 50/50 + val balanced 50/50 (n=2300, val=258)"),
    ("mrpc", "pos50", 2300, "valimb300_pos70",  "mild",
     "MRPC train 50/50 + val 70/30 label1-maj (n=2300, val=300)"),
    ("mrpc", "pos50", 2300, "valimb300_pos90",  "extreme",
     "MRPC train 50/50 + val 90/10 label1-maj (n=2300, val=300)"),
    # QQP: train 50/50, val balanced (no _balanced key — balanced cells will be missing)
    ("qqp",  "pos50", 5000, "valbal1000",       "balanced",
     "QQP  train 50/50 + val balanced 50/50 (n=5000, val=1000)"),
    ("qqp",  "pos50", 5000, "valimb1000_pos30", "mild",
     "QQP  train 50/50 + val 70/30 label0-maj (n=5000, val=1000)"),
    ("qqp",  "pos50", 5000, "valimb1000_pos10", "extreme",
     "QQP  train 50/50 + val 90/10 label0-maj (n=5000, val=1000)"),
    # RTE (binary, valbal 와 다른 size — valimb cap 작아 n=145)
    ("rte",  "pos50", 1300, "valimb145_pos70",  "mild",
     "RTE  train 50/50 + val 70/30 label1-maj (n=1300, val=145)"),
    ("rte",  "pos50", 1300, "valimb145_pos90",  "extreme",
     "RTE  train 50/50 + val 90/10 label1-maj (n=1300, val=145)"),
    # AG News (4-class, both fully done)
    ("ag_news", "cls25_25_25_25", 5000, "valimb1000_cls55_15_15_15", "mild",
     "AG News train cls25 + val cls55_15_15_15 label0-maj (n=5000, val=1000)"),
    ("ag_news", "cls25_25_25_25", 5000, "valimb1000_cls85_05_05_05", "extreme",
     "AG News train cls25 + val cls85_05_05_05 label0-maj (n=5000, val=1000)"),
    # MNLI (3-class, both fully done)
    ("mnli", "cls33_33_33", 5000, "valimb1000_cls60_20_20", "mild",
     "MNLI train cls33 + val cls60_20_20 label0-maj (n=5000, val=1000)"),
    ("mnli", "cls33_33_33", 5000, "valimb1000_cls90_05_05", "extreme",
     "MNLI train cls33 + val cls90_05_05 label0-maj (n=5000, val=1000)"),
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


# Each loader returns a 4-tuple:
#   (top_naive, random_naive, top_balanced, random_balanced)
# Any missing piece is returned as None — caller handles propagation.
def fs_inv(ds, rtag, n, val_tag):
    p = (f"{BASE}/{ds}/{rtag}/lrfshap/inv/sidecar/"
         f"bert_seed2026_num{n}_{val_tag}_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500.json")
    d = _load(p)
    if d is None:
        return None, None, None, None
    return (d.get("top_results_inv"),
            d.get("random_results_inv"),
            d.get("top_results_inv_balanced"),
            d.get("random_results_inv_balanced"))


def lr_eig(ds, rtag, n, val_tag, r):
    p = (f"{BASE}/{ds}/{rtag}/lrfshap/eigen/sidecar/"
         f"bert_seed2026_num{n}_{val_tag}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_"
         f"lrfshap_signFalse_earlystopTrue_tmc500.json")
    d = _load(p)
    if d is None:
        return None, None, None, None
    return (d.get("top_results_inv"),
            d.get("random_results_inv"),
            d.get("top_results_inv_balanced"),
            d.get("random_results_inv_balanced"))


def a1_eig(ds, rtag, n, val_tag, r):
    p = (f"{BASE}/{ds}/{rtag}/a1/eigen/sidecar/"
         f"bert_seed2026_num{n}_{val_tag}_eig{r}.0_eiglam1e-02_invlam1e-06_cholesky_float32_"
         f"a1_signFalse_earlystopTrue_tmc500.json")
    d = _load(p)
    if d is None:
        return None, None, None, None
    return (d.get("top_results_inv"),
            d.get("random_results_inv"),
            d.get("top_results_inv_balanced"),
            d.get("random_results_inv_balanced"))


# ---------- table cell marker logic ----------
# A1-centric scheme, computed on BALANCED accuracy:
#   ! : A1 ≤ random (A1 harmful, no contribution)
#   O : A1 > LR AND A1 > random
#   X : A1 < LR  (A1 still > random)
#   - : A1 == LR (A1 still > random)
# A trailing dagger (†) is appended whenever LR ≤ random (LR collapsed at
# that cell — exactly the regime A1 is supposed to rescue).
# Returns "*" when any of the three balanced inputs is missing.
def cell_marker(lr_v, a1_v, rnd_v):
    """Return (text, facecolor) for one (rank,sel%) cell."""
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
def make_page(stg, pdf, summary):
    ds, rtag, n, val_tag, val_label, label = stg
    v = val_tag
    fs_top_n, fs_rnd_n, fs_top_b, fs_rnd_b = fs_inv(ds, rtag, n, val_tag)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1], hspace=0.55, wspace=0.35)

    # Random baseline curves for the figure — separate naive and balanced.
    # Prefer FS-INV random; else average rank-wise random curves.
    def _pick_rand(naive_or_balanced):
        """naive_or_balanced ∈ {'naive', 'balanced'}."""
        idx_n, idx_b = (1, 3)
        slot_top = 0 if naive_or_balanced == "naive" else 2
        slot_rnd = idx_n if naive_or_balanced == "naive" else idx_b
        # Try FS-INV random first.
        fs = fs_rnd_n if naive_or_balanced == "naive" else fs_rnd_b
        if fs is not None:
            return fs
        rand_curves = []
        for r in RANKS:
            tup = lr_eig(ds, rtag, n, v, r)
            if tup[slot_rnd] is not None:
                rand_curves.append(tup[slot_rnd])
            tup = a1_eig(ds, rtag, n, v, r)
            if tup[slot_rnd] is not None:
                rand_curves.append(tup[slot_rnd])
        if rand_curves:
            return np.array(rand_curves, dtype=float).mean(axis=0).tolist()
        return None

    rand_naive_fig    = _pick_rand("naive")
    rand_balanced_fig = _pick_rand("balanced")

    # ---- per-rank panels: both naive (dotted) and balanced (solid) curves ----
    for i, r in enumerate(RANKS):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        lr_top_n, _, lr_top_b, _ = lr_eig(ds, rtag, n, v, r)
        a1_top_n, _, a1_top_b, _ = a1_eig(ds, rtag, n, v, r)

        # Helper to plot one curve safely (skip None points).
        def _plot(ax, ys, *, color, label, marker, linestyle, linewidth):
            if ys is None:
                return
            xs = []; vals = []
            for s in SEL_RANGE:
                y = ys[s-1] if s-1 < len(ys) else None
                if y is None:
                    continue
                xs.append(s); vals.append(y/100.0)
            if xs:
                ax.plot(xs, vals, color=color, label=label, marker=marker,
                        linestyle=linestyle, linewidth=linewidth, markersize=3.2)

        # Naive curves (dotted)
        _plot(ax, fs_top_n,        color='red',   label='FS naive',   marker=None, linestyle=':',  linewidth=1.2)
        _plot(ax, lr_top_n,        color='blue',  label=f'LR naive r={r}%', marker='o', linestyle=':', linewidth=1.0)
        _plot(ax, a1_top_n,        color='green', label=f'A1 naive r={r}%', marker='s', linestyle=':', linewidth=1.0)
        _plot(ax, rand_naive_fig,  color='gray',  label='rand naive', marker=None, linestyle=':',  linewidth=1.0)

        # Balanced curves (solid) — the primary metric.
        _plot(ax, fs_top_b,        color='red',   label='FS bal',    marker=None, linestyle='-',  linewidth=1.8)
        _plot(ax, lr_top_b,        color='blue',  label=f'LR bal r={r}%', marker='o', linestyle='-', linewidth=1.3)
        _plot(ax, a1_top_b,        color='green', label=f'A1 bal r={r}%', marker='s', linestyle='-', linewidth=1.3)
        _plot(ax, rand_balanced_fig, color='gray', label='rand bal',  marker=None, linestyle='--', linewidth=1.2)

        ax.set_title(f"eigen r = {r}%", fontsize=10)
        ax.set_xlabel("selected %", fontsize=8)
        ax.set_ylabel("val accuracy (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.0, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)

    ax_blank = fig.add_subplot(gs[1, 3])
    ax_blank.axis('off')
    legend_text = (
        "PRIMARY metric: BALANCED accuracy (solid).\n"
        "Naive accuracy shown for reference (dotted).\n"
        "\n"
        "Markers in lower table (balanced acc, A1-centric):\n"
        "  O : A1 > LR  (A1 also beats random)\n"
        "  X : A1 < LR  (A1 still beats random)\n"
        "  - : A1 = LR  (A1 still beats random)\n"
        "  ! : A1 ≤ random  (A1 itself harmful)\n"
        "  † : LR ≤ random  (LR collapsed —\n"
        "      regime A1 is designed for)\n"
        "  * : missing _balanced data\n"
        "\nRandom baseline (figure curve) = FS-INV\n"
        "random_results_inv[_balanced].  Missing\n"
        "_balanced data is skipped point-by-point\n"
        "(no curve drawn for that metric)."
    )
    ax_blank.text(0.0, 0.95, legend_text, fontsize=8.5, va='top', ha='left',
                  family='monospace')

    # ---- table (BALANCED accuracy) ----
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis('off')

    if fs_top_b is None:
        ax_t.text(0.5, 0.5, "(balanced INV baseline missing — table unavailable)",
                  ha='center', va='center', fontsize=10)
        # Still register an empty summary row so the global tables don't trip.
        summary[(ds, rtag, val_tag)] = {
            "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
            "lr_harm": 0, "a1_harm": 0, "n_cells": 0,
            "by_rank": {r: {"a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
                            "lr_harm": 0, "a1_harm": 0, "n_cells": 0}
                        for r in RANKS},
            "by_rank_sel": {(r, s): {"valid": 0, "a1_gt_lr": 0,
                                      "a1_gt_inv": 0, "a1_gt_rnd": 0,
                                      "a1_gt_all": 0}
                            for r in RANKS for s in TABLE_SELS},
        }
    else:
        col_labels = [""] + [f"sel{s}%" for s in TABLE_SELS]

        rows = []
        cell_colors = []

        # Row 0: INV (balanced accuracy + harm flag vs FS-INV balanced random)
        inv_row = ["INV (bal acc)"]
        inv_colors = ["#dddddd"]
        for s in TABLE_SELS:
            val = fs_top_b[s-1] / 100.0
            text = f"{val:.2f}"
            bg = "#ffffff"
            if fs_rnd_b is not None and fs_top_b[s-1] <= fs_rnd_b[s-1]:
                text += "\n[!]"
                bg = "#ffd6d6"
            inv_row.append(text)
            inv_colors.append(bg)
        rows.append(inv_row)
        cell_colors.append(inv_colors)

        per_setting_stats = {
            "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
            "lr_harm": 0, "a1_harm": 0, "n_cells": 0,
            "by_rank": {r: {"a1_gt_lr": 0, "a1_lt_lr": 0,
                            "a1_eq_lr": 0, "lr_harm": 0,
                            "a1_harm": 0, "n_cells": 0}
                        for r in RANKS},
            "by_rank_sel": {(r, s): {"valid": 0,
                                      "a1_gt_lr": 0,
                                      "a1_gt_inv": 0,
                                      "a1_gt_rnd": 0,
                                      "a1_gt_all": 0}
                            for r in RANKS for s in TABLE_SELS},
        }

        # Per-cell baseline = FS-INV BALANCED random (single well-defined baseline).
        base_rnd_b = fs_rnd_b

        for r in RANKS:
            _, _, lr_top_b, _ = lr_eig(ds, rtag, n, v, r)
            _, _, a1_top_b, _ = a1_eig(ds, rtag, n, v, r)

            row = [f"r={r}%"]
            colors = ["#eeeeee"]
            for s in TABLE_SELS:
                lr_v = lr_top_b[s-1] if lr_top_b is not None else None
                a1_v = a1_top_b[s-1] if a1_top_b is not None else None
                rnd_v = base_rnd_b[s-1] if base_rnd_b is not None else None
                marker, bg = cell_marker(lr_v, a1_v, rnd_v)
                row.append(marker)
                colors.append(bg)

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

                # Per-cell raw indicators (all balanced).
                inv_v = fs_top_b[s-1] if fs_top_b is not None else None
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

        for (i, j), cell in tbl.get_celld().items():
            if i == 0:
                cell.set_facecolor("#dddddd")
                cell.set_text_props(weight='bold')
            elif j == 0:
                cell.set_text_props(weight='bold')

        summary[(ds, rtag, val_tag)] = per_setting_stats

    fig.suptitle(label + f"   |   n_train={n}, val_tag={val_tag}, val_imb={val_label}"
                 + "   |   metric = balanced accuracy",
                 fontsize=13, y=0.995)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- imbalance group mapping ----------
IMB_GROUP = {
    (stg[0], stg[1], stg[3]): stg[4] for stg in SETTINGS
}


# ---------- summary page ----------
def _stat_row(label, st):
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
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 0.35], hspace=0.20)
    ax_top = fig.add_subplot(gs[0, 0]); ax_top.axis('off')
    ax_bot = fig.add_subplot(gs[1, 0]); ax_bot.axis('off')
    ax_note = fig.add_subplot(gs[2, 0]); ax_note.axis('off')

    fig.suptitle("Summary (BALANCED acc): A1 vs LRFShap vs random  (per-cell breakdown)",
                 fontsize=15, y=0.99)

    headers = ["group", "cells", "A1 > LR", "A1 < LR", "A1 = LR",
               "LR ≤ rnd (harm)", "A1 ≤ rnd (harm)",
               "A1 win %", "LR-safe %", "A1-safe %"]

    rows = []
    tot = {"n_cells": 0, "a1_gt_lr": 0, "a1_lt_lr": 0, "a1_eq_lr": 0,
           "lr_harm": 0, "a1_harm": 0}
    for (ds, rtag, val_tag), st in summary.items():
        rows.append(_stat_row(f"{ds}/{rtag}/{val_tag}", st))
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
    ax_top.set_title("Breakdown by setting (rows = dataset/ratio/val_tag; cols aggregate over rank × sel%)",
                     fontsize=11, loc='left', pad=12)

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

    note = (
        "Notes:\n"
        "  - PRIMARY metric: balanced accuracy (top_results_inv_balanced).\n"
        "  - n_cells = #valid (rank, sel%) pairs per setting (drops cells with\n"
        "    missing _balanced data — affects qqp/valbal1000 entirely).\n"
        "  - 'A1 win %' = #cells where A1 strictly beats LR (and beats random — '!' excluded).\n"
        "  - 'LR-safe %' = #cells where LR > random; 'A1-safe %' = #cells where A1 > random.\n"
        "  - Per-cell baseline = FS-INV balanced random (random_results_inv_balanced)."
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

    fig.suptitle("Per-rank breakdown grouped by val imbalance level "
                 "(balanced / mild / extreme) — BALANCED acc",
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
        member_labels = ", ".join(f"{ds}/{rtag}/{val_tag}"
                                   for (ds, rtag, val_tag) in group_settings)

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
        "Group definitions in this trainbal+valvary fork:\n"
        "  - BALANCED: val_tag = valbal*    (val composition matches train 50/50)\n"
        "  - MILD:     val_tag = valimb*_pos70  (val label1 70%)\n"
        "  - EXTREME:  val_tag = valimb*_pos90  (val label1 90%)\n"
        "\nEach row aggregates 7 sel% × all settings in that group at the given rank.\n"
        "Random baseline = FS-INV balanced random.  qqp/valbal1000 contributes\n"
        "0 cells because its sidecar predates the balanced-acc rerun."
    )
    ax_note.text(0.02, 0.95, note, fontsize=9.5, va='top', ha='left', family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- indicator tables (group × rank × sel) ----------
INDICATORS = [
    ("A1 > LR  (balanced)",                       "a1_gt_lr"),
    ("A1 > INV (full kernel, balanced)",          "a1_gt_inv"),
    ("A1 > random  (balanced)",                   "a1_gt_rnd"),
    ("A1 > max(LR, INV, random)  (balanced)",     "a1_gt_all"),
]


def make_indicator_page(pdf, summary, ind_label, ind_key):
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
            cells = [f"r={r}%"] + [
                (f"{num_by_sel[s]}/{den_by_sel[s]}"
                 if den_by_sel[s] > 0 else "—")
                for s in TABLE_SELS
            ]
            num_all = sum(num_by_sel[s] for s in TABLE_SELS)
            den_all = sum(den_by_sel[s] for s in TABLE_SELS)
            cells.append(f"{num_all}/{den_all}" if den_all > 0 else "—")
            rows.append(cells)
            row_colors.append([grp_palette[group_name]] * len(cells))
            for s in TABLE_SELS:
                grp_total_num[s] += num_by_sel[s]
                grp_total_den[s] += den_by_sel[s]

        cells = [f"{group_name.upper()} ALL (n={n_grp})"] + [
            (f"{grp_total_num[s]}/{grp_total_den[s]}"
             if grp_total_den[s] > 0 else "—")
            for s in TABLE_SELS
        ]
        num_all = sum(grp_total_num[s] for s in TABLE_SELS)
        den_all = sum(grp_total_den[s] for s in TABLE_SELS)
        cells.append(f"{num_all}/{den_all}" if den_all > 0 else "—")
        rows.append(cells)
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
        "Reading guide (BALANCED accuracy):\n"
        f"  - Each cell is '<numerator>/<denominator>' where the numerator counts how\n"
        f"    many settings in that group satisfy `{ind_label}` at that (rank, sel%).\n"
        f"  - Denominator drops settings whose _balanced sidecar key is missing\n"
        f"    at that cell, so qqp/valbal1000 (no _balanced) is dropped everywhere\n"
        f"    and the partial mrpc cells drop one column each.\n"
        f"  - 'ALL sel%' sums numerators / denominators across the 7 sel% columns.\n"
        f"  - The '<GROUP> ALL (n=N)' row sums over that group's N settings and\n"
        f"    the 7 ranks for each sel%.  Trainbal+valvary fork has 4 BALANCED\n"
        f"    (mr/sst2/mrpc/qqp; qqp drops out), 3 MILD (mr/sst2/mrpc), and 3\n"
        f"    EXTREME (mr/sst2/mrpc) settings."
    )
    ax.text(0.02, 0.05, note, fontsize=9, va='top', ha='left', family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- top-Shapley class composition tables ----------
# Index-based, metric-independent — unchanged by the balanced-acc switch.
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
    if ds not in _LABEL_CACHE:
        d = _DS_LOADERS[ds]()
        _LABEL_CACHE[ds] = np.asarray(d["train"]["label"])
    return _LABEL_CACHE[ds]


def _load_indices(path):
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


def _inv_indices_path(ds, rtag, n, val_tag):
    return (f"{BASE}/{ds}/{rtag}/lrfshap/inv/indices/"
            f"bert_seed2026_num{n}_{val_tag}_lam1e-06_lrfshap_signFalse_"
            f"earlystopTrue_tmc500_indices.txt")


def _lr_eig_indices_path(ds, rtag, n, val_tag, r):
    return (f"{BASE}/{ds}/{rtag}/lrfshap/eigen/indices/"
            f"bert_seed2026_num{n}_{val_tag}_eig{r}.0_eiglam1e-02_invlam1e-06_"
            f"cholesky_float32_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt")


def _a1_eig_indices_path(ds, rtag, n, val_tag, r):
    return (f"{BASE}/{ds}/{rtag}/a1/eigen/indices/"
            f"bert_seed2026_num{n}_{val_tag}_eig{r}.0_eiglam1e-02_invlam1e-06_"
            f"cholesky_float32_a1_signFalse_earlystopTrue_tmc500_indices.txt")


# Intended majority direction (label, frac%) per setting for the random row.
# mr/sst2: label1 is natural majority; valimb pos70/pos90 → L1=70 / L1=90.
# mrpc: label1 ≈ 68% natural majority on the train pool.
# qqp: label0 is natural majority direction.
RANDOM_MAJ = {
    ("mr",   "pos50", "valbal1000"):       (1, 50),
    ("mr",   "pos50", "valimb500_pos70"):  (1, 70),
    ("mr",   "pos50", "valimb500_pos90"):  (1, 90),
    ("sst2", "pos50", "valbal856"):        (1, 50),
    ("sst2", "pos50", "valimb400_pos70"):  (1, 70),
    ("sst2", "pos50", "valimb400_pos90"):  (1, 90),
    ("mrpc", "pos50", "valbal258"):        (1, 68),
    ("mrpc", "pos50", "valimb300_pos70"):  (1, 70),
    ("mrpc", "pos50", "valimb300_pos90"):  (1, 90),
    ("qqp",  "pos50", "valbal1000"):       (0, 50),
    ("qqp",  "pos50", "valimb1000_pos30"): (0, 70),
    ("qqp",  "pos50", "valimb1000_pos10"): (0, 90),
    ("rte",  "pos50", "valimb145_pos70"):  (1, 70),
    ("rte",  "pos50", "valimb145_pos90"):  (1, 90),
    ("ag_news", "cls25_25_25_25", "valimb1000_cls55_15_15_15"): (0, 55),
    ("ag_news", "cls25_25_25_25", "valimb1000_cls85_05_05_05"): (0, 85),
    ("mnli", "cls33_33_33", "valimb1000_cls60_20_20"):           (0, 60),
    ("mnli", "cls33_33_33", "valimb1000_cls90_05_05"):           (0, 90),
}


def _top_class_str(indices_arr, labels_arr, k):
    info = _top_class_info(indices_arr, labels_arr, k)
    if info is None:
        return "—"
    top_class, frac = info
    return f"L{top_class}={frac}"


def _top_class_info(indices_arr, labels_arr, k):
    if indices_arr is None or labels_arr is None:
        return None
    if k <= 0 or k > len(indices_arr):
        return None
    sl = indices_arr[:k]
    if (sl < 0).any() or (sl >= len(labels_arr)).any():
        return None
    sub_labels = labels_arr[sl]
    if len(sub_labels) == 0:
        return None
    top_class, top_count = Counter(sub_labels.tolist()).most_common(1)[0]
    frac = int(round(100.0 * top_count / k))
    return int(top_class), frac


def make_top_class_page(stg, pdf, missing_log):
    ds, rtag, n, val_tag, val_label, label = stg
    labels_arr = _train_labels(ds)

    inv_arr = _load_indices(_inv_indices_path(ds, rtag, n, val_tag))
    if inv_arr is None:
        missing_log.append(f"{ds}/{rtag}/{val_tag} INV indices missing")
    lr_arrs = {}
    a1_arrs = {}
    for r in RANKS:
        p_lr = _lr_eig_indices_path(ds, rtag, n, val_tag, r)
        p_a1 = _a1_eig_indices_path(ds, rtag, n, val_tag, r)
        a_lr = _load_indices(p_lr)
        a_a1 = _load_indices(p_a1)
        if a_lr is None:
            missing_log.append(f"{ds}/{rtag}/{val_tag} LR r={r}% indices missing")
        if a_a1 is None:
            missing_log.append(f"{ds}/{rtag}/{val_tag} A1 r={r}% indices missing")
        lr_arrs[r] = a_lr
        a1_arrs[r] = a_a1

    rnd_lbl, rnd_frac = RANDOM_MAJ.get((ds, rtag, val_tag), (None, None))
    rnd_cell = f"L{rnd_lbl}={rnd_frac}" if rnd_lbl is not None else "—"

    sel_pcts = TABLE_SELS
    col_labels = [""] + [f"sel {s}%" for s in sel_pcts]

    rows = []
    cell_colors = []

    inv_fracs = []
    lr_fracs  = np.full((len(RANKS), len(sel_pcts)), np.nan, dtype=float)
    a1_fracs  = np.full((len(RANKS), len(sel_pcts)), np.nan, dtype=float)

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

    grp_row = ["LR / A1"] + [""] * len(sel_pcts)
    grp_colors = ["#cfcfcf"] + ["#eeeeee"] * len(sel_pcts)
    rows.append(grp_row)
    cell_colors.append(grp_colors)

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

    pool_frac = float(rnd_frac) if rnd_frac is not None else np.nan

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

    fig.suptitle(f"{ds} {rtag} {val_tag} — top-Shapley class composition",
                 fontsize=14, y=0.995)
    ax_tbl.text(0.5, 1.02, label, ha='center', va='bottom',
                fontsize=10.5, transform=ax_tbl.transAxes, color="#333333")

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       loc='upper center', cellLoc='center',
                       colWidths=[0.16] + [0.105] * len(sel_pcts),
                       cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)

    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#cccccc")
            cell.set_text_props(weight='bold')
        elif j == 0:
            cell.set_text_props(weight='bold')
        if i == 2:
            cell.set_text_props(weight='bold', style='italic')

    import matplotlib.colors as _mc
    cmap = _mc.LinearSegmentedColormap.from_list(
        "vivid_yorrd",
        ["#ffffff", "#ffeda0", "#feb24c", "#f03b20", "#bd0026", "#67000d"],
    )
    cmap.set_bad(color="lightgray")
    vmin, vmax = 0.0, 100.0

    sel_tick_labels = [f"sel{s}%" for s in sel_pcts]
    rank_tick_labels = [f"r={r}%" for r in RANKS]

    from matplotlib.patches import Rectangle

    def _draw_heatmap(ax, data2d, *, title, yticklabels):
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
        ax.set_ylim(nrows - 0.5, -0.5)
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
        import matplotlib.cm as _cm_local
        sm = _cm_local.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        return sm

    rnd_data = np.full((1, len(sel_pcts)), pool_frac, dtype=float)
    _draw_heatmap(
        ax_rnd, rnd_data,
        title=(f"random  (pool ref ≈ {rnd_frac}%, label L{rnd_lbl})"
               if rnd_lbl is not None
               else "random  (pool ref unavailable)"),
        yticklabels=["random"],
    )

    inv_data = np.array([inv_fracs], dtype=float)
    _draw_heatmap(
        ax_inv, inv_data,
        title="INV (full kernel)",
        yticklabels=["INV"],
    )

    _draw_heatmap(
        ax_lr, lr_fracs,
        title="LRFShap eigen",
        yticklabels=rank_tick_labels,
    )

    last_im = _draw_heatmap(
        ax_a1, a1_fracs,
        title="A1 eigen",
        yticklabels=rank_tick_labels,
    )

    import matplotlib.cm as _cm2
    sm = _cm2.ScalarMappable(norm=_mc.Normalize(vmin=0.0, vmax=100.0, clip=True),
                              cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.06, 0.015, 0.42])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("dominant class fraction (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    note = (
        "Heatmap shows the fraction (%) of the dominant class in the top-(sel%) Shapley selection.\n"
        "Label identity is intentionally ignored — only the *concentration* is shown.\n"
        "random row = pool ratio (constant). 100 → dark red (#67000d). shared figure-level colorbar."
    )
    fig.text(0.07, 0.015, note, fontsize=9, va='bottom', ha='left',
             family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------- Shapley wall-clock + speedup table ----------
LOGDIR = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/experiments"

_BANNER_RE = re.compile(
    r"^\[(?P<dstag>[a-z0-9_-]+-(?:valbal|valimb))\]\s+(?P<body>.*?)\s*$"
)

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
_BODY_INV_VALIMB_INDEP = re.compile(
    r"^train\s+pos=(?P<tp>[0-9.]+)\s+val\s+pos=(?P<vp>[0-9.]+)\s+"
    r"INV\s+baseline\s+\(method-independent\)"
)
_BODY_INV_VALIMB_PERMETHOD = re.compile(
    r"^train\s+pos=(?P<tp>[0-9.]+)\s+val\s+pos=(?P<vp>[0-9.]+)\s+"
    r"method=(?P<m>lrfshap|a1)\s+inv\s+baseline\b"
)
_BODY_EIG_VALIMB = re.compile(
    r"^train\s+pos=(?P<tp>[0-9.]+)\s+val\s+pos=(?P<vp>[0-9.]+)\s+"
    r"method=(?P<m>lrfshap|a1)\s+eigen\s+r=(?P<r>\d+)%"
)

_TMC_RE = re.compile(
    r"500/500\s*\[(?P<elapsed>\d+(?::\d{1,2}){1,2})<"
)

_BANNER_DS = {
    "mrpc-n2300-valbal":   "mrpc",
    "sst2-n5000-valbal":   "sst2",
    "mr-n4500-valbal":     "mr",
    "qqp-n5000-valbal":    "qqp",
    "rte-n1300-valbal":    "rte",
    "agnews-n5000-valbal": "ag_news",
    "mnli-n5000-valbal":   "mnli",
    "mr-n4500-valimb":     "mr",
    "sst2-n5000-valimb":   "sst2",
    "qqp-n5000-valimb":    "qqp",
    "mrpc-n2300-valimb":   "mrpc",
}


def _pos_to_rtag(pos_str):
    try:
        f = float(pos_str)
    except ValueError:
        return None
    pct = int(round(f * 100))
    return f"pos{pct}"


def _val_pos_to_valimb_tag(ds, vp_str):
    try:
        f = float(vp_str)
    except ValueError:
        return None
    pct = int(round(f * 100))
    val_size = {"mr": 500, "sst2": 400, "mrpc": 300, "qqp": None}.get(ds)
    if val_size is None:
        return None
    return f"valimb{val_size}_pos{pct}"


def _parse_banner(line):
    m = _BANNER_RE.match(line)
    if not m:
        return None
    dstag = m.group("dstag")
    body  = m.group("body")
    ds    = _BANNER_DS.get(dstag)
    if ds is None:
        return None
    is_valimb = dstag.endswith("-valimb")

    if is_valimb:
        mm = _BODY_EIG_VALIMB.match(body)
        if mm:
            rtag = _pos_to_rtag(mm.group("tp"))
            val_tag = _val_pos_to_valimb_tag(ds, mm.group("vp"))
            role = "LR" if mm.group("m") == "lrfshap" else "A1"
            return (ds, rtag, val_tag, role, int(mm.group("r")))
        mm = _BODY_INV_VALIMB_INDEP.match(body)
        if mm:
            rtag = _pos_to_rtag(mm.group("tp"))
            val_tag = _val_pos_to_valimb_tag(ds, mm.group("vp"))
            return (ds, rtag, val_tag, "INV", None)
        mm = _BODY_INV_VALIMB_PERMETHOD.match(body)
        if mm:
            rtag = _pos_to_rtag(mm.group("tp"))
            val_tag = _val_pos_to_valimb_tag(ds, mm.group("vp"))
            return (ds, rtag, val_tag, "INV", None)
        return None

    val_tag_default = {
        "mr": "valbal1000", "sst2": "valbal856", "qqp": "valbal1000",
        "mrpc": "valbal258", "rte": "valbal262",
        "ag_news": "valbal1000", "mnli": "valbal1000",
    }.get(ds)

    mm = _BODY_EIG_POS.match(body)
    if mm:
        rtag = _pos_to_rtag(mm.group("pos"))
        role = "LR" if mm.group("m") == "lrfshap" else "A1"
        return (ds, rtag, val_tag_default, role, int(mm.group("r")))
    mm = _BODY_INV_POS_INDEP.match(body)
    if mm:
        return (ds, _pos_to_rtag(mm.group("pos")), val_tag_default, "INV", None)
    mm = _BODY_INV_POS_PERMETHOD.match(body)
    if mm:
        return (ds, _pos_to_rtag(mm.group("pos")), val_tag_default, "INV", None)
    mm = _BODY_EIG_CLS.match(body)
    if mm:
        role = "LR" if mm.group("m") == "lrfshap" else "A1"
        return (ds, mm.group("tag"), val_tag_default, role, int(mm.group("r")))
    mm = _BODY_INV_CLS_INDEP.match(body)
    if mm:
        return (ds, mm.group("tag"), val_tag_default, "INV", None)
    mm = _BODY_INV_CLS_PERMETHOD.match(body)
    if mm:
        return (ds, mm.group("tag"), val_tag_default, "INV", None)
    return None


def _elapsed_to_seconds(s):
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
    if sec is None:
        return "—"
    if sec >= 3600:
        return f"{sec/3600.0:.1f} h"
    return f"{sec/60.0:.1f} min"


def parse_log_for_times(log_path):
    out = {}
    if not os.path.exists(log_path):
        return out
    current_key = None
    last_elapsed = None
    with open(log_path, errors="replace") as f:
        for raw in f:
            for chunk in raw.split("\r"):
                key = _parse_banner(chunk)
                if key is not None:
                    if current_key is not None and last_elapsed is not None:
                        out[current_key] = last_elapsed
                    current_key = key
                    last_elapsed = None
                    continue
                if current_key is None:
                    continue
                for tm in _TMC_RE.finditer(chunk):
                    secs = _elapsed_to_seconds(tm.group("elapsed"))
                    if secs is not None:
                        last_elapsed = secs
    if current_key is not None and last_elapsed is not None:
        out[current_key] = last_elapsed
    return out


# (ds, rtag, val_tag) -> (list_of_log_filenames, node_label, gpu_label).
# Trainbal+valvary fork — 10 settings.
TIMING_SOURCE = {
    ("mr",   "pos50", "valbal1000"):       (["run_imbalance_mr_qqp_valbal.log"],
                                            "node05", "RTX A6000"),
    ("mr",   "pos50", "valimb500_pos70"):  (["run_imbalance_mr_valimb_pos70.log"],
                                            "node04", "RTX 3090"),
    ("mr",   "pos50", "valimb500_pos90"):  (["run_imbalance_mr_valimb_pos90.log"],
                                            "node04", "RTX 3090"),
    ("sst2", "pos50", "valbal856"):        (["run_imbalance_sst2_n5000_valbal.log"],
                                            "node04", "RTX 3090"),
    ("sst2", "pos50", "valimb400_pos70"):  (["run_imbalance_sst2_valimb_pos70.log"],
                                            "node04", "RTX 3090"),
    ("sst2", "pos50", "valimb400_pos90"):  (["run_imbalance_sst2_valimb_pos90.log"],
                                            "node04", "RTX 3090"),
    ("mrpc", "pos50", "valbal258"):        (["run_imbalance_mrpc_n2300_valbal.log",
                                             "run_imbalance_mrpc_n2300_valbal_resume.log"],
                                            "node01", "RTX 3090"),
    ("mrpc", "pos50", "valimb300_pos70"):  (["run_imbalance_mrpc_valimb_pos70.log"],
                                            "node04", "RTX 3090"),
    ("mrpc", "pos50", "valimb300_pos90"):  (["run_imbalance_mrpc_valimb_pos90.log"],
                                            "node05", "RTX A6000"),
    ("qqp",  "pos50", "valbal1000"):       (["run_imbalance_mr_qqp_valbal.log"],
                                            "node05", "RTX A6000"),
    ("qqp",  "pos50", "valimb1000_pos30"): (["run_imbalance_qqp_valimb_pos30.log"],
                                            "node01", "RTX 2080 Ti"),
    ("qqp",  "pos50", "valimb1000_pos10"): (["run_imbalance_qqp_valimb_pos10.log"],
                                            "node03", "RTX 3090"),
    ("rte",  "pos50", "valimb145_pos70"):  (["run_imbalance_rte_trainbal_valimb.log"],
                                            "node03", "RTX 3090"),
    ("rte",  "pos50", "valimb145_pos90"):  (["run_imbalance_rte_trainbal_valimb.log"],
                                            "node03", "RTX 3090"),
    ("ag_news", "cls25_25_25_25", "valimb1000_cls55_15_15_15"):
        (["run_imbalance_agnews_trainbal_valimb.log"], "node05", "RTX A6000"),
    ("ag_news", "cls25_25_25_25", "valimb1000_cls85_05_05_05"):
        (["run_imbalance_agnews_trainbal_valimb.log"], "node05", "RTX A6000"),
    ("mnli", "cls33_33_33", "valimb1000_cls60_20_20"):
        (["run_imbalance_mnli_trainbal_valimb_pos60.log"], "node04", "RTX A6000"),
    ("mnli", "cls33_33_33", "valimb1000_cls90_05_05"):
        (["run_imbalance_mnli_trainbal_valimb_cls90_lrfshap_inv.log",
          "run_imbalance_mnli_trainbal_valimb_cls90_a1.log",
          "run_imbalance_mnli_trainbal_valimb_cls90_lrfshap_eigen.log"],
         "node03+node01+node05", "RTX 3090 / RTX 2080Ti / RTX A6000"),
}


TIMING_RANKS = RANKS


def _has_full_coverage(times, ds, rtag, val_tag):
    if (ds, rtag, val_tag, "INV", None) not in times:
        return False
    for r in RANKS:
        if (ds, rtag, val_tag, "LR", r) not in times:
            return False
        if (ds, rtag, val_tag, "A1", r) not in times:
            return False
    return True


def collect_timings():
    out = {}
    missing = []
    for (ds, rtag, val_tag), entry in TIMING_SOURCE.items():
        logs, node, gpu = entry
        times = {}
        for lg in logs:
            times.update(parse_log_for_times(os.path.join(LOGDIR, lg)))
        used_logs = logs
        status = "ok" if _has_full_coverage(times, ds, rtag, val_tag) else "partial"
        entry_out = {"_status": status,
                     "_node": node, "_gpu": gpu,
                     "_logs": used_logs}
        inv_key = (ds, rtag, val_tag, "INV", None)
        entry_out["INV"] = times.get(inv_key)
        if entry_out["INV"] is None:
            missing.append(f"{ds}/{rtag}/{val_tag}: INV time missing")
        for r in TIMING_RANKS:
            for role in ("LR", "A1"):
                k = (ds, rtag, val_tag, role, r)
                v = times.get(k)
                entry_out[(role, r)] = v
                if v is None:
                    missing.append(f"{ds}/{rtag}/{val_tag}: {role} r={r}% time missing")
        out[(ds, rtag, val_tag)] = entry_out
    return out, missing


# ---------- timing table page ----------
def make_timing_page(pdf, timings):
    headers = [
        "setting (node, GPU)",
        "INV",
    ] + [f"LR r={r}%" for r in TIMING_RANKS] + [f"A1 r={r}%" for r in TIMING_RANKS]

    rows = []
    cell_colors = []
    for stg in SETTINGS:
        ds, rtag, n, val_tag, val_label, _label = stg
        entry = timings.get((ds, rtag, val_tag))
        if entry is None:
            continue
        status = entry["_status"]
        node = entry.get("_node", "—")
        gpu  = entry.get("_gpu", "—")
        setting_label = f"{ds}/{rtag}/{val_tag}\n({node}, {gpu})"
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
        inv_sec = entry.get("INV")
        row = [setting_label]
        colors = ["#eeeeee"]
        if inv_sec is None:
            row.append("—"); colors.append("#f3f3f3")
        else:
            row.append(f"{_seconds_to_human(inv_sec)}\n1.0x"); colors.append("#dddddd")
        for r in TIMING_RANKS:
            s = entry.get(("LR", r))
            if s is None or inv_sec is None:
                row.append("—"); colors.append("#f3f3f3")
            else:
                spd = inv_sec / s if s > 0 else float("inf")
                row.append(f"{_seconds_to_human(s)}\n{spd:.1f}x")
                colors.append("#ffffff")
        for r in TIMING_RANKS:
            s = entry.get(("A1", r))
            if s is None or inv_sec is None:
                row.append("—"); colors.append("#f3f3f3")
            else:
                spd = inv_sec / s if s > 0 else float("inf")
                row.append(f"{_seconds_to_human(s)}\n{spd:.1f}x")
                colors.append("#ffffff")
        rows.append(row); cell_colors.append(colors)

    fig = plt.figure(figsize=(22, 14))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.suptitle(
        "Shapley wall-clock time and speedup (= INV time / method time)",
        fontsize=14, y=0.995,
    )

    ncols = len(headers)
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
        "Hardware labels reflect the node where each setting was actually run; speedups are NOT\n"
        "comparable across rows because different GPUs were used.\n"
        "Timing is independent of the accuracy metric — these numbers do not change between\n"
        "the naive-acc and balanced-acc editions of this PDF."
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
    # Track which (ds, rtag, val_tag) had any missing _balanced.
    bal_missing_log = []
    with PdfPages(OUT) as pdf:
        for stg in SETTINGS:
            print(f"[plot] {stg[0]}/{stg[1]}  n={stg[2]}  val_tag={stg[3]}  group={stg[4]}")
            # Pre-check balanced key coverage for reporting.
            ds, rtag, n_, val_tag = stg[0], stg[1], stg[2], stg[3]
            _, _, fs_b, fs_rb = fs_inv(ds, rtag, n_, val_tag)
            n_missing_eig = 0
            n_total_eig = 0
            for r in RANKS:
                _, _, lb, _ = lr_eig(ds, rtag, n_, val_tag, r)
                _, _, ab, _ = a1_eig(ds, rtag, n_, val_tag, r)
                for tag, x in (("LR", lb), ("A1", ab)):
                    n_total_eig += 1
                    if x is None:
                        n_missing_eig += 1
            bal_missing_log.append({
                "setting": f"{ds}/{rtag}/{val_tag}",
                "fs_inv_bal_top": fs_b is not None,
                "fs_inv_bal_rnd": fs_rb is not None,
                "eig_bal_missing": n_missing_eig,
                "eig_bal_total": n_total_eig,
            })
            make_page(stg, pdf, summary)
            n_pages += 1
        make_summary_page(pdf, summary)
        n_pages += 1
        make_group_rank_page(pdf, summary)
        n_pages += 1
        for ind_label, ind_key in INDICATORS:
            make_indicator_page(pdf, summary, ind_label, ind_key)
            n_pages += 1

        missing_log = []
        for stg in SETTINGS:
            print(f"[top-class] {stg[0]}/{stg[1]}  n={stg[2]}  val_tag={stg[3]}  group={stg[4]}")
            make_top_class_page(stg, pdf, missing_log)
            n_pages += 1

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
        n_full     = sum(1 for v in timings.values() if v.get("_status") == "ok")
        n_partial  = sum(1 for v in timings.values() if v.get("_status") == "partial")
        n_cross    = sum(1 for v in timings.values() if v.get("_status") == "cross_node")
        n_incompl  = sum(1 for v in timings.values() if v.get("_status") == "incomplete")
        print(f"[timing] full rows: {n_full}, partial rows: {n_partial}, "
              f"cross-node skipped: {n_cross}, incomplete skipped: {n_incompl}")
        for (ds, rtag, val_tag), v in timings.items():
            st = v.get("_status")
            if st in ("ok", "partial"):
                used = ", ".join(v.get("_logs", [])) or "—"
                tag  = "FULL" if st == "ok" else "PARTIAL"
                print(f"  - [{tag}] {ds}/{rtag}/{val_tag} -> {v['_node']} ({v['_gpu']}) "
                      f"[logs: {used}]")
            else:
                print(f"  - [SKIP/{st}] {ds}/{rtag}/{val_tag}")

    if missing_log:
        print(f"[top-class] missing files ({len(missing_log)}):")
        for m in missing_log:
            print(f"  - {m}")
    else:
        print("[top-class] all indices files present")

    print()
    print("[balanced-key coverage]")
    for e in bal_missing_log:
        print(f"  - {e['setting']}: FS bal top={e['fs_inv_bal_top']}, "
              f"FS bal rnd={e['fs_inv_bal_rnd']}, "
              f"eig bal missing={e['eig_bal_missing']}/{e['eig_bal_total']}")

    for k, v in summary.items():
        non_harm = v["a1_gt_lr"] + v["a1_lt_lr"] + v["a1_eq_lr"]
        n_harm_cells += (v["n_cells"] - non_harm)

    print(f"[done] saved -> {OUT}")
    print(f"[summary] pages = {n_pages}")
    print(f"[summary] settings processed = {len(SETTINGS)}")
    print(f"[summary] cells with '!' (harm) marker = {n_harm_cells} / "
          f"{sum(s['n_cells'] for s in summary.values())}")


if __name__ == "__main__":
    main()
