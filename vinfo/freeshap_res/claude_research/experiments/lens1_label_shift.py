#!/usr/bin/env python
"""
lens1_label_shift.py
====================
Iteration 04, plan §2.5 — Lens 1: label-shift / BBSE / weighted risk.

Inputs:
  state/iteration_04/grand_df.csv
  state/iteration_04/grand_meta.csv
  data_selection_test/imbalance/data_selection/<...>/indices/*.txt
    (one index per line; computed from cumulative top selection at sel = S)

Outputs:
  state/iteration_04/lens1_table.csv  — setting-level + cell-level columns
  state/iteration_04/lens1_corr.csv   — H1 / H2 hypothesis decisions
  state/iteration_04/lens1_figs/*.png — H1 / H2 scatter
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

WORK_ROOT = Path("/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research")
SIDECAR_ROOT = WORK_ROOT / "data_selection_test/imbalance/data_selection"
OUT_DIR = WORK_ROOT / "state/iteration_04"
FIG_DIR = OUT_DIR / "lens1_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LABELS_CACHE_DIR = OUT_DIR / "labels_cache"


# ----- helpers reused from build_grand_df pattern -----
def hf_loader_args(dataset_name: str):
    if dataset_name == "sst2":
        return ("sst2",), {}, "validation"
    if dataset_name == "mr":
        return ("rotten_tomatoes",), {}, "validation"
    if dataset_name == "qqp":
        return ("glue", "qqp"), {}, "validation"
    if dataset_name == "mnli":
        return ("glue", "mnli"), {}, "validation_matched"
    if dataset_name == "ag_news":
        return ("ag_news",), {}, "test"
    if dataset_name == "mrpc":
        return ("glue", "mrpc"), {}, "validation"
    if dataset_name == "rte":
        return ("glue", "rte"), {}, "validation"
    raise ValueError(dataset_name)


_LBL_CACHE = {}


def load_labels(dn):
    if dn in _LBL_CACHE:
        return _LBL_CACHE[dn]
    tr = np.load(LABELS_CACHE_DIR / f"{dn}_train.npy").astype(np.int64)
    va = np.load(LABELS_CACHE_DIR / f"{dn}_val.npy").astype(np.int64)
    _LBL_CACHE[dn] = (tr, va)
    return tr, va


def kl_div(p, q):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p / max(p.sum(), eps), eps, 1.0)
    q = np.clip(q / max(q.sum(), eps), eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    eps = 1e-12
    pn = np.clip(p / max(p.sum(), eps), eps, 1.0)
    return float(-np.sum(pn * np.log(pn)))


def fisher_ci(r, n, alpha=0.05):
    """95% Fisher CI for Pearson r."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - zc * se); hi = np.tanh(z + zc * se)
    return float(lo), float(hi)


# ----- find indices for selected subset of a given (setting, method, rank, sel) -----
INDICES_DIR_FOR_METHOD_APPROX = {
    ("A1", "eigen"): ("a1", "eigen"),
    ("LR", "eigen"): ("lrfshap", "eigen"),
    ("FreeShap", "inv"): ("lrfshap", "inv"),
}


def find_indices_file(dataset, train_tag, method, approx, num_train, val_tag,
                     rank_pct: Optional[float]) -> Optional[Path]:
    m_dir, a_dir = INDICES_DIR_FOR_METHOD_APPROX[(method, approx)]
    d = SIDECAR_ROOT / dataset / train_tag / m_dir / a_dir / "indices"
    if not d.exists():
        return None
    if approx == "eigen":
        if rank_pct is None:
            return None
        pat = f"bert_seed2026_num{num_train}_{val_tag}_eig{rank_pct}_eiglam1e-02_invlam1e-06_cholesky_float32_{m_dir}_signFalse_earlystopTrue_tmc500_indices.txt"
    else:
        pat = f"bert_seed2026_num{num_train}_{val_tag}_lam1e-06_{m_dir}_signFalse_earlystopTrue_tmc500_indices.txt"
    p = d / pat
    if p.exists():
        return p
    # fuzzy fall-back
    for cand in d.glob("*.txt"):
        if val_tag in cand.name:
            if approx == "eigen" and rank_pct is not None and f"eig{rank_pct}_" in cand.name:
                return cand
            if approx == "inv" and "lam1e-06" in cand.name and "eig" not in cand.name:
                return cand
    return None


def read_indices(p: Path) -> list:
    """Read whitespace-separated ints from a file; cap by max needed sel."""
    out = []
    with open(p, "r") as f:
        txt = f.read()
    for tok in re.split(r"[\s,]+", txt.strip()):
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            out.append(int(tok))
    return out


# ----- BBSE estimator -----
def bbse_estimator(y_train_local, y_val_local, C, P_train_y, P_val_y):
    """Compute BBSE quantities using a 'naive' classifier ŷ = argmax P_train(y).

    This is the weakest possible classifier — it returns σ_min(Ĉ) ≈ 0 for binary
    cases (good sanity baseline) but lets us verify the BBSE pipeline.  Returns
    dict with sigma_min, cond, oracle w, bbse w, L1 err.
    """
    # Train confusion matrix Ĉ_{ij} = (1/n) Σ 1[ŷ=i, y=j] using a *trivial* classifier
    # that always predicts the majority train class.  This is a baseline; cond_C_hat
    # will be 0 for binary (singular), large for multi-class with imbalanced majority.
    # Better: use top-1 prediction from train marginal — same thing here.
    yhat_class = int(np.argmax(P_train_y))
    C_hat = np.zeros((C, C), dtype=np.float64)
    for j in range(C):
        # all predictions are yhat_class
        C_hat[yhat_class, j] = float((y_train_local == j).sum()) / len(y_train_local)
    svd = np.linalg.svd(C_hat, compute_uv=False)
    sigma_min = float(svd[-1]); sigma_max = float(svd[0])
    cond = sigma_max / max(sigma_min, 1e-12)

    # mu_val(yhat) — naive classifier predicts always yhat_class
    mu_val = np.zeros(C, dtype=np.float64)
    mu_val[yhat_class] = 1.0
    # If C is singular (binary trivial case), pinv
    try:
        q_hat = np.linalg.pinv(C_hat) @ mu_val
    except Exception:
        q_hat = P_val_y.copy()
    # importance weight
    w_oracle = P_val_y / np.maximum(P_train_y, 1e-12)
    w_bbse = q_hat / np.maximum(P_train_y, 1e-12)
    bbse_err = float(np.sum(np.abs(w_oracle - w_bbse)))
    return {
        "sigma_min_C_hat": sigma_min,
        "cond_C_hat": cond,
        "bbse_err": bbse_err,
        "w_oracle": w_oracle.tolist(),
        "w_bbse": w_bbse.tolist(),
    }


# ----- main lens 1 table build -----
def main():
    print("=" * 78)
    print("lens1_label_shift.py — iteration 04 §2 BBSE / weighted risk")
    print("=" * 78)

    grand = pd.read_csv(OUT_DIR / "grand_df.csv")
    meta = pd.read_csv(OUT_DIR / "grand_meta.csv")

    # parse list columns
    for c in ["P_train_y", "P_val_y", "train_class_counts", "val_class_counts"]:
        meta[c] = meta[c].apply(lambda v: json.loads(v) if isinstance(v, str) else v)

    print(f"[info] {len(grand):,} cells, {len(meta)} settings")

    # ===== Step 1: setting-level table =====
    rows = []
    cell_kl_records = []  # for cell-level KL_val_S and KL_S_val per (setting, method, sel)
    for _, m in meta.iterrows():
        dataset = m["dataset"]
        train_tag = m["train_ratio_tag"]
        val_tag = m["val_ratio_tag"]
        regime = m["regime"]
        setting_id = m["setting_id"]
        C = m["C"]
        n = m["num_train"]; n_val = m["num_val"]
        P_train_y = np.asarray(m["P_train_y"], dtype=np.float64)
        P_val_y = np.asarray(m["P_val_y"], dtype=np.float64)

        # KL both directions
        kl_val_train = kl_div(P_val_y, P_train_y)
        kl_train_val = kl_div(P_train_y, P_val_y)
        w_vec = P_val_y / np.maximum(P_train_y, 1e-12)

        H_train = entropy(P_train_y); H_val = entropy(P_val_y)

        # BBSE quantities — need train labels for the trivial classifier
        try:
            tr_lbl_full, va_lbl_full = load_labels(dataset)
            # We need the actual sampled indices — re-load pickle? No, we have counts.
            # The 'naive classifier' uses only the global class counts, so we synthesize
            # y_train_local from the per-class counts.
            train_counts = np.asarray(m["train_class_counts"], dtype=np.int64)
            y_train_local = np.repeat(np.arange(C), train_counts)
            bbse = bbse_estimator(y_train_local, None, C, P_train_y, P_val_y)
        except Exception as e:
            print(f"  [warn] BBSE fail {setting_id}: {e}")
            bbse = {"sigma_min_C_hat": np.nan, "cond_C_hat": np.nan,
                    "bbse_err": np.nan, "w_oracle": [], "w_bbse": []}

        # weighted-risk proxy: (acc_naive − acc_balanced) of FreeShap at sel ∈ {5,10,100}
        risk_proxy = {}
        gsub = grand[(grand["setting_id"] == setting_id) & (grand["method"] == "FreeShap")]
        for s in [5, 10, 100]:
            sub = gsub[gsub["sel"] == s]
            if len(sub) == 0:
                risk_proxy[f"risk_w_proxy_sel{s}"] = np.nan
            else:
                # take any rank (FreeShap is rank-independent for inv); average just in case
                risk_proxy[f"risk_w_proxy_sel{s}"] = float(
                    (sub["acc_top_naive"] - sub["acc_top_balanced"]).mean()
                )

        row = {
            "setting_id": setting_id,
            "dataset": dataset,
            "regime": regime,
            "train_ratio_tag": train_tag,
            "val_ratio_tag": val_tag,
            "imbalance_level": m["imbalance_level"],
            "C": int(C),
            "n": int(n),
            "n_val": int(n_val),
            "P_train_y": json.dumps(P_train_y.tolist()),
            "P_val_y": json.dumps(P_val_y.tolist()),
            "kl_val_train": kl_val_train,
            "kl_train_val": kl_train_val,
            "w_vector": json.dumps(w_vec.tolist()),
            "H_P_train": H_train,
            "H_P_val": H_val,
            "max_P_train": float(P_train_y.max()),
            "max_P_val": float(P_val_y.max()),
            "sigma_min_C_hat": bbse["sigma_min_C_hat"],
            "cond_C_hat": bbse["cond_C_hat"],
            "bbse_err": bbse["bbse_err"],
            **risk_proxy,
        }
        rows.append(row)

        # ===== cell-level KL_val_S, KL_S_val =====
        # For each (method, approx, rank_pct, sel) cell, read the indices file.
        # We sweep methods ∈ {A1, LR, FreeShap} at rank_pct=10 (A1/LR) and rank=None (FreeShap).
        # For each, sels in {1, 2, 3, 5, 10}.
        sels_to_evaluate = [1, 2, 3, 5, 10]
        # Build a label vector to project onto.  Indices files store *global*
        # raw dataset indices (HF row IDs) — they index directly into the full
        # train_labels array.  Sampling has already been applied upstream so
        # the indices listed are guaranteed to be inside sampled_idx, but the
        # values themselves are global dataset positions.
        try:
            train_lbl_full = tr_lbl_full
            train_lbl_local = train_lbl_full  # we'll index by global idx
            train_lbl_max = len(train_lbl_full)
        except Exception as e:
            print(f"  [warn] cannot resolve train labels for {setting_id}: {e}")
            train_lbl_local = None
            train_lbl_max = 0

        for method, approx, rank in [
            ("A1", "eigen", 10.0),
            ("LR", "eigen", 10.0),
            ("FreeShap", "inv", None),
        ]:
            idx_path = find_indices_file(dataset, train_tag, method, approx,
                                          n, val_tag, rank)
            if idx_path is None or train_lbl_local is None:
                continue
            try:
                all_idx = read_indices(idx_path)
            except Exception:
                continue
            for s in sels_to_evaluate:
                top_idx = all_idx[:s]
                if len(top_idx) == 0:
                    continue
                # global raw dataset indices: clip to valid range
                gidx = np.array([i for i in top_idx if 0 <= i < train_lbl_max],
                                 dtype=np.int64)
                if len(gidx) == 0:
                    continue
                lbls = train_lbl_local[gidx]
                cnts = np.bincount(lbls, minlength=C).astype(np.float64)
                P_S = cnts / cnts.sum()
                kl_val_S = kl_div(P_val_y, P_S)
                kl_S_val = kl_div(P_S, P_val_y)
                cell_kl_records.append({
                    "setting_id": setting_id,
                    "method": method, "rank_pct": rank, "sel": s,
                    "kl_val_S": kl_val_S, "kl_S_val": kl_S_val,
                    "P_S": json.dumps(P_S.tolist()),
                })

    table = pd.DataFrame(rows)
    cell_table = pd.DataFrame(cell_kl_records)

    # Merge cell_table back to table wide-form — risk too sparse, store as long-format
    # Write both: setting-level (rows) + cell-level (long, separate file).
    table_path = OUT_DIR / "lens1_table.csv"
    table.to_csv(table_path, index=False)
    cell_path = OUT_DIR / "lens1_cell_kl.csv"
    cell_table.to_csv(cell_path, index=False)
    print(f"[step 1] wrote {table_path}: {len(table)} settings")
    print(f"[step 1] wrote {cell_path}: {len(cell_table)} cell-level KL entries")

    # ===== Step 2: H1, H2 correlations =====
    from scipy.stats import pearsonr, spearmanr
    corr_rows = []

    def _add_corr(hyp, predictor_col, outcome_col, n_settings, x, y,
                  regime_label, imb_label, mixed_beta=np.nan, mixed_ci=(np.nan, np.nan),
                  decision=""):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]
        if len(x) < 3:
            return
        try:
            pr = pearsonr(x, y); sr = spearmanr(x, y)
            pear_r, pear_p = pr.statistic, pr.pvalue
            sp_r, sp_p = sr.statistic, sr.pvalue
        except Exception:
            return
        lo, hi = fisher_ci(pear_r, len(x))
        corr_rows.append({
            "hypothesis_id": hyp,
            "predictor": predictor_col,
            "outcome": outcome_col,
            "regime": regime_label,
            "imbalance_level": imb_label,
            "n_settings": len(x),
            "pearson_r": pear_r,
            "pearson_p": pear_p,
            "pearson_ci_lo": lo, "pearson_ci_hi": hi,
            "spearman_r": sp_r, "spearman_p": sp_p,
            "mixed_beta": mixed_beta,
            "mixed_ci_lo": mixed_ci[0], "mixed_ci_hi": mixed_ci[1],
            "decision": decision,
        })

    # ----- H1: trainbal regime, predictor = kl_val_S averaged over (sel in {1,2,3,5}, method=A1),
    #              outcome = gap_top_random_balanced averaged over the same cells -----
    H1_grand = grand[(grand["regime"] == "trainbal") &
                     (grand["method"] == "A1") &
                     (grand["rank_pct"] == 10.0) &
                     (grand["sel"].isin([1, 2, 3, 5]))]
    H1_setting = H1_grand.groupby("setting_id").agg(
        outcome=("gap_top_random_balanced", "mean")
    ).reset_index()
    H1_cellkl = cell_table[(cell_table["method"] == "A1") &
                            (cell_table["rank_pct"] == 10.0) &
                            (cell_table["sel"].isin([1, 2, 3, 5]))]
    H1_setting_kl = H1_cellkl.groupby("setting_id").agg(
        kl_val_S=("kl_val_S", "mean"),
        kl_S_val=("kl_S_val", "mean"),
    ).reset_index()
    H1_merged = H1_setting.merge(H1_setting_kl, on="setting_id", how="inner")
    # restrict to trainbal settings
    trainbal_ids = set(table[table["regime"] == "trainbal"]["setting_id"])
    H1_merged = H1_merged[H1_merged["setting_id"].isin(trainbal_ids)]
    H1_merged = H1_merged.merge(table[["setting_id", "imbalance_level"]],
                                  on="setting_id", how="left")
    print(f"[H1] n_settings (trainbal A1 r10 sel∈{{1,2,3,5}}) = {len(H1_merged)}")

    # mixed model (statsmodels MixedLM)
    try:
        import statsmodels.formula.api as smf
        # cell-level: merge cell_kl with grand_df
        cell_long = H1_grand.merge(
            cell_table[(cell_table["method"] == "A1") & (cell_table["rank_pct"] == 10.0)],
            on=["setting_id", "method", "rank_pct", "sel"], how="inner"
        )
        cell_long = cell_long.merge(
            table[["setting_id", "max_P_val", "dataset"]], on="setting_id", how="left",
            suffixes=("", "_meta")
        )
        if len(cell_long) >= 10 and cell_long["dataset"].nunique() >= 2:
            md = smf.mixedlm("gap_top_random_balanced ~ kl_val_S + max_P_val",
                              cell_long, groups=cell_long["dataset"])
            mf = md.fit(reml=False, disp=False)
            beta = float(mf.params.get("kl_val_S", np.nan))
            ci = mf.conf_int().loc["kl_val_S"] if "kl_val_S" in mf.params.index else None
            mci = (float(ci[0]), float(ci[1])) if ci is not None else (np.nan, np.nan)
        else:
            beta, mci = np.nan, (np.nan, np.nan)
    except Exception as e:
        print(f"  [warn] H1 mixed model: {e}")
        beta, mci = np.nan, (np.nan, np.nan)

    if len(H1_merged) >= 3:
        _add_corr("H1", "kl_val_S", "gap_top_random_balanced",
                  len(H1_merged), H1_merged["kl_val_S"], H1_merged["outcome"],
                  "trainbal", "all", beta, mci,
                  decision="see thresholds")
        # stratified
        for imb in ["mild", "extreme"]:
            sub = H1_merged[H1_merged["imbalance_level"] == imb]
            if len(sub) >= 3:
                _add_corr("H1_strat", "kl_val_S", "gap_top_random_balanced",
                          len(sub), sub["kl_val_S"], sub["outcome"],
                          "trainbal", imb)

    # ----- H2: valbal regime, predictor = max_P_train,
    #              outcome = acc_top_balanced[A1, r=10, sel=5] − acc_top_balanced[LR, r=10, sel=5] -----
    H2_A1 = grand[(grand["regime"] == "valbal") & (grand["method"] == "A1") &
                  (grand["rank_pct"] == 10.0) & (grand["sel"] == 5)] \
                  [["setting_id", "acc_top_balanced"]].rename(
                      columns={"acc_top_balanced": "acc_A1"})
    H2_LR = grand[(grand["regime"] == "valbal") & (grand["method"] == "LR") &
                  (grand["rank_pct"] == 10.0) & (grand["sel"] == 5)] \
                  [["setting_id", "acc_top_balanced"]].rename(
                      columns={"acc_top_balanced": "acc_LR"})
    H2 = H2_A1.merge(H2_LR, on="setting_id", how="inner")
    H2 = H2.merge(table[["setting_id", "max_P_train", "imbalance_level", "dataset"]],
                  on="setting_id", how="left")
    H2["recovery"] = H2["acc_A1"] - H2["acc_LR"]
    print(f"[H2] n_settings (valbal sel=5) = {len(H2)}")

    # mixed model H2: collapse needs more cells. Use sel ∈ {1,3,5,10}, r=10.
    try:
        import statsmodels.formula.api as smf
        cells_A1 = grand[(grand["regime"] == "valbal") & (grand["method"] == "A1") &
                         (grand["rank_pct"] == 10.0) & (grand["sel"].isin([1,3,5,10]))]
        cells_LR = grand[(grand["regime"] == "valbal") & (grand["method"] == "LR") &
                         (grand["rank_pct"] == 10.0) & (grand["sel"].isin([1,3,5,10]))]
        cells = cells_A1.merge(cells_LR[["setting_id", "sel", "acc_top_balanced"]],
                                on=["setting_id", "sel"], suffixes=("_A1", "_LR"))
        cells["recovery"] = cells["acc_top_balanced_A1"] - cells["acc_top_balanced_LR"]
        cells = cells.merge(table[["setting_id", "max_P_train", "dataset"]],
                              on="setting_id", how="left", suffixes=("", "_m"))
        if len(cells) >= 10 and cells["dataset"].nunique() >= 2:
            md2 = smf.mixedlm("recovery ~ max_P_train", cells, groups=cells["dataset"])
            mf2 = md2.fit(reml=False, disp=False)
            beta2 = float(mf2.params.get("max_P_train", np.nan))
            ci2 = mf2.conf_int().loc["max_P_train"]
            mci2 = (float(ci2[0]), float(ci2[1]))
        else:
            beta2, mci2 = np.nan, (np.nan, np.nan)
    except Exception as e:
        print(f"  [warn] H2 mixed model: {e}")
        beta2, mci2 = np.nan, (np.nan, np.nan)

    if len(H2) >= 3:
        _add_corr("H2", "max_P_train", "A1-LR balanced acc gap",
                  len(H2), H2["max_P_train"], H2["recovery"],
                  "valbal", "all", beta2, mci2)
        for imb in ["mild", "extreme"]:
            sub = H2[H2["imbalance_level"] == imb]
            if len(sub) >= 3:
                _add_corr("H2_strat", "max_P_train", "A1-LR balanced acc gap",
                          len(sub), sub["max_P_train"], sub["recovery"],
                          "valbal", imb)

    # ----- Decision rule (Bonferroni α/4 = 0.0125) -----
    def decide(row):
        r = row["pearson_r"]; p = row["pearson_p"]
        if not np.isfinite(r) or not np.isfinite(p):
            return "n/a"
        if abs(r) > 0.3 and p < 0.0125:
            return "confirm"
        if abs(r) < 0.2 and p > 0.1:
            return "reject"
        return "partial"

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["decision"] = corr_df.apply(decide, axis=1)
    corr_path = OUT_DIR / "lens1_corr.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"[step 2] wrote {corr_path}: {len(corr_df)} rows")
    if not corr_df.empty:
        print(corr_df.to_string())

    # ===== Step 3: figures =====
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # H1 scatter
    if len(H1_merged) >= 3:
        fig, ax = plt.subplots(figsize=(6, 5))
        for ds, sub in H1_merged.merge(table[["setting_id", "dataset"]],
                                         on="setting_id").groupby("dataset"):
            ax.scatter(sub["kl_val_S"], sub["outcome"], label=ds, s=60)
        ax.set_xlabel("KL(P_val ‖ P_S)  averaged over sel ∈ {1,2,3,5} (A1, r=10%)")
        ax.set_ylabel("gap_top_random_balanced (A1 − random)")
        ax.axhline(-0.02, ls="--", color="k", alpha=0.4, label="−0.02 threshold")
        ax.set_title(f"H1 — trainbal random-loss vs label shift  (n={len(H1_merged)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "H1_scatter.png", dpi=130)
        plt.close(fig)
        print(f"[fig] H1_scatter.png")

    if len(H2) >= 3:
        fig, ax = plt.subplots(figsize=(6, 5))
        for ds, sub in H2.groupby("dataset"):
            ax.scatter(sub["max_P_train"], sub["recovery"], label=ds, s=60)
        ax.set_xlabel("max_c P_train(c)  (train majority probability)")
        ax.set_ylabel("acc_top_balanced[A1] − acc_top_balanced[LR]  at r=10%, sel=5")
        ax.axhline(0, ls="--", color="k", alpha=0.4)
        ax.set_title(f"H2 — valbal A1 recovery vs train imbalance  (n={len(H2)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "H2_scatter.png", dpi=130)
        plt.close(fig)
        print(f"[fig] H2_scatter.png")

    print()
    print("[head] lens1_table.csv (first 5 rows):")
    print(table.head().to_string())
    print()
    print("[head] lens1_corr.csv:")
    print(corr_df.to_string() if not corr_df.empty else "(empty)")
    print()
    print(f"[done] lens1_table rows: {len(table)}")
    print(f"[done] lens1_cell_kl rows: {len(cell_table)}")
    print(f"[done] lens1_corr rows: {len(corr_df)}")


if __name__ == "__main__":
    main()
