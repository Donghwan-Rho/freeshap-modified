#!/usr/bin/env python
"""
lens2_kta_spectral.py
=====================
Iteration 04, plan §3.4 — Lens 2: spectral target alignment + Nyström val
projection + lens 3 (d_eff) appended.

Inputs:
  state/iteration_04/grand_df.csv
  state/iteration_04/grand_meta.csv
  state/iteration_04/eig_cache/<setting_id>.npz   (eigvals, eigvecs, Y_tilde,
                                                    Y_val_tilde, K_train_val)

Outputs:
  state/iteration_04/lens2_table.csv  — per-setting kta/lc/pr/d_eff
  state/iteration_04/lens2_corr.csv   — H3, H4 hypothesis decisions
  state/iteration_04/lens2_figs/*.png — H3, H4 scatter
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORK_ROOT = Path("/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research")
OUT_DIR = WORK_ROOT / "state/iteration_04"
EIG_DIR = OUT_DIR / "eig_cache"
FIG_DIR = OUT_DIR / "lens2_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANK_PCT_LIST = [1, 5, 10, 15, 20, 25, 30]
RHO = 1e-2

# Nyström small-λ clip
LAMBDA_CLIP = 1e-6


def fisher_ci(r, n, alpha=0.05):
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def kta_top_r(eigvals, c2, r_idx):
    """KTA(K_r, Ỹ) = (Σ_{i<=r} λ_i c²_i) / (sqrt(Σ_{i<=r} λ_i²) · ‖ỸỸ⊤‖_F).

    Note: ‖ỸỸ⊤‖_F is constant across r (it depends only on Ỹ), so we omit it
    when comparing across r — we return numerator / sqrt(Σ λ²).  For *absolute*
    KTA the caller can divide by ‖ỸỸ⊤‖_F.
    """
    e = eigvals[:r_idx]
    c = c2[:r_idx]
    num = float(np.sum(e * c))
    den = float(np.sqrt(np.sum(e * e)))
    return num / max(den, 1e-30)


def kta_full(eigvals, c2, frob_yyt):
    num = float(np.sum(eigvals * c2))
    den = float(np.sqrt(np.sum(eigvals * eigvals)) * frob_yyt)
    return num / max(den, 1e-30)


def main():
    print("=" * 78)
    print("lens2_kta_spectral.py — iteration 04 §3 KTA spectral + Nyström val")
    print("=" * 78)

    grand = pd.read_csv(OUT_DIR / "grand_df.csv")
    meta = pd.read_csv(OUT_DIR / "grand_meta.csv")
    for c in ["P_train_y", "P_val_y", "train_class_counts", "val_class_counts"]:
        meta[c] = meta[c].apply(lambda v: json.loads(v) if isinstance(v, str) else v)

    rows = []
    n_clip_total = 0
    print(f"[step 1] computing KTA + d_eff for {len(meta)} settings ...")
    for _, m in meta.iterrows():
        setting_id = m["setting_id"]
        cache_path = EIG_DIR / f"{setting_id}.npz"
        if not cache_path.exists():
            print(f"  [skip] no eig cache: {setting_id}")
            continue
        t0 = time.time()
        d = np.load(cache_path)
        eigvals = d["eigvals"].astype(np.float64)
        eigvecs = d["eigvecs"].astype(np.float64)  # (n, k)
        Y_tilde = d["Y_tilde"].astype(np.float64)
        Y_val_tilde = d["Y_val_tilde"].astype(np.float64)
        K_train_val = d["K_train_val"].astype(np.float64)
        n = int(d["n_train"]); n_val = int(d["n_val"]); C = int(d["C"])
        k = eigvecs.shape[1]

        # ----- train-side c²_i -----
        # uᵢ⊤ Ỹ : (k, C) = eigvecs.T @ Y_tilde
        UY = eigvecs.T @ Y_tilde  # (k, C)
        c2_train = np.sum(UY * UY, axis=1)  # (k,)

        # ----- val-side c²_i via Nyström extension (plan §3.2(2)) -----
        # ψ̂_i(x_val,j) = (sqrt(n) / λ_i) · uᵢ⊤ K_{train,val}[:, j]
        # c_i_val,c = (1/n_val) Σ_j ψ̂_i(x_val,j) · ỹ_val,c(j)
        #           = (sqrt(n) / (λ_i · n_val)) · uᵢ⊤ K_{train,val} · ỹ_val,c
        # cᵢ²_val = Σ_c c_i_val,c²
        UKv = eigvecs.T @ K_train_val  # (k, n_val)
        UKvYv = UKv @ Y_val_tilde      # (k, C)
        # safe scale factor — clip small λ
        scale = np.where(eigvals[:k] > LAMBDA_CLIP,
                         np.sqrt(n) / (eigvals[:k] * n_val), 0.0)
        n_clip = int((eigvals[:k] <= LAMBDA_CLIP).sum())
        n_clip_total += n_clip
        c_val_mat = scale[:, None] * UKvYv  # (k, C)
        c2_val = np.sum(c_val_mat * c_val_mat, axis=1)

        # ----- train label using val marginal — "Ỹ_train_via_val_y" -----
        # Y_tilde_via_val = Y - P_val[None, :]
        P_val_y = np.asarray(m["P_val_y"], dtype=np.float64)
        P_train_y = np.asarray(m["P_train_y"], dtype=np.float64)
        # actual Y (one-hot, before centering) = Y_tilde + P_train_y[None,:]
        Y_oh = Y_tilde + P_train_y[None, :]
        Y_via_val = Y_oh - P_val_y[None, :]
        UYv = eigvecs.T @ Y_via_val  # (k, C)
        c2_train_via_val = np.sum(UYv * UYv, axis=1)

        # ----- learnability score for A1 vs LR top-r overlap -----
        s_score = (eigvals[:k] / (eigvals[:k] + RHO)) ** 2 * c2_train

        # ----- KTA quantities -----
        # ‖Ỹ Ỹ⊤‖_F^2 = ‖Ỹ⊤ Ỹ‖_F^2 = Σ_{c,c'} (Σ_i Ỹ_ic Ỹ_ic')² = ‖Y_tilde.T @ Y_tilde‖_F
        YTY = Y_tilde.T @ Y_tilde  # (C, C)
        frob_yyt = float(np.linalg.norm(YTY, "fro"))
        YTY_val = Y_via_val.T @ Y_via_val
        frob_yyt_via_val = float(np.linalg.norm(YTY_val, "fro"))

        kta_train_full_v = kta_full(eigvals[:k], c2_train, frob_yyt)
        kta_train_via_val_v = kta_full(eigvals[:k], c2_train_via_val, frob_yyt_via_val)

        # top-r KTA (use bare ratio, normalised by full to make cross-dataset comparable)
        # Per plan §3.2(5): KTA(K_r, Ỹ) = (Σ_{i≤r} λ_i c²_i) / (sqrt(Σ_{i≤r} λ²) · ‖ỸỸ⊤‖_F)
        kta_train_r = {}
        kta_train_via_val_r = {}
        for r_pct in RANK_PCT_LIST:
            r_idx = max(1, int(round(r_pct / 100.0 * n)))
            r_idx = min(r_idx, k)
            kta_train_r[r_pct] = kta_top_r(eigvals[:k], c2_train, r_idx) / max(frob_yyt, 1e-30)
            kta_train_via_val_r[r_pct] = kta_top_r(eigvals[:k], c2_train_via_val, r_idx) / max(frob_yyt_via_val, 1e-30)

        kta_gap_r = {r: kta_train_r[r] - kta_train_via_val_r[r] for r in RANK_PCT_LIST}

        # ----- LC_train(r), LC_val(r), LC_gap(r) -----
        total_c2_train = float(c2_train.sum())
        total_c2_val = float(c2_val.sum())
        lc_train_r = {}
        lc_val_r = {}
        lc_gap_r = {}
        for r_pct in RANK_PCT_LIST:
            r_idx = max(1, int(round(r_pct / 100.0 * n)))
            r_idx = min(r_idx, k)
            lct = float(c2_train[:r_idx].sum()) / max(total_c2_train, 1e-30)
            lcv = float(c2_val[:r_idx].sum()) / max(total_c2_val, 1e-30)
            lc_train_r[r_pct] = lct
            lc_val_r[r_pct] = lcv
            lc_gap_r[r_pct] = lct - lcv

        # ----- PR(c²) -----
        def PR(x):
            x = np.asarray(x, dtype=np.float64)
            num = float(x.sum()) ** 2
            den = float((x * x).sum())
            return num / max(den, 1e-30)
        pr_c2_train = PR(c2_train)
        pr_c2_val = PR(c2_val)

        # ----- d_eff(ρ) = Σ_i λ_i / (λ_i + ρ) -----
        d_eff_rho = float(np.sum(eigvals / (eigvals + RHO)))
        d_eff_r = {}
        d_eff_ratio = {}
        for r_pct in RANK_PCT_LIST:
            r_idx = max(1, int(round(r_pct / 100.0 * n)))
            r_idx = min(r_idx, len(eigvals))
            v = float(np.sum(eigvals[:r_idx] / (eigvals[:r_idx] + RHO)))
            d_eff_r[r_pct] = v
            d_eff_ratio[r_pct] = v / max(d_eff_rho, 1e-30)

        # ----- A1-LR overlap: top-r by λ_i vs top-r by s_score -----
        # NB: indices into eigvecs (already sorted descending by λ)
        overlap_r = {}
        order_LR = np.argsort(-eigvals[:k])  # already descending
        order_A1 = np.argsort(-s_score)
        for r_pct in RANK_PCT_LIST:
            r_idx = max(1, int(round(r_pct / 100.0 * n)))
            r_idx = min(r_idx, k)
            I_LR = set(order_LR[:r_idx].tolist())
            I_A1 = set(order_A1[:r_idx].tolist())
            overlap_r[r_pct] = len(I_LR & I_A1) / r_idx

        # ----- outcome metrics from grand_df: A1 recovery + LR loss vs random -----
        sub_A1 = grand[(grand["setting_id"] == setting_id) & (grand["method"] == "A1")]
        sub_LR = grand[(grand["setting_id"] == setting_id) & (grand["method"] == "LR")]
        recovery_scores = {}
        lr_loss = {}
        for sel in [1, 3, 5, 10]:
            a1 = sub_A1[(sub_A1["rank_pct"] == 10.0) & (sub_A1["sel"] == sel)]
            lr = sub_LR[(sub_LR["rank_pct"] == 10.0) & (sub_LR["sel"] == sel)]
            if len(a1) and len(lr):
                recovery_scores[f"recovery_sel{sel}"] = float(
                    a1["acc_top_balanced"].mean() - lr["acc_top_balanced"].mean()
                )
            else:
                recovery_scores[f"recovery_sel{sel}"] = np.nan
            if len(lr):
                lr_loss[f"LR_loss_vs_random_balanced_sel{sel}"] = -float(
                    lr["gap_top_random_balanced"].mean()
                )
            else:
                lr_loss[f"LR_loss_vs_random_balanced_sel{sel}"] = np.nan

        row = {
            "setting_id": setting_id,
            "dataset": m["dataset"],
            "regime": m["regime"],
            "train_ratio_tag": m["train_ratio_tag"],
            "val_ratio_tag": m["val_ratio_tag"],
            "imbalance_level": m["imbalance_level"],
            "C": int(C),
            "n": int(n),
            "n_val": int(n_val),
            "kta_train_full": kta_train_full_v,
            "kta_train_via_val_full": kta_train_via_val_v,
            **{f"kta_train_r{r}": v for r, v in kta_train_r.items()},
            **{f"kta_train_via_val_r{r}": v for r, v in kta_train_via_val_r.items()},
            **{f"kta_gap_r{r}": v for r, v in kta_gap_r.items()},
            **{f"lc_train_r{r}": v for r, v in lc_train_r.items()},
            **{f"lc_val_r{r}": v for r, v in lc_val_r.items()},
            **{f"lc_gap_r{r}": v for r, v in lc_gap_r.items()},
            "pr_c2_train": pr_c2_train,
            "pr_c2_val": pr_c2_val,
            "d_eff_rho": d_eff_rho,
            **{f"d_eff_r{r}": v for r, v in d_eff_r.items()},
            **{f"d_eff_ratio_r{r}": v for r, v in d_eff_ratio.items()},
            **{f"overlap_LR_A1_r{r}": v for r, v in overlap_r.items()},
            **recovery_scores,
            **lr_loss,
            "nystrom_clip_count": int(n_clip),
        }
        rows.append(row)
        print(f"  [{len(rows)}/{len(meta)}] {setting_id}  ({time.time()-t0:.2f}s)  "
              f"KTA_train_full={kta_train_full_v:.4f}  d_eff(ρ)={d_eff_rho:.1f}  clip={n_clip}")

    table = pd.DataFrame(rows)
    table_path = OUT_DIR / "lens2_table.csv"
    table.to_csv(table_path, index=False)
    print(f"[step 1] wrote {table_path}: {len(table)} settings  (total Nyström clip = {n_clip_total})")

    # ===== Step 2: H3, H4 correlations =====
    from scipy.stats import pearsonr, spearmanr
    corr_rows = []

    def _add(hyp, predictor, outcome, n, x, y, regime, imb,
             mb=np.nan, mci=(np.nan, np.nan)):
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
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
            "hypothesis_id": hyp, "predictor": predictor, "outcome": outcome,
            "regime": regime, "imbalance_level": imb, "n_settings": len(x),
            "pearson_r": pear_r, "pearson_p": pear_p,
            "pearson_ci_lo": lo, "pearson_ci_hi": hi,
            "spearman_r": sp_r, "spearman_p": sp_p,
            "mixed_beta": mb, "mixed_ci_lo": mci[0], "mixed_ci_hi": mci[1],
        })

    # H3: valbal, predictor = kta_gap_r10, outcome = recovery_sel5
    H3 = table[table["regime"] == "valbal"].copy()
    if len(H3) >= 3:
        _add("H3", "kta_gap_r10", "recovery_sel5",
             len(H3), H3["kta_gap_r10"], H3["recovery_sel5"], "valbal", "all")
        for imb in ["mild", "extreme"]:
            sub = H3[H3["imbalance_level"] == imb]
            if len(sub) >= 3:
                _add("H3_strat", "kta_gap_r10", "recovery_sel5",
                     len(sub), sub["kta_gap_r10"], sub["recovery_sel5"],
                     "valbal", imb)

    # H4: trainbal, predictor = kta_train_r10, outcome = LR_loss_vs_random_balanced_sel5
    H4 = table[table["regime"] == "trainbal"].copy()
    if len(H4) >= 3:
        _add("H4", "kta_train_r10", "LR_loss_vs_random_balanced_sel5",
             len(H4), H4["kta_train_r10"], H4["LR_loss_vs_random_balanced_sel5"],
             "trainbal", "all")
        _add("H4b", "kta_gap_r10", "LR_loss_vs_random_balanced_sel5",
             len(H4), H4["kta_gap_r10"], H4["LR_loss_vs_random_balanced_sel5"],
             "trainbal", "all")
        for imb in ["mild", "extreme"]:
            sub = H4[H4["imbalance_level"] == imb]
            if len(sub) >= 3:
                _add("H4_strat", "kta_train_r10", "LR_loss_vs_random_balanced_sel5",
                     len(sub), sub["kta_train_r10"], sub["LR_loss_vs_random_balanced_sel5"],
                     "trainbal", imb)

    # H5 exploratory: d_eff_ratio_r10 vs recovery
    if len(H3) >= 3:
        _add("H5", "d_eff_ratio_r10", "recovery_sel5",
             len(H3), H3["d_eff_ratio_r10"], H3["recovery_sel5"], "valbal", "all")

    # decision
    def decide(row):
        r, p = row["pearson_r"], row["pearson_p"]
        if not np.isfinite(r) or not np.isfinite(p): return "n/a"
        if abs(r) > 0.3 and p < 0.0125: return "confirm"
        if abs(r) < 0.2 and p > 0.1: return "reject"
        return "partial"

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["decision"] = corr_df.apply(decide, axis=1)
    corr_path = OUT_DIR / "lens2_corr.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"[step 2] wrote {corr_path}: {len(corr_df)} rows")
    if not corr_df.empty:
        print(corr_df.to_string())

    # ===== Step 3: figures =====
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(H3) >= 3:
        fig, ax = plt.subplots(figsize=(6, 5))
        for ds, sub in H3.groupby("dataset"):
            ax.scatter(sub["kta_gap_r10"], sub["recovery_sel5"], label=ds, s=60)
        ax.set_xlabel("kta_gap_r10  (train-Ỹ KTA − train-Ỹ_via_val KTA, r=10%)")
        ax.set_ylabel("recovery_sel5  (A1 − LR balanced acc at r=10%, sel=5)")
        ax.axhline(0, ls="--", color="k", alpha=0.4)
        ax.set_title(f"H3 — valbal A1 recovery vs KTA gap  (n={len(H3)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "H3_scatter.png", dpi=130)
        plt.close(fig)
        print("[fig] H3_scatter.png")

    if len(H4) >= 3:
        fig, ax = plt.subplots(figsize=(6, 5))
        for ds, sub in H4.groupby("dataset"):
            ax.scatter(sub["kta_train_r10"], sub["LR_loss_vs_random_balanced_sel5"],
                       label=ds, s=60)
        ax.set_xlabel("kta_train_r10  (top-r KTA on train side, r=10%)")
        ax.set_ylabel("LR_loss_vs_random_balanced_sel5  (random − LR, larger = LR loses more)")
        ax.axhline(0, ls="--", color="k", alpha=0.4)
        ax.set_title(f"H4 — trainbal LR loss vs KTA top-r  (n={len(H4)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "H4_scatter.png", dpi=130)
        plt.close(fig)
        print("[fig] H4_scatter.png")

    print()
    print("[head] lens2_table.csv (first 5 rows, selected columns):")
    cols = ["setting_id", "regime", "kta_train_full", "kta_gap_r10",
            "lc_train_r10", "lc_val_r10", "pr_c2_train", "pr_c2_val",
            "d_eff_rho", "d_eff_ratio_r10", "overlap_LR_A1_r10",
            "recovery_sel5", "LR_loss_vs_random_balanced_sel5"]
    cols = [c for c in cols if c in table.columns]
    print(table[cols].head().to_string())
    print()
    print(f"[done] lens2_table rows: {len(table)}")
    print(f"[done] lens2_corr rows: {len(corr_df)}")


if __name__ == "__main__":
    main()
