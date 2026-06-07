"""Compute rigorous quantitative measurements for the 5 conditions at n=5000.

For each (dataset, seed) in {bert} × {2024, 2025, 2026}:
  - Load K_tt (train-train kernel) from NTK pkl.
  - Eigendecompose K_tt = U Λ U^T.
  - Load labels Y, build centered one-hot Y_c.
  - Compute c_i^2 = ||u_i^T Y_c||^2 (Frobenius across class dim).
  - For each rank r in {1%, 5%, 10%, 15%, 20%, 25%, 30%} of n:
      - I_LR = top-r by λ_i        (LRFShap selection)
      - I_A1 = top-r by c_i^2      (A1 selection in our λ>>ρ regime)
      - LC(I_LR), FC(I_LR), LC(I_A1), FC(I_A1)
      - miss_LC_by_LR = 1 - LC(I_LR)
      - miss_FC_by_A1 = 1 - FC(I_A1)
      - overlap |I_LR ∩ I_A1| / r
  - Dataset-level summaries:
      - Spearman(λ_i, c_i^2)
      - Pearson(log λ_i, log c_i^2)
      - PR(λ)  = (Σλ)^2 / Σλ^2     (participation ratio of kernel spectrum)
      - PR(c^2) = (Σc^2)^2 / Σc^4    (participation ratio of label energy)
      - effective_rank_lambda = exp(H(λ_normalized))
      - effective_rank_c2 = exp(H(c^2_normalized))

Save to: state/iteration_03/precomputed_n5000.json
"""
import os, sys, pickle, json, time
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr

RHO = 1e-2
RANKS_PCT = [1, 5, 10, 15, 20, 25, 30]
SEEDS = [2024, 2025, 2026]
OUT = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/state/iteration_03/precomputed_n5000.json"

# (ds_name, n_train, val_n, C, val_split, hf_args)
specs = [
    ("rte",     2490,  277, 2, "validation",         {"path": "glue", "name": "rte"}),
    ("sst2",    5000,  872, 2, "validation",         {"path": "sst2"}),
    ("ag_news", 5000, 1000, 4, "test",               {"path": "ag_news"}),
    ("mnli",    5000, 1000, 3, "validation_matched", {"path": "glue", "name": "mnli"}),
    ("mr",      5000, 1000, 2, "validation",         {"path": "rotten_tomatoes"}),
    ("mrpc",    3668,  408, 2, "validation",         {"path": "glue", "name": "mrpc"}),
    ("qqp",     5000, 1000, 2, "validation",         {"path": "glue", "name": "qqp"}),
]

os.makedirs(os.path.dirname(OUT), exist_ok=True)


def participation_ratio(x):
    """PR(x) = (Σx)^2 / Σx^2 — 'effective number' of non-zero entries."""
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    s2 = (x ** 2).sum()
    return float(s * s / max(s2, 1e-300))


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), 1e-300)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def majority_fraction(y):
    """Fraction of majority class — what a degenerate predictor gets."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def run_one(ds_name, n_train, val_n, C, val_split, hf_args, seed):
    ntk_path = f"./freeshap_res/ntk/{ds_name}/bert_seed{seed}_num{n_train}_val{val_n}_signFalse.pkl"
    bundle = pickle.load(open(ntk_path, "rb"))
    ntk = bundle["ntk"]
    sampled_idx = np.asarray(bundle["sampled_idx"])
    if isinstance(ntk, torch.Tensor):
        ntk = ntk.detach().cpu().numpy()
    ntk = np.asarray(ntk)
    if ntk.ndim == 3:
        ntk = ntk[0] if ntk.shape[0] == 1 else ntk.mean(0)
    K = 0.5 * (ntk[:n_train, :n_train] + ntk[:n_train, :n_train].T).astype(np.float64)

    # labels
    ds_tr = load_dataset(**hf_args, split="train")
    y = np.asarray([ds_tr[int(i)]["label"] for i in sampled_idx], dtype=np.int64)
    Y = np.zeros((n_train, C), dtype=np.float64)
    Y[np.arange(n_train), y] = 1.0
    Y_c = Y - Y.mean(0, keepdims=True)

    # eigendecomp
    eigs_asc, U_asc = np.linalg.eigh(K)
    eigs = eigs_asc[::-1]
    U = U_asc[:, ::-1]
    eigs_pos = np.maximum(eigs, 0.0)

    # mode-wise label projection
    coeffs = U.T @ Y_c                  # (n, C)
    c2 = (coeffs ** 2).sum(1)            # (n,)

    total_lam = float(eigs_pos.sum())
    total_c2 = float(c2.sum())
    filt = (eigs_pos / (eigs_pos + RHO)) ** 2
    s_A1 = filt * c2

    # spectrum-label correlations
    sp_rank = float(spearmanr(eigs_pos, c2)[0])
    # Pearson on log-log (only on positive c2 and eigs)
    mask = (eigs_pos > 0) & (c2 > 0)
    pe_loglog = float(pearsonr(np.log(eigs_pos[mask]), np.log(c2[mask]))[0]) if mask.sum() > 2 else float("nan")

    # full-ridge accuracy and val majority class — proxies for Condition B
    K_vt = ntk[n_train:, :n_train].astype(np.float64)
    ds_val = load_dataset(**hf_args, split=val_split)
    sampled_val_idx = np.asarray(bundle["sampled_val_idx"])
    y_val = np.asarray([ds_val[int(i)]["label"] for i in sampled_val_idx], dtype=np.int64)
    Y_val = np.zeros((len(sampled_val_idx), C))
    Y_val[np.arange(len(sampled_val_idx)), y_val] = 1.0
    Y_val_c = Y_val - Y.mean(0, keepdims=True)

    # full-ridge prediction
    alpha_full = U @ (coeffs / (eigs_pos[:, None] + RHO))   # (n_train, C)
    pred_full = K_vt @ alpha_full
    recon = Y.mean(0, keepdims=True) + pred_full
    full_acc = float((recon.argmax(1) == y_val).mean())
    full_mse = float(((Y_val_c - pred_full) ** 2).mean())

    # majority class accuracy (degenerate predictor)
    train_majority = majority_fraction(y)
    val_majority_class = int(Y.mean(0).argmax())
    val_majority_acc = float((y_val == val_majority_class).mean())

    # PR / effective rank
    PR_lam = participation_ratio(eigs_pos)
    PR_c2 = participation_ratio(c2)
    eff_rank_lam = float(np.exp(entropy(eigs_pos)))
    eff_rank_c2 = float(np.exp(entropy(c2)))

    # spectrum stats
    eigvals_stats = {
        "lambda_min": float(eigs_pos.min()),
        "lambda_max": float(eigs_pos.max()),
        "lambda_median": float(np.median(eigs_pos)),
        "lambda_total": total_lam,
        "frac_lambda_gt_10rho": float((eigs_pos > 10*RHO).mean()),
        "frac_lambda_lt_rho": float((eigs_pos < RHO).mean()),
        "filter_top1pct_min": float(filt[:max(1, n_train//100)].min()),
    }

    # per-rank quantities
    per_rank = {}
    for pct in RANKS_PCT:
        r = max(1, int(round(n_train * pct / 100)))
        I_LR = np.argsort(-eigs_pos)[:r]
        # in our regime filter ≈ 1, so I_A1 ≈ argsort -c2
        # use full A1 score (= filt * c2) for correctness, but it's nearly same as c2 ordering
        I_A1 = np.argsort(-s_A1)[:r]
        I_pureC = np.argsort(-c2)[:r]

        FC_LR = float(eigs_pos[I_LR].sum() / total_lam)
        LC_LR = float(c2[I_LR].sum() / total_c2)
        FC_A1 = float(eigs_pos[I_A1].sum() / total_lam)
        LC_A1 = float(c2[I_A1].sum() / total_c2)

        # Per-rank predictions: top-r-by-λ ridge predictor evaluation
        # f_LR(x) = K_vt @ U[:, I_LR] @ diag(1/(lam + rho)) @ U[:, I_LR]^T @ Y
        alpha_LR = U[:, I_LR] @ (coeffs[I_LR] / (eigs_pos[I_LR, None] + RHO))
        pred_LR = K_vt @ alpha_LR
        LR_acc = float(((Y.mean(0, keepdims=True) + pred_LR).argmax(1) == y_val).mean())
        # fraction of LR predictions equal to majority class
        LR_pred_class = (Y.mean(0, keepdims=True) + pred_LR).argmax(1)
        LR_majority_collapse = float((LR_pred_class == val_majority_class).mean())

        # A1's predictor (analogous)
        alpha_A1 = U[:, I_A1] @ (coeffs[I_A1] / (eigs_pos[I_A1, None] + RHO))
        pred_A1 = K_vt @ alpha_A1
        A1_acc = float(((Y.mean(0, keepdims=True) + pred_A1).argmax(1) == y_val).mean())
        A1_pred_class = (Y.mean(0, keepdims=True) + pred_A1).argmax(1)
        A1_majority_collapse = float((A1_pred_class == val_majority_class).mean())

        per_rank[str(pct)] = {
            "r_abs": r,
            "FC_LR": FC_LR, "LC_LR": LC_LR,
            "FC_A1": FC_A1, "LC_A1": LC_A1,
            "miss_LC_by_LR": 1.0 - LC_LR,
            "miss_FC_by_A1": 1.0 - FC_A1,
            "overlap_LR_A1": float(len(set(I_LR.tolist()) & set(I_A1.tolist())) / r),
            "overlap_LR_pureC": float(len(set(I_LR.tolist()) & set(I_pureC.tolist())) / r),
            "LR_full_train_predictor_val_acc": LR_acc,
            "A1_full_train_predictor_val_acc": A1_acc,
            "LR_predict_majority_frac": LR_majority_collapse,
            "A1_predict_majority_frac": A1_majority_collapse,
        }
    return {
        "n_train": n_train, "n_val": len(sampled_val_idx), "C": C,
        "train_majority_fraction": train_majority,
        "val_majority_class": val_majority_class,
        "val_majority_acc": val_majority_acc,
        "full_ridge_val_acc": full_acc,
        "full_ridge_val_mse": full_mse,
        "spectrum_label_correlation": {
            "spearman_lambda_c2": sp_rank,
            "pearson_loglog_lambda_c2": pe_loglog,
        },
        "PR_lambda": PR_lam,
        "PR_c2": PR_c2,
        "effective_rank_lambda": eff_rank_lam,
        "effective_rank_c2": eff_rank_c2,
        "eigvals_summary": eigvals_stats,
        "per_rank": per_rank,
    }


def main():
    os.chdir("/extdata1/donghwan/freeshap/vinfo")
    all_results = {}
    for ds_args in specs:
        ds_name = ds_args[0]
        n_train = ds_args[1]
        print(f"\n===== {ds_name} (n={n_train}) =====")
        per_seed = {}
        for seed in SEEDS:
            print(f"  seed={seed} ...", end=" ", flush=True)
            t0 = time.time()
            res = run_one(*ds_args, seed=seed)
            per_seed[seed] = res
            print(f"done in {time.time()-t0:.1f}s, "
                  f"PR(λ)={res['PR_lambda']:.1f}  PR(c²)={res['PR_c2']:.1f}  "
                  f"Spearman={res['spectrum_label_correlation']['spearman_lambda_c2']:+.3f}")
        # average per-seed scalars and per-rank values
        avg = {}
        scalar_keys = [
            ("train_majority_fraction", None),
            ("val_majority_acc", None),
            ("full_ridge_val_acc", None),
            ("full_ridge_val_mse", None),
            ("PR_lambda", None),
            ("PR_c2", None),
            ("effective_rank_lambda", None),
            ("effective_rank_c2", None),
        ]
        for k, _ in scalar_keys:
            vals = [per_seed[s][k] for s in SEEDS]
            avg[k] = float(np.mean(vals))
            avg[k + "_std"] = float(np.std(vals))
        # spectrum correlations
        for k in ("spearman_lambda_c2", "pearson_loglog_lambda_c2"):
            vals = [per_seed[s]["spectrum_label_correlation"][k] for s in SEEDS]
            avg["spec_corr_" + k] = float(np.mean(vals))
            avg["spec_corr_" + k + "_std"] = float(np.std(vals))
        # per-rank
        avg["per_rank"] = {}
        for pct in [str(p) for p in RANKS_PCT]:
            ravg = {}
            for k in per_seed[SEEDS[0]]["per_rank"][pct]:
                vals = [per_seed[s]["per_rank"][pct][k] for s in SEEDS]
                ravg[k] = float(np.mean(vals))
                ravg[k + "_std"] = float(np.std(vals))
            avg["per_rank"][pct] = ravg
        avg["n_train"] = per_seed[SEEDS[0]]["n_train"]
        avg["C"] = per_seed[SEEDS[0]]["C"]
        avg["seeds"] = SEEDS
        all_results[ds_name] = avg
    json.dump(all_results, open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}  ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
