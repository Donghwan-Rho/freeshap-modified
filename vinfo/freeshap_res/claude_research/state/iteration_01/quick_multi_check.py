"""Quick multi-dataset check — A1 (top-r by s_i) vs LRFShap baseline (top-r by lambda_i).

Same analysis as quick_rte_check.py, but extended to 6 datasets to test whether
the RTE finding (A1 hurts test MSE due to bad kernel-task fit) generalizes or
is dataset-specific.

Datasets:
- sst2     n=5000  val=872   (binary, GLUE 'validation')
- ag_news  n=5000  val=1000  (4-class, HF 'test' since no validation)
- mnli     n=5000  val=1000  (3-class, GLUE 'validation_matched')
- mr       n=5000  val=1000  (binary, HF 'validation')
- mrpc     n=3668  val=408   (binary, GLUE 'validation')
- qqp      n=5000  val=1000  (binary, GLUE 'validation')

Mode: one-hot Frobenius LC (covers binary + multi-class uniformly).
"""

import os, sys, pickle, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RHO = 1e-2
SEED = 2026

OUT_DIR = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/state/iteration_01"
NTK_ROOT = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/ntk"

# (dataset, n_train, val_sample_num, num_classes, val_split, hf_args)
DATASETS = [
    ("sst2",    5000,  872, 2, "validation",         {"path": "sst2"}),
    ("ag_news", 5000, 1000, 4, "test",               {"path": "ag_news"}),
    ("mnli",    5000, 1000, 3, "validation_matched", {"path": "glue", "name": "mnli"}),
    ("mr",      5000, 1000, 2, "validation",         {"path": "rotten_tomatoes"}),
    ("mrpc",    3668,  408, 2, "validation",         {"path": "glue", "name": "mrpc"}),
    ("qqp",     5000, 1000, 2, "validation",         {"path": "glue", "name": "qqp"}),
]


def load_labels(hf_args, train_split, val_split, sampled_idx, sampled_val_idx, ds_name):
    """Return (y_train, y_val) as int numpy arrays.

    Handles SST-2 / MR / RTE-style 'sentence' or 'text', and GLUE pair tasks.
    Label column is uniformly 'label'.
    """
    ds_train = load_dataset(**hf_args, split=train_split)
    ds_val   = load_dataset(**hf_args, split=val_split)

    # GLUE 'qqp' has 363846 examples; MNLI has 392702; SST-2 has 67349.
    # We index by sampled_idx into the train split.
    y_train = np.asarray(
        [ds_train[int(i)]["label"] for i in sampled_idx], dtype=np.int64
    )
    y_val = np.asarray(
        [ds_val[int(i)]["label"] for i in sampled_val_idx], dtype=np.int64
    )
    return y_train, y_val


def onehot_centered(y, num_classes, col_mean=None):
    """Return centered one-hot of shape (n, C). If col_mean given, use it; else compute."""
    n = len(y)
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    if col_mean is None:
        col_mean = Y.mean(axis=0, keepdims=True)
    return Y - col_mean, col_mean


def run_one(ds_name, n_train, val_n, num_classes, val_split, hf_args, seed=SEED):
    print(f"\n{'='*70}\n=== {ds_name}  n={n_train}  val={val_n}  C={num_classes}\n{'='*70}")
    ntk_path = os.path.join(
        NTK_ROOT, ds_name,
        f"bert_seed{seed}_num{n_train}_val{val_n}_signFalse.pkl"
    )
    print(f"[info] loading {ntk_path}")
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    ntk = bundle["ntk"]
    sampled_idx = np.asarray(bundle["sampled_idx"])
    sampled_val_idx = np.asarray(bundle["sampled_val_idx"])

    if isinstance(ntk, torch.Tensor):
        ntk = ntk.detach().cpu().numpy()
    ntk = np.asarray(ntk)
    if ntk.ndim == 3:
        ntk = ntk[0] if ntk.shape[0] == 1 else ntk.mean(axis=0)

    K_tt = 0.5 * (ntk[:n_train, :n_train] + ntk[:n_train, :n_train].T)
    K_tt = K_tt.astype(np.float64)
    K_vt = ntk[n_train:, :n_train].astype(np.float64)
    print(f"[info] K_tt {K_tt.shape}, K_vt {K_vt.shape}")

    # Labels
    print("[info] loading labels ...")
    y_train, y_val = load_labels(hf_args, "train", val_split,
                                  sampled_idx, sampled_val_idx, ds_name)
    Y_train, col_mean = onehot_centered(y_train, num_classes)
    Y_val, _ = onehot_centered(y_val, num_classes, col_mean=col_mean)
    norm_train_F = float((Y_train ** 2).sum())
    print(f"[info] norm_tilde_F^2 train = {norm_train_F:.2f}")

    # Eigendecomp
    print(f"[info] eigendecomp K_tt ({n_train}x{n_train}) ...")
    t0 = time.time()
    eigvals_asc, U_asc = np.linalg.eigh(K_tt)
    eigvals = eigvals_asc[::-1]
    U = U_asc[:, ::-1]
    print(f"[info] eigh done in {time.time()-t0:.1f}s; "
          f"lambda range [{eigvals[-1]:.2e}, {eigvals[0]:.2e}]")

    # Mode-wise
    # c[i] = U[:,i].T @ Y_train  shape (n_train, C)
    C_mat = U.T @ Y_train                     # (n_train, C)
    c2 = (C_mat ** 2).sum(axis=1)              # per-mode squared norm
    filt = (eigvals / (eigvals + RHO)) ** 2
    s = filt * c2

    # Spectrum stats
    frac_gt_10rho = float((eigvals > 10*RHO).mean())
    frac_gt_100rho = float((eigvals > 100*RHO).mean())
    frac_lt_rho = float((eigvals < RHO).mean())
    filt_top1pct_min = float(filt[: max(1, n_train // 100)].min())
    print(f"  frac lambda > 10 rho  : {frac_gt_10rho:.4f}")
    print(f"  frac lambda < rho     : {frac_lt_rho:.4f}")
    print(f"  filter@top1% min      : {filt_top1pct_min:.6f}")

    # Selections
    order_lambda = np.argsort(-eigvals)
    order_s      = np.argsort(-s)
    order_LC     = np.argsort(-c2)

    # Rank grid 1% step
    step = max(1, n_train // 100)
    r_grid = list(range(step, n_train + 1, step))
    if r_grid[-1] != n_train:
        r_grid.append(n_train)
    r_grid = np.asarray(r_grid)

    # Precompute G = K_vt @ U, alpha_full = c / (lambda + rho)
    print("[info] precomputing G ...")
    t0 = time.time()
    G = K_vt @ U                              # (n_val, n_train)
    alpha = C_mat / (eigvals[:, None] + RHO)  # (n_train, C)
    print(f"[info] G done in {time.time()-t0:.1f}s")

    n_grid = len(r_grid)
    arr = lambda: np.zeros(n_grid)
    out = {
        "lc_lambda": arr(), "lc_s": arr(), "lc_LC": arr(),
        "gap_lambda": arr(), "gap_s": arr(), "gap_LC": arr(),
        "mse_lambda": arr(), "mse_s": arr(), "mse_LC": arr(),
        "acc_lambda": arr(), "acc_s": arr(), "acc_LC": arr(),
        "overlap_ls": arr(), "overlap_lLC": arr(), "overlap_sLC": arr(),
    }

    n_val_actual = K_vt.shape[0]

    for idx, r in enumerate(r_grid):
        m_l  = np.zeros(n_train, dtype=bool); m_l [order_lambda[:r]] = True
        m_s  = np.zeros(n_train, dtype=bool); m_s [order_s[:r]]      = True
        m_LC = np.zeros(n_train, dtype=bool); m_LC[order_LC[:r]]     = True

        out["lc_lambda"][idx] = c2[m_l].sum()  / norm_train_F
        out["lc_s"][idx]      = c2[m_s].sum()  / norm_train_F
        out["lc_LC"][idx]     = c2[m_LC].sum() / norm_train_F

        out["gap_lambda"][idx] = s[~m_l].sum()
        out["gap_s"][idx]      = s[~m_s].sum()
        out["gap_LC"][idx]     = s[~m_LC].sum()

        # f_I(X_val) = G[:, I] @ alpha[I, :]
        pred_l  = G[:, m_l]  @ alpha[m_l,  :]
        pred_s_ = G[:, m_s]  @ alpha[m_s,  :]
        pred_LC = G[:, m_LC] @ alpha[m_LC, :]

        out["mse_lambda"][idx] = float(((Y_val - pred_l) ** 2).mean())
        out["mse_s"][idx]      = float(((Y_val - pred_s_) ** 2).mean())
        out["mse_LC"][idx]     = float(((Y_val - pred_LC) ** 2).mean())

        # Predicted class via argmax of (col_mean + pred)
        for key, pred in [("acc_lambda", pred_l), ("acc_s", pred_s_), ("acc_LC", pred_LC)]:
            recon = col_mean + pred
            pred_cls = recon.argmax(axis=1)
            out[key][idx] = float((pred_cls == y_val).mean())

        out["overlap_ls"][idx]  = (m_l & m_s ).sum() / r
        out["overlap_lLC"][idx] = (m_l & m_LC).sum() / r
        out["overlap_sLC"][idx] = (m_s & m_LC).sum() / r

    # Full ridge reference
    pred_full = G @ alpha
    mse_full = float(((Y_val - pred_full) ** 2).mean())
    recon_full = col_mean + pred_full
    acc_full = float((recon_full.argmax(axis=1) == y_val).mean())
    mse_null = float((Y_val ** 2).mean())
    acc_null = float((y_val == col_mean.ravel().argmax()).mean())
    print(f"[info] full ridge:  MSE={mse_full:.4f}  acc={acc_full:.4f}")
    print(f"[info] null/majority: MSE={mse_null:.4f}  acc={acc_null:.4f}")

    spec = {
        "lambda_min":   float(eigvals.min()),
        "lambda_max":   float(eigvals.max()),
        "lambda_median": float(np.median(eigvals)),
        "rho": RHO,
        "frac_lambda_gt_10rho":  frac_gt_10rho,
        "frac_lambda_gt_100rho": frac_gt_100rho,
        "frac_lambda_lt_rho":    frac_lt_rho,
        "filter_top1pct_min":    filt_top1pct_min,
    }
    return {
        "ds_name": ds_name, "n_train": n_train, "n_val": n_val_actual,
        "num_classes": num_classes,
        "r_grid": r_grid, "eigvals": eigvals, **out,
        "mse_full": mse_full, "mse_null": mse_null,
        "acc_full": acc_full, "acc_null": acc_null,
        "spec": spec,
    }


def per_dataset_plot(res):
    ds_name, n, r = res["ds_name"], res["n_train"], res["r_grid"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes.flat

    # spectrum
    ax[0].semilogy(np.arange(1, n+1), res["eigvals"], color="navy", lw=1)
    ax[0].axhline(RHO,    color="red", ls="--", lw=1, label=f"rho={RHO}")
    ax[0].axhline(10*RHO, color="orange", ls=":", lw=1, label="10 rho")
    ax[0].set_xlabel("eigen index i"); ax[0].set_ylabel("lambda_i (log)")
    ax[0].set_title(f"{ds_name} spectrum (n={n})"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # LC
    ax[1].plot(r, res["lc_lambda"], color="tab:red",   lw=2, label="top-r by lambda")
    ax[1].plot(r, res["lc_s"],      color="tab:blue",  lw=2, label="A1 (top-r by s_i)")
    ax[1].plot(r, res["lc_LC"],     color="tab:green", lw=1.5, ls="--", label="max LC")
    ax[1].set_xlabel("rank r"); ax[1].set_ylabel("LC(I)")
    ax[1].set_title("Label concentration"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # gap
    ax[2].semilogy(r, res["gap_lambda"] + 1e-12, color="tab:red",   lw=2, label="top-r by lambda")
    ax[2].semilogy(r, res["gap_s"]      + 1e-12, color="tab:blue",  lw=2, label="A1")
    ax[2].semilogy(r, res["gap_LC"]     + 1e-12, color="tab:green", lw=1.5, ls="--", label="max LC")
    ax[2].set_xlabel("rank r"); ax[2].set_ylabel("eq.(9) LHS (log)")
    ax[2].set_title("In-sample predictor gap"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    # MSE
    ax[3].plot(r, res["mse_lambda"], color="tab:red",   lw=2, label="top-r by lambda")
    ax[3].plot(r, res["mse_s"],      color="tab:blue",  lw=2, label="A1")
    ax[3].plot(r, res["mse_LC"],     color="tab:green", lw=1.5, ls="--", label="max LC")
    ax[3].axhline(res["mse_full"], color="black", ls=":", lw=1, label=f"full ridge {res['mse_full']:.3f}")
    ax[3].axhline(res["mse_null"], color="gray",  ls=":", lw=1, label=f"null {res['mse_null']:.3f}")
    ax[3].set_xlabel("rank r"); ax[3].set_ylabel("val MSE")
    ax[3].set_title(f"Test MSE (val={res['n_val']})"); ax[3].legend(fontsize=7); ax[3].grid(alpha=0.3)

    # accuracy
    ax[4].plot(r, res["acc_lambda"], color="tab:red",   lw=2, label="top-r by lambda")
    ax[4].plot(r, res["acc_s"],      color="tab:blue",  lw=2, label="A1")
    ax[4].plot(r, res["acc_LC"],     color="tab:green", lw=1.5, ls="--", label="max LC")
    ax[4].axhline(res["acc_full"], color="black", ls=":", lw=1, label=f"full ridge {res['acc_full']:.3f}")
    ax[4].axhline(res["acc_null"], color="gray",  ls=":", lw=1, label=f"majority {res['acc_null']:.3f}")
    ax[4].set_xlabel("rank r"); ax[4].set_ylabel("val accuracy")
    ax[4].set_title("Test accuracy"); ax[4].legend(fontsize=7); ax[4].grid(alpha=0.3)

    # overlap
    ax[5].plot(r, res["overlap_ls"],  color="tab:purple", lw=2, label="|I_lam ∩ I_s|/r")
    ax[5].plot(r, res["overlap_lLC"], color="tab:olive",  lw=2, label="|I_lam ∩ I_LC|/r")
    ax[5].plot(r, res["overlap_sLC"], color="tab:cyan",   lw=1.5, ls="--", label="|I_s ∩ I_LC|/r")
    ax[5].set_xlabel("rank r"); ax[5].set_ylabel("overlap ratio")
    ax[5].set_title("Selection overlap"); ax[5].legend(fontsize=8); ax[5].grid(alpha=0.3)
    ax[5].set_ylim(0, 1.05)

    plt.suptitle(f"{ds_name} (n={n}, val={res['n_val']}, C={res['num_classes']}, rho={RHO}, seed={SEED})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = os.path.join(OUT_DIR, f"quick_{ds_name}_n{res['n_train']}.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def per_dataset_table(res):
    key_rs_pct = [1, 2, 5, 10, 20, 30, 50, 75, 100]
    n = res["n_train"]
    key_rs = sorted(set(max(1, int(round(p * n / 100))) for p in key_rs_pct))
    out_txt = os.path.join(OUT_DIR, f"quick_{res['ds_name']}_n{n}.txt")
    with open(out_txt, "w") as f:
        f.write(f"# {res['ds_name']}  n={n}  val={res['n_val']}  C={res['num_classes']}  rho={RHO}  seed={SEED}\n")
        f.write(f"# full ridge: MSE={res['mse_full']:.6f}  acc={res['acc_full']:.6f}\n")
        f.write(f"# null/majority: MSE={res['mse_null']:.6f}  acc={res['acc_null']:.6f}\n")
        f.write(f"# spec: lambda_min={res['spec']['lambda_min']:.3e}  "
                f"lambda_max={res['spec']['lambda_max']:.3e}  "
                f"frac(lambda>10rho)={res['spec']['frac_lambda_gt_10rho']:.4f}  "
                f"filter_top1%min={res['spec']['filter_top1pct_min']:.6f}\n\n")
        f.write(f"{'r':>5} {'r%':>4}  "
                f"{'LCλ':>6} {'LC_s':>6}  "
                f"{'gap_λ':>9} {'gap_s':>9}  "
                f"{'MSEλ':>7} {'MSE_s':>7}  "
                f"{'accλ':>6} {'acc_s':>6}  "
                f"{'ovr_ls':>6}\n")
        f.write("-" * 95 + "\n")
        r_grid = res["r_grid"]
        for r in key_rs:
            i = int(np.argmin(np.abs(r_grid - r)))
            rr = r_grid[i]
            f.write(f"{rr:>5} {100*rr/n:>4.0f}  "
                    f"{res['lc_lambda'][i]:>6.3f} {res['lc_s'][i]:>6.3f}  "
                    f"{res['gap_lambda'][i]:>9.2f} {res['gap_s'][i]:>9.2f}  "
                    f"{res['mse_lambda'][i]:>7.4f} {res['mse_s'][i]:>7.4f}  "
                    f"{res['acc_lambda'][i]:>6.4f} {res['acc_s'][i]:>6.4f}  "
                    f"{res['overlap_ls'][i]:>6.3f}\n")
    return out_txt


def summary_plot(all_res):
    """Overlay 6 datasets: LC, gap, MSE for baseline vs A1."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_res)))
    for ax in axes.flat:
        ax.set_xlabel("rank r% (fraction of n)")
        ax.grid(alpha=0.3)

    # row 0: LC, gap, MSE — baseline vs A1 overlay
    for k, res in enumerate(all_res):
        r_pct = 100.0 * res["r_grid"] / res["n_train"]
        c = colors[k]
        lbl = f"{res['ds_name']} (n={res['n_train']})"

        axes[0,0].plot(r_pct, res["lc_lambda"], color=c, lw=1, ls="--")
        axes[0,0].plot(r_pct, res["lc_s"],      color=c, lw=2, label=lbl)

        axes[0,1].semilogy(r_pct, res["gap_lambda"]+1e-12, color=c, lw=1, ls="--")
        axes[0,1].semilogy(r_pct, res["gap_s"]+1e-12,      color=c, lw=2, label=lbl)

        # MSE — relative to full ridge for cross-dataset comparison
        axes[0,2].plot(r_pct, res["mse_lambda"] / res["mse_full"], color=c, lw=1, ls="--")
        axes[0,2].plot(r_pct, res["mse_s"]      / res["mse_full"], color=c, lw=2, label=lbl)

        # row 1: accuracy curves + spectrum + scatter (full ridge vs null table)
        axes[1,0].plot(r_pct, res["acc_lambda"], color=c, lw=1, ls="--")
        axes[1,0].plot(r_pct, res["acc_s"],      color=c, lw=2, label=lbl)

        # spectrum overlay (eigenvalue at i-th relative rank)
        idx_pct = np.linspace(1, res["n_train"], 200).astype(int) - 1
        axes[1,1].semilogy(100.0*(idx_pct+1)/res["n_train"], res["eigvals"][idx_pct],
                            color=c, lw=1.2, label=lbl)

    axes[0,0].set_ylabel("LC(I)")
    axes[0,0].set_title("Label concentration  (solid = A1, dashed = top-r by lambda)")
    axes[0,0].legend(fontsize=8, loc="lower right")

    axes[0,1].set_ylabel("eq.(9) LHS  (log)")
    axes[0,1].set_title("In-sample gap")

    axes[0,2].axhline(1.0, color="black", ls=":", lw=1)
    axes[0,2].set_ylabel("MSE / full-ridge MSE")
    axes[0,2].set_title("Relative test MSE (1.0 = full ridge)")

    axes[1,0].set_ylabel("val accuracy")
    axes[1,0].set_title("Test accuracy")

    axes[1,1].axhline(RHO, color="black", ls=":", lw=1, label=f"rho={RHO}")
    axes[1,1].set_ylabel("lambda (log)"); axes[1,1].set_title("Kernel spectra")
    axes[1,1].legend(fontsize=8, loc="upper right")

    # row1 col2: dataset summary text
    axes[1,2].axis("off")
    lines = ["dataset summary (val MSE / acc)", "—" * 40]
    lines.append(f"{'dataset':>10} {'full':>7} {'null':>7} {'A1@10%':>8} {'baseline@10%':>13}")
    for res in all_res:
        r_grid = res["r_grid"]; n = res["n_train"]
        idx10 = int(np.argmin(np.abs(r_grid - n // 10)))
        lines.append(
            f"{res['ds_name']:>10} {res['mse_full']:>7.3f} {res['mse_null']:>7.3f} "
            f"{res['mse_s'][idx10]:>8.3f} {res['mse_lambda'][idx10]:>13.3f}"
        )
    axes[1,2].text(0.0, 0.95, "\n".join(lines),
                    fontfamily="monospace", fontsize=9, va="top", ha="left")

    plt.suptitle(f"Cross-dataset summary — A1 vs LRFShap baseline (rho={RHO}, seed={SEED})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = os.path.join(OUT_DIR, "quick_summary_6datasets.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_png


def summary_table(all_res):
    """Write a comparison table with full-ridge-vs-null status + A1 deltas."""
    out_txt = os.path.join(OUT_DIR, "quick_summary_6datasets.txt")
    with open(out_txt, "w") as f:
        f.write(f"# Cross-dataset summary — A1 vs LRFShap baseline  (rho={RHO}, seed={SEED})\n\n")
        f.write("## Full ridge vs null/majority — does the kernel fit the task?\n\n")
        f.write(f"{'dataset':>10} {'n':>6} {'C':>2}  "
                f"{'full_MSE':>9} {'null_MSE':>9}  {'full<null?':>10}  "
                f"{'full_acc':>9} {'maj_acc':>9}  {'fit?':>5}\n")
        f.write("-" * 95 + "\n")
        for r in all_res:
            fit_mse = "YES" if r["mse_full"] < r["mse_null"] else "NO"
            fit_acc = "YES" if r["acc_full"] > r["acc_null"] else "NO"
            f.write(f"{r['ds_name']:>10} {r['n_train']:>6} {r['num_classes']:>2}  "
                    f"{r['mse_full']:>9.4f} {r['mse_null']:>9.4f}  {fit_mse:>10}  "
                    f"{r['acc_full']:>9.4f} {r['acc_null']:>9.4f}  {fit_acc:>5}\n")
        f.write("\n## A1 vs baseline at key ranks (val accuracy)\n\n")
        f.write(f"{'dataset':>10}  ")
        pcts = [1, 2, 5, 10, 20, 30, 50]
        for p in pcts:
            f.write(f"{f'r={p}%':>12}  ")
        f.write("\n" + "-" * 100 + "\n")
        for r in all_res:
            f.write(f"{r['ds_name']:>10}  ")
            r_grid = r["r_grid"]; n = r["n_train"]
            for p in pcts:
                ridx = int(np.argmin(np.abs(r_grid - p * n / 100)))
                la = r["acc_lambda"][ridx]; s = r["acc_s"][ridx]
                f.write(f"{la:>5.3f}/{s:>5.3f}  ")
            f.write("    (baseline / A1)\n")
        f.write("\n## A1 vs baseline at key ranks (val MSE)\n\n")
        f.write(f"{'dataset':>10}  ")
        for p in pcts:
            f.write(f"{f'r={p}%':>14}  ")
        f.write("\n" + "-" * 120 + "\n")
        for r in all_res:
            f.write(f"{r['ds_name']:>10}  ")
            r_grid = r["r_grid"]; n = r["n_train"]
            for p in pcts:
                ridx = int(np.argmin(np.abs(r_grid - p * n / 100)))
                la = r["mse_lambda"][ridx]; s = r["mse_s"][ridx]
                f.write(f"{la:>6.4f}/{s:>6.4f}  ")
            f.write("\n")
    return out_txt


# ---------- main ----------
all_res = []
all_pngs = []
all_txts = []
for ds_args in DATASETS:
    res = run_one(*ds_args)
    png = per_dataset_plot(res)
    txt = per_dataset_table(res)
    all_res.append(res)
    all_pngs.append(png)
    all_txts.append(txt)
    print(f"[saved] {png}")
    print(f"[saved] {txt}")

print("\n[done] generating cross-dataset summary ...")
summary_png = summary_plot(all_res)
summary_txt = summary_table(all_res)
print(f"[saved] {summary_png}")
print(f"[saved] {summary_txt}")
