"""κ-aware A1 multi-dataset check.

Solves Simon/Canatar self-consistency:
  rho = kappa * (1 - (1/n) * sum_i lambda_i / (lambda_i + kappa))

then uses s_i^kappa = (lambda_i / (lambda_i + kappa))^2 * ||u_i^T Y||_F^2 as
the supervised selection criterion. Compares 4 selections:
- I_lambda:   top-r by lambda_i               (LRFShap baseline)
- I_s_rho:    top-r by s_i^rho   = (lambda/(lambda+rho))^2 * (u^T y)^2  (A1 original)
- I_s_kappa:  top-r by s_i^kappa = (lambda/(lambda+kappa))^2 * (u^T y)^2 (A1 kappa-aware)
- I_LC:       top-r by (u^T y)^2                (max-LC)

Same datasets / metrics as quick_multi_check.py.
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

DATASETS = [
    ("sst2",    5000,  872, 2, "validation",         {"path": "sst2"}),
    ("ag_news", 5000, 1000, 4, "test",               {"path": "ag_news"}),
    ("mnli",    5000, 1000, 3, "validation_matched", {"path": "glue", "name": "mnli"}),
    ("mr",      5000, 1000, 2, "validation",         {"path": "rotten_tomatoes"}),
    ("mrpc",    3668,  408, 2, "validation",         {"path": "glue", "name": "mrpc"}),
    ("qqp",     5000, 1000, 2, "validation",         {"path": "glue", "name": "qqp"}),
]


def solve_kappa(eigvals, rho, n, tol=1e-9, max_iter=200):
    """Solve  kappa - rho - (kappa/n) * sum(lambda/(lambda+kappa)) = 0
    via bisection in [rho_eff_lo, lambda_max].

    Equivalent form:
      rho = kappa * (1 - (1/n) * sum lambda_i / (lambda_i + kappa))
    """
    lam = np.asarray(eigvals, dtype=np.float64)
    # Clamp tiny negative numerical eigenvalues to zero.
    lam = np.maximum(lam, 0.0)

    def g(k):
        # g(k) = k - rho - (k/n) * sum(lam/(lam+k))
        return k - rho - (k / n) * np.sum(lam / (lam + k))

    # g(rho) ≈ rho - rho - (rho/n)*sum ≈ -(rho/n)*sum which is negative.
    # We expand `hi` until g(hi) > 0.
    lo = rho
    hi = max(lam.max(), 1.0)
    # Make sure g(hi) > 0
    safety = 0
    while g(hi) < 0:
        hi *= 10
        safety += 1
        if safety > 30:
            raise RuntimeError(f"cannot bracket kappa: g(hi)={g(hi):.3e}")
    # If g(lo) > 0 already, kappa < rho (shouldn't happen normally)
    if g(lo) > 0:
        return lo

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm) < tol or (hi - lo) < tol * max(mid, 1.0):
            return mid
        if gm < 0:
            lo = mid
        else:
            hi = mid
    return mid


def load_labels(hf_args, train_split, val_split, sampled_idx, sampled_val_idx):
    ds_train = load_dataset(**hf_args, split=train_split)
    ds_val   = load_dataset(**hf_args, split=val_split)
    y_train = np.asarray([ds_train[int(i)]["label"] for i in sampled_idx], dtype=np.int64)
    y_val   = np.asarray([ds_val[int(i)]["label"] for i in sampled_val_idx], dtype=np.int64)
    return y_train, y_val


def onehot_centered(y, C, col_mean=None):
    n = len(y)
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    if col_mean is None:
        col_mean = Y.mean(axis=0, keepdims=True)
    return Y - col_mean, col_mean


def run_one(ds_name, n_train, val_n, C, val_split, hf_args, seed=SEED):
    print(f"\n{'='*70}\n=== {ds_name}  n={n_train}  val={val_n}  C={C}\n{'='*70}")
    ntk_path = os.path.join(
        NTK_ROOT, ds_name,
        f"bert_seed{seed}_num{n_train}_val{val_n}_signFalse.pkl"
    )
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

    K_tt = 0.5 * (ntk[:n_train, :n_train] + ntk[:n_train, :n_train].T).astype(np.float64)
    K_vt = ntk[n_train:, :n_train].astype(np.float64)

    y_train, y_val = load_labels(hf_args, "train", val_split, sampled_idx, sampled_val_idx)
    Y_train, col_mean = onehot_centered(y_train, C)
    Y_val, _ = onehot_centered(y_val, C, col_mean=col_mean)
    norm_train_F = float((Y_train ** 2).sum())

    print(f"[info] eigh K_tt ({n_train}x{n_train}) ...")
    t0 = time.time()
    eigvals_asc, U_asc = np.linalg.eigh(K_tt)
    eigvals = eigvals_asc[::-1]
    U = U_asc[:, ::-1]
    print(f"[info] eigh done in {time.time()-t0:.1f}s; "
          f"lambda range [{eigvals[-1]:.2e}, {eigvals[0]:.2e}]")

    # Solve kappa
    kappa = solve_kappa(eigvals, RHO, n_train)
    print(f"[info] kappa solved: {kappa:.4e}  (rho={RHO}, ratio kappa/rho={kappa/RHO:.2f})")

    # Mode-wise label energy
    C_mat = U.T @ Y_train                       # (n_train, C)
    c2 = (C_mat ** 2).sum(axis=1)                # (n_train,)

    # Filters
    filt_rho   = (eigvals / (eigvals + RHO))   ** 2
    filt_kappa = (eigvals / (eigvals + kappa)) ** 2
    s_rho   = filt_rho   * c2
    s_kappa = filt_kappa * c2

    # Spectrum stats
    frac_gt_10rho = float((eigvals > 10*RHO).mean())
    frac_gt_kappa = float((eigvals > kappa).mean())
    frac_gt_10kappa = float((eigvals > 10*kappa).mean())
    print(f"  frac lambda > 10 rho       : {frac_gt_10rho:.4f}")
    print(f"  frac lambda > kappa        : {frac_gt_kappa:.4f}")
    print(f"  frac lambda > 10 kappa     : {frac_gt_10kappa:.4f}")
    print(f"  filter_rho   @top1%  min   : {filt_rho  [: max(1, n_train // 100)].min():.6f}")
    print(f"  filter_kappa @top1%  min   : {filt_kappa[: max(1, n_train // 100)].min():.6f}")
    print(f"  filter_kappa @top10% min   : {filt_kappa[: max(1, n_train // 10)].min():.6f}")
    print(f"  filter_kappa @top30% min   : {filt_kappa[: max(1, 3*n_train // 10)].min():.6f}")

    # Orderings
    order_lambda = np.argsort(-eigvals)
    order_s_rho   = np.argsort(-s_rho)
    order_s_kappa = np.argsort(-s_kappa)
    order_LC      = np.argsort(-c2)

    # Rank grid (1% step)
    step = max(1, n_train // 100)
    r_grid = list(range(step, n_train + 1, step))
    if r_grid[-1] != n_train:
        r_grid.append(n_train)
    r_grid = np.asarray(r_grid)

    print("[info] precomputing G ...")
    t0 = time.time()
    G = K_vt @ U                                # (n_val, n_train)
    alpha = C_mat / (eigvals[:, None] + RHO)    # (n_train, C); predictor uses rho
    print(f"[info] G done in {time.time()-t0:.1f}s")

    n_grid = len(r_grid)
    arr = lambda: np.zeros(n_grid)
    out = {
        "lc_lambda":   arr(), "lc_s_rho":   arr(), "lc_s_kappa":  arr(), "lc_LC":   arr(),
        "gap_lambda":  arr(), "gap_s_rho":  arr(), "gap_s_kappa": arr(), "gap_LC":  arr(),
        "mse_lambda":  arr(), "mse_s_rho":  arr(), "mse_s_kappa": arr(), "mse_LC":  arr(),
        "acc_lambda":  arr(), "acc_s_rho":  arr(), "acc_s_kappa": arr(), "acc_LC":  arr(),
        "overlap_l_s_rho":   arr(),
        "overlap_l_s_kappa": arr(),
        "overlap_s_rho_kappa": arr(),
    }

    for idx, r in enumerate(r_grid):
        m_l       = np.zeros(n_train, dtype=bool); m_l      [order_lambda[:r]]   = True
        m_s_rho   = np.zeros(n_train, dtype=bool); m_s_rho  [order_s_rho[:r]]    = True
        m_s_kappa = np.zeros(n_train, dtype=bool); m_s_kappa[order_s_kappa[:r]]  = True
        m_LC      = np.zeros(n_train, dtype=bool); m_LC     [order_LC[:r]]       = True

        out["lc_lambda"  ][idx] = c2[m_l].sum()       / norm_train_F
        out["lc_s_rho"   ][idx] = c2[m_s_rho].sum()   / norm_train_F
        out["lc_s_kappa" ][idx] = c2[m_s_kappa].sum() / norm_train_F
        out["lc_LC"      ][idx] = c2[m_LC].sum()      / norm_train_F

        # In-sample gap using rho-filter (the LRFShap eq. (9) LHS)
        out["gap_lambda" ][idx] = s_rho[~m_l].sum()
        out["gap_s_rho"  ][idx] = s_rho[~m_s_rho].sum()
        out["gap_s_kappa"][idx] = s_rho[~m_s_kappa].sum()
        out["gap_LC"     ][idx] = s_rho[~m_LC].sum()

        # Test MSE / accuracy via rank-r ridge with the rho-based alpha
        pred_l       = G[:, m_l      ] @ alpha[m_l,       :]
        pred_s_rho   = G[:, m_s_rho  ] @ alpha[m_s_rho,   :]
        pred_s_kappa = G[:, m_s_kappa] @ alpha[m_s_kappa, :]
        pred_LC      = G[:, m_LC     ] @ alpha[m_LC,      :]

        out["mse_lambda" ][idx] = float(((Y_val - pred_l)       ** 2).mean())
        out["mse_s_rho"  ][idx] = float(((Y_val - pred_s_rho)   ** 2).mean())
        out["mse_s_kappa"][idx] = float(((Y_val - pred_s_kappa) ** 2).mean())
        out["mse_LC"     ][idx] = float(((Y_val - pred_LC)      ** 2).mean())

        for key, pred in [("acc_lambda", pred_l),
                          ("acc_s_rho",  pred_s_rho),
                          ("acc_s_kappa",pred_s_kappa),
                          ("acc_LC",     pred_LC)]:
            recon = col_mean + pred
            out[key][idx] = float((recon.argmax(axis=1) == y_val).mean())

        out["overlap_l_s_rho"     ][idx] = (m_l       & m_s_rho).sum() / r
        out["overlap_l_s_kappa"   ][idx] = (m_l       & m_s_kappa).sum() / r
        out["overlap_s_rho_kappa" ][idx] = (m_s_rho   & m_s_kappa).sum() / r

    pred_full = G @ alpha
    mse_full = float(((Y_val - pred_full) ** 2).mean())
    acc_full = float(((col_mean + pred_full).argmax(axis=1) == y_val).mean())
    mse_null = float((Y_val ** 2).mean())
    acc_null = float((y_val == col_mean.ravel().argmax()).mean())
    print(f"[info] full ridge:  MSE={mse_full:.4f}  acc={acc_full:.4f}")
    print(f"[info] null/majority: MSE={mse_null:.4f}  acc={acc_null:.4f}")

    return {
        "ds_name": ds_name, "n_train": n_train, "n_val": K_vt.shape[0], "C": C,
        "r_grid": r_grid, "eigvals": eigvals, "kappa": kappa,
        **out,
        "mse_full": mse_full, "mse_null": mse_null,
        "acc_full": acc_full, "acc_null": acc_null,
        "frac_gt_kappa": frac_gt_kappa,
    }


def per_dataset_plot(res):
    ds_name, n, r = res["ds_name"], res["n_train"], res["r_grid"]
    kappa = res["kappa"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes.flat

    # spectrum
    ax[0].semilogy(np.arange(1, n+1), res["eigvals"], color="navy", lw=1)
    ax[0].axhline(RHO,    color="red", ls="--", lw=1, label=f"rho={RHO}")
    ax[0].axhline(kappa,  color="purple", ls="-", lw=1.5, label=f"kappa={kappa:.2e}")
    ax[0].set_xlabel("eigen index i"); ax[0].set_ylabel("lambda_i (log)")
    ax[0].set_title(f"{ds_name} spectrum (n={n})"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # LC
    ax[1].plot(r, res["lc_lambda"],  color="tab:red",    lw=2, label="top-r by lambda")
    ax[1].plot(r, res["lc_s_rho"],   color="tab:blue",   lw=2, label="A1 rho")
    ax[1].plot(r, res["lc_s_kappa"], color="tab:purple", lw=2, label="A1 kappa")
    ax[1].plot(r, res["lc_LC"],      color="tab:green",  lw=1.5, ls="--", label="max LC")
    ax[1].set_xlabel("rank r"); ax[1].set_ylabel("LC(I)")
    ax[1].set_title("Label concentration"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # gap
    ax[2].semilogy(r, res["gap_lambda"]  + 1e-12, color="tab:red",    lw=2, label="top-r by lambda")
    ax[2].semilogy(r, res["gap_s_rho"]   + 1e-12, color="tab:blue",   lw=2, label="A1 rho")
    ax[2].semilogy(r, res["gap_s_kappa"] + 1e-12, color="tab:purple", lw=2, label="A1 kappa")
    ax[2].semilogy(r, res["gap_LC"]      + 1e-12, color="tab:green",  lw=1.5, ls="--", label="max LC")
    ax[2].set_xlabel("rank r"); ax[2].set_ylabel("eq.(9) LHS (log)")
    ax[2].set_title("In-sample predictor gap"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    # MSE
    ax[3].plot(r, res["mse_lambda"],  color="tab:red",    lw=2, label="top-r by lambda")
    ax[3].plot(r, res["mse_s_rho"],   color="tab:blue",   lw=2, label="A1 rho")
    ax[3].plot(r, res["mse_s_kappa"], color="tab:purple", lw=2, label="A1 kappa")
    ax[3].plot(r, res["mse_LC"],      color="tab:green",  lw=1.5, ls="--", label="max LC")
    ax[3].axhline(res["mse_full"], color="black", ls=":", lw=1, label=f"full {res['mse_full']:.3f}")
    ax[3].axhline(res["mse_null"], color="gray",  ls=":", lw=1, label=f"null {res['mse_null']:.3f}")
    ax[3].set_xlabel("rank r"); ax[3].set_ylabel("val MSE")
    ax[3].set_title(f"Test MSE (val={res['n_val']})"); ax[3].legend(fontsize=7); ax[3].grid(alpha=0.3)

    # accuracy
    ax[4].plot(r, res["acc_lambda"],  color="tab:red",    lw=2, label="top-r by lambda")
    ax[4].plot(r, res["acc_s_rho"],   color="tab:blue",   lw=2, label="A1 rho")
    ax[4].plot(r, res["acc_s_kappa"], color="tab:purple", lw=2, label="A1 kappa")
    ax[4].plot(r, res["acc_LC"],      color="tab:green",  lw=1.5, ls="--", label="max LC")
    ax[4].axhline(res["acc_full"], color="black", ls=":", lw=1, label=f"full {res['acc_full']:.3f}")
    ax[4].axhline(res["acc_null"], color="gray",  ls=":", lw=1, label=f"maj {res['acc_null']:.3f}")
    ax[4].set_xlabel("rank r"); ax[4].set_ylabel("val accuracy")
    ax[4].set_title("Test accuracy"); ax[4].legend(fontsize=7); ax[4].grid(alpha=0.3)

    # overlap
    ax[5].plot(r, res["overlap_l_s_rho"],     color="tab:blue",   lw=2,
                label="|I_lam ∩ A1_rho|/r")
    ax[5].plot(r, res["overlap_l_s_kappa"],   color="tab:purple", lw=2,
                label="|I_lam ∩ A1_kappa|/r")
    ax[5].plot(r, res["overlap_s_rho_kappa"], color="tab:olive",  lw=1.5, ls="--",
                label="|A1_rho ∩ A1_kappa|/r")
    ax[5].set_xlabel("rank r"); ax[5].set_ylabel("overlap ratio")
    ax[5].set_title("Selection overlap"); ax[5].legend(fontsize=8); ax[5].grid(alpha=0.3)
    ax[5].set_ylim(0, 1.05)

    plt.suptitle(f"{ds_name} (n={n}, val={res['n_val']}, C={res['C']}, rho={RHO}, "
                 f"kappa={kappa:.2e}, seed={SEED})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = os.path.join(OUT_DIR, f"kappa_{res['ds_name']}_n{n}.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def summary_plot(all_res):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_res)))
    for ax in axes.flat:
        ax.set_xlabel("rank r% (fraction of n)")
        ax.grid(alpha=0.3)

    # row 0: LC, gap, MSE  for baseline (dashed) vs A1_kappa (solid)
    # row 1: accuracy, spectrum, summary text
    for k, res in enumerate(all_res):
        r_pct = 100.0 * res["r_grid"] / res["n_train"]
        c = colors[k]
        lbl = f"{res['ds_name']} (n={res['n_train']})"

        axes[0,0].plot(r_pct, res["lc_lambda"],  color=c, lw=1, ls="--")
        axes[0,0].plot(r_pct, res["lc_s_kappa"], color=c, lw=2, label=lbl)

        axes[0,1].semilogy(r_pct, res["gap_lambda"]  + 1e-12, color=c, lw=1, ls="--")
        axes[0,1].semilogy(r_pct, res["gap_s_kappa"] + 1e-12, color=c, lw=2, label=lbl)

        axes[0,2].plot(r_pct, res["mse_lambda"]   / res["mse_full"], color=c, lw=1, ls="--")
        axes[0,2].plot(r_pct, res["mse_s_kappa"]  / res["mse_full"], color=c, lw=2, label=lbl)

        axes[1,0].plot(r_pct, res["acc_lambda"],  color=c, lw=1, ls="--")
        axes[1,0].plot(r_pct, res["acc_s_kappa"], color=c, lw=2, label=lbl)

        idx_pct = np.linspace(1, res["n_train"], 200).astype(int) - 1
        axes[1,1].semilogy(100.0*(idx_pct+1)/res["n_train"], res["eigvals"][idx_pct],
                            color=c, lw=1.2, label=lbl)
        axes[1,1].axhline(res["kappa"], color=c, lw=1, ls=":", alpha=0.7)

    axes[0,0].set_ylabel("LC(I)")
    axes[0,0].set_title("Label concentration  (solid = A1_kappa, dashed = top-r by lambda)")
    axes[0,0].legend(fontsize=8, loc="lower right")
    axes[0,1].set_ylabel("eq.(9) LHS  (log)")
    axes[0,1].set_title("In-sample gap")
    axes[0,2].axhline(1.0, color="black", ls=":", lw=1)
    axes[0,2].set_ylabel("MSE / full-ridge MSE")
    axes[0,2].set_title("Relative test MSE (1.0 = full ridge)")
    axes[1,0].set_ylabel("val accuracy")
    axes[1,0].set_title("Test accuracy")
    axes[1,1].axhline(RHO, color="black", ls=":", lw=1, label=f"rho={RHO}")
    axes[1,1].set_ylabel("lambda (log)")
    axes[1,1].set_title("Kernel spectra (dotted = each dataset's kappa)")
    axes[1,1].legend(fontsize=8, loc="upper right")

    axes[1,2].axis("off")
    lines = ["A1 (kappa) summary @ r=10%", "—" * 50]
    lines.append(f"{'dataset':>10} {'kappa':>9} {'k/rho':>6}  "
                  f"{'fullMSE':>7} {'nullMSE':>7}  "
                  f"{'A1_κ':>6} {'A1_ρ':>6} {'base':>6}")
    for res in all_res:
        r_grid = res["r_grid"]; n = res["n_train"]
        idx10 = int(np.argmin(np.abs(r_grid - n // 10)))
        lines.append(
            f"{res['ds_name']:>10} {res['kappa']:>9.3e} {res['kappa']/RHO:>6.2f}  "
            f"{res['mse_full']:>7.3f} {res['mse_null']:>7.3f}  "
            f"{res['mse_s_kappa'][idx10]:>6.3f} "
            f"{res['mse_s_rho'][idx10]:>6.3f} "
            f"{res['mse_lambda'][idx10]:>6.3f}"
        )
    axes[1,2].text(0.0, 0.95, "\n".join(lines),
                    fontfamily="monospace", fontsize=8.5, va="top", ha="left")

    plt.suptitle(f"Cross-dataset summary — A1 kappa-aware vs LRFShap baseline (rho={RHO}, seed={SEED})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = os.path.join(OUT_DIR, "kappa_summary_6datasets.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_png


def summary_table(all_res):
    out_txt = os.path.join(OUT_DIR, "kappa_summary_6datasets.txt")
    with open(out_txt, "w") as f:
        f.write(f"# kappa-aware A1 vs LRFShap baseline summary (rho={RHO}, seed={SEED})\n\n")
        f.write("## kappa and kernel-task fit\n\n")
        f.write(f"{'dataset':>10} {'n':>5} {'C':>2}  "
                f"{'kappa':>10} {'kappa/rho':>10}  "
                f"{'frac λ>κ':>10}  "
                f"{'fullMSE':>8} {'nullMSE':>8}  {'fit?':>5}\n")
        f.write("-" * 95 + "\n")
        for r in all_res:
            fit = "YES" if r["mse_full"] < r["mse_null"] else "NO"
            f.write(f"{r['ds_name']:>10} {r['n_train']:>5} {r['C']:>2}  "
                    f"{r['kappa']:>10.3e} {r['kappa']/RHO:>10.3f}  "
                    f"{r['frac_gt_kappa']:>10.4f}  "
                    f"{r['mse_full']:>8.4f} {r['mse_null']:>8.4f}  {fit:>5}\n")

        f.write("\n## Test accuracy at key ranks (baseline / A1_rho / A1_kappa)\n\n")
        pcts = [1, 2, 5, 10, 20, 30, 50]
        f.write(f"{'dataset':>10}  ")
        for p in pcts:
            f.write(f"{f'r={p}%':>22}  ")
        f.write("\n" + "-" * 200 + "\n")
        for r in all_res:
            f.write(f"{r['ds_name']:>10}  ")
            r_grid = r["r_grid"]; n = r["n_train"]
            for p in pcts:
                ridx = int(np.argmin(np.abs(r_grid - p * n / 100)))
                la = r["acc_lambda"][ridx]
                sr = r["acc_s_rho"][ridx]
                sk = r["acc_s_kappa"][ridx]
                f.write(f"{la:>5.3f}/{sr:>5.3f}/{sk:>5.3f}    ")
            f.write("\n")

        f.write("\n## Test MSE at key ranks (baseline / A1_rho / A1_kappa)\n\n")
        f.write(f"{'dataset':>10}  ")
        for p in pcts:
            f.write(f"{f'r={p}%':>24}  ")
        f.write("\n" + "-" * 200 + "\n")
        for r in all_res:
            f.write(f"{r['ds_name']:>10}  ")
            r_grid = r["r_grid"]; n = r["n_train"]
            for p in pcts:
                ridx = int(np.argmin(np.abs(r_grid - p * n / 100)))
                la = r["mse_lambda"][ridx]
                sr = r["mse_s_rho"][ridx]
                sk = r["mse_s_kappa"][ridx]
                f.write(f"{la:>6.4f}/{sr:>6.4f}/{sk:>6.4f}  ")
            f.write("\n")
    return out_txt


# ---------- main ----------
all_res = []
for ds_args in DATASETS:
    res = run_one(*ds_args)
    png = per_dataset_plot(res)
    all_res.append(res)
    print(f"[saved] {png}")

print("\n[done] generating cross-dataset summary ...")
summary_png = summary_plot(all_res)
summary_txt = summary_table(all_res)
print(f"[saved] {summary_png}")
print(f"[saved] {summary_txt}")
