"""Quick RTE check — A1 (top-r by s_i) vs LRFShap baseline (top-r by lambda_i).

Goal: empirically test whether A1's supervised selection meaningfully improves
the rank-r approximation on RTE, where LC(r) is known to be low at small r.

Setup:
- RTE seed=2026, n_train=2490 (full), n_val=277
- rho = 1e-2 (LRFShap default)
- scalar binary label mode (matches cached *_LCr_scalar.txt)
- rank grid: 1% step of n = {25, 50, ..., 2475, 2490}

Three selections compared:
- I_lambda: top-r by lambda_i (LRFShap original)
- I_s:      top-r by s_i = (lambda_i/(lambda_i+rho))^2 * (u_i^T tilde_y)^2 (A1)
- I_LC:     top-r by (u_i^T tilde_y)^2 (max-LC ceiling)

Quantities computed at each r:
- LC(I) = sum_{i in I} (u_i^T tilde_y)^2 / ||tilde_y||^2
- In-sample predictor gap (eq. (9) LHS):
    sum_{i not in I} (lambda_i/(lambda_i+rho))^2 * (u_i^T tilde_y)^2
- Test MSE on val: || tilde_y_val - f_I(X_val) ||^2 / |val|
- Overlap |I_lambda intersect I_s| / r
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------- Config ----------
NTK_PATH = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/ntk/rte/bert_seed2026_num2490_val277_signFalse.pkl"
DATASET = "rte"
SEED = 2026
N_TRAIN = 2490
N_VAL = 277
RHO = 1e-2

OUT_DIR = "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/state/iteration_01"
PNG_PATH = os.path.join(OUT_DIR, "quick_rte_results.png")
TXT_PATH = os.path.join(OUT_DIR, "quick_rte_results.txt")


# ---------- Load cached NTK + indices ----------
print(f"[info] loading NTK bundle: {NTK_PATH}")
with open(NTK_PATH, "rb") as f:
    bundle = pickle.load(f)

ntk = bundle["ntk"]
sampled_idx = np.asarray(bundle["sampled_idx"])
sampled_val_idx = np.asarray(bundle["sampled_val_idx"])

if isinstance(ntk, torch.Tensor):
    ntk = ntk.detach().cpu().numpy()
ntk = np.asarray(ntk)
if ntk.ndim == 3:
    if ntk.shape[0] == 1:
        ntk = ntk[0]
    else:
        ntk = ntk.mean(axis=0)
print(f"[info] NTK shape: {ntk.shape}, train={N_TRAIN}, val={N_VAL}")

# Train-train kernel (top N_TRAIN rows of the (n_train+n_val) x n_train matrix)
K_tt = 0.5 * (ntk[:N_TRAIN, :N_TRAIN] + ntk[:N_TRAIN, :N_TRAIN].T)
K_tt = K_tt.astype(np.float64)
# Val-train kernel
K_vt = ntk[N_TRAIN:, :N_TRAIN].astype(np.float64)
print(f"[info] K_tt: {K_tt.shape}, K_vt: {K_vt.shape}")


# ---------- Load labels (train + val) ----------
print(f"[info] loading labels for {DATASET}")
ds_train = load_dataset("glue", "rte")["train"]
# RTE uses validation split as the val set in the FreeShap protocol
ds_val = load_dataset("glue", "rte")["validation"]

y_train = np.asarray([ds_train[int(i)]["label"] for i in sampled_idx], dtype=np.float64)
y_val = np.asarray([ds_val[int(i)]["label"] for i in sampled_val_idx], dtype=np.float64)

# Scalar centered: tilde_y = y - mean(y), using train mean for both train + val
y_bar = float(y_train.mean())
tilde_y_train = y_train - y_bar
tilde_y_val = y_val - y_bar
norm_tilde_sq = float(tilde_y_train @ tilde_y_train)
print(f"[info] y_train mean = {y_bar:.4f}, ||tilde_y_train||^2 = {norm_tilde_sq:.4f}")


# ---------- Eigendecomposition K_tt = U Lambda U^T ----------
print("[info] eigendecomposition of K_tt (2490 x 2490) ...")
import time
t0 = time.time()
eigvals_asc, U_asc = np.linalg.eigh(K_tt)
eigvals = eigvals_asc[::-1]            # descending
U = U_asc[:, ::-1]                      # columns aligned to descending lambda
print(f"[info] eigendecomp done in {time.time()-t0:.1f}s")
print(f"[info] lambda range: min={eigvals[-1]:.3e}  max={eigvals[0]:.3e}")


# ---------- Mode-wise coefficients ----------
c = U.T @ tilde_y_train               # (n_train,)
c2 = c ** 2                             # (u_i^T tilde_y)^2
filt = (eigvals / (eigvals + RHO)) ** 2  # (lambda/(lambda+rho))^2
s = filt * c2                          # supervised score per mode


# ---------- Spectrum stats ----------
spec_stats = {
    "lambda_min":   float(eigvals.min()),
    "lambda_max":   float(eigvals.max()),
    "lambda_median": float(np.median(eigvals)),
    "rho":          RHO,
    "frac_lambda_gt_10rho":  float((eigvals > 10*RHO).mean()),
    "frac_lambda_gt_100rho": float((eigvals > 100*RHO).mean()),
    "frac_lambda_lt_rho":    float((eigvals < RHO).mean()),
    "filter_at_top1pct_min":  float(filt[: max(1, N_TRAIN // 100)].min()),
    "filter_at_top10pct_min": float(filt[: max(1, N_TRAIN // 10)].min()),
    "filter_at_top30pct_min": float(filt[: max(1, 3 * N_TRAIN // 10)].min()),
}
for k, v in spec_stats.items():
    print(f"  {k:35s}: {v:.6g}")


# ---------- Rank grid (1% step) ----------
step = max(1, N_TRAIN // 100)            # = 24 for N=2490
r_grid = list(range(step, N_TRAIN + 1, step))
if r_grid[-1] != N_TRAIN:
    r_grid.append(N_TRAIN)
r_grid = np.asarray(r_grid)
print(f"[info] rank grid: {len(r_grid)} points, step={step}, "
      f"min={r_grid[0]}, max={r_grid[-1]}")


# ---------- Index sets for each selection ----------
# Sorted indices (descending) for each criterion. Indices into [0..N_TRAIN-1]
# where 0 corresponds to the largest lambda eigenvector.
order_lambda = np.argsort(-eigvals)       # top-r by lambda
order_s      = np.argsort(-s)              # top-r by supervised score
order_LC     = np.argsort(-c2)             # top-r by (u^T y)^2


def top_r_mask(order, r, n):
    m = np.zeros(n, dtype=bool)
    m[order[:r]] = True
    return m


# Pre-compute coefficients in eigenbasis for test prediction
# f_I(X_val) = K_vt @ U_I @ diag(1/(lambda_I+rho)) @ U_I^T @ tilde_y_train
# Precompute alpha = U^T tilde_y_train / (lambda + rho)   (n_train,)
# Then f_full(X_val) = K_vt @ U @ alpha,
#      f_I(X_val)    = K_vt @ U[:,I] @ alpha[I]
# We can speed this by precomputing G = K_vt @ U   (n_val, n_train)
# Then f_I(X_val) = G[:, I] @ alpha[I]
print("[info] precomputing G = K_vt @ U ...")
t0 = time.time()
G = K_vt @ U                              # (n_val, n_train)
alpha = c / (eigvals + RHO)               # (n_train,) — full ridge mode coefficients
print(f"[info] G precomputed in {time.time()-t0:.1f}s")


# ---------- Sweep ----------
lc_lambda  = np.zeros(len(r_grid))
lc_s       = np.zeros(len(r_grid))
lc_LC      = np.zeros(len(r_grid))
gap_lambda = np.zeros(len(r_grid))
gap_s      = np.zeros(len(r_grid))
gap_LC     = np.zeros(len(r_grid))
mse_lambda = np.zeros(len(r_grid))
mse_s      = np.zeros(len(r_grid))
mse_LC     = np.zeros(len(r_grid))
overlap_ls = np.zeros(len(r_grid))      # |I_lambda intersect I_s| / r
overlap_lLC = np.zeros(len(r_grid))     # |I_lambda intersect I_LC| / r
overlap_sLC = np.zeros(len(r_grid))     # |I_s intersect I_LC| / r

# Precompute filter * c2 (used in gap)
s_full = filt * c2                       # equals `s`, kept for clarity

n_val = K_vt.shape[0]

for idx, r in enumerate(r_grid):
    mask_lambda = top_r_mask(order_lambda, r, N_TRAIN)
    mask_s      = top_r_mask(order_s,      r, N_TRAIN)
    mask_LC     = top_r_mask(order_LC,     r, N_TRAIN)

    # LC(I) = sum_{i in I} c2 / norm_tilde_sq
    lc_lambda[idx] = c2[mask_lambda].sum() / norm_tilde_sq
    lc_s[idx]      = c2[mask_s].sum() / norm_tilde_sq
    lc_LC[idx]     = c2[mask_LC].sum() / norm_tilde_sq

    # In-sample predictor gap (eq. (9) LHS) = sum_{i not in I} s_full[i]
    gap_lambda[idx] = s_full[~mask_lambda].sum()
    gap_s[idx]      = s_full[~mask_s].sum()
    gap_LC[idx]     = s_full[~mask_LC].sum()

    # Test MSE for each truncation:
    # f_I(X_val) = G[:, I] @ alpha[I]
    pred_lambda = G[:, mask_lambda] @ alpha[mask_lambda]
    pred_s      = G[:, mask_s]      @ alpha[mask_s]
    pred_LC     = G[:, mask_LC]     @ alpha[mask_LC]
    mse_lambda[idx] = float(((tilde_y_val - pred_lambda) ** 2).mean())
    mse_s[idx]      = float(((tilde_y_val - pred_s)      ** 2).mean())
    mse_LC[idx]     = float(((tilde_y_val - pred_LC)     ** 2).mean())

    # Overlap (Jaccard-like, but normalized by r)
    overlap_ls[idx]  = (mask_lambda & mask_s).sum() / r
    overlap_lLC[idx] = (mask_lambda & mask_LC).sum() / r
    overlap_sLC[idx] = (mask_s      & mask_LC).sum() / r


# ---------- Reference: full-rank ridge test MSE ----------
pred_full = G @ alpha
mse_full = float(((tilde_y_val - pred_full) ** 2).mean())
mse_null = float((tilde_y_val ** 2).mean())   # predict 0 (i.e., return baseline mean)
print(f"[info] full-rank ridge val MSE = {mse_full:.6f}")
print(f"[info] null (predict 0) val MSE = {mse_null:.6f}")


# ---------- Plot ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
ax_spec, ax_lc, ax_gap = axes[0]
ax_mse, ax_overlap, ax_score = axes[1]

# (1) Spectrum
ax_spec.semilogy(np.arange(1, N_TRAIN + 1), eigvals, color="navy", lw=1.2)
ax_spec.axhline(RHO, color="red", ls="--", lw=1, label=f"rho={RHO}")
ax_spec.axhline(10*RHO, color="orange", ls=":", lw=1, label="10 rho")
ax_spec.set_xlabel("eigen index i (descending)")
ax_spec.set_ylabel("lambda_i  (log scale)")
ax_spec.set_title(f"RTE kernel spectrum (n={N_TRAIN})")
ax_spec.legend(fontsize=9)
ax_spec.grid(alpha=0.3)

# (2) LC(r)
ax_lc.plot(r_grid, lc_lambda, label="top-r by lambda (LRFShap)", color="tab:red", lw=2)
ax_lc.plot(r_grid, lc_s,      label="top-r by s_i (A1)",          color="tab:blue", lw=2)
ax_lc.plot(r_grid, lc_LC,     label="top-r by (u^T y)^2 (max-LC)", color="tab:green", lw=2, ls="--")
ax_lc.set_xlabel("rank r")
ax_lc.set_ylabel("LC(I) = label energy in I  /  ||y~||^2")
ax_lc.set_title("Label concentration vs rank")
ax_lc.legend(fontsize=9)
ax_lc.grid(alpha=0.3)

# (3) Gap (in-sample)
ax_gap.semilogy(r_grid, gap_lambda + 1e-12, label="top-r by lambda", color="tab:red", lw=2)
ax_gap.semilogy(r_grid, gap_s + 1e-12,      label="top-r by s_i",    color="tab:blue", lw=2)
ax_gap.semilogy(r_grid, gap_LC + 1e-12,     label="top-r by LC",     color="tab:green", lw=2, ls="--")
ax_gap.set_xlabel("rank r")
ax_gap.set_ylabel("eq. (9) LHS  (log scale)")
ax_gap.set_title("In-sample predictor gap vs rank")
ax_gap.legend(fontsize=9)
ax_gap.grid(alpha=0.3)

# (4) Test MSE
ax_mse.plot(r_grid, mse_lambda, label="top-r by lambda", color="tab:red", lw=2)
ax_mse.plot(r_grid, mse_s,      label="top-r by s_i",    color="tab:blue", lw=2)
ax_mse.plot(r_grid, mse_LC,     label="top-r by LC",     color="tab:green", lw=2, ls="--")
ax_mse.axhline(mse_full, color="black", ls=":", lw=1, label=f"full ridge MSE = {mse_full:.4f}")
ax_mse.axhline(mse_null, color="gray", ls=":", lw=1, label=f"null (predict 0) = {mse_null:.4f}")
ax_mse.set_xlabel("rank r")
ax_mse.set_ylabel("val MSE")
ax_mse.set_title(f"Test MSE on RTE val set (n_val={N_VAL})")
ax_mse.legend(fontsize=8)
ax_mse.grid(alpha=0.3)

# (5) Overlap
ax_overlap.plot(r_grid, overlap_ls, label="|I_lambda ∩ I_s| / r",  color="tab:purple", lw=2)
ax_overlap.plot(r_grid, overlap_lLC, label="|I_lambda ∩ I_LC| / r", color="tab:olive", lw=2)
ax_overlap.plot(r_grid, overlap_sLC, label="|I_s ∩ I_LC| / r",       color="tab:cyan",  lw=2, ls="--")
ax_overlap.set_xlabel("rank r")
ax_overlap.set_ylabel("overlap ratio")
ax_overlap.set_title("Selection overlap")
ax_overlap.legend(fontsize=9)
ax_overlap.grid(alpha=0.3)
ax_overlap.set_ylim(0, 1.05)

# (6) Score scatter at a representative r
r_focus = 250 if 250 <= N_TRAIN else N_TRAIN // 10
sel_l = top_r_mask(order_lambda, r_focus, N_TRAIN)
sel_s = top_r_mask(order_s,      r_focus, N_TRAIN)
ax_score.scatter(np.arange(N_TRAIN), c2, s=4, color="lightgray", label="all modes")
ax_score.scatter(np.where(sel_l)[0], c2[sel_l], s=10, color="tab:red",
                  label=f"top-{r_focus} by lambda")
ax_score.scatter(np.where(sel_s)[0], c2[sel_s], s=10, color="tab:blue", marker="x",
                  label=f"top-{r_focus} by s_i")
ax_score.set_yscale("log")
ax_score.set_xlabel("eigen index i (descending by lambda)")
ax_score.set_ylabel("(u_i^T tilde_y)^2  (log)")
ax_score.set_title(f"Mode-wise label projection (r={r_focus})")
ax_score.legend(fontsize=9)
ax_score.grid(alpha=0.3)

plt.suptitle(f"Quick RTE A1 check — seed={SEED}, n={N_TRAIN}, rho={RHO}",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(PNG_PATH, dpi=140, bbox_inches="tight")
print(f"[info] saved PNG: {PNG_PATH}")


# ---------- Save table ----------
# Highlight rows at key r values
key_rs = [25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2490]
with open(TXT_PATH, "w") as f:
    f.write(f"# Quick RTE A1 check\n")
    f.write(f"# seed={SEED}, n_train={N_TRAIN}, n_val={N_VAL}, rho={RHO}\n")
    f.write(f"# y_bar = {y_bar:.4f}, ||tilde_y_train||^2 = {norm_tilde_sq:.4f}\n")
    f.write(f"# full-rank ridge val MSE = {mse_full:.6f}\n")
    f.write(f"# null (predict 0) val MSE = {mse_null:.6f}\n")
    f.write("\n# Spectrum statistics\n")
    for k, v in spec_stats.items():
        f.write(f"#   {k:35s} = {v:.6g}\n")
    f.write("\n")
    hdr = (f"{'r':>5}  {'LC_lam':>8}  {'LC_s':>8}  {'LC_LC':>8}  "
           f"{'gap_lam':>10}  {'gap_s':>10}  {'gap_LC':>10}  "
           f"{'MSE_lam':>10}  {'MSE_s':>10}  {'MSE_LC':>10}  "
           f"{'ovr_ls':>7}  {'ovr_sLC':>7}\n")
    f.write(hdr)
    f.write("-" * (len(hdr) - 1) + "\n")
    for r in key_rs:
        idx = np.where(r_grid == r)[0]
        if len(idx) == 0:
            # find closest
            idx = np.argmin(np.abs(r_grid - r))
        else:
            idx = idx[0]
        f.write(f"{r_grid[idx]:>5}  "
                f"{lc_lambda[idx]:>8.4f}  {lc_s[idx]:>8.4f}  {lc_LC[idx]:>8.4f}  "
                f"{gap_lambda[idx]:>10.4f}  {gap_s[idx]:>10.4f}  {gap_LC[idx]:>10.4f}  "
                f"{mse_lambda[idx]:>10.6f}  {mse_s[idx]:>10.6f}  {mse_LC[idx]:>10.6f}  "
                f"{overlap_ls[idx]:>7.3f}  {overlap_sLC[idx]:>7.3f}\n")
print(f"[info] saved table: {TXT_PATH}")
print("[done]")
