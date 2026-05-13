"""
task_feature_concentration.py
-----------------------------

Feature Concentration FC(r) on the training-kernel spectrum.

Given the cached NTK bundle produced by ``task_ntk.py``, this script

    1. loads the train-train kernel block   K_S  \in R^{k x k},
    2. performs an eigendecomposition         K_S = U diag(lambda) U^T,
    3. evaluates (label-independent counterpart of LC(r))

           FC(r) = sum_{j<=r} lambda_j  /  sum_j lambda_j

       for every r on a user-supplied grid (default 1%, 2%, ..., 100% of k).

FC(r) measures how much of the **data covariance** (kernel trace) lives in
the top-r eigendirections; LC(r) in ``task_label_concentration.py`` measures
the analogous quantity for **labels**. Their comparison reveals whether the
kernel is task-aligned (LC ≈ FC) or not (LC ≪ FC).

The output is a PNG plot + a text table stored under
``./freeshap_res/feature_concentration/{dataset_name}/``.

Usage
-----
    # First make sure the NTK cache exists, e.g.
    python task_ntk.py --seed 2025 --dataset_name sst2 \
        --num_train_dp 5000 --val_sample_num 872

    # Then (default: r grid step = 1% of k, i.e. 50 for k=5000)
    python task_feature_concentration.py --seed 2025 --dataset_name sst2 \
        --num_train_dp 5000 --val_sample_num 872

    # Or override the step (as a percentage of k):
    python task_feature_concentration.py --seed 2025 --dataset_name sst2 \
        --num_train_dp 5000 --val_sample_num 872 \
        --r_pct 1.0
"""

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import pickle
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# arg parsing
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_train_dp", type=int, default=5000)
    parser.add_argument("--val_sample_num", type=int, default=872,
                        help="Used only to disambiguate the cache filename.")
    parser.add_argument("--r_pct", type=float, default=1.0,
                        help="Grid step as a percentage of k (default 1.0 -> "
                             "r = ceil(k/100), 2*ceil(k/100), ..., k).")
    parser.add_argument("--r_max", type=int, default=None,
                        help="Upper bound of r. Defaults to k (train size).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# NTK cache loading (identical to task_label_concentration.py)
# --------------------------------------------------------------------------- #
def find_ntk_file(dataset_name, seed, num_train_dp, val_sample_num):
    ntk_dir = f"./freeshap_res/ntk/{dataset_name}"
    if not os.path.isdir(ntk_dir):
        raise FileNotFoundError(f"NTK directory not found: {ntk_dir}")

    pattern = os.path.join(
        ntk_dir,
        f"*_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign*.pkl",
    )
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No NTK cache matches pattern:\n  {pattern}\n"
            f"Run task_ntk.py --seed {seed} --dataset_name {dataset_name} "
            f"--num_train_dp {num_train_dp} --val_sample_num {val_sample_num} first."
        )
    if len(candidates) > 1:
        print("[warn] multiple NTK caches matched; using most recent:")
        for p in candidates[:5]:
            print(f"  - {p}")
    return candidates[0]


def load_bundle(ntk_path):
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise ValueError(
            "Expected the new bundle format "
            "(dict with keys 'ntk', 'sampled_idx', 'sampled_val_idx'). "
            f"Got: type={type(bundle).__name__}"
        )
    return bundle


def extract_train_kernel(ntk, n_train):
    """Return a symmetric (k, k) train-train kernel as float64 numpy."""
    if isinstance(ntk, torch.Tensor):
        ntk = ntk.detach().cpu().numpy()
    ntk = np.asarray(ntk)

    if ntk.ndim == 3:
        if ntk.shape[0] == 1:
            ntk = ntk[0]
        else:
            print(f"[info] 3D NTK with C={ntk.shape[0]}; averaging over class axis.")
            ntk = ntk.mean(axis=0)

    if ntk.ndim != 2:
        raise ValueError(f"Unexpected NTK shape: {ntk.shape}")

    rows, cols = ntk.shape
    if rows < cols:
        raise ValueError(f"Unexpected NTK shape (rows<cols): {ntk.shape}")

    assert cols == n_train, (
        f"cols={cols} != n_train={n_train}; cache may be mismatched."
    )
    k_trtr = ntk[:n_train, :n_train].astype(np.float64)
    k_sym = 0.5 * (k_trtr + k_trtr.T)
    return k_sym


# --------------------------------------------------------------------------- #
# FC(r) computation
# --------------------------------------------------------------------------- #
def compute_fc_curve(K_sym, r_grid):
    """
    FC(r) = sum_{j<=r} lambda_j  /  sum_j lambda_j

    Only non-negative eigenvalues are retained (tiny negatives from
    numerical noise are clipped to 0).
    """
    k = K_sym.shape[0]
    print(f"[info] eigendecomposition of K_S ({k} x {k}) ...")
    eigvals_asc = np.linalg.eigvalsh(K_sym)
    eigvals = eigvals_asc[::-1]  # descending
    eigvals = np.clip(eigvals, 0.0, None)

    denom = float(eigvals.sum())
    if denom <= 0:
        raise ValueError("trace(K_S) = 0; kernel is degenerate.")
    cumsum = np.cumsum(eigvals)
    fc_full = cumsum / denom

    fc_values = np.asarray([fc_full[r - 1] for r in r_grid])
    return r_grid, fc_values, eigvals, {"trace": denom}


def build_r_grid(k, step, r_max):
    if r_max is None:
        r_max = k
    r_max = min(r_max, k)
    grid = list(range(step, r_max + 1, step))
    if grid[-1] != r_max:
        grid.append(r_max)
    return grid


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def save_plot(r_grid, fc_values, eigvals, out_png, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # FC(r)
    ax1.plot(r_grid, fc_values, marker="o", markersize=4, linewidth=1.3,
             color="tab:green")
    ax1.set_xlabel("r (truncation rank)", fontsize=14)
    ax1.set_ylabel("FC(r)", fontsize=14)
    ax1.set_title(f"Feature Concentration FC(r) {title_suffix}",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.02)
    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)

    for thr in [0.5, 0.7, 0.9, 0.95, 0.99]:
        idx = np.searchsorted(fc_values, thr)
        if idx < len(r_grid):
            ax1.axhline(thr, color="red", linestyle="--", linewidth=0.6, alpha=0.4)
            ax1.plot(r_grid[idx], fc_values[idx], "ro", markersize=6)
            ax1.text(r_grid[idx], thr + 0.01,
                     f"r={r_grid[idx]}\n({fc_values[idx]:.3f})",
                     fontsize=9, color="red")

    # eigenvalue spectrum
    eig_pos = eigvals[eigvals > 0]
    ax2.plot(np.arange(1, len(eig_pos) + 1), eig_pos, linewidth=1.0)
    ax2.set_yscale("log")
    ax2.set_xlabel("index j", fontsize=14)
    ax2.set_ylabel(r"$\lambda_j$ (log)", fontsize=14)
    ax2.set_title(f"NTK spectrum {title_suffix}", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    ntk_path = find_ntk_file(
        args.dataset_name, args.seed, args.num_train_dp, args.val_sample_num,
    )
    print(f"[info] NTK cache: {ntk_path}")

    bundle = load_bundle(ntk_path)
    sampled_idx = bundle["sampled_idx"]
    n_train = len(sampled_idx)
    print(f"[info] n_train = {n_train}")

    K_sym = extract_train_kernel(bundle["ntk"], n_train)
    print(f"[info] K_S shape = {K_sym.shape}, "
          f"mean={K_sym.mean():.4e}, diag_mean={np.diag(K_sym).mean():.4e}")

    step = max(1, int(np.ceil(n_train * args.r_pct / 100.0)))
    print(f"[info] r_pct={args.r_pct}% -> step={step} (k={n_train})")
    r_grid = build_r_grid(n_train, step, args.r_max)
    print(f"[info] r grid: {r_grid[0]}, {r_grid[1] if len(r_grid) > 1 else ''}, "
          f"..., {r_grid[-1]}  (count={len(r_grid)})")

    r_grid, fc_values, eigvals, info = compute_fc_curve(K_sym, r_grid)

    # ---- save outputs ---------------------------------------------------- #
    out_dir = f"./freeshap_res/feature_concentration/{args.dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(ntk_path).replace(".pkl", "")
    txt_path = os.path.join(out_dir, f"{stem}_FCr.txt")
    png_path = os.path.join(out_dir, f"{stem}_FCr.png")

    header = (
        f"# dataset={args.dataset_name}  seed={args.seed}  n_train={n_train}\n"
        f"# trace(K_S) = {info['trace']:.6e}\n"
        f"# columns: r  FC(r)"
    )
    np.savetxt(
        txt_path,
        np.stack([np.asarray(r_grid, dtype=np.int64), fc_values], axis=1),
        fmt=["%d", "%.10e"], header=header, comments="",
    )
    print(f"[done] saved table: {txt_path}")

    title_suffix = f"({args.dataset_name}, n={n_train})"
    save_plot(r_grid, fc_values, eigvals, png_path, title_suffix)
    print(f"[done] saved plot:  {png_path}")

    # print a small summary
    print("\n[summary] thresholds:")
    print(f"{'threshold':>10} | {'r':>8} | {'FC(r)':>8}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for thr in [0.5, 0.7, 0.9, 0.95, 0.99]:
        idx = int(np.searchsorted(fc_values, thr))
        if idx < len(r_grid):
            print(f"{thr:>10.2f} | {r_grid[idx]:>8d} | {fc_values[idx]:>8.4f}")
        else:
            print(f"{thr:>10.2f} | {'>max':>8} | {fc_values[-1]:>8.4f}")


if __name__ == "__main__":
    main()
