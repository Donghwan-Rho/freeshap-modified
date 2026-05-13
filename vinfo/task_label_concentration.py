"""
task_label_concentration.py
---------------------------

Label Concentration LC(r) on the training-kernel spectrum.

Given the cached NTK bundle produced by ``task_ntk.py``, this script

    1. loads the train-train kernel block   K_S  \in R^{k x k},
    2. loads the corresponding training labels y_S,
    3. computes the centered label vector    \tilde y_S = y_S - \bar y_S 1,
    4. performs an eigendecomposition         K_S = U diag(lambda) U^T,
    5. evaluates

           LC(r) = || U_r^T \tilde y_S ||_F^2  /  || \tilde y_S ||_F^2

       for every r on a user-supplied grid (default 50, 100, ..., k).

The output is a PNG plot + a text table stored under
``./freeshap_res/label_concentration/{dataset_name}/``.

Usage
-----
    # First make sure the NTK cache exists, e.g.
    python task_ntk.py --seed 2025 --dataset_name sst2 \
        --num_train_dp 5000 --val_sample_num 872

    # Then (default: r grid step = 1% of k, i.e. 50 for k=5000)
    python task_label_concentration.py --seed 2025 --dataset_name sst2 \
        --num_train_dp 5000 --val_sample_num 872

    # Or override the step (as a percentage of k):
    python task_label_concentration.py --seed 2025 --dataset_name sst2 \
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
from datasets import load_dataset

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
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "scalar", "onehot"],
                        help="'scalar': binary scalar label. "
                             "'onehot': Frobenius LC on centered one-hot. "
                             "'auto': scalar if C=2 else onehot.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# NTK cache loading
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
        # shape (C, n_train + n_val, n_train)
        if ntk.shape[0] == 1:
            ntk = ntk[0]
        else:
            # single_kernel approximation: average over class axis
            print(f"[info] 3D NTK with C={ntk.shape[0]}; averaging over class axis.")
            ntk = ntk.mean(axis=0)

    if ntk.ndim != 2:
        raise ValueError(f"Unexpected NTK shape: {ntk.shape}")

    rows, cols = ntk.shape
    if rows < cols:
        raise ValueError(f"Unexpected NTK shape (rows<cols): {ntk.shape}")

    # train-train block: first n_train rows, all cols
    assert cols == n_train, (
        f"cols={cols} != n_train={n_train}; cache may be mismatched."
    )
    k_trtr = ntk[:n_train, :n_train].astype(np.float64)
    k_sym = 0.5 * (k_trtr + k_trtr.T)
    return k_sym


# --------------------------------------------------------------------------- #
# label loading
# --------------------------------------------------------------------------- #
def load_labels(dataset_name, sampled_idx):
    """Return y_S of shape (k,) as int numpy."""
    if dataset_name == "sst2":
        ds = load_dataset("sst2")["train"]
    elif dataset_name == "mr":
        ds = load_dataset("rotten_tomatoes")["train"]
    elif dataset_name == "rte":
        ds = load_dataset("glue", "rte")["train"]
    elif dataset_name == "mrpc":
        ds = load_dataset("glue", "mrpc")["train"]
    elif dataset_name == "qqp":
        ds = load_dataset("glue", "qqp")["train"]
    elif dataset_name == "mnli":
        ds = load_dataset("glue", "mnli")["train"]
    elif dataset_name == "ag_news":
        ds = load_dataset("ag_news")["train"]
    else:
        raise ValueError(f"unknown dataset_name={dataset_name}")

    idx = list(sampled_idx)
    labels = [ds[int(i)]["label"] for i in idx]
    return np.asarray(labels, dtype=np.int64)


# --------------------------------------------------------------------------- #
# LC(r) computation
# --------------------------------------------------------------------------- #
def compute_lc_curve(K_sym, y, num_labels, mode, r_grid):
    """
    Return (r_grid, lc_values, extra_info).

    scalar  : ỹ = y - ȳ 1  \in R^k;   LC(r) = ||U_r^T ỹ||² / ||ỹ||²
    onehot  : Y = one_hot(y) \in R^{k x C};
              Ỹ = Y - 1 Ȳ^T;
              LC(r) = ||U_r^T Ỹ||_F² / ||Ỹ||_F²
    """
    k = K_sym.shape[0]
    print(f"[info] eigendecomposition of K_S ({k} x {k}) ...")
    # eigh returns ascending; flip to descending
    eigvals_asc, U_asc = np.linalg.eigh(K_sym)
    eigvals = eigvals_asc[::-1]
    U = U_asc[:, ::-1]  # columns are u_1, u_2, ... sorted desc by lambda_j

    if mode == "scalar":
        y_bar = float(y.mean())
        tilde = y.astype(np.float64) - y_bar  # (k,)
        denom = float(tilde @ tilde)
        if denom <= 0:
            raise ValueError("||tilde y||^2 = 0; label is constant.")
        coeffs = U.T @ tilde                   # (k,)
        cumsum = np.cumsum(coeffs ** 2)
        lc_full = cumsum / denom
        info = {"y_bar": y_bar, "norm_tilde_sq": denom}
    else:  # onehot
        # build one-hot, then center columns
        C = int(num_labels)
        Y = np.zeros((k, C), dtype=np.float64)
        Y[np.arange(k), y] = 1.0
        col_mean = Y.mean(axis=0, keepdims=True)  # (1, C)
        tilde = Y - col_mean                       # (k, C)
        denom = float((tilde * tilde).sum())
        if denom <= 0:
            raise ValueError("||tilde Y||_F^2 = 0; label is constant.")
        coeffs = U.T @ tilde                       # (k, C)
        per_row_sq = (coeffs ** 2).sum(axis=1)     # (k,)
        cumsum = np.cumsum(per_row_sq)
        lc_full = cumsum / denom
        info = {"col_mean": col_mean.ravel(), "norm_tilde_sq": denom}

    # restrict to grid
    lc_values = np.asarray([lc_full[r - 1] for r in r_grid])
    return r_grid, lc_values, eigvals, info


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
def save_plot(r_grid, lc_values, eigvals, out_png, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # LC(r)
    ax1.plot(r_grid, lc_values, marker="o", markersize=4, linewidth=1.3)
    ax1.set_xlabel("r (truncation rank)", fontsize=14)
    ax1.set_ylabel("LC(r)", fontsize=14)
    ax1.set_title(f"Label Concentration LC(r) {title_suffix}", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.02)
    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)

    # annotate a few thresholds
    for thr in [0.5, 0.7, 0.9, 0.95, 0.99]:
        idx = np.searchsorted(lc_values, thr)
        if idx < len(r_grid):
            ax1.axhline(thr, color="red", linestyle="--", linewidth=0.6, alpha=0.4)
            ax1.plot(r_grid[idx], lc_values[idx], "ro", markersize=6)
            ax1.text(r_grid[idx], thr + 0.01,
                     f"r={r_grid[idx]}\n({lc_values[idx]:.3f})",
                     fontsize=9, color="red")

    # eigenvalue spectrum (for context)
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

    y = load_labels(args.dataset_name, sampled_idx)
    num_labels = int(y.max()) + 1
    class_counts = np.bincount(y, minlength=num_labels)
    print(f"[info] num_labels detected = {num_labels}, counts = {class_counts.tolist()}")

    if args.mode == "auto":
        mode = "scalar" if num_labels == 2 else "onehot"
    else:
        mode = args.mode
    print(f"[info] mode = {mode}")

    step = max(1, int(np.ceil(n_train * args.r_pct / 100.0)))
    print(f"[info] r_pct={args.r_pct}% -> step={step} (k={n_train})")
    r_grid = build_r_grid(n_train, step, args.r_max)
    print(f"[info] r grid: {r_grid[0]}, {r_grid[1] if len(r_grid) > 1 else ''}, "
          f"..., {r_grid[-1]}  (count={len(r_grid)})")

    r_grid, lc_values, eigvals, info = compute_lc_curve(
        K_sym, y, num_labels, mode, r_grid,
    )

    # ---- save outputs ---------------------------------------------------- #
    out_dir = f"./freeshap_res/label_concentration/{args.dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(ntk_path).replace(".pkl", "")
    txt_path = os.path.join(out_dir, f"{stem}_LCr_{mode}.txt")
    png_path = os.path.join(out_dir, f"{stem}_LCr_{mode}.png")

    header = (
        f"# dataset={args.dataset_name}  seed={args.seed}  "
        f"n_train={n_train}  mode={mode}\n"
        f"# num_labels={num_labels}  class_counts={class_counts.tolist()}\n"
        f"# ||tilde||_F^2 = {info['norm_tilde_sq']:.6e}\n"
        f"# columns: r  LC(r)"
    )
    np.savetxt(
        txt_path,
        np.stack([np.asarray(r_grid, dtype=np.int64), lc_values], axis=1),
        fmt=["%d", "%.10e"], header=header, comments="",
    )
    print(f"[done] saved table: {txt_path}")

    title_suffix = (
        f"({args.dataset_name}, n={n_train}, mode={mode}, "
        f"C={num_labels})"
    )
    save_plot(r_grid, lc_values, eigvals, png_path, title_suffix)
    print(f"[done] saved plot:  {png_path}")

    # print a small summary
    print("\n[summary] thresholds:")
    print(f"{'threshold':>10} | {'r':>8} | {'LC(r)':>8}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for thr in [0.5, 0.7, 0.9, 0.95, 0.99]:
        idx = int(np.searchsorted(lc_values, thr))
        if idx < len(r_grid):
            print(f"{thr:>10.2f} | {r_grid[idx]:>8d} | {lc_values[idx]:>8.4f}")
        else:
            print(f"{thr:>10.2f} | {'>max':>8} | {lc_values[-1]:>8.4f}")


if __name__ == "__main__":
    main()
