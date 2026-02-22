import warnings
warnings.filterwarnings('ignore')

import os
import glob
import pickle
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mr")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_train_dp", type=int, default=8530)
    parser.add_argument("--val_sample_num", type=int, default=None,
                        help="Number of validation samples. If None, use entire validation set.")
    return parser.parse_args()


def find_ntk_file(base_path, seed, num_train_dp, val_sample_num):
    ntk_dir = os.path.join(base_path, "ntk")
    if not os.path.isdir(ntk_dir):
        raise FileNotFoundError(f"NTK directory not found: {ntk_dir}")

    if val_sample_num is None:
        pattern = os.path.join(
            ntk_dir,
            f"*_seed{seed}_num{num_train_dp}_val*_signFalse.pkl"
        )
    else:
        pattern = os.path.join(
            ntk_dir,
            f"*_seed{seed}_num{num_train_dp}_val{val_sample_num}_signFalse.pkl"
        )

    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            "No matching NTK cache found. "
            f"pattern={pattern}"
        )

    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    if len(candidates) > 1:
        print("[warn] Multiple NTK caches matched. Using the most recent file:")
        for path in candidates[:5]:
            print(f"  - {path}")

    return candidates[0]


def load_ntk(ntk_path):
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict) and "ntk" in bundle:
        ntk = bundle["ntk"]
    else:
        ntk = bundle

    if isinstance(ntk, torch.Tensor):
        ntk = ntk.detach().cpu().numpy()
    else:
        ntk = np.asarray(ntk)

    if ntk.ndim == 3:
        # task_ntk.py often stores NTK as (num_class, train+val, train)
        if ntk.shape[0] == 1:
            print(f"[info] 3D NTK detected. Using ntk[0] with shape {ntk[0].shape}")
            ntk = ntk[0]
        else:
            print(
                "[info] 3D NTK detected with multiple class kernels. "
                f"Reducing by mean over class axis: {ntk.shape} -> {ntk.mean(axis=0).shape}"
            )
            ntk = ntk.mean(axis=0)

    if ntk.ndim != 2:
        raise ValueError(f"Expected 2D NTK matrix, got shape={ntk.shape}")

    return ntk


def compute_eigenvalues(ntk):
    rows, cols = ntk.shape

    # task_ntk/task_shapley_acc convention:
    # ntk shape is typically (n_train + n_val, n_train),
    # and eigen decomposition should be done on K_train,train (n_train x n_train).
    if rows >= cols:
        n_train = cols
        k_trtr = ntk[:n_train, :]
        k_sym = 0.5 * (k_trtr + k_trtr.T)
        print(f"[info] Using train-train NTK block for eigendecomposition: {k_sym.shape}")
        eigvals = np.linalg.eigvalsh(k_sym)
    else:
        print("[warn] Unexpected NTK shape (rows < cols). Falling back to NTK @ NTK.T")
        gram = ntk @ ntk.T
        eigvals = np.linalg.eigvalsh(gram)

    eigvals = np.sort(eigvals)[::-1]
    return eigvals


def save_plot(eigvals, out_png):
    positive_mask = eigvals > 0
    pos_vals = eigvals[positive_mask]

    if len(pos_vals) == 0:
        raise ValueError("No positive eigenvalues found. Cannot draw log-scale plot.")

    x = np.arange(1, len(pos_vals) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, pos_vals, marker='o', markersize=3, linewidth=1.2)
    plt.yscale('log')
    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('NTK Eigenvalue Spectrum')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num

    base_path = f"./freeshap_res/{dataset_name}"
    os.makedirs(base_path, exist_ok=True)

    ntk_path = find_ntk_file(base_path, seed, num_train_dp, val_sample_num)
    print(f"[info] using ntk_path = {ntk_path}")

    ntk = load_ntk(ntk_path)
    print(f"[info] ntk shape = {ntk.shape}")

    eigvals = compute_eigenvalues(ntk)

    eig_txt_path = ntk_path.replace('.pkl', '_eigenvalues.txt')
    np.savetxt(eig_txt_path, eigvals, fmt='%.10e')

    eig_png_path = ntk_path.replace('.pkl', '_eigenvalues_log.png')
    save_plot(eigvals, eig_png_path)

    top_k = min(20, len(eigvals))
    print(f"[done] Saved eigenvalues text: {eig_txt_path}")
    print(f"[done] Saved eigenvalue plot: {eig_png_path}")
    print(f"[done] Top-{top_k} eigenvalues:")
    for i in range(top_k):
        print(f"  {i+1:>2}: {eigvals[i]:.10e}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = '16'
    os.environ["OPENBLAS_NUM_THREADS"] = '16'
    os.environ["MKL_NUM_THREADS"] = '16'
    main()
