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
    total_count = len(eigvals)
    positive_idx = np.where(eigvals > 0)[0]
    pos_vals = eigvals[positive_idx]

    if len(pos_vals) == 0:
        raise ValueError("No positive eigenvalues found. Cannot draw log-scale plot.")

    # x-axis: top percentage over all eigenvalues
    x_percent = (positive_idx + 1) / total_count * 100.0
    
    # Compute cumulative sum and ratio
    cumsum = np.cumsum(pos_vals)
    total_sum = cumsum[-1]
    cumulative_ratio = cumsum / total_sum * 100.0  # percentage

    # Create figure with 2 subplots (horizontal layout)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ===== Subplot 1: Eigenvalue spectrum (log scale) =====
    ax1.plot(x_percent, pos_vals, marker='o', markersize=3, linewidth=1.2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Top Eigenvalue Percentage (%)', fontsize=20)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=20)
    ax1.set_title('NTK Eigenvalue Spectrum', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis='both', labelsize=16)
    
    # ===== Subplot 2: Cumulative explained ratio =====
    ax2.plot(x_percent, cumulative_ratio, marker='o', markersize=3, linewidth=1.2, color='blue')
    ax2.set_xlabel('Top Eigenvalue Percentage (%)', fontsize=14)
    ax2.set_ylabel('Cumulative Explained Ratio (%)', fontsize=14)
    ax2.set_title('Cumulative Variance Explained', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='both', labelsize=12)
    
    # Mark key thresholds (10%, 20%, ..., 90%)
    thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    threshold_info = []
    
    for threshold in thresholds:
        # Find the first index where cumulative ratio >= threshold
        idx = np.searchsorted(cumulative_ratio, threshold)
        if idx < len(x_percent):
            x_pos = x_percent[idx]
            y_pos = threshold  # Use exact threshold value for y
            
            # Draw horizontal and vertical lines
            ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
            ax2.axvline(x=x_pos, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
            
            # Annotate with marker
            ax2.plot(x_pos, cumulative_ratio[idx], 'ro', markersize=7)
            
            # Add text annotation for x and y coordinates
            # Position text slightly offset from the point
            text_offset_x = 2
            text_offset_y = 2
            ax2.text(x_pos + text_offset_x, cumulative_ratio[idx] + text_offset_y, 
                    f'({x_pos:.1f}%, {threshold}%)',
                    fontsize=14, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))
            
            threshold_info.append((threshold, x_pos, idx + 1))
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    return threshold_info


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

    # Create eigenvalues directory
    eigenvalues_dir = os.path.join(base_path, "eigenvalues")
    os.makedirs(eigenvalues_dir, exist_ok=True)
    
    # Extract filename from ntk_path and save in eigenvalues directory
    ntk_basename = os.path.basename(ntk_path)
    eig_txt_path = os.path.join(eigenvalues_dir, ntk_basename.replace('.pkl', '_eigenvalues.txt'))
    np.savetxt(eig_txt_path, eigvals, fmt='%.10e')

    eig_png_path = os.path.join(eigenvalues_dir, ntk_basename.replace('.pkl', '_eigenvalues_log.png'))
    threshold_info = save_plot(eigvals, eig_png_path)

    top_k = min(20, len(eigvals))
    print(f"[done] Saved eigenvalues text: {eig_txt_path}")
    print(f"[done] Saved eigenvalue plot: {eig_png_path}")
    print(f"[done] Top-{top_k} eigenvalues:")
    for i in range(top_k):
        print(f"  {i+1:>2}: {eigvals[i]:.10e}")
    
    # Print cumulative variance explained thresholds
    print(f"\n[info] Cumulative Variance Explained:")
    print(f"{'Threshold':>10} | {'Top %':>8} | {'# Eigenvalues':>15}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*15}")
    for threshold, top_pct, num_eigs in threshold_info:
        print(f"{threshold:>9}% | {top_pct:>7.2f}% | {num_eigs:>15}")
    
    # Save threshold info to text file
    threshold_txt_path = os.path.join(eigenvalues_dir, ntk_basename.replace('.pkl', '_eigenvalues_thresholds.txt'))
    with open(threshold_txt_path, 'w') as f:
        f.write("Cumulative Variance Explained Thresholds\n")
        f.write("="*50 + "\n")
        f.write(f"{'Threshold':>10} | {'Top %':>8} | {'# Eigenvalues':>15}\n")
        f.write(f"{'-'*10}-+-{'-'*8}-+-{'-'*15}\n")
        for threshold, top_pct, num_eigs in threshold_info:
            f.write(f"{threshold:>9}% | {top_pct:>7.2f}% | {num_eigs:>15}\n")
    print(f"[done] Saved threshold info: {threshold_txt_path}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = '16'
    os.environ["OPENBLAS_NUM_THREADS"] = '16'
    os.environ["MKL_NUM_THREADS"] = '16'
    main()
