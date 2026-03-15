import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_indices(file_path):
    """Load indices from a text file (one index per line)"""
    with open(file_path, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    return np.array(indices)


def calculate_overlap_ratios(inv_indices, eigen_indices, num_train_dp):
    """
    Calculate overlap ratios for top 1%, 2%, ..., 100% of data.
    Returns a list of 100 overlap percentages.
    """
    overlap_ratios = []
    
    for pct in range(1, 101):
        # Calculate number of samples for this percentage
        k = int(num_train_dp * pct / 100)
        k = min(k, len(inv_indices), len(eigen_indices))
        
        # Get top k indices from both
        inv_top_k = set(inv_indices[:k])
        eigen_top_k = set(eigen_indices[:k])
        
        # Calculate overlap
        overlap = len(inv_top_k & eigen_top_k)
        overlap_ratio = (overlap / k) * 100 if k > 0 else 0
        
        overlap_ratios.append(overlap_ratio)
    
    return overlap_ratios


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate and plot overlap between INV and Eigen indices')
    parser.add_argument("--dataset_name", type=str, default="mnli")
    parser.add_argument("--num_train_dp", type=int, default=1000)
    parser.add_argument("--val_sample_num", type=int, default=1000)
    parser.add_argument("--tmc_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--inv_lambda", type=float, default=1e-2,
                        help="Lambda for INV mode")
    parser.add_argument("--eigen_lambda", type=float, default=1e-2,
                        help="Lambda for Eigen mode")
    parser.add_argument("--eigen_rank_list", type=int, nargs='+', 
                        default=[1, 5, 10, 15, 20, 25, 30],
                        help="List of eigen ranks (as percentages)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for the plot (default: same as data directory)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    dataset_name = args.dataset_name
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    tmc_iter = args.tmc_iter
    seed = args.seed
    inv_lambda = args.inv_lambda
    eigen_lambda = args.eigen_lambda
    eigen_rank_list = args.eigen_rank_list
    
    # Base path
    base_path = f"./freeshap_res/{dataset_name}"
    
    # Format lambda in scientific notation
    inv_lambda_str = f"{inv_lambda:.0e}"
    eigen_lambda_str = f"{eigen_lambda:.0e}"
    
    # ===== Load INV indices =====
    inv_indices_path = (
        f"{base_path}/shapley/inv/indices/"
        f"bert_seed{seed}_num{num_train_dp}_val{val_sample_num}_"
        f"lam{inv_lambda_str}_signFalse_earlystopTrue_tmc{tmc_iter}_indices.txt"
    )
    
    print(f"Loading INV indices from: {inv_indices_path}")
    if not os.path.exists(inv_indices_path):
        raise FileNotFoundError(f"INV indices file not found: {inv_indices_path}")
    
    inv_indices = load_indices(inv_indices_path)
    print(f"Loaded {len(inv_indices)} INV indices")
    
    # ===== Load Eigen indices for each rank =====
    eigen_indices_dict = {}
    
    for rank in eigen_rank_list:
        eigen_indices_path = (
            f"{base_path}/shapley/eigen/indices/"
            f"bert_seed{seed}_num{num_train_dp}_val{val_sample_num}_"
            f"eig{rank}_lam{eigen_lambda_str}_cholesky_float32_"
            f"signFalse_earlystopTrue_tmc{tmc_iter}_indices.txt"
        )
        
        print(f"Loading Eigen rank {rank}% indices from: {eigen_indices_path}")
        if not os.path.exists(eigen_indices_path):
            print(f"Warning: Eigen indices file not found: {eigen_indices_path}")
            continue
        
        eigen_indices = load_indices(eigen_indices_path)
        eigen_indices_dict[rank] = eigen_indices
        print(f"Loaded {len(eigen_indices)} Eigen rank {rank}% indices")
    
    if not eigen_indices_dict:
        raise ValueError("No Eigen indices files were loaded successfully")
    
    # ===== Calculate overlap ratios =====
    print("\nCalculating overlap ratios...")
    overlap_results = {}
    
    for rank, eigen_indices in eigen_indices_dict.items():
        overlap_ratios = calculate_overlap_ratios(inv_indices, eigen_indices, num_train_dp)
        overlap_results[rank] = overlap_ratios
        print(f"Rank {rank}%: Overlap at 1%={overlap_ratios[0]:.2f}%, at 100%={overlap_ratios[-1]:.2f}%")
    
    # ===== Plot the results =====
    print("\nGenerating plot...")
    
    # Setup colors matching the notebook
    inv_color = 'red'
    eigen_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eigen_rank_list)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot overlap ratios for each eigen rank
    selected_dp_percent = list(range(1, 101))
    
    for i, rank in enumerate(sorted(overlap_results.keys())):
        overlap_ratios = overlap_results[rank]
        ax.plot(selected_dp_percent, overlap_ratios, 
                color=eigen_colors[i], linewidth=2.5,
                linestyle='-', marker='o', markersize=4, 
                label=f'rank {rank}%', alpha=0.85)
    
    # Add baseline: y=x (random selection expected overlap)
    ax.plot(selected_dp_percent, selected_dp_percent, 
            color='red', linestyle='--', linewidth=2.0, 
            label='Random baseline (y=x)', alpha=0.7)
    
    # Add reference line at 100%
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Selected Data Percentage (%)', fontsize=20)
    ax.set_ylabel('Overlap Ratio (%)', fontsize=20)
    ax.set_title(f'INV vs Eigen Indices Overlap | Dataset: {dataset_name}, '
                 f'Data: {num_train_dp}, Val: {val_sample_num}, Iter: {tmc_iter}, Seed: {seed}',
                 fontsize=20)
    ax.legend(fontsize=18, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    
    # Save the figure
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = base_path
    
    # Create overlap directory
    overlap_dir = f"{output_dir}/overlap"
    os.makedirs(overlap_dir, exist_ok=True)
    
    output_path = (
        f"{overlap_dir}/overlap_inv_eigen_seed{seed}_num{num_train_dp}_"
        f"val{val_sample_num}_invlam{inv_lambda_str}_eigenlam{eigen_lambda_str}_tmc{tmc_iter}.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    plt.show()
    
    # ===== Print summary statistics =====
    print("\n" + "="*80)
    print("OVERLAP SUMMARY")
    print("="*80)
    for rank in sorted(overlap_results.keys()):
        overlap_ratios = overlap_results[rank]
        mean_overlap = np.mean(overlap_ratios)
        min_overlap = np.min(overlap_ratios)
        max_overlap = np.max(overlap_ratios)
        print(f"Rank {rank}%: Mean={mean_overlap:.2f}%, Min={min_overlap:.2f}%, Max={max_overlap:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
