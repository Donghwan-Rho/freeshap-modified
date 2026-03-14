import pickle
import numpy as np
import yaml
import torch
import random
from datasets import load_dataset
import time

import sys, os
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *
from probe import *
from dvutils.Data_Shapley import Fast_Data_Shapley  # YAML tag 해석용

import argparse


def log_gpu_info():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"[gpu] CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(f"[gpu] torch.cuda.is_available()={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[gpu] CUDA is not available. Running on CPU.")
        return

    device_count = torch.cuda.device_count()
    print(f"[gpu] torch.cuda.device_count()={device_count}")

    for gpu_idx in range(device_count):
        props = torch.cuda.get_device_properties(gpu_idx)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(
            f"[gpu] cuda:{gpu_idx} name={props.name}, total_memory={total_mem_gb:.2f} GB"
        )

    current_idx = torch.cuda.current_device()
    print(f"[gpu] torch.cuda.current_device()={current_idx}")
    print(f"[gpu] current_device_name={torch.cuda.get_device_name(current_idx)}")


def get_gpu_info():
    """Collect GPU information for timing records"""
    gpu_info = {}
    
    if not torch.cuda.is_available():
        gpu_info['device_type'] = 'CPU'
        gpu_info['used_gpu_count'] = 0
        return gpu_info
    
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    current_idx = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_idx)
    
    gpu_info['used_gpu_count'] = 1  # Single GPU usage (no DataParallel)
    gpu_info['gpu_index'] = f"cuda:{current_idx}"
    gpu_info['gpu_model'] = current_device_name
    if cuda_visible:
        gpu_info['cuda_visible_devices'] = cuda_visible
    
    return gpu_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mr")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_train_dp", type=int, default=8530)
    parser.add_argument("--val_sample_num", type=int, default=1066)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="inv",
                        choices=["inv", "eigen", "none"])
    parser.add_argument("--eigen_rank", type=int, default=30,
                        help="Eigen rank as percentage of num_train_dp (e.g., 10 means 10% of data)")
    parser.add_argument("--inv_lambda_", type=float, default=1e-6,
                        help="Lambda (regularization parameter) for INV mode")
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2,
                        help="Lambda (regularization parameter) for Eigen mode")
    parser.add_argument("--num_train_selected", type=int, default=100)
    parser.add_argument("--num_train_selected_list", type=int, nargs='+', 
                        default=[i for i in range(1, 101)],
                        help="List of percentages (1-100) of num_train_dp to use as training data")
    parser.add_argument("--log_early_stopping", action="store_true",
                        help="Enable logging of early stopping points in TMC iterations")
    return parser.parse_args()


def main():
    args = parse_args()
    log_gpu_info()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num

    tmc_iter = args.tmc_iter

    approximate = args.approximate
    num_train_selected = args.num_train_selected
    num_train_selected_list = args.num_train_selected_list
    eigen_rank_pct = args.eigen_rank  # Now interpreted as percentage of num_dp
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_
    log_early_stopping = args.log_early_stopping
    
    # Initialize timing info dictionary
    timing_info = {}
    
    # Collect GPU information
    gpu_info = get_gpu_info()
    timing_info['gpu_info'] = gpu_info
    
    # Calculate actual eigen rank from percentage
    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% of num_dp={num_train_dp} -> actual rank={eigen_rank}")

    prompt = True
    signgd = False

    # eigen 파라미터
    eigen_solver = "cholesky"  # or "lstsq"
    eigen_dtype = "float32"

    per_point = True
    early_stopping = "True"

    # Dynamic path construction based on dataset_name
    yaml_path = f"../configs/dshap/{dataset_name}/ntk_prompt.yaml"
    base_path = f"./freeshap_res/{dataset_name}"

    # seed 고정 (재현성 보장)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN 최적화 비활성화
    np.random.seed(seed)
    random.seed(seed)

    # ===== YAML 로드 =====
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    probe_model = yaml_args["probe_com"]
    dshap_com = yaml_args["dshap_com"]
    
    # Enable early stopping logging if requested
    if log_early_stopping:
        dshap_com.log_early_stopping = True
        print(f"[info] Early stopping logging enabled")

    if prompt:
        probe_model.model.init(list_dataset.label_word_list)

    if approximate != "none":
        probe_model.approximate(approximate)

    # # 정책: ntk_normalize는 모든 approximate에 적용
    # if hasattr(probe_model, "normalize_ntk"):
    #     probe_model.normalize_ntk()

    if approximate == "eigen":
        probe_model.set_eigen_params(
            rank=eigen_rank,
            lam=eigen_lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed
        )
    elif approximate == "inv":
        probe_model.set_inv_params(lam=inv_lambda_)

    if signgd:
        probe_model.signgd()

    # ===== model_name 결정 =====
    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    # ===== 1) NTK 캐시 로드 (indices도 같이 로드) =====
    ntk_path = (
        f"{base_path}/ntk/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")

    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)

    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError(
            "NTK cache format is old (ntk only). "
            "Please regenerate NTK with the updated script that stores indices."
        )

    ntk = bundle["ntk"]
    sampled_idx = np.array(bundle["sampled_idx"])
    sampled_val_idx = np.array(bundle["sampled_val_idx"])
    print(f"[info] loaded NTK + indices: |train|={len(sampled_idx)}, |val|={len(sampled_val_idx)}")

    # probe에 NTK 주입
    probe_model.get_cached_ntk(ntk)

    # ===== 2) train/val set 구성 (indices 재샘플링 금지) =====
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    # eigen이면 feature 준비 (train labels 설정 후에 호출해야 함)
    if approximate == "eigen":
        print("[info] Preparing eigen regression features...")
        import time as time_module
        eigen_start_time = time_module.time()
        probe_model.prepare_eigen_regression()
        eigen_total_time = time_module.time() - eigen_start_time
        
        # Get eigendecomposition time from the regression model
        if hasattr(probe_model, 'eigen_regression') and probe_model.eigen_regression is not None:
            eigendecom_time = probe_model.eigen_regression.eigen_decomposition_time
            timing_info['eigendecomposition'] = eigendecom_time
            timing_info['eigen_preparation_total'] = eigen_total_time
            timing_info['eigen_preparation_overhead'] = eigen_total_time - eigendecom_time
            print(f"[TIMING] Eigendecomposition time: {eigendecom_time:.4f}s")
            print(f"[TIMING] Total eigen preparation time: {eigen_total_time:.4f}s")
            print(f"[TIMING] Overhead (non-decomposition): {eigen_total_time - eigendecom_time:.4f}s")

    print("len(train_set) =", len(train_set))
    print("len(val_set)   =", len(val_set))

    # ===== 3) Shapley 결과 경로 =====
    method_dir = approximate  # 'inv' or 'eigen'
    
    # Format lambda in scientific notation for filename (always use 1e-X format)
    if approximate == "eigen":
        lambda_str = f"{eigen_lambda_:.0e}"
        # Use the percentage value itself (eigen_rank_pct) in filename
        extra_tag = f"_eig{eigen_rank_pct}_lam{lambda_str}_{eigen_solver}_{eigen_dtype}"
    else:
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}"

    shapley_path = (
        f"{base_path}/shapley/{method_dir}/results/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")
    
    # Setup early stopping log file if enabled
    if log_early_stopping:
        setting_name = os.path.basename(shapley_path).replace('.pkl', '')
        early_stop_dir = f"{base_path}/shapley/{method_dir}/early_stopping"
        os.makedirs(early_stop_dir, exist_ok=True)
        early_stop_path = f"{early_stop_dir}/{setting_name}.txt"
        
        # Write header
        k_width = len(str(num_train_dp))
        n_width = len(str(num_train_dp))
        with open(early_stop_path, 'a') as f:
            f.write(f"TMC Iteration Log\n")
            f.write(f"{'='*60}\n")
            f.write(f"Dataset: {dataset_name}, Train: {num_train_dp}, Val: {val_sample_num}\n")
            f.write(f"Mode: {approximate}, TMC iterations: {tmc_iter}, Seed: {seed}\n")
            f.write(f"Early stopping enabled: {early_stopping}\n")
            f.write(f"\n{'Iter':<6} {'k':>{k_width}} / {'n':<{n_width}} {'Pct':>7} {'Status':<12}\n")
            f.write(f"{'-'*6} {'-'*k_width}   {'-'*n_width} {'-'*7} {'-'*12}\n")
        
        # Set the log path and formatting widths in dshap_com
        dshap_com.early_stopping_log_path = early_stop_path
        dshap_com._log_k_width = k_width
        dshap_com._log_n_width = n_width
        dshap_com._log_iter = 0  # Counter for iteration number
        print(f"[info] Early stopping log will be written to: {early_stop_path}")
    
    # ===== 4) Shapley 로드 or 계산 =====
    shapley_computation_time = None
    try:
        with open(shapley_path, "rb") as f:
            result = pickle.load(f)
        print(f"[info] loaded Shapley from {shapley_path}")

        dv_result = result["dv_result"]
        if "sampled_idx" in result:
            sampled_idx = np.array(result["sampled_idx"])
        if "sampled_val_idx" in result:
            sampled_val_idx = np.array(result["sampled_val_idx"])
        
        # Load timing info if available
        if "timing_info" in result:
            timing_info = result["timing_info"]
            print(f"[info] loaded timing info from cache: {timing_info}")

    except Exception as e:
        print("[info] shapley cache miss -> computing Shapley")
        print(f"[info] reason: {e}")
        
        shapley_start_time = time.time()

        dv_result = dshap_com.run(
            data_idx=sampled_idx.tolist(),
            val_data_idx=sampled_val_idx.tolist(),
            iteration=tmc_iter,
            use_cache_ntk=True,
            prompt=prompt,
            seed=seed,
            num_dp=num_train_dp,
            checkpoint=False,
            per_point=per_point,
            early_stopping=early_stopping
        )
        
        shapley_computation_time = time.time() - shapley_start_time
        timing_info['shapley_computation'] = shapley_computation_time
        print(f"[TIMING] Shapley computation time: {shapley_computation_time:.4f}s")
        
        # Append statistics to early stopping log if enabled
        if log_early_stopping and os.path.exists(early_stop_path):
            # Count lines to get statistics
            with open(early_stop_path, 'r') as f:
                lines = f.readlines()
            
            # Find data lines (skip header)
            data_lines = []
            for line in lines:
                if line.strip() and not line.startswith('TMC') and not line.startswith('=') and \
                   not line.startswith('Dataset') and not line.startswith('Mode') and \
                   not line.startswith('Early stopping') and not line.startswith('Iter') and \
                   not line.startswith('-'):
                    data_lines.append(line.strip())
            
            if data_lines:
                # Parse statistics
                early_stopped_count = sum(1 for line in data_lines if 'Early Stop' in line)
                completed_count = sum(1 for line in data_lines if 'Complete' in line)
                total_count = len(data_lines)
                
                # Calculate averages
                k_values = []
                n_values = []
                early_k_values = []
                percentages = []
                for line in data_lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        k = int(parts[1])
                        n = int(parts[3])
                        # Extract percentage (remove % sign)
                        pct_str = parts[4].rstrip('%')
                        pct = float(pct_str)
                        k_values.append(k)
                        n_values.append(n)
                        percentages.append(pct)
                        if 'Early Stop' in line:
                            early_k_values.append(k)
                
                # Calculate distribution by percentage ranges
                bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                distribution = {f"{bins[i]}-{bins[i+1]}%": 0 for i in range(len(bins)-1)}
                
                for pct in percentages:
                    for i in range(len(bins)-1):
                        if bins[i] <= pct < bins[i+1] or (i == len(bins)-2 and pct == 100):
                            distribution[f"{bins[i]}-{bins[i+1]}%"] += 1
                            break
                
                # Append statistics
                with open(early_stop_path, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Statistics:\n")
                    f.write(f"  Total iterations: {total_count}\n")
                    f.write(f"  - Early stopped: {early_stopped_count} ({early_stopped_count/total_count*100:.1f}%)\n")
                    f.write(f"  - Completed: {completed_count} ({completed_count/total_count*100:.1f}%)\n")
                    if k_values:
                        avg_k = np.mean(k_values)
                        avg_n = np.mean(n_values)
                        avg_pct = (avg_k / avg_n) * 100
                        f.write(f"  Overall average: {avg_k:.1f} / {avg_n:.1f} ({avg_pct:.2f}%)\n")
                    if early_k_values:
                        avg_early_k = np.mean(early_k_values)
                        avg_early_pct = (avg_early_k / n_values[0]) * 100
                        f.write(f"  Early stop avg: {avg_early_k:.1f} / {n_values[0]:.1f} ({avg_early_pct:.2f}%)\n")
                    
                    # Write distribution
                    f.write(f"\n{'-'*60}\n")
                    f.write(f"Early Stopping Distribution:\n")
                    for range_label in [f"{bins[i]}-{bins[i+1]}%" for i in range(len(bins)-1)]:
                        count = distribution[range_label]
                        pct_of_total = (count / total_count) * 100 if total_count > 0 else 0
                        bar = '█' * int(pct_of_total / 2)  # Visual bar (each █ = 2%)
                        f.write(f"  {range_label:>8}: {count:>4} ({pct_of_total:>5.1f}%) {bar}\n")
            
            print(f"[info] Early stopping log saved to {early_stop_path}")

        result = {
            "dv_result": dv_result,
            "sampled_idx": sampled_idx,
            "sampled_val_idx": sampled_val_idx,
            "args": vars(args),
            "timing_info": timing_info,
        }
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(shapley_path), exist_ok=True)
        with open(shapley_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[info] saved Shapley to {shapley_path}")

    dv_result = np.array(dv_result)
    print("dv_result shape:", dv_result.shape)

    # ===== 5) Prepare data subsets =====
    # dv_result: (num_train_dp, 2, val_sample_num)
    acc_contrib = dv_result[:, 1, :]            # (num_train_dp, val_sample_num)
    acc_sum_per_train = acc_contrib.sum(axis=1) # (num_train_dp,)

    sorted_indices = np.argsort(acc_sum_per_train)[::-1]  # 큰 값이 앞
    all_indices = np.arange(len(sampled_idx))
    
    # Save sorted indices (top to bottom) to txt file in indices/ directory
    indices_filename = shapley_path.split('/')[-1].replace('.pkl', '_indices.txt')
    indices_txt_path = f"{base_path}/shapley/{method_dir}/indices/{indices_filename}"
    # Convert internal indices to original dataset indices
    original_indices = sampled_idx[sorted_indices]
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(indices_txt_path), exist_ok=True)
    with open(indices_txt_path, 'a') as f:
        for idx in original_indices:
            f.write(f"{idx}\n")
    print(f"[info] saved sorted indices to {indices_txt_path}")

    # # ===== 6) Evaluate KRR for all num_train_selected values =====
    # print(f"\n{'='*80}")
    # print(f"Running evaluations for num_train_selected_list: {num_train_selected_list}")
    # print(f"{'='*80}\n")

    # Initialize result containers for each data_mode
    if approximate == "inv":
        top_results = []
        random_results = []
    elif approximate == "eigen":
        top_results_eigen = []
        random_results_eigen = []
        top_results_inv = []
        random_results_inv = []

    # Prepare all indices for random selection
    all_indices = np.arange(len(sampled_idx))

    # ===== Stage 1: INV mode predictions (if applicable) =====
    if approximate == "inv":
        print("[info] Running INV mode predictions (top)...")
        for num_train_selected_pct in num_train_selected_list:
            # Convert percentage to actual number of samples
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                print(f"Skipping num_train_selected_pct={num_train_selected_pct}% (k={k}, invalid value)")
                continue

            # Prepare data selection
            top_indices = sorted_indices[:k]
            
            # Reset pre_inv before each kernel_regression to prevent dimension mismatch
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(top_indices, dtype=int),
                test_set=val_set,
            )
            top_results.append(int(torch.round(acc * 10000).item()))
        
        print("[info] Running INV mode predictions (random)...")
        for num_train_selected_pct in num_train_selected_list:
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                continue

            # Random selection
            random_indices = np.random.choice(all_indices, size=k, replace=False)
            
            # Reset pre_inv before each kernel_regression
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(random_indices, dtype=int),
                test_set=val_set,
            )
            random_results.append(int(torch.round(acc * 10000).item()))
    
    # ===== Stage 2: EIGEN mode predictions (all k values) =====
    elif approximate == "eigen":
        probe_model.eigen_decom_mode = "top"
        
        print("[info] Running EIGEN mode predictions (top)...")
        for num_train_selected_pct in num_train_selected_list:
            # Convert percentage to actual number of samples
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                print(f"Skipping num_train_selected_pct={num_train_selected_pct}% (k={k}, invalid value)")
                continue

            # Prepare data selection
            top_indices = sorted_indices[:k]
            
            # Eigen prediction for top
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(top_indices, dtype=int),
                test_set=val_set,
            )
            top_results_eigen.append(int(torch.round(acc * 10000).item()))
        
        print("[info] Running EIGEN mode predictions (random)...")
        for num_train_selected_pct in num_train_selected_list:
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                continue

            # Random selection
            random_indices = np.random.choice(all_indices, size=k, replace=False)
            
            # Eigen prediction for random
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(random_indices, dtype=int),
                test_set=val_set,
            )
            random_results_eigen.append(int(torch.round(acc * 10000).item()))
        
        # ===== Stage 3: INV mode predictions (all k values) =====
        print("[info] Switching to INV mode for dual-mode evaluation...")
        
        # Switch to INV mode with inv_lambda_
        original_approx = probe_model.approximate_ntk
        probe_model.approximate_ntk = "inv"
        probe_model.set_inv_params(lam=inv_lambda_)
        
        print("[info] Running INV mode predictions (top)...")
        for num_train_selected_pct in num_train_selected_list:
            # Convert percentage to actual number of samples
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                continue

            # Prepare data selection (same as eigen stage)
            top_indices = sorted_indices[:k]
            
            # Reset INV caches before each prediction
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            # INV prediction for top
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(top_indices, dtype=int),
                test_set=val_set,
            )
            top_results_inv.append(int(torch.round(acc * 10000).item()))
        
        print("[info] Running INV mode predictions (random)...")
        for num_train_selected_pct in num_train_selected_list:
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                continue

            # Random selection
            random_indices = np.random.choice(all_indices, size=k, replace=False)
            
            # Reset INV caches before each prediction
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            # INV prediction for random
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(random_indices, dtype=int),
                test_set=val_set,
            )
            random_results_inv.append(int(torch.round(acc * 10000).item()))
        
        # Restore original EIGEN mode
        print("[info] Restoring EIGEN mode...")
        probe_model.approximate_ntk = original_approx
        probe_model.set_eigen_params(
            rank=eigen_rank,
            lam=eigen_lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed
        )
        
        # Store results for compatibility
        top_results = top_results_eigen
    
    # Print results by data_mode
    if approximate == "inv":
        print(f"\n[INV mode with lambda={inv_lambda_}]")
        print(f"top: {top_results}")
        print(f"random:\n{random_results}")
    elif approximate == "eigen":
        print(f"\n[Eigen mode with lambda={eigen_lambda_}]")
        print(f"top: {top_results_eigen}")
        print(f"random:\n{random_results_eigen}")
        print(f"\n[INV mode with lambda={inv_lambda_}]")
        print(f"top: {top_results_inv}")
        print(f"random:\n{random_results_inv}")
    
    # ===== Save predictions to txt file =====
    predictions_filename = shapley_path.split('/')[-1].replace('.pkl', '_predictions.txt')
    predictions_txt_path = f"{base_path}/shapley/{method_dir}/predictions/{predictions_filename}"
    os.makedirs(os.path.dirname(predictions_txt_path), exist_ok=True)
    
    # Helper function to add early stopping info
    def add_early_stopping_info(f, base_path, method_dir, shapley_path, log_early_stopping):
        if not log_early_stopping:
            return
        
        setting_name = os.path.basename(shapley_path).replace('.pkl', '')
        early_stop_dir = f"{base_path}/shapley/{method_dir}/early_stopping"
        early_stop_path = f"{early_stop_dir}/{setting_name}.txt"
        
        if not os.path.exists(early_stop_path):
            return
        
        with open(early_stop_path, 'r') as es_f:
            lines = es_f.readlines()
        
        # Find data lines (skip header and statistics)
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip all header and footer lines
            if any(stripped.startswith(x) for x in ['TMC', '=', 'Dataset:', 'Mode:', 'Early stopping', 
                                                      'Iter', '----', 'Statistics:', 'Total', 'Overall', 
                                                      'Early stop', 'Early Stopping Distribution']):
                continue
            # Skip statistics section lines
            if stripped.startswith('- '):
                continue
            # Skip distribution bar chart lines (contains █ or ends with %)
            parts = stripped.split()
            if '█' in stripped and not (len(parts) >= 5 and parts[0].isdigit()):
                continue
            # Valid iteration line: starts with digit and has k/n format
            if len(parts) >= 5 and parts[0].isdigit() and '/' in parts[2]:
                data_lines.append(stripped)
        
        if not data_lines:
            return
        
        # Parse statistics
        early_stopped_count = sum(1 for line in data_lines if 'Early Stop' in line)
        completed_count = sum(1 for line in data_lines if 'Complete' in line)
        total_count = len(data_lines)
        
        # Calculate averages
        percentages = []
        iter_stop_points = []
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 5:
                iter_num = int(parts[0])
                pct_str = parts[4].rstrip('%')
                pct = float(pct_str)
                percentages.append(pct)
                iter_stop_points.append((iter_num, pct))
        
        # Write early stopping statistics
        f.write(f"\n")
        f.write(f"early stopping statistics:\n")
        f.write(f"total iterations: {total_count}\n")
        f.write(f"early stopped: {early_stopped_count} ({early_stopped_count/total_count*100:.1f}%)\n")
        f.write(f"completed: {completed_count} ({completed_count/total_count*100:.1f}%)\n")
        
        if percentages:
            avg_pct = np.mean(percentages)
            f.write(f"average early stop: {avg_pct:.2f}%\n")
        
        # Distribution by percentage range
        f.write(f"\n")
        f.write(f"distribution by percentage range:\n")
        bins = [(i, i+10) for i in range(0, 100, 10)]
        bin_counts = [0] * len(bins)
        
        for pct in percentages:
            bin_idx = min(int(pct // 10), 9)  # 100% goes to last bin
            bin_counts[bin_idx] += 1
        
        # Find max count for scaling progress bar
        max_count = max(bin_counts) if bin_counts else 1
        bar_width = 40  # max bar width in characters
        
        for i, (start, end) in enumerate(bins):
            count = bin_counts[i]
            pct_of_total = (count / total_count * 100) if total_count > 0 else 0
            # Scale bar length
            bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = '█' * bar_length
            f.write(f"{start:3d}-{end:3d}%: {count:4d} ({pct_of_total:5.1f}%) {bar}\n")
        
        # Write each iteration's early stop point with aligned formatting
        f.write(f"\n")
        f.write(f"iteration early stop points (%):\n")
        # Calculate width needed for iteration numbers
        max_iter = max(iter_num for iter_num, _ in iter_stop_points) if iter_stop_points else 0
        iter_width = len(str(max_iter))
        
        for iter_num, pct in iter_stop_points:
            f.write(f"{iter_num:{iter_width}d}: {pct:5.1f}%\n")
    
    with open(predictions_txt_path, 'a') as f:
        if approximate == "inv":
            f.write(f"\n{'='*80}\n")
            f.write(f"dataset: {dataset_name}\n")
            f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"\n")
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n")
            f.write(f"{top_results}\n")
            f.write(f"random:\n")
            f.write(f"{random_results}\n")
            
            # Add early stopping statistics if flag is enabled
            add_early_stopping_info(f, base_path, method_dir, shapley_path, log_early_stopping)
            
        elif approximate == "eigen":
            f.write(f"\n{'='*80}\n")
            f.write(f"eigen rank: {eigen_rank_pct}% (actual: {eigen_rank})\n")
            f.write(f"dataset: {dataset_name}\n")
            f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"solver: {eigen_solver}, dtype: {eigen_dtype}\n")
            f.write(f"\n")
            f.write(f"eigen mode lambda={eigen_lambda_:.0e}\n")
            f.write(f"top:\n")
            f.write(f"{top_results_eigen}\n")
            f.write(f"random:\n")
            f.write(f"{random_results_eigen}\n")
            f.write(f"\n")
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n")
            f.write(f"{top_results_inv}\n")
            f.write(f"random:\n")
            f.write(f"{random_results_inv}\n")
            
            # Add early stopping statistics if flag is enabled
            add_early_stopping_info(f, base_path, method_dir, shapley_path, log_early_stopping)
    
    print(f"[info] saved predictions to {predictions_txt_path}")
    
    # ===== Print timing summary =====
    if timing_info:
        print("\n" + "="*80)
        print("TIMING SUMMARY")
        print("="*80)
        
        # Print GPU info first
        if 'gpu_info' in timing_info:
            gpu_info = timing_info['gpu_info']
            print("\nGPU Information:")
            print("-" * 40)
            for key, value in gpu_info.items():
                print(f"{key:.<40} {value}")
            print("-" * 40 + "\n")
        
        # Print timing measurements
        print("Timing Measurements:")
        print("-" * 40)
        for key, value in timing_info.items():
            if key != 'gpu_info':
                print(f"{key:.<40} {value:.4f}s")
        print("="*80 + "\n")
        
        # Save timing info to a separate log file
        timing_log_path = shapley_path.replace('.pkl', '_timing.txt')
        with open(timing_log_path, 'a') as f:
            f.write("TIMING SUMMARY\n")
            f.write("="*80 + "\n")
            
            # Write GPU info
            if 'gpu_info' in timing_info:
                f.write("\nGPU Information:\n")
                f.write("-" * 40 + "\n")
                for key, value in timing_info['gpu_info'].items():
                    f.write(f"{key}: {value}\n")
                f.write("-" * 40 + "\n\n")
            
            # Write timing measurements
            f.write("Timing Measurements:\n")
            f.write("-" * 40 + "\n")
            for key, value in timing_info.items():
                if key != 'gpu_info':
                    f.write(f"{key}: {value:.4f}s\n")
            f.write("="*80 + "\n")
        print(f"[info] saved timing info to {timing_log_path}")

if __name__ == "__main__":
    main()
