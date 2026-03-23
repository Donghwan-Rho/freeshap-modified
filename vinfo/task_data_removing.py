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
                        default=[i for i in range(0, 101)],
                        help="List of percentages (0-100) of num_train_dp to remove as training data")
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
    data_removing_base_path = f"./freeshap_res/data_removing/{dataset_name}"

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
    
    # ===== 4) Shapley 로드 (캐시에서만 로드, 계산 안 함) =====
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
        print("[ERROR] shapley cache not found. Please run task_shapley_acc.py first to compute Shapley values.")
        print(f"[info] reason: {e}")
        return

    dv_result = np.array(dv_result)
    print("dv_result shape:", dv_result.shape)

    # ===== 5) Prepare data subsets =====
    # dv_result: (num_train_dp, 2, val_sample_num)
    acc_contrib = dv_result[:, 1, :]            # (num_train_dp, val_sample_num)
    acc_sum_per_train = acc_contrib.sum(axis=1) # (num_train_dp,)

    sorted_indices = np.argsort(acc_sum_per_train)[::-1]  # 큰 값이 앞 (high Shapley)
    all_indices = np.arange(len(sampled_idx))

    # Initialize result containers for each removal strategy
    if approximate == "inv":
        top_removal_results = []
        bottom_removal_results = []
        random_removal_results = []
    elif approximate == "eigen":
        top_removal_results_eigen = []
        bottom_removal_results_eigen = []
        random_removal_results_eigen = []
        top_removal_results_inv = []
        bottom_removal_results_inv = []
        random_removal_results_inv = []

    # ===== Stage 1: EIGEN mode predictions (if applicable) =====
    if approximate == "eigen":
        print("[info] Running EIGEN mode predictions...")
        
        # Top removal (remove high Shapley value data)
        print("[info] EIGEN - Top removal (removing high Shapley value data)...")
        for remove_pct in num_train_selected_list:
            # Convert percentage to actual number of samples to REMOVE
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(sorted_indices):
                print(f"Skipping remove_pct={remove_pct}% (k_remove={k_remove}, would remove all data)")
                continue
            
            # Remove top k_remove indices, keep the rest
            remaining_indices = sorted_indices[k_remove:]
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            top_removal_results_eigen.append(int(torch.round(acc * 10000).item()))
        
        # Bottom removal (remove low Shapley value data)
        print("[info] EIGEN - Bottom removal (removing low Shapley value data)...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(sorted_indices):
                continue
            
            # Remove bottom k_remove indices, keep the rest
            remaining_indices = sorted_indices[:-k_remove] if k_remove > 0 else sorted_indices
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            bottom_removal_results_eigen.append(int(torch.round(acc * 10000).item()))
        
        # Random removal
        print("[info] EIGEN - Random removal...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(all_indices):
                continue
            
            # Randomly select k_remove indices to remove
            indices_to_remove = np.random.choice(all_indices, size=k_remove, replace=False)
            remaining_indices = np.setdiff1d(all_indices, indices_to_remove)
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            random_removal_results_eigen.append(int(torch.round(acc * 10000).item()))
        
        # ===== Stage 2: INV mode predictions for EIGEN approximate =====
        print("[info] Switching to INV mode for dual-mode evaluation...")
        
        # Switch to INV mode with inv_lambda_
        original_approx = probe_model.approximate_ntk
        probe_model.approximate_ntk = "inv"
        probe_model.set_inv_params(lam=inv_lambda_)
        
        # Top removal (INV)
        print("[info] INV - Top removal (removing high Shapley value data)...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(sorted_indices):
                continue
            
            remaining_indices = sorted_indices[k_remove:]
            
            # Reset INV caches before each prediction
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            top_removal_results_inv.append(int(torch.round(acc * 10000).item()))
        
        # Bottom removal (INV)
        print("[info] INV - Bottom removal (removing low Shapley value data)...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(sorted_indices):
                continue
            
            remaining_indices = sorted_indices[:-k_remove] if k_remove > 0 else sorted_indices
            
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            bottom_removal_results_inv.append(int(torch.round(acc * 10000).item()))
        
        # Random removal (INV)
        print("[info] INV - Random removal...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(all_indices):
                continue
            
            indices_to_remove = np.random.choice(all_indices, size=k_remove, replace=False)
            remaining_indices = np.setdiff1d(all_indices, indices_to_remove)
            
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            random_removal_results_inv.append(int(torch.round(acc * 10000).item()))
        
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
    
    # ===== Stage 3: INV only mode predictions =====
    elif approximate == "inv":
        print("[info] Running INV mode predictions...")
        
        # Top removal
        print("[info] INV - Top removal (removing high Shapley value data)...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(sorted_indices):
                continue
            
            remaining_indices = sorted_indices[k_remove:]
            
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            top_removal_results.append(int(torch.round(acc * 10000).item()))
        
        # Bottom removal
        print("[info] INV - Bottom removal (removing low Shapley value data)...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(sorted_indices):
                continue
            
            remaining_indices = sorted_indices[:-k_remove] if k_remove > 0 else sorted_indices
            
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            bottom_removal_results.append(int(torch.round(acc * 10000).item()))
        
        # Random removal
        print("[info] INV - Random removal...")
        for remove_pct in num_train_selected_list:
            k_remove = int(num_train_dp * remove_pct / 100)
            if k_remove >= len(all_indices):
                continue
            
            indices_to_remove = np.random.choice(all_indices, size=k_remove, replace=False)
            remaining_indices = np.setdiff1d(all_indices, indices_to_remove)
            
            probe_model.pre_inv = None
            probe_model.kr_model = None
            
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(remaining_indices, dtype=int),
                test_set=val_set,
            )
            random_removal_results.append(int(torch.round(acc * 10000).item()))
    
    # ===== Print results =====
    print("\n" + "="*80)
    print("DATA REMOVAL RESULTS")
    print("="*80 + "\n")
    
    if approximate == "eigen":
        print(f"eigen mode lambda={eigen_lambda_:.0e}")
        print(f"top removal (removing high Shapley):")
        print(f"{top_removal_results_eigen}")
        print(f"bottom removal (removing low Shapley):")
        print(f"{bottom_removal_results_eigen}")
        print(f"random:")
        print(f"{random_removal_results_eigen}\n")
        
        print(f"inv mode lambda={inv_lambda_:.0e}")
        print(f"top removal (removing high Shapley):")
        print(f"{top_removal_results_inv}")
        print(f"bottom removal (removing low Shapley):")
        print(f"{bottom_removal_results_inv}")
        print(f"random:")
        print(f"{random_removal_results_inv}")
    elif approximate == "inv":
        print(f"inv mode lambda={inv_lambda_:.0e}")
        print(f"top removal (removing high Shapley):")
        print(f"{top_removal_results}")
        print(f"bottom removal (removing low Shapley):")
        print(f"{bottom_removal_results}")
        print(f"random:")
        print(f"{random_removal_results}")
    
    print("\n" + "="*80)
    
    # ===== Save predictions to txt file =====
    predictions_filename = shapley_path.split('/')[-1].replace('.pkl', '_predictions.txt')
    predictions_txt_path = f"{data_removing_base_path}/shapley/{method_dir}/predictions/{predictions_filename}"
    os.makedirs(os.path.dirname(predictions_txt_path), exist_ok=True)
    
    with open(predictions_txt_path, 'a') as f:
        if approximate == "inv":
            f.write(f"\n{'='*80}\n")
            f.write(f"dataset: {dataset_name}\n")
            f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"\n")
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top removal (removing high Shapley):\n")
            f.write(f"{top_removal_results}\n")
            f.write(f"bottom removal (removing low Shapley):\n")
            f.write(f"{bottom_removal_results}\n")
            f.write(f"random:\n")
            f.write(f"{random_removal_results}\n")
            
        elif approximate == "eigen":
            f.write(f"\n{'='*80}\n")
            f.write(f"eigen rank: {eigen_rank_pct}% (actual: {eigen_rank})\n")
            f.write(f"dataset: {dataset_name}\n")
            f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"solver: {eigen_solver}, dtype: {eigen_dtype}\n")
            f.write(f"\n")
            f.write(f"eigen mode lambda={eigen_lambda_:.0e}\n")
            f.write(f"top removal (removing high Shapley):\n")
            f.write(f"{top_removal_results_eigen}\n")
            f.write(f"bottom removal (removing low Shapley):\n")
            f.write(f"{bottom_removal_results_eigen}\n")
            f.write(f"random:\n")
            f.write(f"{random_removal_results_eigen}\n")
            f.write(f"\n")
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top removal (removing high Shapley):\n")
            f.write(f"{top_removal_results_inv}\n")
            f.write(f"bottom removal (removing low Shapley):\n")
            f.write(f"{bottom_removal_results_inv}\n")
            f.write(f"random:\n")
            f.write(f"{random_removal_results_inv}\n")
    
    print(f"[info] saved predictions to {predictions_txt_path}")


if __name__ == "__main__":
    main()
