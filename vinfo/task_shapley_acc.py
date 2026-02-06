import pickle
import numpy as np
import yaml
import torch
import random
from datasets import load_dataset

import sys, os
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *
from probe import *
from dvutils.Data_Shapley import Fast_Data_Shapley  # YAML tag 해석용

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mnli")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_train_dp", type=int, default=1000)
    parser.add_argument("--val_sample_num", type=int, default=1000)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--tmc_seed", type=int, default=2023)
    parser.add_argument("--approximate", type=str, default="inv",
                        choices=["inv", "eigen", "none"])
    parser.add_argument("--eigen_rank", type=int, default=30,
                        help="Eigen rank as percentage of num_train_dp (e.g., 10 means 10% of data)")
    parser.add_argument("--lambda_", type=float, default=1e-6,
                        help="Lambda (regularization parameter) for eigen regression")
    parser.add_argument("--num_train_selected", type=int, default=100)
    parser.add_argument("--num_train_selected_list", type=int, nargs='+', 
                        default=[i for i in range(1, 101)],
                        help="List of percentages (1-100) of num_train_dp to use as training data")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num

    tmc_iter = args.tmc_iter
    tmc_seed = args.tmc_seed

    approximate = args.approximate
    num_train_selected = args.num_train_selected
    num_train_selected_list = args.num_train_selected_list
    eigen_rank_pct = args.eigen_rank  # Now interpreted as percentage of num_dp
    lambda_ = args.lambda_
    
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
            lam=lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed
        )
    elif approximate == "inv":
        probe_model.set_inv_params(lam=lambda_)

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
        probe_model.prepare_eigen_regression()

    print("len(train_set) =", len(train_set))
    print("len(val_set)   =", len(val_set))

    # ===== 3) Shapley 결과 경로 =====
    method_dir = approximate  # 'inv' or 'eigen'
    
    # Format lambda in scientific notation for filename (always use 1e-X format)
    lambda_str = f"{lambda_:.0e}"
    
    if approximate == "eigen":
        # Use the percentage value itself (eigen_rank_pct) in filename
        extra_tag = f"_eig{eigen_rank_pct}_lam{lambda_str}_{eigen_solver}_{eigen_dtype}"
    else:
        extra_tag = f"_lam{lambda_str}"

    shapley_path = (
        f"{base_path}/shapley/{method_dir}/results/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_seed}_iter{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")

    # ===== 4) Shapley 로드 or 계산 =====
    try:
        with open(shapley_path, "rb") as f:
            result = pickle.load(f)
        print(f"[info] loaded Shapley from {shapley_path}")

        dv_result = result["dv_result"]
        if "sampled_idx" in result:
            sampled_idx = np.array(result["sampled_idx"])
        if "sampled_val_idx" in result:
            sampled_val_idx = np.array(result["sampled_val_idx"])

    except Exception as e:
        print("[info] shapley cache miss -> computing Shapley")
        print(f"[info] reason: {e}")

        dv_result = dshap_com.run(
            data_idx=sampled_idx.tolist(),
            val_data_idx=sampled_val_idx.tolist(),
            iteration=tmc_iter,
            use_cache_ntk=True,
            prompt=prompt,
            seed=tmc_seed,
            num_dp=num_train_dp,
            checkpoint=False,
            per_point=per_point,
            early_stopping=early_stopping
        )

        result = {
            "dv_result": dv_result,
            "sampled_idx": sampled_idx,
            "sampled_val_idx": sampled_val_idx,
            "args": vars(args),
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
    with open(indices_txt_path, 'w') as f:
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
        bottom_results = []
        random_results = []
        all_results = []
    elif approximate == "eigen":
        top_results = []
        bottom_results = []
        random_results = []

    for num_train_selected_pct in num_train_selected_list:
        # Convert percentage to actual number of samples
        k = int(num_train_dp * num_train_selected_pct / 100)
        k = int(min(k, len(acc_sum_per_train)))
        if k <= 0:
            print(f"Skipping num_train_selected_pct={num_train_selected_pct}% (k={k}, invalid value)")
            continue

        # Reset cached models to prevent dimension mismatch across iterations
        probe_model.kr_model = None
        probe_model.pre_inv = None
        if hasattr(probe_model, 'eigen_regression'):
            probe_model.eigen_regression = None

        # Prepare three types of data selection
        top_indices = sorted_indices[:k]
        bottom_indices = sorted_indices[-k:]
        # Reset RNG for each num_train_selected to ensure reproducibility
        rng = np.random.RandomState(seed)
        random_indices = rng.choice(len(sampled_idx), size=k, replace=False)
        
        if approximate == "inv":
            # INV: 4 evaluations (top, bottom, random, all)
            for data_mode, indices, result_list in [
                ("top", top_indices, top_results), 
                # ("bottom", bottom_indices, bottom_results), 
                ("random", random_indices, random_results), 
                # ("all", all_indices, all_results)
            ]:
                # Reset pre_inv before each kernel_regression to prevent dimension mismatch
                probe_model.pre_inv = None
                _, acc = probe_model.kernel_regression(
                    train_indices=np.array(indices, dtype=int),
                    test_set=val_set,
                )
                result_list.append(int(torch.round(acc * 10000).item()))
            
        elif approximate == "eigen":
            # EIGEN: 3 evaluations (top, bottom, random) with eigen_mode="top" only
            probe_model.eigen_decom_mode = "top"
            for data_mode, indices, result_list in [
                ("top", top_indices, top_results), 
                # ("bottom", bottom_indices, bottom_results), 
                ("random", random_indices, random_results)
            ]:
                probe_model.eigen_regression = None  # Reset cache
                
                _, acc = probe_model.kernel_regression(
                    train_indices=np.array(indices, dtype=int),
                    test_set=val_set,
                )
                result_list.append(int(torch.round(acc * 10000).item()))
    
    # Print results by data_mode
    if approximate == "inv":
        print(f"top: {top_results}")
        # print(f"bottom: {bottom_results}")
        print(f"random: {random_results}")
        # print(f"all: {all_results}")
    elif approximate == "eigen":
        print(f"top: {top_results}")
        # print(f"bottom: {bottom_results}")
        print(f"random: {random_results}")
if __name__ == "__main__":
    main()
