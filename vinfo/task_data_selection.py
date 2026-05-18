import pickle
import numpy as np
import yaml
import torch
import random
import time

import sys, os
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *
from probe import *
from dvutils.Data_Shapley import Fast_Data_Shapley  # YAML tag 해석용

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mr")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_train_dp", type=int, default=8530)
    parser.add_argument("--val_sample_num", type=int, default=1066)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="inv",
                        choices=["inv", "eigen", "none"])
    parser.add_argument("--eigen_rank", type=float, default=30,
                        help="Eigen rank as percentage of num_train_dp (e.g., 10 means 10% of data)")
    parser.add_argument("--inv_lambda_", type=float, default=1e-6,
                        help="Lambda (regularization parameter) for INV mode")
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2,
                        help="Lambda (regularization parameter) for Eigen mode")
    parser.add_argument("--num_train_selected_list", type=int, nargs='+', 
                        default=[i for i in range(1, 101)],
                        help="List of percentages (1-100) of num_train_dp to use as training data")
    parser.add_argument("--config", type=str, default="ntk_prompt",
                        help="YAML config name without .yaml extension (e.g., ntk_prompt, ntk_llama)")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num

    tmc_iter = args.tmc_iter

    approximate = args.approximate
    num_train_selected_list = args.num_train_selected_list
    eigen_rank_pct = args.eigen_rank
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_
    
    # Calculate actual eigen rank from percentage
    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% of num_dp={num_train_dp} -> actual rank={eigen_rank}")

    prompt = True
    signgd = False

    # eigen 파라미터
    eigen_solver = "cholesky"
    eigen_dtype = "float32"

    early_stopping = "True"

    # Dynamic path construction based on dataset_name
    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"

    # seed 고정 (재현성 보장)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # ===== YAML 로드 =====
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    probe_model = yaml_args["probe_com"]

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

    # ===== 1) Shapley pkl 경로 구성 및 로드 =====
    method_dir = approximate  # 'inv' or 'eigen'

    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_invlam{inv_lam_str}_{eigen_solver}_{eigen_dtype}"
    else:
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}"

    shapley_base = f"./freeshap_res/shapley/{dataset_name}"
    shapley_path = (
        f"{shapley_base}/{method_dir}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")

    with open(shapley_path, "rb") as f:
        result = pickle.load(f)
    print(f"[info] loaded Shapley from {shapley_path}")

    dv_result = np.array(result["dv_result"])
    sampled_idx = np.array(result["sampled_idx"])
    sampled_val_idx = np.array(result["sampled_val_idx"])
    timing_info = result.get("timing_info", {})
    early_stopping_records = result.get("early_stopping_records", None)

    print("dv_result shape:", dv_result.shape)
    print(f"[info] |train|={len(sampled_idx)}, |val|={len(sampled_val_idx)}")

    # ===== 2) NTK 캐시 로드 =====
    ntk_path = (
        f"./freeshap_res/ntk/{dataset_name}/{model_name}"
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
    probe_model.get_cached_ntk(ntk)

    # ===== 3) train/val set 구성 =====
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    # eigen이면 feature 준비
    if approximate == "eigen":
        print("[info] Preparing eigen regression features...")
        probe_model.prepare_eigen_regression()

    print("len(train_set) =", len(train_set))
    print("len(val_set)   =", len(val_set))

    # ===== 4) Shapley 정렬 및 indices 저장 =====
    acc_contrib = dv_result[:, 1, :]            # (num_train_dp, val_sample_num)
    acc_sum_per_train = acc_contrib.sum(axis=1)  # (num_train_dp,)

    sorted_indices = np.argsort(acc_sum_per_train)[::-1]  # 큰 값이 앞
    all_indices = np.arange(len(sampled_idx))

    # 출력 경로: freeshap_res/data_selection/{dataset}/{method}/
    ds_base = f"./freeshap_res/data_selection/{dataset_name}"
    setting_name = os.path.basename(shapley_path).replace('.pkl', '')

    # Save sorted indices (original dataset indices)
    indices_txt_path = f"{ds_base}/{method_dir}/indices/{setting_name}_indices.txt"
    original_indices = sampled_idx[sorted_indices]
    os.makedirs(os.path.dirname(indices_txt_path), exist_ok=True)
    with open(indices_txt_path, 'a') as f:
        for idx in original_indices:
            f.write(f"{idx}\n")
    print(f"[info] saved sorted indices to {indices_txt_path}")

    # ===== 5) kernel_regression 평가 =====
    if approximate == "inv":
        top_results = []
        random_results = []
    elif approximate == "eigen":
        top_results_eigen = []
        random_results_eigen = []
        top_results_inv = []
        random_results_inv = []

    all_indices = np.arange(len(sampled_idx))

    if approximate == "inv":
        print("[info] Running INV mode predictions (top)...")
        for num_train_selected_pct in num_train_selected_list:
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                print(f"Skipping num_train_selected_pct={num_train_selected_pct}% (k={k}, invalid value)")
                continue

            top_indices = sorted_indices[:k]
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

            random_indices = np.random.choice(all_indices, size=k, replace=False)
            probe_model.pre_inv = None
            probe_model.kr_model = None

            _, acc = probe_model.kernel_regression(
                train_indices=np.array(random_indices, dtype=int),
                test_set=val_set,
            )
            random_results.append(int(torch.round(acc * 10000).item()))

    elif approximate == "eigen":
        probe_model.eigen_decom_mode = "top"

        print("[info] Running EIGEN mode predictions (top)...")
        for num_train_selected_pct in num_train_selected_list:
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                print(f"Skipping num_train_selected_pct={num_train_selected_pct}% (k={k}, invalid value)")
                continue

            top_indices = sorted_indices[:k]
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

            random_indices = np.random.choice(all_indices, size=k, replace=False)
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(random_indices, dtype=int),
                test_set=val_set,
            )
            random_results_eigen.append(int(torch.round(acc * 10000).item()))

        # INV mode predictions (dual-mode evaluation)
        print("[info] Switching to INV mode for dual-mode evaluation...")
        original_approx = probe_model.approximate_ntk
        probe_model.approximate_ntk = "inv"
        probe_model.set_inv_params(lam=inv_lambda_)

        print("[info] Running INV mode predictions (top)...")
        for num_train_selected_pct in num_train_selected_list:
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                continue

            top_indices = sorted_indices[:k]
            probe_model.pre_inv = None
            probe_model.kr_model = None

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

            random_indices = np.random.choice(all_indices, size=k, replace=False)
            probe_model.pre_inv = None
            probe_model.kr_model = None

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

    # Print results
    if approximate == "inv":
        print(f"\n[INV mode with lambda={inv_lambda_}]")
        print(f"top: {top_results}")
        print(f"random:\n{random_results}")
    elif approximate == "eigen":
        print(f"\n[INV mode with lambda={inv_lambda_}]")
        print(f"top: {top_results_inv}")
        print(f"random:\n{random_results_inv}")
        print(f"\n[Eigen mode with lambda={eigen_lambda_}]")
        print(f"top: {top_results_eigen}")
        print(f"random:\n{random_results_eigen}")

    # ===== 6) predictions.txt 저장 =====
    predictions_txt_path = f"{ds_base}/{method_dir}/predictions/{setting_name}_predictions.txt"
    os.makedirs(os.path.dirname(predictions_txt_path), exist_ok=True)

    with open(predictions_txt_path, 'a') as f:
        f.write(f"{'='*80}\n")
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
        f.write(f"seed: {seed}\n")

        if approximate == "eigen":
            f.write(f"eigen rank: {eigen_rank_pct}% (actual: {eigen_rank})\n")
            f.write(f"solver: {eigen_solver}, dtype: {eigen_dtype}\n")

        f.write(f"\n")

        if approximate == "inv":
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n{top_results}\n")
            f.write(f"random:\n{random_results}\n")
        elif approximate == "eigen":
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n{top_results_inv}\n")
            f.write(f"random:\n{random_results_inv}\n")
            f.write(f"\n")
            f.write(f"eigen mode lambda={eigen_lambda_:.0e}\n")
            f.write(f"top:\n{top_results_eigen}\n")
            f.write(f"random:\n{random_results_eigen}\n")

        # Timing info from pkl
        if timing_info:
            f.write(f"\n{'='*80}\n")
            f.write(f"timing info (from shapley pkl):\n")
            if 'gpu_info' in timing_info:
                for key, value in timing_info['gpu_info'].items():
                    f.write(f"  {key}: {value}\n")
            for key, value in timing_info.items():
                if key != 'gpu_info':
                    f.write(f"  {key}: {value:.4f}s\n")

        # Early stopping info from pkl
        if early_stopping_records:
            records = early_stopping_records
            total_count = len(records)
            early_stopped_count = sum(1 for r in records if r['status'] == 'Early Stop')
            completed_count = sum(1 for r in records if r['status'] == 'Complete')
            percentages = [r['percentage'] for r in records]

            f.write(f"\n{'='*80}\n")
            f.write(f"early stopping statistics:\n")
            f.write(f"  total iterations: {total_count}\n")
            f.write(f"  early stopped: {early_stopped_count} ({early_stopped_count/total_count*100:.1f}%)\n")
            f.write(f"  completed: {completed_count} ({completed_count/total_count*100:.1f}%)\n")

            if percentages:
                avg_pct = np.mean(percentages)
                f.write(f"  average stop point: {avg_pct:.2f}%\n")

            # Distribution
            f.write(f"\n  distribution:\n")
            bins = [(i, i+10) for i in range(0, 100, 10)]
            bin_counts = [0] * len(bins)
            for pct in percentages:
                bin_idx = min(int(pct // 10), 9)
                bin_counts[bin_idx] += 1
            max_count = max(bin_counts) if bin_counts else 1
            for i, (start, end) in enumerate(bins):
                count = bin_counts[i]
                pct_of_total = (count / total_count * 100) if total_count > 0 else 0
                bar_length = int((count / max_count) * 40) if max_count > 0 else 0
                bar = '█' * bar_length
                f.write(f"  {start:3d}-{end:3d}%: {count:4d} ({pct_of_total:5.1f}%) {bar}\n")

    print(f"[info] saved predictions to {predictions_txt_path}")

    # ===== 7) base_accuracy 파일 저장 =====
    # Records f=0 (chance) + f∈{1,5,10,15,20,25,30} in the same
    # "label:\nvalue\n" format produced by task_base_accuracy.py, so that
    # notebooks like lc_fc_analysis_f{10,20,30}.ipynb can paste directly.
    #
    # Output paths:
    #   inv   mode  ->  data_selection/{ds}/inv/base_accuracy/{setting}_base.txt
    #   eigen mode  ->  data_selection/{ds}/eigen/base_accuracy/{setting}_base.txt
    #
    # Content for inv mode:
    #   inv_base_accuracy:
    #   <int at f=0>
    #   inv_top_f{1,5,10,15,20,25,30}_accuracy:
    #   <int at f>
    # Content for eigen mode (same eigen-r Shapley ranking, dual-mode
    # prediction — inv block first, eigen block second):
    #   inv_base_accuracy:
    #   <int at f=0>
    #   inv_top_f{...}_accuracy:
    #   <int at f>
    #   eigen_base_accuracy:
    #   <int at f=0>
    #   eigen_top_f{...}_accuracy:
    #   <int at f>
    F_LIST = [1, 5, 10, 15, 20, 25, 30]

    # f=0 chance accuracy: empty train -> zero predictor -> argmax=class 0.
    # Accuracy equals the fraction of val examples whose label is class 0.
    val_labels = torch.tensor([ex['label'] for ex in val_set])
    acc0 = (val_labels == 0).float().mean()
    acc0_int = int(torch.round(acc0 * 10000).item())

    # pct -> index into top_results lists (top_results was appended in the
    # same order as num_train_selected_list, skipping only k<=0 which does
    # not happen for num_train_dp>=100).
    pct_to_idx = {pct: i for i, pct in enumerate(num_train_selected_list)}

    if approximate in ("inv", "eigen"):
        base_dir = f"{ds_base}/{method_dir}/base_accuracy"
        os.makedirs(base_dir, exist_ok=True)
        base_txt_path = f"{base_dir}/{setting_name}_base.txt"

        with open(base_txt_path, 'w') as f:
            if approximate == "inv":
                f.write("inv_base_accuracy:\n")
                f.write(f"{acc0_int}\n")
                for fpct in F_LIST:
                    idx = pct_to_idx.get(fpct)
                    if idx is not None and idx < len(top_results):
                        f.write(f"inv_top_f{fpct}_accuracy:\n")
                        f.write(f"{top_results[idx]}\n")
            else:  # eigen
                # Top block: eigen-Shapley ranking + INV kernel-regression prediction.
                # Bottom block: eigen-Shapley ranking + EIGEN kernel-regression prediction.
                f.write("inv_base_accuracy:\n")
                f.write(f"{acc0_int}\n")
                for fpct in F_LIST:
                    idx = pct_to_idx.get(fpct)
                    if idx is not None and idx < len(top_results_inv):
                        f.write(f"inv_top_f{fpct}_accuracy:\n")
                        f.write(f"{top_results_inv[idx]}\n")
                f.write("eigen_base_accuracy:\n")
                f.write(f"{acc0_int}\n")
                for fpct in F_LIST:
                    idx = pct_to_idx.get(fpct)
                    if idx is not None and idx < len(top_results_eigen):
                        f.write(f"eigen_top_f{fpct}_accuracy:\n")
                        f.write(f"{top_results_eigen[idx]}\n")
        print(f"[info] saved base_accuracy to {base_txt_path}")

if __name__ == "__main__":
    main()
