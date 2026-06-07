"""task_imbalance_data_selection.py — Data selection eval for imbalance run.

Mirror of `task_label_data_selection.py` with method-flag (lrfshap | a1) and
imbalanced NTK / Shapley paths.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import os
import sys
import time
import random
import argparse
import json

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml

import entks.ntk_regression as _ntk_reg

_ORIG_PRECOMPUTE = _ntk_reg.EigenNTKRegression._precompute_eigen_features


def _label_aware_precompute(self, ntk_full):
    if getattr(self, "eigen_decom_mode", "top") != "top":
        return _ORIG_PRECOMPUTE(self, ntk_full)

    t0 = time.time()
    if ntk_full.ndim == 3:
        ntk0 = ntk_full[0].detach()
    else:
        ntk0 = ntk_full.detach()
    ntk0 = ntk0.to("cpu", dtype=torch.float64)
    n_total, n_train = ntk0.shape

    K_trtr = ntk0[:n_train, :].numpy()
    K_tetr = ntk0[n_train:, :].numpy()
    K_sym = 0.5 * (K_trtr + K_trtr.T)

    d = int(min(self.rank, n_train))
    evals, evecs = np.linalg.eigh(K_sym)

    y_int = self.y.detach().cpu().numpy() if torch.is_tensor(self.y) else np.asarray(self.y)
    y_int = y_int.astype(np.int64)
    n = y_int.shape[0]
    C = int(getattr(self, "n_class", None) or (int(y_int.max()) + 1))
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y_int] = 1.0
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    coeffs = evecs.T @ Y_centered
    c2 = (coeffs ** 2).sum(axis=1)
    evals_pos = np.maximum(evals, 0.0)
    rho_filter = float(self.lam)
    filt = (evals_pos / (evals_pos + rho_filter)) ** 2
    score = filt * c2
    idx = np.argsort(score)[::-1][:d]

    lam_kept = evals[idx]
    U_kept = evecs[:, idx]
    lam_kept = np.clip(lam_kept, 1e-8, None)
    Phi_tr = U_kept * np.sqrt(lam_kept)[None, :]
    Phi_te = K_tetr @ (U_kept / np.sqrt(lam_kept)[None, :])

    self.eigen_decomposition_time = time.time() - t0
    phi_tr = torch.from_numpy(Phi_tr).to(device=self.device, dtype=self.dtype)
    phi_te = torch.from_numpy(Phi_te).to(device=self.device, dtype=self.dtype)
    return phi_tr, phi_te


sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *           # noqa: E402,F401,F403
from probe import *             # noqa: E402,F401,F403
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: E402,F401


IMBALANCE_ROOT = "./freeshap_res/claude_research/data_selection_test/imbalance"


def parse_ratios(pos_ratio, class_ratios_str):
    if class_ratios_str is not None:
        ratios = [float(x) for x in class_ratios_str.split(",")]
    elif pos_ratio is not None:
        ratios = [1.0 - pos_ratio, pos_ratio]
    else:
        raise ValueError("Need --pos_ratio or --class_ratios.")
    return ratios


def ratio_tag(class_ratios):
    if len(class_ratios) == 2:
        return f"pos{int(round(class_ratios[1] * 100)):02d}"
    return "cls" + "_".join(f"{int(round(r * 100)):02d}" for r in class_ratios)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["sst2", "mr", "qqp", "mnli", "ag_news", "mrpc", "rte"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_train_dp", type=int, default=2000)
    parser.add_argument("--val_sample_num", type=int, required=True)
    parser.add_argument("--pos_ratio", type=float, default=None)
    parser.add_argument("--class_ratios", type=str, default=None)
    parser.add_argument("--val_balance", action="store_true",
                        help="Must match the value used for task_imbalance_ntk.py.")
    parser.add_argument("--val_pos_ratio", type=float, default=None,
                        help="Must match the value used for task_imbalance_ntk.py "
                             "(binary-only imbalanced val).")
    parser.add_argument("--val_class_ratios", type=str, default=None,
                        help="Must match the value used for task_imbalance_ntk.py "
                             "(multi-class imbalanced val).")
    parser.add_argument("--method", type=str, required=True,
                        choices=["lrfshap", "a1"])
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="eigen",
                        choices=["inv", "eigen", "nystrom"])
    parser.add_argument("--eigen_rank", type=float, default=10)
    parser.add_argument("--inv_lambda_", type=float, default=1e-6)
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2)
    parser.add_argument("--nystrom_d", type=int, default=512)
    parser.add_argument("--landmark_seed", type=int, default=1234)
    parser.add_argument("--nystrom_lambda_", type=float, default=1e-2)
    parser.add_argument("--num_train_selected_list", type=int, nargs='+',
                        default=[i for i in range(1, 101)])
    parser.add_argument("--config", type=str, default="ntk_prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.method == "a1":
        _ntk_reg.EigenNTKRegression._precompute_eigen_features = _label_aware_precompute

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    class_ratios = parse_ratios(args.pos_ratio, args.class_ratios)
    tmc_iter = args.tmc_iter
    approximate = args.approximate
    num_train_selected_list = args.num_train_selected_list
    eigen_rank_pct = args.eigen_rank
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_

    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)

    prompt = True
    signgd = False
    eigen_solver = "cholesky"
    eigen_dtype = "float32"
    early_stopping = "True"

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"
    rtag = ratio_tag(class_ratios)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    probe_model = yaml_args["probe_com"]

    if prompt:
        probe_model.model.init(list_dataset.label_word_list)

    probe_model.approximate(approximate)
    if approximate == "eigen":
        probe_model.set_eigen_params(
            rank=eigen_rank, lam=eigen_lambda_,
            solver=eigen_solver, dtype=eigen_dtype, seed=seed,
        )
    elif approximate == "nystrom":
        probe_model.set_nystrom_params(
            d=int(args.nystrom_d), lam=float(args.nystrom_lambda_),
            solver=eigen_solver, dtype=eigen_dtype,
            landmark_seed=int(args.landmark_seed), jitter=1e-8,
        )
    else:
        probe_model.set_inv_params(lam=inv_lambda_)

    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    method_dir = approximate
    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (
            f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_invlam{inv_lam_str}"
            f"_{eigen_solver}_{eigen_dtype}_{args.method}"
        )
    elif approximate == "nystrom":
        nys_lam_str = f"{float(args.nystrom_lambda_):.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (
            f"_nys{int(args.nystrom_d)}_landseed{int(args.landmark_seed)}"
            f"_nyslam{nys_lam_str}_invlam{inv_lam_str}"
            f"_{eigen_solver}_{eigen_dtype}_{args.method}"
        )
    else:
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}_{args.method}"

    if args.val_pos_ratio is not None:
        val_tag = f"valimb{val_sample_num}_pos{int(round(args.val_pos_ratio * 100)):02d}"
    elif args.val_class_ratios is not None:
        parts = [int(round(float(r) * 100)) for r in args.val_class_ratios.split(",")]
        val_tag = f"valimb{val_sample_num}_cls" + "_".join(f"{p:02d}" for p in parts)
    elif args.val_balance:
        val_tag = f"valbal{val_sample_num}"
    else:
        val_tag = f"val{val_sample_num}"
    shapley_base = f"{IMBALANCE_ROOT}/shapley/{dataset_name}/{rtag}/{args.method}"
    shapley_path = (
        f"{shapley_base}/{method_dir}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_{val_tag}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")

    with open(shapley_path, "rb") as f:
        result = pickle.load(f)
    dv_result = np.array(result["dv_result"])
    sampled_idx = np.array(result["sampled_idx"])
    sampled_val_idx = np.array(result["sampled_val_idx"])
    timing_info = result.get("timing_info", {})
    early_stopping_records = result.get("early_stopping_records", None)

    # Imbalance NTK
    ntk_path = (
        f"./freeshap_res/claude_research/imbalance_ntk/{dataset_name}/{rtag}/"
        f"{model_name}_seed{seed}_num{num_train_dp}_{val_tag}"
        f"_sign{signgd}.pkl"
    )
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    ntk = bundle["ntk"]
    probe_model.get_cached_ntk(ntk)

    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)
    if approximate == "eigen":
        probe_model.prepare_eigen_regression()
    elif approximate == "nystrom":
        probe_model.prepare_nystrom_regression()

    acc_contrib = dv_result[:, 1, :]
    acc_sum_per_train = acc_contrib.sum(axis=1)
    sorted_indices = np.argsort(acc_sum_per_train)[::-1]
    all_indices = np.arange(len(sampled_idx))

    ds_base = f"{IMBALANCE_ROOT}/data_selection/{dataset_name}/{rtag}/{args.method}"
    setting_name = os.path.basename(shapley_path).replace('.pkl', '')

    indices_txt_path = f"{ds_base}/{method_dir}/indices/{setting_name}_indices.txt"
    os.makedirs(os.path.dirname(indices_txt_path), exist_ok=True)
    with open(indices_txt_path, 'a') as f:
        for idx in sampled_idx[sorted_indices]:
            f.write(f"{idx}\n")

    # ----- val labels (used for balanced accuracy) -----
    val_labels_full = torch.tensor([ex['label'] for ex in val_set]).long()
    val_n_class = int(val_labels_full.max().item()) + 1
    # cache per-class mask + count to avoid recomputation
    val_class_masks = [(val_labels_full == c) for c in range(val_n_class)]
    val_class_counts = [int(m.sum().item()) for m in val_class_masks]

    def _balanced_acc_int(per_point_correct):
        # per_point_correct: float tensor on cpu, shape (N_val,), 1.0 if correct else 0.0
        per_class_recall = []
        for c in range(val_n_class):
            m = val_class_masks[c]
            cnt = val_class_counts[c]
            if cnt == 0:
                continue
            per_class_recall.append(per_point_correct[m].sum().item() / cnt)
        if len(per_class_recall) == 0:
            return 0
        bal_acc = sum(per_class_recall) / len(per_class_recall)
        return int(round(bal_acc * 10000))

    if approximate == "inv":
        top_results = []
        random_results = []
        top_results_balanced = []
        random_results_balanced = []
    else:
        top_results_eigen = []
        random_results_eigen = []
        top_results_inv = []
        random_results_inv = []
        top_results_eigen_balanced = []
        random_results_eigen_balanced = []
        top_results_inv_balanced = []
        random_results_inv_balanced = []

    def _kr(indices):
        probe_model.pre_inv = None
        probe_model.kr_model = None
        # per_point=True returns (loss_per_point[np], acc_per_point[torch.float])
        _, acc_per_point = probe_model.kernel_regression(
            train_indices=np.array(indices, dtype=int),
            test_set=val_set,
            per_point=True,
        )
        acc_per_point_cpu = acc_per_point.detach().to('cpu').float()
        naive_acc = acc_per_point_cpu.mean()
        naive_acc_int = int(torch.round(naive_acc * 10000).item())
        balanced_acc_int = _balanced_acc_int(acc_per_point_cpu)
        return naive_acc_int, balanced_acc_int

    if approximate == "inv":
        print("[info] INV mode predictions ...")
        for pct in num_train_selected_list:
            k = int(min(num_train_dp * pct / 100, len(acc_sum_per_train)))
            if k <= 0:
                continue
            n_acc, b_acc = _kr(sorted_indices[:k])
            top_results.append(n_acc)
            top_results_balanced.append(b_acc)
        for pct in num_train_selected_list:
            k = int(min(num_train_dp * pct / 100, len(acc_sum_per_train)))
            if k <= 0:
                continue
            n_acc, b_acc = _kr(np.random.choice(all_indices, size=k, replace=False))
            random_results.append(n_acc)
            random_results_balanced.append(b_acc)
    else:
        # eigen and nystrom share the same downstream evaluation path:
        # low-rank approximation prediction (saved as *_results_eigen*) then
        # dual evaluation with full-kernel INV (saved as *_results_inv*).
        if approximate == "eigen":
            probe_model.eigen_decom_mode = "top"
        approx_label = "EIGEN" if approximate == "eigen" else "NYSTROM"
        print(f"[info] {approx_label} mode predictions ({args.method}) ...")
        for pct in num_train_selected_list:
            k = int(min(num_train_dp * pct / 100, len(acc_sum_per_train)))
            if k <= 0:
                continue
            n_acc, b_acc = _kr(sorted_indices[:k])
            top_results_eigen.append(n_acc)
            top_results_eigen_balanced.append(b_acc)
        for pct in num_train_selected_list:
            k = int(min(num_train_dp * pct / 100, len(acc_sum_per_train)))
            if k <= 0:
                continue
            n_acc, b_acc = _kr(np.random.choice(all_indices, size=k, replace=False))
            random_results_eigen.append(n_acc)
            random_results_eigen_balanced.append(b_acc)

        # Switch to INV mode for dual-axis evaluation
        print("[info] Switching to INV mode for dual evaluation ...")
        original_approx = probe_model.approximate_ntk
        probe_model.approximate_ntk = "inv"
        probe_model.set_inv_params(lam=inv_lambda_)
        for pct in num_train_selected_list:
            k = int(min(num_train_dp * pct / 100, len(acc_sum_per_train)))
            if k <= 0:
                continue
            n_acc, b_acc = _kr(sorted_indices[:k])
            top_results_inv.append(n_acc)
            top_results_inv_balanced.append(b_acc)
        for pct in num_train_selected_list:
            k = int(min(num_train_dp * pct / 100, len(acc_sum_per_train)))
            if k <= 0:
                continue
            n_acc, b_acc = _kr(np.random.choice(all_indices, size=k, replace=False))
            random_results_inv.append(n_acc)
            random_results_inv_balanced.append(b_acc)
        probe_model.approximate_ntk = original_approx
        if approximate == "eigen":
            probe_model.set_eigen_params(
                rank=eigen_rank, lam=eigen_lambda_,
                solver=eigen_solver, dtype=eigen_dtype, seed=seed,
            )
        elif approximate == "nystrom":
            probe_model.set_nystrom_params(
                d=int(args.nystrom_d), lam=float(args.nystrom_lambda_),
                solver=eigen_solver, dtype=eigen_dtype,
                landmark_seed=int(args.landmark_seed), jitter=1e-8,
            )

    # ----- predictions txt + sidecar -----
    predictions_txt_path = f"{ds_base}/{method_dir}/predictions/{setting_name}_predictions.txt"
    os.makedirs(os.path.dirname(predictions_txt_path), exist_ok=True)
    with open(predictions_txt_path, 'a') as f:
        f.write(f"{'='*80}\n")
        f.write(f"dataset: {dataset_name}, class_ratios: {class_ratios}, method: {args.method}\n")
        f.write(f"train: {num_train_dp}, val: {val_sample_num}, seed: {seed}\n")
        if approximate in ("eigen", "nystrom"):
            if approximate == "eigen":
                f.write(f"eigen rank: {eigen_rank_pct}% (actual: {eigen_rank})\n\n")
                f.write(f"eigen mode lambda={eigen_lambda_:.0e}\n")
            else:
                f.write(f"nystrom d: {int(args.nystrom_d)}, landmark_seed: {int(args.landmark_seed)}\n\n")
                f.write(f"nystrom mode lambda={float(args.nystrom_lambda_):.0e}\n")
            f.write(f"top:\n{top_results_eigen}\n")
            f.write(f"random:\n{random_results_eigen}\n")
            f.write(f"top_balanced:\n{top_results_eigen_balanced}\n")
            f.write(f"random_balanced:\n{random_results_eigen_balanced}\n\n")
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n{top_results_inv}\n")
            f.write(f"random:\n{random_results_inv}\n")
            f.write(f"top_balanced:\n{top_results_inv_balanced}\n")
            f.write(f"random_balanced:\n{random_results_inv_balanced}\n")
        else:
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n{top_results}\n")
            f.write(f"random:\n{random_results}\n")
            f.write(f"top_balanced:\n{top_results_balanced}\n")
            f.write(f"random_balanced:\n{random_results_balanced}\n")

    F_LIST = [1, 5, 10, 15, 20, 25, 30]
    # val_labels_full already computed above
    acc0 = (val_labels_full == 0).float().mean()
    acc0_int = int(torch.round(acc0 * 10000).item())
    pct_to_idx = {pct: i for i, pct in enumerate(num_train_selected_list)}

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
                    f.write(f"inv_top_f{fpct}_accuracy:\n{top_results[idx]}\n")
        else:
            f.write("inv_base_accuracy:\n")
            f.write(f"{acc0_int}\n")
            for fpct in F_LIST:
                idx = pct_to_idx.get(fpct)
                if idx is not None and idx < len(top_results_inv):
                    f.write(f"inv_top_f{fpct}_accuracy:\n{top_results_inv[idx]}\n")
            f.write("eigen_base_accuracy:\n")
            f.write(f"{acc0_int}\n")
            for fpct in F_LIST:
                idx = pct_to_idx.get(fpct)
                if idx is not None and idx < len(top_results_eigen):
                    f.write(f"eigen_top_f{fpct}_accuracy:\n{top_results_eigen[idx]}\n")

    sidecar = {
        "dataset_name": dataset_name,
        "class_ratios": class_ratios,
        "ratio_tag": rtag,
        "method": args.method,
        "seed": seed,
        "num_train_dp": num_train_dp,
        "val_sample_num": val_sample_num,
        "tmc_iter": tmc_iter,
        "approximate": approximate,
        "inv_lambda_": inv_lambda_,
        "eigen_lambda_": eigen_lambda_ if approximate == "eigen" else None,
        "eigen_rank_pct": eigen_rank_pct if approximate == "eigen" else None,
        "nystrom_d": int(args.nystrom_d) if approximate == "nystrom" else None,
        "landmark_seed": int(args.landmark_seed) if approximate == "nystrom" else None,
        "nystrom_lambda_": float(args.nystrom_lambda_) if approximate == "nystrom" else None,
        "setting_name": setting_name,
        "num_train_selected_list": list(num_train_selected_list),
        "acc_at_f0": acc0_int,
    }
    if approximate == "inv":
        sidecar["top_results_inv"] = list(top_results)
        sidecar["random_results_inv"] = list(random_results)
        sidecar["top_results_inv_balanced"] = list(top_results_balanced)
        sidecar["random_results_inv_balanced"] = list(random_results_balanced)
    else:
        sidecar["top_results_eigen"] = list(top_results_eigen)
        sidecar["random_results_eigen"] = list(random_results_eigen)
        sidecar["top_results_inv"] = list(top_results_inv)
        sidecar["random_results_inv"] = list(random_results_inv)
        sidecar["top_results_eigen_balanced"] = list(top_results_eigen_balanced)
        sidecar["random_results_eigen_balanced"] = list(random_results_eigen_balanced)
        sidecar["top_results_inv_balanced"] = list(top_results_inv_balanced)
        sidecar["random_results_inv_balanced"] = list(random_results_inv_balanced)

    sidecar_dir = f"{ds_base}/{method_dir}/sidecar"
    os.makedirs(sidecar_dir, exist_ok=True)
    sidecar_path = f"{sidecar_dir}/{setting_name}.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[done] sidecar saved -> {sidecar_path}")


if __name__ == "__main__":
    main()
