"""task_label_data_selection.py — Data selection evaluation for A1 Shapley.

Mirror of `vinfo/task_data_selection.py` with these differences:

1. Same monkey-patch on `entks.ntk_regression.EigenNTKRegression._precompute_eigen_features`
   as in `task_label_shapley.py`, so that any rank-r eigendecomp performed during
   evaluation (e.g., inside `probe_model.kernel_regression(top_indices, val_set)`)
   uses the A1 supervised selection criterion. This keeps the truncation rule
   consistent between Shapley computation and prediction-time evaluation.

2. Reads Shapley pickles from

       ./freeshap_res/claude_research/data_selection_test/shapley/{dataset}/{method}/

   and writes downstream outputs (indices/, predictions/, base_accuracy/) under

       ./freeshap_res/claude_research/data_selection_test/data_selection/{dataset}/{method}/

   so the entire A1 experimental output tree lives under `claude_research/`.

3. Appends `_label` to `extra_tag` so the loaded shapley pkl filename matches
   what `task_label_shapley.py` wrote.

Invoke from `vinfo/` directory just like `task_data_selection.py`:

    python ./freeshap_res/claude_research/experiments/task_label_data_selection.py \
        --seed 2026 --dataset_name rte --num_train_dp 2490 \
        --val_sample_num 277 --approximate eigen --eigen_rank 10 \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
"""

import pickle
import os
import sys
import time
import random
import argparse

# Ensure cwd (expected to be vinfo/) is on sys.path so the upstream modules
# are importable. See task_label_shapley.py for the same fix.
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml

# ----- monkey-patch BEFORE importing probe -----
import entks.ntk_regression as _ntk_reg

_ORIG_PRECOMPUTE = _ntk_reg.EigenNTKRegression._precompute_eigen_features


def _label_aware_precompute(self, ntk_full):
    """A1 selection: top-r by  s_i = (lambda/(lambda+rho))^2 * || u_i^T Y_centered ||^2."""
    if getattr(self, "eigen_decom_mode", "top") != "top":
        return _ORIG_PRECOMPUTE(self, ntk_full)

    print("[A1 label-aware] Computing eigen decomposition with supervised selection ...")
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
    if y_int.ndim != 1:
        raise ValueError(f"Expected 1-D integer labels, got shape {y_int.shape}")
    n = y_int.shape[0]
    if n != n_train:
        raise ValueError(f"len(y) {n} != n_train {n_train}")

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
    method = f"a1_label(top-{d})"

    lam_kept = evals[idx]
    U_kept = evecs[:, idx]
    lam_kept = np.clip(lam_kept, 1e-8, None)

    Phi_tr = U_kept * np.sqrt(lam_kept)[None, :]
    Phi_te = K_tetr @ (U_kept / np.sqrt(lam_kept)[None, :])

    decomp_time = time.time() - t0
    self.eigen_decomposition_time = decomp_time
    print(
        f"[A1 label-aware] Done via {method} (n_train={n_train}, d={d}, "
        f"d/n_train(%)={d / n_train * 100:.2f}%) in {decomp_time:.4f}s"
    )

    phi_tr = torch.from_numpy(Phi_tr).to(device=self.device, dtype=self.dtype)
    phi_te = torch.from_numpy(Phi_te).to(device=self.device, dtype=self.dtype)
    return phi_tr, phi_te


_ntk_reg.EigenNTKRegression._precompute_eigen_features = _label_aware_precompute
print("[A1 label-aware] EigenNTKRegression._precompute_eigen_features monkey-patched.")

# ----- rest of imports (parity with task_data_selection.py) -----

sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *           # noqa: E402,F401,F403
from probe import *             # noqa: E402,F401,F403
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: E402,F401


LABEL_OUTPUT_ROOT = "./freeshap_res/claude_research/data_selection_test"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="rte")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_train_dp", type=int, default=2490)
    parser.add_argument("--val_sample_num", type=int, default=277)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="eigen",
                        choices=["inv", "eigen", "none"])
    parser.add_argument("--eigen_rank", type=float, default=10)
    parser.add_argument("--inv_lambda_", type=float, default=1e-6)
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2)
    parser.add_argument("--num_train_selected_list", type=int, nargs='+',
                        default=[i for i in range(1, 101)])
    parser.add_argument("--config", type=str, default="ntk_prompt")
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

    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% of num_dp={num_train_dp} -> actual rank={eigen_rank}")

    prompt = True
    signgd = False
    eigen_solver = "cholesky"
    eigen_dtype = "float32"
    early_stopping = "True"

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"

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

    if approximate != "none":
        probe_model.approximate(approximate)

    if approximate == "eigen":
        probe_model.set_eigen_params(
            rank=eigen_rank,
            lam=eigen_lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed,
        )
    elif approximate == "inv":
        probe_model.set_inv_params(lam=inv_lambda_)

    if signgd:
        probe_model.signgd()

    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    # ===== 1) Shapley pkl path (A1 layout under claude_research/) =====
    method_dir = approximate

    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (
            f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_invlam{inv_lam_str}"
            f"_{eigen_solver}_{eigen_dtype}_label"
        )
    else:
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}_label"

    shapley_base = f"{LABEL_OUTPUT_ROOT}/shapley/{dataset_name}"
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

    # ===== 2) NTK cache (shared with upstream) =====
    ntk_path = (
        f"./freeshap_res/ntk/{dataset_name}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError("NTK cache format is old; regenerate via task_ntk.py.")
    ntk = bundle["ntk"]
    probe_model.get_cached_ntk(ntk)

    # ===== 3) train/val sets =====
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    if approximate == "eigen":
        print("[info] Preparing eigen regression features (label-aware) ...")
        probe_model.prepare_eigen_regression()

    print("len(train_set) =", len(train_set))
    print("len(val_set)   =", len(val_set))

    # ===== 4) Shapley sorting and indices =====
    acc_contrib = dv_result[:, 1, :]
    acc_sum_per_train = acc_contrib.sum(axis=1)
    sorted_indices = np.argsort(acc_sum_per_train)[::-1]
    all_indices = np.arange(len(sampled_idx))

    # Outputs under claude_research/.
    ds_base = f"{LABEL_OUTPUT_ROOT}/data_selection/{dataset_name}"
    setting_name = os.path.basename(shapley_path).replace('.pkl', '')

    indices_txt_path = f"{ds_base}/{method_dir}/indices/{setting_name}_indices.txt"
    original_indices = sampled_idx[sorted_indices]
    os.makedirs(os.path.dirname(indices_txt_path), exist_ok=True)
    with open(indices_txt_path, 'a') as f:
        for idx in original_indices:
            f.write(f"{idx}\n")
    print(f"[info] saved sorted indices to {indices_txt_path}")

    # ===== 5) kernel_regression evaluation =====
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
        print("[info] Running INV mode predictions (top) ...")
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
            top_results.append(int(torch.round(acc * 10000).item()))

        print("[info] Running INV mode predictions (random) ...")
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

        print("[info] Running EIGEN mode predictions (top, A1 selection) ...")
        for num_train_selected_pct in num_train_selected_list:
            k = int(num_train_dp * num_train_selected_pct / 100)
            k = int(min(k, len(acc_sum_per_train)))
            if k <= 0:
                continue
            top_indices = sorted_indices[:k]
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(top_indices, dtype=int),
                test_set=val_set,
            )
            top_results_eigen.append(int(torch.round(acc * 10000).item()))

        print("[info] Running EIGEN mode predictions (random) ...")
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
        print("[info] Switching to INV mode for dual-mode evaluation ...")
        original_approx = probe_model.approximate_ntk
        probe_model.approximate_ntk = "inv"
        probe_model.set_inv_params(lam=inv_lambda_)

        print("[info] Running INV mode predictions (top) ...")
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

        print("[info] Running INV mode predictions (random) ...")
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

        print("[info] Restoring EIGEN mode ...")
        probe_model.approximate_ntk = original_approx
        probe_model.set_eigen_params(
            rank=eigen_rank,
            lam=eigen_lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed,
        )

    # ===== print results =====
    if approximate == "inv":
        print(f"\n[INV mode with lambda={inv_lambda_}]")
        print(f"top: {top_results}")
        print(f"random:\n{random_results}")
    elif approximate == "eigen":
        print(f"\n[Eigen mode (A1 label) with lambda={eigen_lambda_}]")
        print(f"top: {top_results_eigen}")
        print(f"random:\n{random_results_eigen}")
        print(f"\n[INV mode with lambda={inv_lambda_}]")
        print(f"top: {top_results_inv}")
        print(f"random:\n{random_results_inv}")

    # ===== 6) predictions.txt =====
    predictions_txt_path = f"{ds_base}/{method_dir}/predictions/{setting_name}_predictions.txt"
    os.makedirs(os.path.dirname(predictions_txt_path), exist_ok=True)

    with open(predictions_txt_path, 'a') as f:
        f.write(f"{'='*80}\n")
        f.write(f"dataset: {dataset_name}  (A1 label-aware)\n")
        f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
        f.write(f"seed: {seed}\n")

        if approximate == "eigen":
            f.write(f"eigen rank: {eigen_rank_pct}% (actual: {eigen_rank})\n")
            f.write(f"solver: {eigen_solver}, dtype: {eigen_dtype}\n")
        f.write("\n")

        if approximate == "inv":
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n{top_results}\n")
            f.write(f"random:\n{random_results}\n")
        elif approximate == "eigen":
            f.write(f"eigen mode lambda={eigen_lambda_:.0e}\n")
            f.write(f"top:\n{top_results_eigen}\n")
            f.write(f"random:\n{random_results_eigen}\n")
            f.write("\n")
            f.write(f"inv mode lambda={inv_lambda_:.0e}\n")
            f.write(f"top:\n{top_results_inv}\n")
            f.write(f"random:\n{random_results_inv}\n")

        if timing_info:
            f.write(f"\n{'='*80}\n")
            f.write("timing info (from shapley pkl):\n")
            if 'gpu_info' in timing_info:
                for key, value in timing_info['gpu_info'].items():
                    f.write(f"  {key}: {value}\n")
            for key, value in timing_info.items():
                if key != 'gpu_info':
                    f.write(f"  {key}: {value:.4f}s\n")

        if early_stopping_records:
            records = early_stopping_records
            total_count = len(records)
            early_stopped_count = sum(1 for r in records if r['status'] == 'Early Stop')
            completed_count = sum(1 for r in records if r['status'] == 'Complete')
            percentages = [r['percentage'] for r in records]
            f.write(f"\n{'='*80}\n")
            f.write("early stopping statistics:\n")
            f.write(f"  total iterations: {total_count}\n")
            f.write(f"  early stopped: {early_stopped_count} ({early_stopped_count/total_count*100:.1f}%)\n")
            f.write(f"  completed: {completed_count} ({completed_count/total_count*100:.1f}%)\n")
            if percentages:
                avg_pct = np.mean(percentages)
                f.write(f"  average stop point: {avg_pct:.2f}%\n")
            f.write("\n  distribution:\n")
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

    # ===== 7) base_accuracy file =====
    F_LIST = [1, 5, 10, 15, 20, 25, 30]
    val_labels = torch.tensor([ex['label'] for ex in val_set])
    acc0 = (val_labels == 0).float().mean()
    acc0_int = int(torch.round(acc0 * 10000).item())
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
            else:
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

    # ===== 8) sidecar JSON for cross-run aggregation =====
    # `aggregate_label_summary.py` later reads every sidecar in this dataset's
    # tree and produces a single 8-line comparison file (inv baseline + 7 eigen
    # ranks, all on the same INV-prediction axis).
    import json
    sidecar = {
        "dataset_name": dataset_name,
        "seed": seed,
        "num_train_dp": num_train_dp,
        "val_sample_num": val_sample_num,
        "tmc_iter": tmc_iter,
        "approximate": approximate,
        "inv_lambda_": inv_lambda_,
        "eigen_lambda_": eigen_lambda_ if approximate == "eigen" else None,
        "eigen_rank_pct": eigen_rank_pct if approximate == "eigen" else None,
        "setting_name": setting_name,
        "num_train_selected_list": list(num_train_selected_list),
        # accuracy_int values are int(acc * 10000) to match upstream convention.
        "acc_at_f0": acc0_int,
    }
    if approximate == "inv":
        sidecar["top_results_inv"] = list(top_results)
        sidecar["random_results_inv"] = list(random_results)
    elif approximate == "eigen":
        sidecar["top_results_eigen"] = list(top_results_eigen)
        sidecar["random_results_eigen"] = list(random_results_eigen)
        sidecar["top_results_inv"] = list(top_results_inv)
        sidecar["random_results_inv"] = list(random_results_inv)

    sidecar_dir = f"{ds_base}/{method_dir}/sidecar"
    os.makedirs(sidecar_dir, exist_ok=True)
    sidecar_path = f"{sidecar_dir}/{setting_name}.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[info] saved sidecar JSON to {sidecar_path}")


if __name__ == "__main__":
    main()
