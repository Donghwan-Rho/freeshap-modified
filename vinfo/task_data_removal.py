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

# ============================================================================
# task_data_removal.py  —  task_data_selection.py 와 동일 양식.
#   차이: downstream 을 "top-k% 선택 후 예측" → "top/bottom/random k% 제거 후
#         남은 데이터로 예측" 으로 교체.
#   * SV(shapley pkl) 및 NTK 는 selection 과 **완전히 동일한 파일 재사용** (재계산 X).
#   * 랭킹: SV 내림차순(sorted_indices, 큰 값이 앞).
#       - top removal    : 고가치(높은 SV) k% 제거 → remaining = sorted[k:]
#       - bottom removal : 저가치(낮은 SV) k% 제거 → remaining = sorted[:N-k]
#       - random removal : 무작위 k% 제거
#   * eigen/nystrom 은 dual-mode(approx + inv) 예측. 분석엔 inv-mode 사용
#     (approx SV 로 랭킹, exact 커널로 예측 — selection/detection 과 같은 규약).
#   출력: {out_root}/data_removal/{ds}/{method}/predictions/{setting}_removal.txt
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mr")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_train_dp", type=int, default=8530)
    parser.add_argument("--val_sample_num", type=int, default=1066)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="inv",
                        choices=["inv", "eigen", "nystrom", "none"])
    parser.add_argument("--eigen_rank", type=float, default=30,
                        help="Eigen rank as percentage of num_train_dp (e.g., 10 means 10%% of data)")
    parser.add_argument("--inv_lambda_", type=float, default=1e-6,
                        help="Lambda (regularization parameter) for INV mode")
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2,
                        help="Lambda (regularization parameter) for Eigen mode")
    parser.add_argument("--nystrom_d", type=float, default=30,
                        help="Nystrom landmark count as percentage of num_train_dp, same convention as --eigen_rank")
    parser.add_argument("--nystrom_lambda_", type=float, default=1e-3,
                        help="Lambda (ridge regularization) for Nystrom mode")
    parser.add_argument("--nyseps", type=str, default="1e+1",
                        help="Nystrom feature-construction jitter ε in (W+εI). Float 또는 'auto'. Default 1e+1.")
    parser.add_argument("--eigeps", type=str, default="1e-8",
                        help="Eigen eigenvalue jitter ε: lam -> max(lam,0)+ε. Default 1e-8.")
    parser.add_argument("--out_root", type=str, default="./freeshap_res",
                        help="Root dir for shapley/data_removal I/O (NTK read from ./freeshap_res/ntk).")
    parser.add_argument("--num_train_removed_list", type=int, nargs='+',
                        default=[i for i in range(0, 100)],
                        help="제거할 num_train_dp 퍼센트 목록 (0=제거없음 baseline ~ 99).")
    parser.add_argument("--config", type=str, default="ntk_prompt",
                        help="YAML config name without .yaml extension")
    return parser.parse_args()


def _fmt_nyseps(eps):
    """decade 표기: 1e+01 → '1e+1', 1e-08 → '1e-8'."""
    return f"{eps:.0e}".replace('e+0', 'e+').replace('e-0', 'e-')


def _quantize_nyseps(eps):
    """Filename(1 sig-fig 과학표기)에 맞춰 값 quantize. 1e+01 → 10.0."""
    return float(f"{eps:.0e}")


def _compute_nyseps_auto(ntk, num_train_dp, nystrom_d, landmark_seed):
    """σ_min(W) 계산. W = K[S,S], S = landmark_seed 로 뽑은 subset."""
    K = ntk[0] if ntk.ndim == 3 else ntk
    K_np = K.to("cpu", dtype=torch.float64).numpy()[:num_train_dp, :num_train_dp]
    K_np = 0.5 * (K_np + K_np.T)
    rng = np.random.RandomState(int(landmark_seed))
    S = np.sort(rng.choice(num_train_dp, size=nystrom_d, replace=False))
    W = K_np[np.ix_(S, S)]
    return float(np.linalg.eigvalsh(W)[0])


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    tmc_iter = args.tmc_iter

    approximate = args.approximate
    remove_pct_list = args.num_train_removed_list
    eigen_rank_pct = args.eigen_rank
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_
    eigen_eps = _quantize_nyseps(float(args.eigeps))
    eigeps_str = _fmt_nyseps(eigen_eps)

    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% of num_dp={num_train_dp} -> actual rank={eigen_rank}")

    nystrom_d_pct = args.nystrom_d
    nystrom_d = int(num_train_dp * nystrom_d_pct / 100)
    if approximate == "nystrom":
        print(f"[info] nystrom_d={nystrom_d_pct}% of num_dp={num_train_dp} -> actual landmarks={nystrom_d}")

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
            rank=eigen_rank, lam=eigen_lambda_, solver=eigen_solver,
            dtype=eigen_dtype, seed=seed, floor=eigen_eps)
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

    # ===== 0.5) nystrom 이면 NTK peek + --nyseps 처리 =====
    nystrom_eps = None
    nyseps_str = None
    if approximate == "nystrom":
        ntk_path_peek = (
            f"./freeshap_res/ntk/{dataset_name}/{model_name}"
            f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
        )
        with open(ntk_path_peek, "rb") as f:
            ntk_peek = pickle.load(f)["ntk"]
        if str(args.nyseps).lower() == "auto":
            _raw = _compute_nyseps_auto(ntk_peek, num_train_dp, nystrom_d, seed)
            print(f"[nyseps auto] σ_min(W) raw = {_raw:.6e}")
            nystrom_eps = _quantize_nyseps(_raw)
        else:
            nystrom_eps = _quantize_nyseps(float(args.nyseps))
        nyseps_str = _fmt_nyseps(nystrom_eps)
        print(f"[nyseps] value={nystrom_eps:.4e}  tag='{nyseps_str}'")
        probe_model.set_nystrom_params(
            d=nystrom_d, lam=float(args.nystrom_lambda_), solver=eigen_solver,
            dtype=eigen_dtype, landmark_seed=seed, jitter=nystrom_eps)
        del ntk_peek

    # ===== 1) Shapley pkl 경로 (selection 과 동일한 파일 재사용) =====
    method_dir = approximate
    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_eigeps{eigeps_str}_invlam{inv_lam_str}_{eigen_solver}_{eigen_dtype}"
    elif approximate == "nystrom":
        nys_lam_str = f"{float(args.nystrom_lambda_):.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (f"_nys{nystrom_d_pct}_nyslam{nys_lam_str}_nyseps{nyseps_str}"
                     f"_invlam{inv_lam_str}_{eigen_solver}_{eigen_dtype}")
    else:
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}"

    shapley_base = f"{args.out_root}/shapley/{dataset_name}"
    shapley_path = (
        f"{shapley_base}/{method_dir}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")
    with open(shapley_path, "rb") as f:
        result = pickle.load(f)
    print(f"[info] loaded Shapley from {shapley_path}")

    dv_result = np.array(result["dv_result"])
    sampled_idx = np.array(result["sampled_idx"])
    sampled_val_idx = np.array(result["sampled_val_idx"])
    timing_info = result.get("timing_info", {})
    print("dv_result shape:", dv_result.shape)

    # ===== 2) NTK 캐시 로드 (selection 과 동일) =====
    ntk_path = (
        f"./freeshap_res/ntk/{dataset_name}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError("NTK cache format is old (ntk only). Regenerate NTK with indices.")
    ntk = bundle["ntk"]
    probe_model.get_cached_ntk(ntk)

    # ===== 3) train/val set =====
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    if approximate == "eigen":
        print("[info] Preparing eigen regression features...")
        probe_model.prepare_eigen_regression()
    elif approximate == "nystrom":
        print("[info] Preparing Nystrom regression features...")
        probe_model.prepare_nystrom_regression()

    print("len(train_set) =", len(train_set), "len(val_set) =", len(val_set))

    # ===== 4) Shapley 정렬 =====
    acc_contrib = dv_result[:, 1, :]
    acc_sum_per_train = acc_contrib.sum(axis=1)
    sorted_indices = np.argsort(acc_sum_per_train)[::-1]   # 큰 값(고가치)이 앞
    N = len(sorted_indices)
    all_indices = np.arange(N)

    ds_base = f"{args.out_root}/data_removing/{dataset_name}"
    setting_name = os.path.basename(shapley_path).replace('.pkl', '')

    # ===== 5) 제거 후 kernel_regression =====
    def keep_indices(strategy, k):
        """제거 k개 후 남기는 인덱스."""
        if k <= 0:
            return all_indices
        if strategy == "top":       # 고가치 제거
            return sorted_indices[k:]
        if strategy == "bottom":    # 저가치 제거
            return sorted_indices[:N - k]
        # random
        rem = np.random.choice(all_indices, size=k, replace=False)
        return np.setdiff1d(all_indices, rem)

    def run_curve(strategy, reset_inv):
        out = []
        for pct in remove_pct_list:
            k = int(min(int(num_train_dp * pct / 100), N))
            if N - k <= 0:      # 다 제거하면 skip
                continue
            keep = keep_indices(strategy, k)
            if reset_inv:
                probe_model.pre_inv = None
                probe_model.kr_model = None
            _, acc = probe_model.kernel_regression(
                train_indices=np.array(keep, dtype=int), test_set=val_set)
            out.append(int(torch.round(acc * 10000).item()))
        return out

    curves = {}   # (mode, strategy) -> list
    if approximate == "inv":
        for strat in ["top", "bottom", "random"]:
            print(f"[info] INV - {strat} removal ...")
            curves[("inv", strat)] = run_curve(strat, reset_inv=True)
    else:
        approx_label = approximate.upper()
        if approximate == "eigen":
            probe_model.eigen_decom_mode = "top"
        # approx-mode
        for strat in ["top", "bottom", "random"]:
            print(f"[info] {approx_label} - {strat} removal ...")
            curves[(approximate, strat)] = run_curve(strat, reset_inv=False)
        # inv-mode (dual)
        print("[info] Switching to INV mode for dual-mode evaluation...")
        original_approx = probe_model.approximate_ntk
        probe_model.approximate_ntk = "inv"
        probe_model.set_inv_params(lam=inv_lambda_)
        for strat in ["top", "bottom", "random"]:
            print(f"[info] INV - {strat} removal ...")
            curves[("inv", strat)] = run_curve(strat, reset_inv=True)
        # restore
        probe_model.approximate_ntk = original_approx
        if approximate == "eigen":
            probe_model.set_eigen_params(rank=eigen_rank, lam=eigen_lambda_, solver=eigen_solver,
                                         dtype=eigen_dtype, seed=seed, floor=eigen_eps)
        else:
            probe_model.set_nystrom_params(d=nystrom_d, lam=float(args.nystrom_lambda_),
                                           solver=eigen_solver, dtype=eigen_dtype,
                                           landmark_seed=seed, jitter=nystrom_eps)

    # ===== 6) removal.txt 저장 =====
    out_path = f"{ds_base}/{method_dir}/predictions/{setting_name}_predictions.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lr_lam = eigen_lambda_ if approximate == "eigen" else (
        float(args.nystrom_lambda_) if approximate == "nystrom" else inv_lambda_)

    def _block(f, mode, lam):
        f.write(f"{mode} mode lambda={lam:.0e}\n")
        f.write(f"top_removal:\n{curves[(mode, 'top')]}\n")
        f.write(f"bottom_removal:\n{curves[(mode, 'bottom')]}\n")
        f.write(f"random:\n{curves[(mode, 'random')]}\n\n")

    with open(out_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"remove_pct_list: {list(remove_pct_list)}\n")
        if approximate == "eigen":
            f.write(f"eigen rank: {eigen_rank_pct}% (actual: {eigen_rank})\n")
        elif approximate == "nystrom":
            f.write(f"nystrom d: {nystrom_d_pct}% (actual: {nystrom_d}), nyseps: {nyseps_str}\n")
        f.write("\n")
        _block(f, "inv", inv_lambda_)
        if approximate in ("eigen", "nystrom"):
            _block(f, approximate, lr_lam)
        if timing_info:
            f.write(f"{'='*80}\ntiming info (from shapley pkl):\n")
            for key, value in timing_info.items():
                if key == 'gpu_info':
                    for k2, v2 in value.items():
                        f.write(f"  {k2}: {v2}\n")
                else:
                    try: f.write(f"  {key}: {value:.4f}s\n")
                    except Exception: f.write(f"  {key}: {value}\n")
    print(f"[info] saved removal curves to {out_path}")


if __name__ == "__main__":
    main()
