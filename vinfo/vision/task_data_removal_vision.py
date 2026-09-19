import pickle
import numpy as np
import yaml
import torch
import random

import sys, os
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

# vision 폴더에서 실행 → vinfo/를 sys.path에 추가해 probe/dataset 임포트 가능하게
_HERE  = os.path.dirname(os.path.abspath(__file__))
_VINFO = os.path.dirname(_HERE)
if _VINFO not in sys.path:
    sys.path.insert(0, _VINFO)
import heldout_common as HO

from dataset import *
from probe import *
from dvutils.Data_Shapley import Fast_Data_Shapley  # YAML tag 해석용

# vision 신규 클래스 (yaml_tag 등록)
from vision.vision_dataset import VisionReader, VisionDataset  # noqa: F401
from vision.vision_probe   import NTKVisionProbe                # noqa: F401

import argparse

from landmark_sv import resolve_landmark_sv, add_landmark_args   # SV 기반 Nystrom landmark (nystrom_q0..q4)

# ============================================================================
# task_data_removal_vision.py — text 판 task_data_removal.py 의 vision 포트.
#   downstream: top/bottom/random k% 제거 후 남은 데이터로 kernel_regression 예측.
#   * SV(shapley pkl)/NTK 는 task_shapley_vision 과 동일 파일 재사용 (재계산 X).
#   * eigen/nystrom(+pinv/lev) 은 dual-mode(approx + inv) 예측 — text 판과 동일 규약.
#   출력: {out_root}/data_removing/{ds}/{method}/predictions/{setting}_predictions.txt
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_train_dp", type=int, default=2000)
    parser.add_argument("--val_sample_num", type=int, default=1000)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="inv",
                        choices=["inv", "eigen", "nystrom", "nystrom_pinv", "nystrom_lev",
                                 "nystrom_q4", "nystrom_q3", "nystrom_q2",
                                 "nystrom_q1", "nystrom_q0", "none"])
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
    parser.add_argument("--eigeps", type=str, default="1e-8",
                        help="Eigen eigenvalue floor (filename tag: eigeps)")
    parser.add_argument("--nyseps", type=str, default="1e-8",
                        help="Nystrom jitter/eps (filename tag: nyseps)")
    parser.add_argument("--out_root", type=str, default="./freeshap_res",
                        help="Root dir for shapley/data_removing I/O (NTK read from ./freeshap_res/ntk).")
    parser.add_argument("--num_train_removed_list", type=int, nargs='+',
                        default=[i for i in range(0, 100)],
                        help="제거할 num_train_dp 퍼센트 목록 (0=제거없음 baseline ~ 99).")
    # --- held-out 평가 (selection 과 동일 규약) ---
    parser.add_argument("--heldout", action="store_true",
                        help="held-out 집합에서 평가 (vision/task_ntk_heldout_vision.py 블록 필요)")
    parser.add_argument("--heldout_size", type=int, default=None)
    parser.add_argument("--heldout_seed", type=int, default=HO.HO_SEED_DEFAULT)
    parser.add_argument("--config", type=str, default="ntk_vision",
                        help="YAML config name without .yaml extension")
    add_landmark_args(parser)
    return parser.parse_args()


def _fmt_nyseps(eps):
    """1e+02 → '1e+2', 1e-08 → '1e-8' (text 판 파일명 포맷과 동일)."""
    return f"{eps:.0e}".replace('e+0', 'e+').replace('e-0', 'e-')


def _quantize_nyseps(eps):
    """파일명 표기와 실제 사용값을 일치시키기 위한 quantize (text 판과 동일)."""
    return float(f"{eps:.0e}")


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    tmc_iter = args.tmc_iter

    approximate = args.approximate
    # nystrom_pinv/nystrom_lev: alias to "nystrom" for all param/tag/path logic; only the
    # regression CLASS (probe_model.nystrom_use_pinv/lev) and method_dir differ. (text 판과 동일)
    _is_pinv = (approximate == "nystrom_pinv")
    _is_lev = (approximate == "nystrom_lev")   # leverage-score landmarks (pinv construction)
    # SV 기반 결정적 landmark (oracle diagnostic). 구성은 pinv 와 동일하고 landmark 선택만 다르다.
    _lm_mode = ("q4" if approximate == "nystrom_q4"
                else "q0" if approximate == "nystrom_q0"
                else "q2" if approximate == "nystrom_q2"
                else "q1" if approximate == "nystrom_q1"
                else "q3" if approximate == "nystrom_q3" else "uniform")
    _is_svlm = (_lm_mode != "uniform")
    if _is_pinv or _is_lev or _is_svlm:
        approximate = "nystrom"
    remove_pct_list = args.num_train_removed_list
    eigen_rank_pct = args.eigen_rank
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_

    # eps 처리 (파일명 태그와 실제 사용값 일치; text 판과 동일)
    eigen_eps = _quantize_nyseps(float(args.eigeps))
    eigeps_str = _fmt_nyseps(eigen_eps)
    nystrom_eps = _quantize_nyseps(float(args.nyseps))
    nyseps_str = _fmt_nyseps(nystrom_eps)

    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% of num_dp={num_train_dp} -> actual rank={eigen_rank}")

    nystrom_d_pct = args.nystrom_d
    nystrom_d = int(num_train_dp * nystrom_d_pct / 100)
    if approximate == "nystrom":
        print(f"[info] nystrom_d={nystrom_d_pct}% of num_dp={num_train_dp} -> actual landmarks={nystrom_d}")

    prompt = False   # vision은 prompt 없음
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
    probe_model.nystrom_use_pinv = _is_pinv   # pinv -> pseudoinverse-Nystrom class
    probe_model.nystrom_use_lev = _is_lev    # lev -> leverage-score Nystrom class
    probe_model.nystrom_landmark_mode = _lm_mode   # q0..q4 -> SV 기반 결정적 landmark

    if approximate == "eigen":
        probe_model.set_eigen_params(
            rank=eigen_rank, lam=eigen_lambda_, solver=eigen_solver,
            dtype=eigen_dtype, seed=seed, floor=eigen_eps)
    elif approximate == "nystrom":
        probe_model.set_nystrom_params(
            d=nystrom_d, lam=float(args.nystrom_lambda_), solver=eigen_solver,
            dtype=eigen_dtype, landmark_seed=seed, jitter=nystrom_eps)
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
    elif 'resnet' in probe_model.args['model']:
        model_name = 'resnet'
    elif 'resnext' in probe_model.args['model']:
        model_name = 'resnext'
    else:
        model_name = 'model'

    # ===== SV 기반 landmark: 같은 seed 의 inv Shapley 를 읽어 probe 에 주입 =====
    #   nyseps auto 는 랜덤 subset 으로 sigma_min(W) 를 추정하므로 실제 landmark 와 어긋난다.
    if _is_svlm:
        if str(args.nyseps).lower() == "auto":
            raise SystemExit("[nystrom_q0..q4] --nyseps auto 는 지원하지 않습니다. "
                             "--nyseps 1e-8 처럼 명시하세요.")
        probe_model.nystrom_landmark_sv = resolve_landmark_sv(
            args, dataset_name, model_name, seed, num_train_dp,
            val_sample_num, inv_lambda_, tmc_iter=tmc_iter)

    # ===== 1) Shapley pkl 경로 (task_shapley_vision 과 동일한 파일 재사용) =====
    method_dir = (f"nystrom_{_lm_mode}" if _is_svlm else
                  "nystrom_lev" if _is_lev else
                  ("nystrom_pinv" if _is_pinv else approximate))
    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_eigeps{eigeps_str}_invlam{inv_lam_str}_{eigen_solver}_{eigen_dtype}"
    elif approximate == "nystrom":
        nys_lam_str = f"{float(args.nystrom_lambda_):.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (f"_nys{nystrom_d_pct}_nyslam{nys_lam_str}_nyseps{nyseps_str}"
                     f"_invlam{inv_lam_str}_{eigen_solver}_{eigen_dtype}")
        if _is_pinv:
            extra_tag = extra_tag.replace("_nys", "_nyspinv", 1)  # filename marker: pseudoinverse variant
        elif _is_lev:
            extra_tag = extra_tag.replace("_nys", "_nyslev", 1)   # filename marker: leverage-score variant
        elif _is_svlm:
            extra_tag = extra_tag.replace("_nys", f"_nys{_lm_mode}", 1)
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

    # ===== 2) NTK 캐시 로드 =====
    if args.heldout:
        _ho_tag = HO.ho_tag(HO.resolve_size(dataset_name, args.heldout_size), args.heldout_seed, "train")
        ntk_path = HO.heldout_ntk_path("./freeshap_res", dataset_name, model_name,
                                       seed, num_train_dp, _ho_tag, signgd)
    else:
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
    if args.heldout:
        if not np.array_equal(np.array(bundle["sampled_idx"]), sampled_idx):
            raise RuntimeError("held-out 블록의 train subset 이 Shapley 결과와 다르다 — 중단")
        _eval_idx = [int(i) for i in bundle["heldout_idx"]]
        val_set = list_dataset.get_idx_dataset(_eval_idx, split=bundle["heldout_split"])
        print(f"[info] held-out 평가: split={bundle['heldout_split']} |H|={len(_eval_idx)}")
    else:
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

    ds_base = f"{HO.removal_base(args.out_root, 'heldout' if args.heldout else 'insample')}/{dataset_name}"
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
