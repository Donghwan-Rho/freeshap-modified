"""
task_wrong_label_detection_vision.py
------------------------------------
NLP 판 task_wrong_label_detection.py 의 vision 포트.

  * {config}_poison.yaml (ntk_vision_poison) 로 train 라벨 10% flip 한 데이터에서 SV 계산
  * SV 오름차순(저가치 우선)으로 inspect 하며 flip 된 샘플을 얼마나 찾는지 = detection rate
  * NTK 는 라벨과 무관하므로 기존 clean 캐시 재사용 (freeshap_res/ntk/...)

지원 방법: inv / eigen / nystrom / nystrom_pinv / nystrom_lev
출력: {out_root}/wrong_label_detection/{ds}/{method}/predictions/{setting}_detection.{txt,pkl}
      (SV pkl 은 {out_root}/shapley/{ds}/{method}/{setting}.pkl — poison 태그 포함)
"""

import os
import sys
import pickle
import random
import argparse


import numpy as np
import yaml
import torch

sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

# vision 폴더에서 실행 → vinfo/를 sys.path에 추가
_HERE  = os.path.dirname(os.path.abspath(__file__))
_VINFO = os.path.dirname(_HERE)
if _VINFO not in sys.path:
    sys.path.insert(0, _VINFO)

from dataset import *          # noqa: F401,F403
from probe import *            # noqa: F401,F403
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: F401 — YAML tag 등록

from vision.vision_dataset import VisionReader, VisionDataset  # noqa: F401
from vision.vision_probe   import NTKVisionProbe                # noqa: F401
from landmark_sv import resolve_landmark_sv, add_landmark_args   # SV 기반 Nystrom landmark (nystrom_q0..q4)


# 전체 train set 크기 (poison 인덱스 재현용)
DATASET_FULL_TRAIN_SIZE = {
    "cifar10": 50000,
    "cifar100": 50000,
}


def get_poison_indices(dataset_name, poison_pct, poison_seed=2023):
    """vision_dataset.VisionReader.flip_indices() 와 동일한 인덱스 재현."""
    if dataset_name not in DATASET_FULL_TRAIN_SIZE:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: {list(DATASET_FULL_TRAIN_SIZE.keys())}")
    nrows = DATASET_FULL_TRAIN_SIZE[dataset_name]
    num_to_flip = int(nrows * poison_pct / 100)
    rng = random.Random(poison_seed)
    return set(rng.sample(range(nrows), num_to_flip)), nrows, num_to_flip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="cifar10")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--num_train_dp", type=int, default=5000)
    p.add_argument("--val_sample_num", type=int, default=1000)
    p.add_argument("--tmc_iter", type=int, default=500)
    p.add_argument("--approximate", type=str, default="inv",
                   choices=["inv", "eigen", "nystrom", "nystrom_pinv", "nystrom_lev",
                                 "nystrom_q4", "nystrom_q3", "nystrom_q2",
                                 "nystrom_q1", "nystrom_q0", "none"])
    p.add_argument("--eigen_rank", type=float, default=30,
                   help="Eigen rank as percentage of num_train_dp")
    p.add_argument("--inv_lambda_", type=float, default=1e-6)
    p.add_argument("--eigen_lambda_", type=float, default=1e-2)
    p.add_argument("--nystrom_d", type=float, default=30,
                   help="Nystrom landmark count as percentage of num_train_dp")
    p.add_argument("--nystrom_lambda_", type=float, default=1e-2)
    p.add_argument("--eigeps", type=str, default="1e-8")
    p.add_argument("--nyseps", type=str, default="1e-8")
    p.add_argument("--poison_pct", type=int, default=10,
                   help="flip 할 train 라벨 비율 (%%)")
    p.add_argument("--poison_seed", type=int, default=None,
                   help="None 이면 --seed 와 통합 (NLP 판과 동일 규약)")
    p.add_argument("--num_train_selected_list", type=int, nargs='+',
                   default=[i for i in range(1, 101)],
                   help="inspect 비율 목록 (%%)")
    p.add_argument("--out_root", type=str, default="./freeshap_res")
    p.add_argument("--config", type=str, default="ntk_vision",
                   help="YAML 이름 (poison 변형 '{config}_poison.yaml' 을 로드)")
    add_landmark_args(p)
    return p.parse_args()


def _fmt_eps(eps):
    """1e-08 → '1e-8' (파일명 태그, text 판과 동일)."""
    return f"{eps:.0e}".replace('e+0', 'e+').replace('e-0', 'e-')


def _quantize_eps(eps):
    return float(f"{eps:.0e}")


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    tmc_iter = args.tmc_iter
    out_root = args.out_root
    poison_pct = args.poison_pct
    poison_seed = args.poison_seed if args.poison_seed is not None else seed
    print(f"[info] seed={seed}  poison_seed={poison_seed}  "
          f"({'unified with --seed' if args.poison_seed is None else 'explicit override'})")

    approximate = args.approximate
    # nystrom_pinv/nystrom_lev: 파라미터/경로는 nystrom 취급, 회귀 CLASS 와 method_dir 만 다름
    _is_pinv = (approximate == "nystrom_pinv")
    _is_lev = (approximate == "nystrom_lev")
    # SV 기반 결정적 landmark (oracle diagnostic). 구성은 pinv 와 동일하고 landmark 선택만 다르다.
    _lm_mode = ("q4" if approximate == "nystrom_q4"
                else "q0" if approximate == "nystrom_q0"
                else "q2" if approximate == "nystrom_q2"
                else "q1" if approximate == "nystrom_q1"
                else "q3" if approximate == "nystrom_q3" else "uniform")
    _is_svlm = (_lm_mode != "uniform")
    if _is_pinv or _is_lev or _is_svlm:
        approximate = "nystrom"

    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_
    eigen_eps = _quantize_eps(float(args.eigeps)); eigeps_str = _fmt_eps(eigen_eps)
    nystrom_eps = _quantize_eps(float(args.nyseps)); nyseps_str = _fmt_eps(nystrom_eps)

    eigen_rank_pct = args.eigen_rank
    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    nystrom_d_pct = args.nystrom_d
    nystrom_d = int(num_train_dp * nystrom_d_pct / 100)

    prompt = False        # vision 은 prompt 없음
    signgd = False
    eigen_solver = "cholesky"
    eigen_dtype = "float32"
    per_point = True
    early_stopping = "True"

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed); random.seed(seed)

    # ===== 1) poison YAML 로드 =====
    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}_poison.yaml"
    print(f"[info] yaml = {yaml_path}")
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    probe_model = yaml_args["probe_com"]
    dshap_com = yaml_args["dshap_com"]

    # VisionReader 에 poison 파라미터 주입 (flip_indices 가 이 값 사용)
    if hasattr(list_dataset, "data_loader"):
        list_dataset.data_loader.poison_seed = poison_seed
        list_dataset.data_loader.poison_pct = poison_pct

    if prompt:
        probe_model.model.init(list_dataset.label_word_list)

    if approximate != "none":
        probe_model.approximate(approximate)
    probe_model.nystrom_use_pinv = _is_pinv
    probe_model.nystrom_use_lev = _is_lev
    probe_model.nystrom_landmark_mode = _lm_mode   # q0..q4 -> SV 기반 결정적 landmark

    if approximate == "eigen":
        probe_model.set_eigen_params(rank=eigen_rank, lam=eigen_lambda_, solver=eigen_solver,
                                     dtype=eigen_dtype, seed=seed, floor=eigen_eps)
    elif approximate == "nystrom":
        probe_model.set_nystrom_params(d=nystrom_d, lam=float(args.nystrom_lambda_),
                                       solver=eigen_solver, dtype=eigen_dtype,
                                       landmark_seed=seed, jitter=nystrom_eps)
    elif approximate == "inv":
        probe_model.set_inv_params(lam=inv_lambda_)

    # ===== model_name =====
    if 'resnet' in probe_model.args['model']:
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

    # ===== 2) NTK 캐시 (라벨 무관 → clean 캐시 재사용) =====
    ntk_path = (f"./freeshap_res/ntk/{dataset_name}/{model_name}"
                f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl")
    print(f"[info] ntk_path = {ntk_path}")
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError("NTK cache format is old (ntk only). Regenerate with indices.")
    ntk = bundle["ntk"]
    sampled_idx = np.array(bundle["sampled_idx"])
    sampled_val_idx = np.array(bundle["sampled_val_idx"])
    probe_model.get_cached_ntk(ntk)

    # ===== 3) poison 된 train/val set =====
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")   # 라벨 flip 적용됨
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")   # val 은 clean
    probe_model.get_train_labels(train_set)

    if approximate == "eigen":
        probe_model.prepare_eigen_regression()
    elif approximate == "nystrom":
        probe_model.prepare_nystrom_regression()

    # ===== 4) Shapley 경로 (poison 태그 포함) =====
    method_dir = (f"nystrom_{_lm_mode}" if _is_svlm else
                  "nystrom_lev" if _is_lev else
                  ("nystrom_pinv" if _is_pinv else approximate))
    if approximate == "eigen":
        extra_tag = (f"_eig{eigen_rank_pct}_eiglam{eigen_lambda_:.0e}_eigeps{eigeps_str}"
                     f"_{eigen_solver}_{eigen_dtype}")
    elif approximate == "nystrom":
        extra_tag = (f"_nys{nystrom_d_pct}_nyslam{float(args.nystrom_lambda_):.0e}"
                     f"_nyseps{nyseps_str}_{eigen_solver}_{eigen_dtype}")
        if _is_pinv:
            extra_tag = extra_tag.replace("_nys", "_nyspinv", 1)
        elif _is_lev:
            extra_tag = extra_tag.replace("_nys", "_nyslev", 1)
        elif _is_svlm:
            extra_tag = extra_tag.replace("_nys", f"_nys{_lm_mode}", 1)
    else:
        extra_tag = f"_lam{inv_lambda_:.0e}"

    poison_tag = f"_poison{poison_pct}"
    if poison_seed != 2023:
        poison_tag += f"_ps{poison_seed}"

    detection_base_path = f"{out_root}/wrong_label_detection/{dataset_name}"
    shapley_path = (f"{out_root}/shapley/{dataset_name}/{method_dir}/{model_name}"
                    f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
                    f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
                    f"_tmc{tmc_iter}{poison_tag}.pkl")
    print(f"[info] shapley_path = {shapley_path}")

    # ===== 5) Shapley 로드 or 계산 =====
    try:
        with open(shapley_path, "rb") as f:
            result = pickle.load(f)
        dv_result = result["dv_result"]
        if "sampled_idx" in result:
            sampled_idx = np.array(result["sampled_idx"])
        print(f"[info] loaded Shapley from {shapley_path}")
    except Exception as e:
        print(f"[info] shapley cache miss -> computing ({e})")
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
            early_stopping=early_stopping,
        )
        result = {"dv_result": dv_result, "sampled_idx": sampled_idx,
                  "sampled_val_idx": sampled_val_idx, "args": vars(args)}
        os.makedirs(os.path.dirname(shapley_path), exist_ok=True)
        with open(shapley_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[info] saved Shapley to {shapley_path}")

    dv_result = np.array(dv_result)

    # ===== 6) SV 오름차순 정렬 + poison 매칭 =====
    acc_sum_per_train = dv_result[:, 1, :].sum(axis=1)
    sorted_indices_ascending = np.argsort(acc_sum_per_train)   # 저가치 우선

    poison_indices_global, full_train_size, num_flipped = get_poison_indices(
        dataset_name, poison_pct, poison_seed)
    poisoned_mask = np.array([int(i) in poison_indices_global for i in sampled_idx])
    total_poisoned = int(poisoned_mask.sum())
    print(f"[info] poison: full {num_flipped}/{full_train_size}, "
          f"sampled {total_poisoned}/{len(sampled_idx)}")
    if total_poisoned == 0:
        print("[WARNING] sampled set 에 poison 이 없음 — 종료"); return

    # ===== 7) detection curve =====
    detection_results, random_detection = [], []
    for pct in args.num_train_selected_list:
        k = min(int(num_train_dp * pct / 100), len(sorted_indices_ascending))
        if k <= 0:
            continue
        detected = int(poisoned_mask[sorted_indices_ascending[:k]].sum())
        detection_results.append({
            'inspect_pct': pct, 'k_inspect': k, 'detected': detected,
            'total_poisoned': total_poisoned,
            'detection_rate': float(detected / total_poisoned),
        })
        random_detection.append({'inspect_pct': pct,
                                 'detection_rate': float(k / len(sampled_idx))})

    rates = [r['detection_rate'] for r in detection_results]
    rnd = [r['detection_rate'] for r in random_detection]
    pcts = [r['inspect_pct'] for r in detection_results]
    print("\ndetection rates (x10000):", [int(round(r * 10000)) for r in rates])

    # ===== 8) 저장 (NLP 판과 동일 포맷) =====
    setting_name = os.path.basename(shapley_path).replace('.pkl', '')
    out_txt = f"{detection_base_path}/{method_dir}/predictions/{setting_name}_detection.txt"
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"train: {num_train_dp}, val: {val_sample_num}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"poison: {poison_pct}% (seed={poison_seed})\n")
        f.write(f"poisoned in sampled set: {total_poisoned}/{len(sampled_idx)}\n")
        if approximate == "eigen":
            f.write(f"eigen rank: {eigen_rank_pct}% (actual: {eigen_rank})\n")
        elif approximate == "nystrom":
            f.write(f"nystrom d: {nystrom_d_pct}% (actual: {nystrom_d}), nyseps: {nyseps_str}\n")
        f.write(f"\ndetection rates (x10000, lowest Shapley first):\n")
        f.write(f"{[int(round(r * 10000)) for r in rates]}\n")
        f.write(f"\nrandom baseline (x10000):\n{[int(round(r * 10000)) for r in rnd]}\n")
        f.write(f"\ninspect percentages:\n{pcts}\n")
    print(f"[info] saved detection results to {out_txt}")

    out_pkl = out_txt.replace("_detection.txt", "_detection.pkl")
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            "args": vars(args), "dataset_name": dataset_name, "seed": seed,
            "poison_seed": poison_seed, "poison_pct": poison_pct,
            "num_train_dp": num_train_dp, "val_sample_num": val_sample_num,
            "approximate": args.approximate,
            "sampled_idx": sampled_idx, "poisoned_mask": poisoned_mask,
            "sorted_indices_ascending": sorted_indices_ascending,
            "detection_results": detection_results,
            "random_detection": random_detection,
        }, f)
    print(f"[info] saved detection bundle to {out_pkl}")


if __name__ == "__main__":
    main()
