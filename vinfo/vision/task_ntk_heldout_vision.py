"""held-out 평가 집합에 대한 eNTK 블록 추가 계산 — vision(ResNet/CIFAR-10) 판.

NLP 판(task_ntk_heldout.py)과 규약이 완전히 같다.
  기존  freeshap_res/ntk/{ds}/{model}_seed{S}_num{N}_val{V}_signFalse.pkl        (N+V, N)
  생성  freeshap_res/ntk_heldout/{ds}/{model}_seed{S}_num{N}_{tag}_signFalse.pkl (N+H, N)

  * held-out 은 seed 와 무관하게 고정 (heldout_seed=42), 3개 seed 의 train 합집합에서 제외.
  * 앞 N행(train x train)은 기존 캐시 값으로 덮어써 Shapley 계산에 쓴 커널과 일치시킨다.
  * Shapley(TMC)는 재계산하지 않는다.

사용:
  python vision/task_ntk_heldout_vision.py --config ntk_vision --dataset_name cifar10 \
      --seed 2024 --num_train_dp 5000 --val_sample_num 1000
"""

import warnings
warnings.filterwarnings('ignore')
import gc
import os
import pickle
import random
import sys

import numpy as np
import torch
import yaml as yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

# vision 폴더에서 실행 → vinfo/ 를 sys.path 에 추가 (다른 vision 스크립트와 동일)
_VINFO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VINFO not in sys.path:
    sys.path.insert(0, _VINFO)

from dataset import *          # noqa: F401,F403  (yaml 태그 해석용)
from probe import *            # noqa: F401,F403
import heldout_common as HO

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_train_dp", type=int, default=5000)
    parser.add_argument("--val_sample_num", type=int, default=1000)
    parser.add_argument("--heldout_size", type=int, default=None,
                        help="기본값은 데이터셋별 규약 (cifar10=1000)")
    parser.add_argument("--heldout_seed", type=int, default=HO.HO_SEED_DEFAULT)
    parser.add_argument("--union_seeds", type=int, nargs="+", default=list(HO.SEEDS_DEFAULT))
    parser.add_argument("--out_root", type=str, default="./freeshap_res")
    parser.add_argument("--config", type=str, default="ntk_vision")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--grad_chunksize", type=int, default=None,
                        help="한 번에 처리할 파라미터 수. 메모리는 (train+heldout) x 이 값 x 4 bytes 다. "
                             "yaml 기본값(2e7)은 6000행 기준 480GB 라 RAM 이 작은 노드에서는 "
                             "80만 정도로 낮춰야 한다 (대신 파라미터 배치 수가 늘어 느려진다).")
    parser.add_argument("--keep_fresh_train", action="store_true",
                        help="train 블록을 새로 계산한 값 그대로 둔다 (기본은 기존 캐시로 교체)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    root = args.out_root
    signgd = False

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed); random.seed(seed)

    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']
    probe_model = yaml_args['probe_com']

    m = probe_model.args['model']
    model_name = 'resnet' if 'resnet' in m else ('resnext' if 'resnext' in m else 'model')

    if (dataset_name, num_train_dp) in HO.NO_ROOM:
        raise SystemExit(f"[stop] {dataset_name} n={num_train_dp} 은 train 을 전부 써서 held-out 불가")

    # ===== 기존 캐시에서 이 seed 의 train subset 을 그대로 읽는다 =====
    base_path = HO.ntk_path(root, dataset_name, model_name, seed, num_train_dp,
                            val_sample_num, signgd)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"기존 NTK 캐시가 없다: {base_path}")
    with open(base_path, "rb") as f:
        base_bundle = pickle.load(f)
    sampled_idx = [int(i) for i in base_bundle["sampled_idx"]]
    print(f"[info] train subset from cache: {base_path}  (|train|={len(sampled_idx)})")

    # ===== held-out 인덱스 (seed 무관, 고정) =====
    total_train = list_dataset.len_train()
    ho_size = HO.resolve_size(dataset_name, args.heldout_size)
    heldout_idx, split = HO.pick_heldout(
        root, dataset_name, model_name, num_train_dp, val_sample_num, total_train,
        size=ho_size, ho_seed=args.heldout_seed, seeds=tuple(args.union_seeds),
        signgd=signgd, split="train")
    overlap = len(set(heldout_idx) & set(sampled_idx))
    print(f"[info] held-out: split={split}  |H|={len(heldout_idx)}  "
          f"(train pool={total_train}, seed={args.heldout_seed})  train 과 겹침={overlap}")
    if overlap:
        raise RuntimeError(f"held-out 과 train 이 {overlap} 개 겹친다 — 중단")

    tag = HO.ho_tag(len(heldout_idx), args.heldout_seed, split)
    out_path = HO.heldout_ntk_path(root, dataset_name, model_name, seed,
                                   num_train_dp, tag, signgd)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] 이미 있음: {out_path}  (--overwrite 로 덮어쓰기)")
        return
    print(f"[info] out_path = {out_path}")

    # --grad_chunksize: yaml 을 건드리지 않고 이 실행에서만 메모리/속도 균형을 바꾼다.
    if args.grad_chunksize:
        probe_model.args['grad_chunksize'] = args.grad_chunksize
        _rows = len(sampled_idx) + len(heldout_idx)
        print(f"[info] grad_chunksize={args.grad_chunksize:,} -> grads 버퍼 "
              f"{_rows * args.grad_chunksize * 4 / 1e9:.1f} GB "
              f"(파라미터 배치 {(109514298 // 2 - 1) // args.grad_chunksize + 1} 개 내외)")

    # ===== eNTK: 행 = train + held-out, 열 = train =====
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    heldout_set = list_dataset.get_idx_dataset(heldout_idx, split="train")
    print(f"len(train_set)   = {len(train_set)}")
    print(f"len(heldout_set) = {len(heldout_set)}")

    ntk = probe_model.compute_ntk(train_set, heldout_set)
    if isinstance(ntk, torch.Tensor):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ntk = ntk.detach().cpu().contiguous()
    print(f"[info] ntk shape = {tuple(ntk.shape)}")

    # ===== train 블록을 기존 캐시 값으로 교체 (Shapley 와 비트 단위 일치) =====
    _n = len(sampled_idx)
    _base = base_bundle["ntk"]
    if not torch.is_tensor(_base):
        _base = torch.as_tensor(np.asarray(_base))
    _rel = float((ntk[..., :_n, :].to(torch.float64) - _base[..., :_n, :].to(torch.float64)).norm()
                 / _base[..., :_n, :].to(torch.float64).norm())
    print(f"[check] train 블록 상대차 (새 계산 vs 기존 캐시) = {_rel:.3e}")
    if _rel > 1e-3:
        raise RuntimeError(f"train 블록이 너무 다르다 (상대차 {_rel:.3e}) — 중단")
    if args.keep_fresh_train:
        print("[info] --keep_fresh_train: 새로 계산한 train 블록 유지")
    else:
        ntk[..., :_n, :] = _base[..., :_n, :].to(ntk.dtype)
        print("[info] train 블록을 기존 캐시 값으로 교체했다")

    bundle = {
        "ntk": ntk,
        "sampled_idx": np.array(sampled_idx),
        "heldout_idx": np.array(heldout_idx),
        "heldout_split": split,
        "meta": {
            "dataset_name": dataset_name, "seed": seed, "num_train_dp": num_train_dp,
            "val_sample_num": val_sample_num, "heldout_size": len(heldout_idx),
            "heldout_seed": args.heldout_seed, "union_seeds": list(args.union_seeds),
            "signgd": signgd, "model_name": model_name, "base_ntk": base_path,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[done] saved -> {out_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
