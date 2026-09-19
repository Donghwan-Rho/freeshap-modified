"""held-out 평가 집합에 대한 eNTK 블록만 추가로 계산한다.

무엇을 만드나
-------------
기존 캐시  freeshap_res/ntk/{ds}/{model}_seed{S}_num{N}_val{V}_signFalse.pkl
  -> (1, N+V, N) : 행 = train N + val V, 열 = train N
새 캐시    freeshap_res/ntk_heldout/{ds}/{model}_seed{S}_num{N}_{tag}_signFalse.pkl
  -> (1, N+H, N) : 행 = train N + held-out H, 열 = train N   (tag 예: ho1000s42)

train 블록(앞 N행)은 기존 캐시와 같은 점들이므로 downstream 은 이 파일 하나만 읽으면 된다.
저장값은 기존과 동일하게 **정규화 전(raw)** 이다 (probe 가 로드할 때 /10000 한다).

왜 Shapley 재계산이 필요 없나
-----------------------------
Shapley 값은 train 과 val 로만 정해진다. 이 스크립트는 평가용 행만 덧붙이므로
freeshap_res/shapley/ 의 결과를 그대로 쓴다.

주의
----
* held-out 은 seed 와 무관하게 고정된다 (heldout_common.pick_heldout).
  단, 커널의 **열**이 그 seed 의 train 5000점이라 블록 자체는 seed 마다 계산해야 한다.
* RTE(2490) / MRPC(3668) 은 train 을 전부 써서 뗄 자리가 없다 -> 에러로 막는다.
* resnet/cifar10 은 vision 파이프라인(vision/task_*_vision.py)이 따로라 여기서 다루지 않는다.

사용 예
-------
  python vinfo/task_ntk_heldout.py --dataset_name=sst2 --seed=2024 \
      --num_train_dp=5000 --val_sample_num=872 --config=ntk_prompt
"""

import warnings
warnings.filterwarnings('ignore')
import gc

from torch.multiprocessing import set_start_method, set_sharing_strategy
from dvutils.Data_Shapley import Fast_Data_Shapley
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
set_sharing_strategy("file_system")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import yaml as yaml
import random
from datasets import load_dataset
import torch
import numpy as np

import sys
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

os.environ["OMP_NUM_THREADS"] = '16'
os.environ["OPENBLAS_NUM_THREADS"] = '16'
os.environ["MKL_NUM_THREADS"] = '16'

from dataset import *
from probe import *

import argparse

import heldout_common as HO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_train_dp", type=int, default=5000)
    parser.add_argument("--val_sample_num", type=int, default=None,
                        help="기존 NTK 캐시를 찾기 위한 값 (그 파일명에 박힌 숫자).")
    parser.add_argument("--heldout_size", type=int, default=None,
                        help="기본값은 데이터셋별 규약 (MR=1066=test 전체, 나머지=1000).")
    parser.add_argument("--heldout_seed", type=int, default=HO.HO_SEED_DEFAULT,
                        help="held-out 추출 시드. train/val 시드와 분리해 고정한다.")
    parser.add_argument("--heldout_split", type=str, default=None,
                        choices=["train", "test"],
                        help="기본값은 데이터셋별 규약 (MR=test, 나머지=train).")
    parser.add_argument("--union_seeds", type=int, nargs="+", default=list(HO.SEEDS_DEFAULT),
                        help="held-out 에서 제외할 train subset 들의 seed 목록.")
    parser.add_argument("--out_root", type=str, default="./freeshap_res")
    parser.add_argument("--config", type=str, default="ntk_prompt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--grad_chunksize", type=int, default=None,
                        help="한 번에 처리할 파라미터 수. 메모리는 (train+heldout) x 이 값 x 4 bytes 다. "
                             "yaml 기본값(2e7)은 6000행 기준 480GB 라 RAM 이 작은 노드에서는 "
                             "80만 정도로 낮춰야 한다 (대신 파라미터 배치 수가 늘어 느려진다).")
    parser.add_argument("--keep_fresh_train", action="store_true",
                        help="train 블록을 이번에 새로 계산한 값 그대로 둔다. "
                             "기본은 기존 캐시의 train 블록으로 교체 — Shapley 를 계산할 때 "
                             "쓴 커널과 비트 단위로 같게 만들기 위해서다.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    root = args.out_root

    prompt = True
    signgd = False

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    # ===== YAML =====
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']
    probe_model = yaml_args['probe_com']
    if prompt:
        probe_model.model.init(list_dataset.label_word_list)
    if signgd:
        probe_model.signgd()

    # ===== HF dataset (크기 확인용) =====
    if dataset_name == "sst2":
        dataset = load_dataset("sst2")
    elif dataset_name == "mr":
        dataset = load_dataset("rotten_tomatoes")
    elif dataset_name == "rte":
        dataset = load_dataset("glue", "rte")
    elif dataset_name == "mnli":
        dataset = load_dataset("glue", "mnli")
    elif dataset_name == "mrpc":
        dataset = load_dataset("glue", "mrpc")
    elif dataset_name == "qqp":
        dataset = load_dataset("glue", "qqp")
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
    else:
        raise ValueError(f"unknown dataset_name={dataset_name}")

    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    if (dataset_name, num_train_dp) in HO.NO_ROOM:
        raise SystemExit(
            f"[stop] {dataset_name} n={num_train_dp} 은 train split 전체를 쓰고 있어 "
            f"held-out 을 뗄 수 없다. train 을 쪼개면 n 이 바뀌어 Shapley 재계산이 필요하다.")

    # ===== 기존 NTK 캐시에서 이 seed 의 train subset 을 그대로 읽는다 =====
    base_path = HO.ntk_path(root, dataset_name, model_name, seed, num_train_dp,
                            val_sample_num, signgd)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"기존 NTK 캐시가 없다: {base_path}")
    with open(base_path, "rb") as f:
        base_bundle = pickle.load(f)
    sampled_idx = [int(i) for i in base_bundle["sampled_idx"]]
    print(f"[info] train subset from cache: {base_path}  (|train|={len(sampled_idx)})")

    # ===== held-out 인덱스 (seed 무관, 고정) =====
    split = args.heldout_split or HO.ho_split_of(dataset_name)
    ho_size = HO.resolve_size(dataset_name, args.heldout_size)
    if split == "test":
        pool_size = dataset['test'].num_rows
    else:
        pool_size = dataset['train'].num_rows
    heldout_idx, split = HO.pick_heldout(
        root, dataset_name, model_name, num_train_dp, val_sample_num, pool_size,
        size=ho_size, ho_seed=args.heldout_seed,
        seeds=tuple(args.union_seeds), signgd=signgd, split=split)
    # split="test" 면 애초에 다른 split 이라 인덱스 번호가 겹쳐도 무관하다.
    overlap = len(set(heldout_idx) & set(sampled_idx)) if split == "train" else 0
    print(f"[info] held-out: split={split}  |H|={len(heldout_idx)}  "
          f"(pool={pool_size}, seed={args.heldout_seed})"
          + (f"  train 과 겹침={overlap}" if split == "train" else "  (train 과 다른 split)"))
    if split == "train" and overlap:
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

    # ===== eNTK 블록 계산: 행 = train + held-out, 열 = train =====
    #   compute_ntk 는 train 과 test 의 gradient 를 모두 구하므로 비용이
    #   원래 NTK 작업과 비슷하다 (train 쪽 gradient 를 다시 계산하기 때문).
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    heldout_set = list_dataset.get_idx_dataset(heldout_idx, split=split)
    print(f"len(train_set)   = {len(train_set)}")
    print(f"len(heldout_set) = {len(heldout_set)}")

    ntk = probe_model.compute_ntk(train_set, heldout_set)
    if isinstance(ntk, torch.Tensor):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ntk = ntk.detach().cpu().contiguous()
    print(f"[info] ntk shape = {tuple(ntk.shape)}")

    # ===== train 블록을 기존 캐시 값으로 교체 =====
    #   compute_ntk 는 train+heldout 을 함께 돌므로 앞 N행(train x train)도 새로 계산된다.
    #   그런데 Shapley 는 **기존 캐시의** train 블록으로 계산됐다. 둘은 수학적으로 같지만
    #   GPU 축약 순서 때문에 비트 단위로는 다를 수 있어, 앞 N행을 기존 값으로 덮어써
    #   "점수를 낼 때 쓴 커널"과 "평가할 때 쓰는 커널"을 완전히 일치시킨다.
    #   (상대차를 함께 찍어 두면 train subset 을 잘못 잡은 경우도 바로 드러난다.)
    _n = len(sampled_idx)
    _base_ntk = base_bundle["ntk"]
    if not torch.is_tensor(_base_ntk):
        _base_ntk = torch.as_tensor(np.asarray(_base_ntk))
    if tuple(_base_ntk.shape[-1:]) != tuple(ntk.shape[-1:]):
        raise RuntimeError(f"열 개수가 다르다: base {tuple(_base_ntk.shape)} vs new {tuple(ntk.shape)}")
    _new_tr = ntk[..., :_n, :].to(torch.float64)
    _old_tr = _base_ntk[..., :_n, :].to(torch.float64)
    _rel = float((_new_tr - _old_tr).norm() / _old_tr.norm())
    print(f"[check] train 블록 상대차 (새 계산 vs 기존 캐시) = {_rel:.3e}")
    if _rel > 1e-3:
        raise RuntimeError(
            f"train 블록이 너무 다르다 (상대차 {_rel:.3e}). train subset 이나 모델 설정이 "
            f"기존 캐시와 다를 수 있다 — 중단.")
    if args.keep_fresh_train:
        print("[info] --keep_fresh_train: 새로 계산한 train 블록을 그대로 둔다")
    else:
        ntk[..., :_n, :] = _base_ntk[..., :_n, :].to(ntk.dtype)
        print("[info] train 블록을 기존 캐시 값으로 교체했다 (Shapley 와 동일)")

    bundle = {
        "ntk": ntk,
        "sampled_idx": np.array(sampled_idx),
        "heldout_idx": np.array(heldout_idx),
        "heldout_split": split,
        "meta": {
            "dataset_name": dataset_name,
            "seed": seed,
            "num_train_dp": num_train_dp,
            "val_sample_num": val_sample_num,
            "heldout_size": len(heldout_idx),
            "heldout_seed": args.heldout_seed,
            "union_seeds": list(args.union_seeds),
            "signgd": signgd,
            "model_name": model_name,
            "base_ntk": base_path,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[done] saved -> {out_path}")

    children = mp.active_children()
    if children:
        for p in children:
            p.join(timeout=5)
        for p in children:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


if __name__ == "__main__":
    main()
