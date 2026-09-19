"""held-out 평가 집합의 정의 — seed 와 무관하게 고정한다.

배경
----
data selection 실험은 "점수를 매긴 집합"과 "성능을 재는 집합"이 달라야 한다
(FreeShap 논문 App. F.2: 점수는 val 에서, 평가는 별도 held-out 에서).
기존 실험은 둘 다 val 이라 in-sample 이었고, 그 결과는
freeshap_res/data_selection_insample/ 에 보관되어 있다.

규약
----
* 점수 계산용 집합(val)은 그대로 둔다  -> Shapley 값은 재계산하지 않는다.
* held-out 은 **seed 와 무관하게 고정**한다. train subset 은 seed 마다 다르므로
  (task_ntk.py 의 shuffle(seed)), 세 seed 의 train 합집합을 모두 제외하고 뽑는다.
  그래야 어느 seed 에서도 학습에 쓰이지 않은 점들만 남고, seed 간 분산이
  "평가 집합이 달라서" 생기는 일이 없다.
* MR 은 train 풀이 8530 인데 3 seed 합집합이 7939 라 1000 개를 못 뽑는다.
  대신 공식 test split(1066, 라벨 공개)이 통째로 안 쓰이고 있어 그것을 쓴다.
  FreeShap 논문이 MR 에서 held-out 실험을 한 것도 이 split 이다.
* RTE(2490) / MRPC(3668) 은 train split 전체를 이미 쓰고 있어 남는 점이 없다.
  이 둘은 train 을 쪼개야 하고, 그러면 n 이 바뀌어 Shapley 재계산이 필요하다 (미결).
"""

import os
import pickle

import numpy as np

HO_SIZE_DEFAULT = 1000
HO_SEED_DEFAULT = 42
SEEDS_DEFAULT = (2024, 2025, 2026)

# held-out 을 어느 split 에서 뽑는가 (여기 없으면 "train")
HO_SPLIT = {"mr": "test"}

# 데이터셋별 기본 크기 (여기 없으면 HO_SIZE_DEFAULT).
#   MR 은 test split(1066) 을 통째로 쓴다 — FreeShap 논문이 쓴 그 집합이다.
#   RTE/MRPC 는 아래 FIXED_SPLIT 의 held-out 크기.
HO_SIZE_BY_DS = {"mr": 1066, "rte": 990, "mrpc": 668}

# train split 전체를 이미 소진해 held-out 을 뗄 수 없는 조합 (num_train_dp 기준)
#   -> 이 n 으로는 불가. 대신 아래 FIXED_SPLIT 의 n 으로 다시 만든다.
NO_ROOM = {("rte", 2490), ("mrpc", 3668)}

# train 을 전부 쓰던 데이터셋: held-out 을 **먼저** seed 42 로 고정하고 train = 나머지 전부.
#   (n_train, n_heldout), 합이 train split 크기와 같아야 한다.
#   train 도 seed 와 무관하게 고정되고, seed 는 TMC 순열(과 순서)만 바꾼다 — 예전 RTE/MRPC 와 같은 구조.
#   n_train 은 기존 캠페인의 num 값(rte 1000/2000/2490, mrpc 1000/2000/3668)과 겹치지 않게 골랐다.
FIXED_SPLIT = {"rte": (1500, 990), "mrpc": (3000, 668)}
FULL_NUM = {"rte": 2490, "mrpc": 3668}          # train split 전체 크기 (그 외 데이터셋은 5000 사용)


def num_of(dataset_name, protocol="heldout", default=5000):
    """selection 결과 파일명의 num 값.
    heldout 프로토콜에서 RTE/MRPC 는 FIXED_SPLIT 의 n_train, 그 외/예전 결과는 FULL_NUM."""
    if protocol == "heldout" and dataset_name in FIXED_SPLIT:
        return FIXED_SPLIT[dataset_name][0]
    return FULL_NUM.get(dataset_name, default)


def fixed_heldout(dataset_name, pool_size, ho_seed=HO_SEED_DEFAULT):
    """FIXED_SPLIT 데이터셋의 held-out 인덱스 (train split 전체에서 seed 42 로 추출, 정렬)."""
    n_tr, n_ho = FIXED_SPLIT[dataset_name]
    if n_tr + n_ho != pool_size:
        raise ValueError(f"{dataset_name}: FIXED_SPLIT {n_tr}+{n_ho} != train split {pool_size}")
    idx = np.sort(np.random.default_rng(ho_seed).choice(pool_size, n_ho, replace=False))
    return [int(i) for i in idx]


def fixed_train_pool(dataset_name, pool_size, ho_seed=HO_SEED_DEFAULT):
    """FIXED_SPLIT 데이터셋의 train 후보 = held-out 을 뺀 나머지 (정렬). task_ntk --exclude_heldout 가 쓴다."""
    H = set(fixed_heldout(dataset_name, pool_size, ho_seed))
    return [i for i in range(pool_size) if i not in H]


def ho_split_of(dataset_name):
    return HO_SPLIT.get(dataset_name, "train")


def resolve_size(dataset_name, size=None):
    """--heldout_size 를 주지 않았을 때의 기본 크기. 세 스크립트가 같은 값을 써야
    파일명(tag)이 일치하므로 여기 한 곳에서 정한다."""
    if size is not None:
        return size
    return HO_SIZE_BY_DS.get(dataset_name, HO_SIZE_DEFAULT)


def ho_tag(size, ho_seed=HO_SEED_DEFAULT, split="train"):
    """파일명 태그. seed 와 무관한 고정 집합임이 드러나게 ho{크기}{t}s{홀드아웃시드}."""
    return f"ho{size}{'t' if split == 'test' else ''}s{ho_seed}"


def ntk_path(root, dataset_name, model_name, seed, num, val, signgd=False):
    return (f"{root}/ntk/{dataset_name}/{model_name}"
            f"_seed{seed}_num{num}_val{val}_sign{signgd}.pkl")


def heldout_ntk_path(root, dataset_name, model_name, seed, num, tag, signgd=False):
    return (f"{root}/ntk_heldout/{dataset_name}/{model_name}"
            f"_seed{seed}_num{num}_{tag}_sign{signgd}.pkl")


def train_union(root, dataset_name, model_name, num, val, seeds=SEEDS_DEFAULT, signgd=False):
    """seeds 각각의 train subset 합집합. NTK 캐시의 sampled_idx 를 그대로 읽는다.

    (여기서 다시 shuffle(seed) 를 돌리지 않는 이유: 실제로 쓰인 인덱스와 100% 일치해야
     겹침이 없다고 보장할 수 있기 때문이다.)
    """
    union, missing = set(), []
    for s in seeds:
        p = ntk_path(root, dataset_name, model_name, s, num, val, signgd)
        if not os.path.exists(p):
            missing.append(p)
            continue
        with open(p, "rb") as f:
            union |= set(int(i) for i in pickle.load(f)["sampled_idx"])
    if missing:
        raise FileNotFoundError(
            "held-out 을 고정하려면 세 seed 의 NTK 캐시가 모두 있어야 한다.\n  "
            + "\n  ".join(missing))
    return union


def pick_heldout(root, dataset_name, model_name, num, val, pool_size,
                 size=HO_SIZE_DEFAULT, ho_seed=HO_SEED_DEFAULT,
                 seeds=SEEDS_DEFAULT, signgd=False, split=None):
    """held-out 인덱스 (정렬된 list) 와 그 출처 split 을 돌려준다.

    pool_size : split="train" 이면 train split 크기, "test" 면 test split 크기.
    """
    split = split or ho_split_of(dataset_name)
    if dataset_name in FIXED_SPLIT and split == "train":
        # held-out 이 먼저 고정된 데이터셋: seed 합집합과 무관하게 항상 같은 집합
        return fixed_heldout(dataset_name, pool_size, ho_seed), split
    if split == "test":
        # 공식 test split 은 학습에 쓰인 적이 없으므로 통째로 쓴다 (크면 size 개만).
        idx = np.arange(pool_size)
        if size < pool_size:
            idx = np.sort(np.random.default_rng(ho_seed).choice(idx, size, replace=False))
        return [int(i) for i in idx], split

    used = train_union(root, dataset_name, model_name, num, val, seeds, signgd)
    pool = np.setdiff1d(np.arange(pool_size), np.fromiter(used, dtype=int, count=len(used)))
    if len(pool) < size:
        raise ValueError(
            f"{dataset_name}: train {pool_size} 중 {len(used)} 개가 이미 쓰여 "
            f"남은 점이 {len(pool)} 개뿐이다 (요청 {size}). "
            f"HO_SPLIT 로 test split 을 쓰거나 크기를 줄여야 한다.")
    idx = np.sort(np.random.default_rng(ho_seed).choice(pool, size, replace=False))
    return [int(i) for i in idx], split


def selection_base(out_root, mode="heldout"):
    """selection 결과 폴더.

      heldout  : val 로 점수, train/test 에서 뗀 고정 집합에서 평가 (현재 프로토콜)
      insample : 점수 집합(val)에서 그대로 평가 — 예전 결과 보관용
    """
    return f"{out_root}/" + {"insample": "data_selection_insample",
                             "heldout": "data_selection"}[mode]


def removal_base(out_root, mode="heldout"):
    """removal 결과 폴더 (selection_base 와 같은 규약).

      heldout  : val 로 점수, held-out 에서 평가 (현재 프로토콜)  -> data_removing
      insample : 점수 집합(val)에서 그대로 평가 — 예전 결과 보관용 -> data_removing_insample
    """
    return f"{out_root}/" + {"insample": "data_removing_insample",
                             "heldout": "data_removing"}[mode]
