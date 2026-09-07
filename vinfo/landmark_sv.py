# -*- coding: utf-8 -*-
"""SV 기반 Nystrom landmark 모드용 공통 헬퍼.

nystrom_top / nystrom_bottom 모드는 **같은 seed 의 exact(inv) Shapley value** 로
학습점을 정렬해 상위/하위 d 개를 landmark 로 쓴다. 그 SV 를 본진(freeshap_res)에서
읽어오는 경로 조립과 로딩을 여기 모아 4 개 task 파일이 공유한다.

주의: nys 결과는 out_root(예: ./jitter_exp/nys_landmark_top_res) 로 격리되지만,
      landmark 재료인 inv SV 는 **본진 out_root(기본 ./freeshap_res)** 에 있으므로
      경로를 따로 받는다 (--landmark_sv_root).
"""
import os
import pickle

import numpy as np


def inv_sv_path(sv_root, dataset_name, model_name, seed, num_train_dp,
                val_sample_num, inv_lambda_, signgd=False,
                early_stopping="True", tmc_iter=500):
    """inv Shapley pkl 경로. task_shapley 의 저장 규약과 동일한 stem 을 조립한다."""
    stem = (f"{model_name}_seed{seed}_num{num_train_dp}_val{val_sample_num}"
            f"_lam{float(inv_lambda_):.0e}_sign{signgd}"
            f"_earlystop{early_stopping}_tmc{tmc_iter}.pkl")
    return f"{sv_root}/shapley/{dataset_name}/inv/{stem}"


def load_inv_sv(path, expect_n=None):
    """inv pkl 에서 학습점별 SV 배열을 읽는다.

    dv_result[:, 1, :] 가 (num_train_dp, val_sample_num) 의 accuracy 기여도이고,
    val 축 합이 task_data_selection 등이 쓰는 것과 동일한 per-point SV 다.
    커널 행 순서(= sampled_idx 순서)와 정렬돼 있으므로 인덱스 매핑이 필요 없다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[landmark_sv] inv Shapley 파일이 없습니다: {path}\n"
            f"  nystrom_top/bottom 은 같은 seed 의 inv 결과가 먼저 있어야 합니다.\n"
            f"  --landmark_sv_root 를 확인하거나 해당 inv 셀을 먼저 돌리세요.")
    with open(path, "rb") as f:
        result = pickle.load(f)
    sv = np.array(result["dv_result"])[:, 1, :].sum(axis=1).astype(np.float64)
    if expect_n is not None and sv.shape[0] != expect_n:
        raise ValueError(f"[landmark_sv] SV 길이 {sv.shape[0]} != num_train_dp {expect_n} ({path})")
    return sv


def resolve_landmark_sv(args, dataset_name, model_name, seed, num_train_dp,
                        val_sample_num, inv_lambda_, tmc_iter=500):
    """CLI 인자에서 SV 를 해석. --landmark_sv_pkl 이 있으면 그것을, 없으면 자동 조립."""
    path = getattr(args, "landmark_sv_pkl", None) or inv_sv_path(
        getattr(args, "landmark_sv_root", "./freeshap_res"),
        dataset_name, model_name, seed, num_train_dp, val_sample_num,
        inv_lambda_, tmc_iter=tmc_iter)
    sv = load_inv_sv(path, expect_n=num_train_dp)
    print(f"[landmark_sv] loaded {path}  (n={sv.shape[0]}, "
          f"min={sv.min():.4e}, median={np.median(sv):.4e}, max={sv.max():.4e})")
    return sv


def add_landmark_args(parser):
    """4 개 task 가 공유하는 CLI 인자."""
    parser.add_argument("--landmark_sv_root", type=str, default="./freeshap_res",
                        help="nystrom_top/bottom 의 landmark 재료인 inv Shapley 결과 루트 "
                             "(out_root 와 별개 — nys 결과는 격리 폴더에 쓰고 SV 는 본진에서 읽는다)")
    parser.add_argument("--landmark_sv_pkl", type=str, default=None,
                        help="inv Shapley pkl 경로를 직접 지정 (지정 시 --landmark_sv_root 무시)")
    return parser
