# -*- coding: utf-8 -*-
"""task_ntk.py의 vision 대응.

원본 대비 변경:
  - HF datasets 라이브러리 import 제거 (torchvision으로 로드)
  - dataset_name == "sst2"/... NLP 분기 제거
  - prompt = False (label_word_list.init 안 부름)
  - model_name = 'resnet' (bert/llama/roberta 대응)
경로/저장 형식은 freeshap과 동일:
  freeshap_res/ntk/{dataset_name}/{model_name}_seed{seed}_num{N}_val{V}_sign{signgd}.pkl
"""
import warnings; warnings.filterwarnings('ignore')
import gc, os, sys, pickle, random
import yaml as yaml
import torch
import numpy as np

# torch multiprocessing (freeshap 원본과 동일 설정)
from torch.multiprocessing import set_sharing_strategy
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
set_sharing_strategy("file_system")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]      = '16'
os.environ["OPENBLAS_NUM_THREADS"] = '16'
os.environ["MKL_NUM_THREADS"]      = '16'

# freeshap 원본 스크립트 경로도 추가 (probe/dataset 모듈 로드용)
_HERE = os.path.dirname(os.path.abspath(__file__))
_VINFO = os.path.dirname(_HERE)
sys.path.insert(0, _VINFO)
sys.path.insert(0, os.path.join(_VINFO, 'lmntk'))

# ── freeshap 기본 인프라 ──
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: F401  (yaml tag 등록)
from dataset import *                               # noqa: F401  (혹시 필요 시)
from probe   import *                               # noqa: F401  (NTKProbe base 등록)

# ── vision 신규 클래스 (yaml tag 등록됨) ──
from vision.vision_dataset import VisionReader, VisionDataset  # noqa: F401
from vision.vision_probe   import NTKVisionProbe                # noqa: F401

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name",   type=str, default="cifar10")
    p.add_argument("--seed",           type=int, default=2024)
    p.add_argument("--num_train_dp",   type=int, default=2500)
    p.add_argument("--val_sample_num", type=int, default=1000)
    p.add_argument("--config",         type=str, default="ntk_vision",
                   help="YAML config name without .yaml (예: ntk_vision)")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_name   = args.dataset_name
    seed           = args.seed
    num_train_dp   = args.num_train_dp
    val_sample_num = args.val_sample_num

    prompt = False   # vision은 prompt 없음
    signgd = False

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"
    base_path = f"./freeshap_res/ntk/{dataset_name}"
    os.makedirs(base_path, exist_ok=True)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed); random.seed(seed)

    # ===== YAML 로드 =====
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']       # VisionDataset
    probe_model  = yaml_args['probe_com']     # NTKVisionProbe

    if prompt:
        # 절대 안 들어옴 — 하위 호환용
        probe_model.model.init(list_dataset.label_word_list)

    # ===== train / val 인덱스 샘플 =====
    total_train = list_dataset.len_train()
    total_val   = list_dataset.len_val()
    print(f'[info] Total train size = {total_train}, val size = {total_val}')

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_train)
    sampled_idx = perm[:min(total_train, num_train_dp)].tolist()
    print(f'[info] Sampled train: {len(sampled_idx)} (of {total_train})')

    if val_sample_num is None or val_sample_num > total_val:
        sampled_val_idx = list(range(total_val))
        val_sample_num = total_val
        print(f"[info] Using entire validation set: {val_sample_num}")
    else:
        sampled_val_idx = rng.choice(total_val, val_sample_num, replace=False).tolist()
        print(f"[info] Sampled val: {val_sample_num}")

    # ===== model_name 결정 (파일명 접두사) =====
    m = probe_model.args['model']
    if 'resnet' in m:
        model_name = 'resnet'
    elif 'resnext' in m:
        model_name = 'resnext'
    else:
        model_name = 'model'

    # ===== NTK 경로 (freeshap과 완전 동일 규약) =====
    ntk_path = (
        f"{base_path}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")

    # ===== 캐시 로드 시도 =====
    try:
        with open(ntk_path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "ntk" in bundle:
            ntk = bundle["ntk"]
            si  = bundle.get("sampled_idx", None)
            svi = bundle.get("sampled_val_idx", None)
        else:
            ntk = bundle; si = svi = None
        print("++++++ using cached ntk ++++++")
        if si is not None and svi is not None:
            sampled_idx     = si
            sampled_val_idx = svi
    except Exception as e:
        print("++++++ no cached ntk, computing ++++++")
        print(f"[info] reason: {e}")

        train_set = list_dataset.get_idx_dataset(sampled_idx,     split="train")
        val_set   = list_dataset.get_idx_dataset(sampled_val_idx, split="val")

        ntk = probe_model.compute_ntk(train_set, val_set)
        if isinstance(ntk, torch.Tensor):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ntk = ntk.detach().cpu().contiguous()

        bundle = {
            "ntk":            ntk,
            "sampled_idx":    sampled_idx,
            "sampled_val_idx": sampled_val_idx,
            "meta": {
                "dataset_name":  dataset_name,
                "seed":          seed,
                "num_train_dp":  num_train_dp,
                "val_sample_num": val_sample_num,
                "signgd":        signgd,
                "model_name":    model_name,
            }
        }
        with open(ntk_path, "wb") as f:
            pickle.dump(bundle, f)
        print("++++++ saved ntk cache ++++++")

    # cleanup
    print("[done] NTK cached with indices.")
    print(f"[done] sampled_idx len = {len(sampled_idx)}")
    print(f"[done] sampled_val_idx len = {len(sampled_val_idx)}")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
    sys.exit(0)
