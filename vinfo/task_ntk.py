import warnings
warnings.filterwarnings('ignore')
import gc

# multi-processing for NTK kernel
from torch.multiprocessing import set_start_method, set_sharing_strategy
from dvutils.Data_Shapley import Fast_Data_Shapley
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
# Use file_system instead of file_descriptor to avoid "Too many open files" error
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mrpc")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_train_dp", type=int, default=1000)
    parser.add_argument("--val_sample_num", type=int, default=None,
                        help="Number of validation samples. If None, use entire validation set.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num

    prompt = True
    signgd = False

    # Dynamic path construction based on dataset_name
    yaml_path = f"../configs/dshap/{dataset_name}/ntk_prompt.yaml"
    base_path = f"./freeshap_res/{dataset_name}"
    os.makedirs(base_path, exist_ok=True)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    # ===== YAML 로드 =====
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']
    probe_model = yaml_args['probe_com']

    if prompt:
        probe_model.model.init(list_dataset.label_word_list)

    if signgd:
        probe_model.signgd()

    # ===== HF dataset 로드 =====
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
    else:
        raise ValueError(f"unknown dataset_name={dataset_name}")

    # ===== train subset 샘플링 =====
    train_data = dataset['train']
    total_train_size = train_data.num_rows
    print(f'[info] Total train dataset size: {total_train_size}')
    
    train_data = train_data.map(lambda example, idx: {'idx': idx}, with_indices=True)
    train_data = train_data.shuffle(seed).select(range(min(train_data.num_rows, num_train_dp)))
    sampled_idx = train_data['idx']
    print(f'[info] Sampled train data: {len(sampled_idx)} samples (from {total_train_size})')

    # ===== validation subset 샘플링 =====
    if dataset_name == "mnli":
        val_num = dataset['validation_matched'].num_rows
    elif dataset_name == "subj" or dataset_name == "ag_news":
        val_num = dataset['test'].num_rows
    else:
        val_num = dataset['validation'].num_rows
    
    print(f'[info] Total validation dataset size: {val_num}')

    # If val_sample_num is None or greater than available, use entire validation set
    if val_sample_num is None or val_sample_num > val_num:
        sampled_val_idx = np.arange(val_num).tolist()
        val_sample_num = val_num  # Update for filename consistency
        print(f"[info] Using entire validation set: {val_sample_num} samples")
    else:
        sampled_val_idx = np.random.choice(np.arange(val_num), val_sample_num, replace=False).tolist()
        print(f"[info] Using sampled validation set: {val_sample_num} samples")

    # ===== 모델 이름 결정 =====
    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    # ===== NTK 캐시 경로 =====
    os.makedirs(f"{base_path}/ntk", exist_ok=True)
    ntk_path = (
        f"{base_path}/ntk/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")

    # ===== 캐시 로드 시도 =====
    try:
        with open(ntk_path, "rb") as f:
            bundle = pickle.load(f)

        # 과거 포맷(그냥 ntk만 저장)도 대응
        if isinstance(bundle, dict) and "ntk" in bundle:
            ntk = bundle["ntk"]
            sampled_idx_loaded = bundle.get("sampled_idx", None)
            sampled_val_idx_loaded = bundle.get("sampled_val_idx", None)
        else:
            ntk = bundle
            sampled_idx_loaded = None
            sampled_val_idx_loaded = None

        print("++++++++++++++++++++++++++++++++++++using cached ntk++++++++++++++++++++++++++++++++++++")
        # indices가 저장되어 있으면 그것을 신뢰 (재현성 보장)
        if sampled_idx_loaded is not None and sampled_val_idx_loaded is not None:
            sampled_idx = sampled_idx_loaded
            sampled_val_idx = sampled_val_idx_loaded

    except Exception as e:
        print("++++++++++++++++++++++++++++++++++no cached ntk, computing+++++++++++++++++++++++++++++++++++")
        print(f"[info] reason: {e}")

        train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
        val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")

        ntk = probe_model.compute_ntk(train_set, val_set)

        if isinstance(ntk, torch.Tensor):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            ntk = ntk.detach().cpu().contiguous()

        bundle = {
            "ntk": ntk,
            "sampled_idx": sampled_idx,
            "sampled_val_idx": sampled_val_idx,
            "meta": {
                "dataset_name": dataset_name,
                "seed": seed,
                "num_train_dp": num_train_dp,
                "val_sample_num": val_sample_num,
                "signgd": signgd,
                "model_name": model_name,
            }
        }
        with open(ntk_path, "wb") as f:
            pickle.dump(bundle, f)
        gc.collect()
        
        children = mp.active_children()
        if children:
            print(f"[cleanup] active children = {len(children)}")
        for p in children:
            p.join(timeout=5)
        for p in children:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        print("++++++++++++++++++++++++++++++++++saving ntk cache+++++++++++++++++++++++++++++++++++")

    # 최종 확인
    # probe_model.ntk.shape 찍고 싶으면 get_cached_ntk 호출해야 하는데,
    # 여기서는 “계산/저장만”이 목적이라 굳이 로드하지 않아도 됨.
    print("[done] NTK cached with indices.")
    print(f"[done] sampled_idx len = {len(sampled_idx)}")
    print(f"[done] sampled_val_idx len = {len(sampled_val_idx)}")


if __name__ == "__main__":
    # try:
    #     mp.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass
    # set_sharing_strategy("file_system")
    mp.set_start_method("spawn", force=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = '16'
    os.environ["OPENBLAS_NUM_THREADS"] = '16'
    os.environ["MKL_NUM_THREADS"] = '16'

    main()
