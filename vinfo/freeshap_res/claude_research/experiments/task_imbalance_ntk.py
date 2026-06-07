"""task_imbalance_ntk.py — NTK with stratified (imbalanced) train sampling.

Fork of `vinfo/task_ntk.py` with one difference:
  - Adds `--pos_ratio` argument (e.g., 0.5, 0.7, 0.9).
  - Train indices are stratified-sampled so that the positive class makes up
    `pos_ratio` of the n_train samples (rounded; minority gets the remainder).
  - NTK cache path includes a `pos_<NN>` tag to disambiguate.

Run from `vinfo/`:
    python ./freeshap_res/claude_research/experiments/task_imbalance_ntk.py \
        --seed 2026 --dataset_name sst2 --num_train_dp 2000 \
        --val_sample_num 872 --pos_ratio 0.7
"""

import warnings
warnings.filterwarnings('ignore')
import gc

import os
import sys
# Make cwd (expected = vinfo/) importable BEFORE we touch any project module.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from torch.multiprocessing import set_start_method, set_sharing_strategy
from dvutils.Data_Shapley import Fast_Data_Shapley
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
set_sharing_strategy("file_system")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import yaml as yaml
import random
from datasets import load_dataset
import torch
import numpy as np

os.environ["OMP_NUM_THREADS"] = '16'
os.environ["OPENBLAS_NUM_THREADS"] = '16'
os.environ["MKL_NUM_THREADS"] = '16'

from dataset import *
from probe import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_train_dp", type=int, default=2000)
    parser.add_argument("--val_sample_num", type=int, default=None)
    parser.add_argument("--pos_ratio", type=float, default=None,
                        help="Binary only. Fraction of label==1 samples. "
                             "e.g., 0.7 -> 1400 label1 + 600 label0.")
    parser.add_argument("--class_ratios", type=str, default=None,
                        help="Comma-separated class fractions indexed by label "
                             "(e.g., '0.8,0.1,0.1' for 3-class).")
    parser.add_argument("--val_balance", action="store_true",
                        help="If set, val sample is stratified to equal per-class "
                             "counts (val_sample_num // n_classes per class).")
    parser.add_argument("--val_pos_ratio", type=float, default=None,
                        help="Binary only. Fraction of label==1 in val (stratified). "
                             "Mutually exclusive with --val_balance. "
                             "Path tag will be valimb{N}_pos{XX}.")
    parser.add_argument("--val_class_ratios", type=str, default=None,
                        help="Comma-separated val class fractions for multi-class "
                             "imbalanced val (e.g., '0.6,0.2,0.2' for 3-class). "
                             "Mutually exclusive with --val_balance and --val_pos_ratio. "
                             "Path tag will be valimb{N}_cls{a}_{b}_...")
    parser.add_argument("--config", type=str, default="ntk_prompt")
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"[info] ignoring unrecognized args: {_unknown}")
    if args.pos_ratio is None and args.class_ratios is None:
        raise ValueError("Provide one of --pos_ratio or --class_ratios.")
    return args


def parse_ratios(args):
    """Return a list of class fractions (summing to ~1)."""
    if args.class_ratios is not None:
        ratios = [float(x) for x in args.class_ratios.split(",")]
    else:
        # binary: pos_ratio = label-1 fraction
        ratios = [1.0 - args.pos_ratio, args.pos_ratio]
    s = sum(ratios)
    if abs(s - 1.0) > 1e-3:
        raise ValueError(f"class_ratios must sum to 1, got {s}: {ratios}")
    return ratios


def stratified_sample_idx_multi(labels, n_total, class_ratios, rng):
    """Stratified sample to target class ratios.
    class_ratios is a list of floats indexed by label (label 0 first).
    Returns (sampled_idx_list, per_class_counts).
    """
    labels = np.asarray(labels)
    counts = [int(round(n_total * r)) for r in class_ratios]
    # Adjust rounding residual on the largest class
    diff = n_total - sum(counts)
    counts[int(np.argmax(counts))] += diff

    chunks = []
    for c, k in enumerate(counts):
        idx_c = np.where(labels == c)[0]
        if k > len(idx_c):
            raise ValueError(f"class {c}: need {k} samples but only {len(idx_c)} available.")
        chunks.append(rng.choice(idx_c, k, replace=False))
    sampled = np.concatenate(chunks)
    rng.shuffle(sampled)
    return sampled.tolist(), counts


def ratio_tag(class_ratios):
    """Binary -> 'pos<NN>' (NN = label-1 frac). Multi-class -> 'cls<NN>_<NN>_...'."""
    if len(class_ratios) == 2:
        pos_pct = int(round(class_ratios[1] * 100))
        return f"pos{pos_pct:02d}"
    return "cls" + "_".join(f"{int(round(r * 100)):02d}" for r in class_ratios)


def main():
    args = parse_args()
    class_ratios = parse_ratios(args)
    tag = ratio_tag(class_ratios)

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num

    prompt = True
    signgd = False

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"
    base_path = (
        f"./freeshap_res/claude_research/imbalance_ntk/{dataset_name}/{tag}"
    )
    os.makedirs(base_path, exist_ok=True)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

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
    elif dataset_name == "qqp":
        dataset = load_dataset("glue", "qqp")
    elif dataset_name == "mnli":
        dataset = load_dataset("glue", "mnli")
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
    elif dataset_name == "mrpc":
        dataset = load_dataset("glue", "mrpc")
    elif dataset_name == "rte":
        dataset = load_dataset("glue", "rte")
    else:
        raise ValueError(f"unsupported dataset_name={dataset_name}.")

    train_data = dataset['train']
    total_train_size = train_data.num_rows
    print(f"[info] Total train dataset size: {total_train_size}")

    # ===== stratified train idx =====
    labels = np.asarray(train_data['label'])
    rng = np.random.RandomState(seed)
    sampled_idx, per_class_counts = stratified_sample_idx_multi(
        labels, num_train_dp, class_ratios, rng,
    )
    print(f"[info] Stratified train: per_class_counts={per_class_counts}, "
          f"total={len(sampled_idx)} (class_ratios={class_ratios})")

    # ===== validation set =====
    if dataset_name == "mnli":
        val_split = "validation_matched"
    elif dataset_name == "ag_news":
        val_split = "test"
    else:
        val_split = "validation"
    val_data = dataset[val_split]
    val_num = val_data.num_rows
    print(f"[info] Total validation dataset size: {val_num} (split={val_split})")

    if args.val_pos_ratio is not None:
        # Binary-only imbalanced val (stratified). label1 fraction = val_pos_ratio.
        if len(class_ratios) != 2:
            raise ValueError("--val_pos_ratio is binary-only.")
        val_labels_arr = np.asarray(val_data['label'])
        val_pos_count = int(round(val_sample_num * args.val_pos_ratio))
        val_neg_count = val_sample_num - val_pos_count
        idx_0 = np.where(val_labels_arr == 0)[0]
        idx_1 = np.where(val_labels_arr == 1)[0]
        if val_neg_count > len(idx_0):
            raise ValueError(
                f"val_pos_ratio: label0 needs {val_neg_count} but only "
                f"{len(idx_0)} available in {val_split}.")
        if val_pos_count > len(idx_1):
            raise ValueError(
                f"val_pos_ratio: label1 needs {val_pos_count} but only "
                f"{len(idx_1)} available in {val_split}.")
        rng_val = np.random.RandomState(seed)
        chunks = [rng_val.choice(idx_0, val_neg_count, replace=False),
                  rng_val.choice(idx_1, val_pos_count, replace=False)]
        sampled_val_idx = np.concatenate(chunks)
        rng_val.shuffle(sampled_val_idx)
        sampled_val_idx = sampled_val_idx.tolist()
        per_class_counts_val = [val_neg_count, val_pos_count]
        print(f"[info] val imbalanced: label0={val_neg_count}, label1={val_pos_count}, "
              f"total={val_sample_num} (val_pos_ratio={args.val_pos_ratio})")
    elif args.val_class_ratios is not None:
        val_class_ratios_list = [float(x) for x in args.val_class_ratios.split(",")]
        if len(val_class_ratios_list) != len(class_ratios):
            raise ValueError(
                f"val_class_ratios length {len(val_class_ratios_list)} "
                f"!= n_classes {len(class_ratios)}")
        val_labels_arr = np.asarray(val_data['label'])
        chunks = []
        per_class_counts_val = []
        rng_val = np.random.RandomState(seed)
        for c, r in enumerate(val_class_ratios_list):
            k = int(round(val_sample_num * r))
            idx_c = np.where(val_labels_arr == c)[0]
            if k > len(idx_c):
                raise ValueError(
                    f"val_class_ratios: class {c} needs {k} but only "
                    f"{len(idx_c)} available in {val_split}.")
            chunks.append(rng_val.choice(idx_c, k, replace=False))
            per_class_counts_val.append(k)
        sampled_val_idx = np.concatenate(chunks)
        rng_val.shuffle(sampled_val_idx)
        sampled_val_idx = sampled_val_idx.tolist()
        val_sample_num = len(sampled_val_idx)
        print(f"[info] val class_ratios: {val_class_ratios_list}, "
              f"per_class={per_class_counts_val}, total={val_sample_num}")
    elif args.val_balance:
        n_classes = len(class_ratios)
        per_class = val_sample_num // n_classes
        remainder = val_sample_num - per_class * n_classes
        val_labels_arr = np.asarray(val_data['label'])
        rng_val = np.random.RandomState(seed)
        # Pick which classes receive +1 (random class selection if remainder > 0)
        extra_for_class = np.zeros(n_classes, dtype=int)
        if remainder > 0:
            extra_classes = rng_val.choice(n_classes, remainder, replace=False)
            for c in extra_classes:
                extra_for_class[c] = 1
        chunks = []
        per_class_counts_val = []
        for c in range(n_classes):
            k = per_class + int(extra_for_class[c])
            idx_c = np.where(val_labels_arr == c)[0]
            if k > len(idx_c):
                raise ValueError(
                    f"val_balance: class {c} needs {k} but only "
                    f"{len(idx_c)} available in {val_split}.")
            chunks.append(rng_val.choice(idx_c, k, replace=False))
            per_class_counts_val.append(k)
        sampled_val_idx = np.concatenate(chunks)
        rng_val.shuffle(sampled_val_idx)
        sampled_val_idx = sampled_val_idx.tolist()
        val_sample_num = len(sampled_val_idx)
        print(f"[info] val_balance: per_class={per_class_counts_val}, "
              f"total={val_sample_num} (remainder={remainder} dispatched to random class)")
    elif val_sample_num is None or val_sample_num > val_num:
        sampled_val_idx = np.arange(val_num).tolist()
        val_sample_num = val_num
        print(f"[info] Using entire validation set: {val_sample_num} samples")
    else:
        sampled_val_idx = np.random.choice(
            np.arange(val_num), val_sample_num, replace=False,
        ).tolist()
        print(f"[info] Using sampled validation set: {val_sample_num} samples")

    # ===== model name =====
    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    # ===== NTK cache path =====
    if args.val_pos_ratio is not None:
        val_tag = f"valimb{val_sample_num}_pos{int(round(args.val_pos_ratio * 100)):02d}"
    elif args.val_class_ratios is not None:
        parts = [int(round(float(r) * 100)) for r in args.val_class_ratios.split(",")]
        val_tag = f"valimb{val_sample_num}_cls" + "_".join(f"{p:02d}" for p in parts)
    elif args.val_balance:
        val_tag = f"valbal{val_sample_num}"
    else:
        val_tag = f"val{val_sample_num}"
    ntk_path = (
        f"{base_path}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_{val_tag}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")

    # ===== cache check =====
    try:
        with open(ntk_path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "ntk" in bundle:
            print(f"[info] cached NTK found, skipping recompute.")
            return
    except Exception as e:
        print(f"[info] no cached ntk -> computing. reason: {e}")

    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")

    ntk = probe_model.compute_ntk(train_set, val_set)

    if isinstance(ntk, torch.Tensor):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
            "class_ratios": class_ratios,
            "per_class_counts": per_class_counts,
            "ratio_tag": tag,
            "val_balance": bool(args.val_balance),
        },
    }
    with open(ntk_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[done] saved NTK to {ntk_path}")

    # cleanup
    children = mp.active_children()
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
