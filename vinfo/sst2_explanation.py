import warnings
warnings.filterwarnings('ignore')

# multi-processing for NTK kernel
from torch.multiprocessing import set_start_method, set_sharing_strategy
import torch.multiprocessing as mp

# Prevent multiprocessing context error
try:
    set_start_method("spawn")
except RuntimeError:
    pass  # Context already set
set_sharing_strategy("file_system")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import yaml as yaml
import click
from datetime import datetime
import random
from datasets import load_dataset, concatenate_datasets, Value
import torch

import sys

sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

os.environ["OMP_NUM_THREADS"] = '16'
os.environ["OPENBLAS_NUM_THREADS"] = '16'
os.environ["MKL_NUM_THREADS"] = '16'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from dataset import *
from probe import *
from dvutils.Data_Shapley import *
import logging

from dataclasses import dataclass, field
from transformers import HfArgumentParser

import time

if __name__ == '__main__':
    dataset_name="sst2"
    seed=2023
    num_dp=5000
    tmc_iter=500
    prompt=True # usually True, whether use prompt fine-tuning
    signgd=False # usually False, whether use signGD kernel; not adopted in FreeShap
    approximate="inv" # can also be "none" (use no approximation, exact inverse); "diagonal" (use block diagonal for inverse)
    per_point=True # if True: get the instance score for each test point; if False: get instance score for test sets
    early_stopping="True"
    tmc_seed=2023
    val_sample_num = 1000
    yaml_path="../configs/dshap/sst2/ntk_prompt.yaml"
    file_path = "./test/"

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']
    probe_model = yaml_args['probe_com']
    dshap_com = yaml_args['dshap_com']
    if prompt:
        probe_model.model.init(list_dataset.label_word_list)
    if approximate != "none":
        probe_model.approximate(approximate)
    if approximate == "inv":
        probe_model.normalize_ntk()
    if signgd:
        probe_model.signgd()
    np.random.seed(seed)
    random.seed(seed)
        
    if dataset_name == "sst2":
        dataset = load_dataset("sst2")
    elif dataset_name == "mr":
        dataset = load_dataset("rotten_tomatoes")
    elif dataset_name == "rte":
        dataset = load_dataset("glue", "rte")
        # 1: not entail; 0: entail
    elif dataset_name == "mnli":
        dataset = load_dataset("glue", "mnli")
    elif dataset_name == "mrpc":
        dataset = load_dataset("glue", "mrpc")
    # Sample 10 data points from the dataset
    train_data = dataset['train']
    train_data = train_data.map(lambda example, idx: {'idx': idx}, with_indices=True)
    train_data = train_data.shuffle(seed).select(range(min(train_data.num_rows, num_dp)))
    sampled_idx = train_data['idx']

    if dataset_name == "mnli":
        val_num = dataset['validation_matched'].num_rows
    elif dataset_name == "subj" or dataset_name == "ag_news":
        val_num = dataset['test'].num_rows
    else:
        val_num = dataset['validation'].num_rows
    if val_sample_num > val_num:
        sampled_val_idx = np.arange(val_num)
    else:
        sampled_val_idx = np.random.choice(np.arange(val_num), val_sample_num, replace=False).tolist()
        
    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    valid_data = dataset['validation']
    reindex_valid_data = []
    for index in sampled_val_idx:
        reindex_valid_data.append(valid_data[int(index)])
        
    print(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl")
    try:
        with open(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl", "rb") as f:
            ntk = pickle.load(f)
        print("++++++++++++++++++++++++++++++++++++using cached ntk++++++++++++++++++++++++++++++++++++")
        probe_model.get_cached_ntk(ntk)
        probe_model.get_train_labels(list_dataset.get_idx_dataset(sampled_idx, split="train"))
    except:
        print("++++++++++++++++++++++++++++++++++no cached ntk, computing+++++++++++++++++++++++++++++++++++")
        train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
        val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
        # Given that train_loader and val_loader are provided in run(), prepare datasets
        # Set parameters for ntk computation
        # compute ntk matrix
        ntk = probe_model.compute_ntk(train_set, val_set)
        # save the ntk matrix
        with open(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl", "wb") as f:
            pickle.dump(ntk, f)
        print("++++++++++++++++++++++++++++++++++saving ntk cache+++++++++++++++++++++++++++++++++++")
        
    shapley_file_path=f"{file_path}{dataset_name}_{model_name}_shapley_result_seed{seed}_num{num_dp}_appro{approximate}_sign{signgd}_earlystop{early_stopping}_tmc{tmc_seed}_iter{tmc_iter}.pkl"
    try:
        with open(shapley_file_path,'rb') as f:
            result_dict = pickle.load(f)
        print(f"Loading FreeShap result from {shapley_file_path}")
    except:
        print("Computing FreeShap Results")
        dv_result = dshap_com.run(data_idx=sampled_idx, val_data_idx=sampled_val_idx, iteration=tmc_iter,
                                      use_cache_ntk=True, prompt=prompt, seed=tmc_seed, num_dp=num_dp,
                                      checkpoint=False, per_point=per_point, early_stopping=early_stopping)

        mc_com = np.array(dshap_com.mc_cache)
        result_dict = {'dv_result': dv_result,  # entropy, accuracy
                       'sampled_idx': sampled_idx}
        with open(shapley_file_path, "wb") as f:
            pickle.dump(result_dict, f)
        print(f"Saving FreeShap result to {shapley_file_path}")
        
    acc = result_dict['dv_result'][:, 1, :]
    acc_sum = np.sum(acc, axis=0)

    top_10_high = {}
    top_10_low = {}

    idx=535
    column_vector = acc[:, idx]
    print(f"{reindex_valid_data[int(idx)]}")
    top_10_high[idx] = np.argsort(column_vector)[-10:][::-1]  # Indices of top 5 highest values
    print("================================ Most influential ==========================")
    for aindex in top_10_high[idx]:
        print(f"score: {column_vector[int(aindex)]}  |  {train_data[int(aindex)]}")
    print("================================ Least influential ==========================")
    top_10_low[idx] = np.argsort(column_vector)[:10]  # Indices of top 5 lowest values
    for aindex in top_10_low[idx]:
        print(f"score: {column_vector[int(aindex)]}  |  {train_data[int(aindex)]}")
    print()
    print()

    acc = result_dict['dv_result'][:, 1, :]
    acc_sum = np.sum(acc, axis=0)

    top_10_high = {}
    top_10_low = {}

    idx=1
    column_vector = acc[:, idx]
    print(f"{reindex_valid_data[int(idx)]}")
    top_10_high[idx] = np.argsort(column_vector)[-10:][::-1]  # Indices of top 5 highest values
    print("================================ Most influential ==========================")
    for aindex in top_10_high[idx]:
        print(f"score: {column_vector[int(aindex)]}  |  {train_data[int(aindex)]}")
    print("================================ Least influential ==========================")
    top_10_low[idx] = np.argsort(column_vector)[:10]  # Indices of top 5 lowest values
    for aindex in top_10_low[idx]:
        print(f"score: {column_vector[int(aindex)]}  |  {train_data[int(aindex)]}")
    print()
    print()

    acc = result_dict['dv_result'][:, 1, :]
    acc_sum = np.sum(acc, axis=1)

    sorted_indices = np.argsort(acc_sum)[::-1]
            
    top = 5
    cur = 0
    # top - sample
    equal_symbol="="* 35
    print(f"{equal_symbol} Most Helpful {equal_symbol}")
    for index in sorted_indices[:top]:
        print(f"score: {acc_sum[int(index)]} | {train_data[int(index)]}")
    print(f"{equal_symbol} Most Harmful {equal_symbol}")
    for index in sorted_indices[-top:]:
        print(f"score: {acc_sum[int(index)]} | {train_data[int(index)]}")

