"""
task_base_accuracy.py
---------------------

Mirror of task_data_selection.py, but restricted to two operating points
that are needed to form ρ(r, f=20%):

  * num_train_selected = 0%  -> inv_base_accuracy
        (empty train; predictor = 0 ∈ R^{|val|×C}; acc = fraction of class-0)
  * num_train_selected = 20% -> inv_top_f20_accuracy
        (inv-top kernel regression at k = 0.2·num_train_dp)

Output
------
    freeshap_res/data_selection/{dataset}/inv/base_accuracy/
        {shapley_setting_name}_base.txt
Contents (same ×10000-int convention as task_data_selection.py):
    inv_base_accuracy:
    <int at f=0>
    inv_top_f20_accuracy:
    <int at f=20>
"""

import os
import sys
import pickle
import argparse
import random

import numpy as np
import yaml
import torch

sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *
from probe import *
from dvutils.Data_Shapley import Fast_Data_Shapley  # YAML tag resolution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_train_dp", type=int, default=5000)
    parser.add_argument("--val_sample_num", type=int, default=872)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="inv",
                        choices=["inv"],
                        help="Only 'inv' is supported (this is the baseline "
                             "used as ρ's denominator in our analysis).")
    parser.add_argument("--inv_lambda_", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name   = args.dataset_name
    seed           = args.seed
    num_train_dp   = args.num_train_dp
    val_sample_num = args.val_sample_num
    tmc_iter       = args.tmc_iter
    approximate    = args.approximate
    inv_lambda_    = args.inv_lambda_

    signgd         = False
    early_stopping = "True"

    yaml_path = f"../configs/dshap/{dataset_name}/ntk_prompt.yaml"

    # ----- seed fixing (identical to task_data_selection.py) -----
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # ----- YAML + probe model (identical to task_data_selection.py) -----
    yaml_args    = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    probe_model  = yaml_args["probe_com"]
    probe_model.model.init(list_dataset.label_word_list)
    probe_model.approximate(approximate)
    probe_model.set_inv_params(lam=inv_lambda_)

    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    # ----- shapley pkl (for sampled_idx / sampled_val_idx) -----
    lambda_str   = f"{inv_lambda_:.0e}"
    extra_tag    = f"_lam{lambda_str}"
    shapley_path = (
        f"./freeshap_res/shapley/{dataset_name}/inv/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")
    with open(shapley_path, "rb") as f:
        result = pickle.load(f)
    sampled_idx     = np.array(result["sampled_idx"])
    sampled_val_idx = np.array(result["sampled_val_idx"])
    dv_result       = np.array(result["dv_result"])
    print(f"[info] |train|={len(sampled_idx)}, |val|={len(sampled_val_idx)}")

    # ----- NTK cache (so pipeline state matches task_data_selection.py) -----
    ntk_path = (
        f"./freeshap_res/ntk/{dataset_name}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")
    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError("NTK cache format is old; regenerate with updated script.")
    probe_model.get_cached_ntk(bundle["ntk"])

    # ----- train / val datasets (identical to task_data_selection.py) -----
    train_set = list_dataset.get_idx_dataset(sampled_idx,     split="train")
    val_set   = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    print("len(train_set) =", len(train_set))
    print("len(val_set)   =", len(val_set))

    # ===== 0% selection: inv-mode predictor with empty train set =========
    # For inv mode, kernel regression produces K_test @ (K_train + λI)^{-1} @ y_train.
    # With n=0, y_train is empty and the product collapses to 0 ∈ R^{|val| × C}.
    # We construct the prediction explicitly to avoid a degenerate
    # matrix-inverse call inside probe_model.kernel_regression.
    test_labels = torch.tensor([ex['label'] for ex in val_set])
    num_labels  = int(probe_model.num_labels) if hasattr(probe_model, "num_labels") \
                  else int(test_labels.max().item()) + 1
    test_preds  = torch.zeros(len(val_set), num_labels)     # empty-train inv prediction
    acc0        = (test_preds.argmax(dim=1) == test_labels).float().mean()
    acc0_int    = int(torch.round(acc0 * 10000).item())
    print(f"[result] inv-mode accuracy @ 0%  selected = {acc0.item():.4f} "
          f"(×10000 -> {acc0_int})")

    # ===== 20% selection: inv-top kernel regression =====================
    # Replicates task_data_selection.py inv/top at f=20% exactly:
    #   1) sort by Shapley acc-contribution (sum over val), descending
    #   2) take top-k with k = int(num_train_dp * 20 / 100)
    #   3) run probe_model.kernel_regression with those indices
    acc_contrib        = dv_result[:, 1, :]             # (num_train_dp, val_sample_num)
    acc_sum_per_train  = acc_contrib.sum(axis=1)
    sorted_indices     = np.argsort(acc_sum_per_train)[::-1]

    k20 = int(num_train_dp * 20 / 100)
    k20 = int(min(k20, len(acc_sum_per_train)))
    top_indices_20 = sorted_indices[:k20]

    probe_model.pre_inv = None
    probe_model.kr_model = None
    _, acc20 = probe_model.kernel_regression(
        train_indices=np.array(top_indices_20, dtype=int),
        test_set=val_set,
    )
    acc20_int = int(torch.round(acc20 * 10000).item())
    print(f"[result] inv-mode accuracy @ 20% selected (top-k, k={k20}) "
          f"= {acc20.item():.4f} (×10000 -> {acc20_int})")

    # ----- write output (same ×10000-int convention as task_data_selection) -----
    setting_name = os.path.basename(shapley_path).replace('.pkl', '')
    out_dir      = f"./freeshap_res/data_selection/{dataset_name}/inv/base_accuracy"
    os.makedirs(out_dir, exist_ok=True)
    out_path     = f"{out_dir}/{setting_name}_base.txt"
    with open(out_path, 'w') as f:
        f.write("inv_base_accuracy:\n")
        f.write(f"{acc0_int}\n")
        f.write("inv_top_f20_accuracy:\n")
        f.write(f"{acc20_int}\n")
    print(f"[info] saved -> {out_path}")


if __name__ == "__main__":
    main()
