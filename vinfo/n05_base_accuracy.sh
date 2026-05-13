#!/bin/sh
# Compute empirical base accuracy (0% selected data) for all seven datasets
# used in the LC/FC analysis: seed=2026, num=max_train (matches cached NTK).
# Requires: shapley pkl + NTK cache to already exist in ./freeshap_res/.

python task_base_accuracy.py --seed 2026 --dataset_name sst2    --num_train_dp 5000 --val_sample_num 872  --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_base_accuracy.py --seed 2026 --dataset_name rte     --num_train_dp 2490 --val_sample_num 277  --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_base_accuracy.py --seed 2026 --dataset_name qqp     --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_base_accuracy.py --seed 2026 --dataset_name mnli    --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_base_accuracy.py --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_base_accuracy.py --seed 2026 --dataset_name mr      --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_base_accuracy.py --seed 2026 --dataset_name mrpc    --num_train_dp 3668 --val_sample_num 408  --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
