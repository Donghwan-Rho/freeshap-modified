#!/bin/sh
python task_ntk.py --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --config ntk_llama
python task_shapley.py --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500 --config ntk_llama
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500 --config ntk_llama