#!/bin/sh
python task_ntk.py --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000
python task_shapley_acc.py --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --lambda_ 1e-5
python task_shapley_acc.py --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --lambda_ 1e-4
python task_shapley_acc.py --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --lambda_ 1e-3
python task_shapley_acc.py --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --lambda_ 1e-2
python task_shapley_acc.py --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --lambda_ 1e-1

# CUDA_VIISBLE_DEVICES=0 python prompt_finetune_train.py --num_train_dp 5000 --tmc_iter 500 --approximate inv --lambda_ 1e-6  --sweep_percentages --sweep_start 1 --sweep_end 100 --sweep_step 1
# CUDA_VIISBLE_DEVICES=0,1 python prompt_finetune_train.py --num_train_dp 10000 --tmc_iter 500 --approximate inv --lambda_ 1e-6  --sweep_percentages --sweep_start 1 --sweep_end 100 --sweep_step 1