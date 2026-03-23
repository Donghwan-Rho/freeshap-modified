#!/bin/sh
python task_ntk.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_ntk.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_ntk.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_ntk.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name mr --num_train_dp 8530 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --eigen_lambda_ 1e-2 --tmc_iter 500 --log_early_stopping

# CUDA_VIISBLE_DEVICES=0 python prompt_finetune_train.py --num_train_dp 5000 --tmc_iter 500 --approximate inv --inv_lambda_ 1e-6  --sweep_percentages --sweep_start 1 --sweep_end 100 --sweep_step 1
# CUDA_VIISBLE_DEVICES=0,1 python prompt_finetune_train.py --num_train_dp 8530 --tmc_iter 500 --approximate inv --inv_lambda_ 1e-6  --sweep_percentages --sweep_start 1 --sweep_end 100 --sweep_step 1