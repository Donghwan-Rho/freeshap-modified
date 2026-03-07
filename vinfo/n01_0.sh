#!/bin/sh
python task_ntk.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate inv --lambda_ 1e-6 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate eigen --eigen_rank 1 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate eigen --eigen_rank 5 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate eigen --eigen_rank 10 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate eigen --eigen_rank 15 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate eigen --eigen_rank 20 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate eigen --eigen_rank 25 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate eigen --eigen_rank 30 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_ntk.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate inv --lambda_ 1e-6 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate eigen --eigen_rank 1 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate eigen --eigen_rank 5 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate eigen --eigen_rank 10 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate eigen --eigen_rank 15 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate eigen --eigen_rank 20 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate eigen --eigen_rank 25 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2000 --val_sample_num 277 --approximate eigen --eigen_rank 30 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_ntk.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate inv --lambda_ 1e-6 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 1 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 5 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 10 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 15 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 20 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 25 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
python task_shapley_acc.py --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 30 --lambda_ 1e-2 --tmc_iter 100 --log_early_stopping
