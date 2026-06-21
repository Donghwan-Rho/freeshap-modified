#!/bin/sh

# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 1000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2025 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

# python task_ntk.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --config ntk_llama --seed 2026 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
