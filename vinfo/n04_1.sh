#!/bin/sh
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-5 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-5 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-4 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-4 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-3 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-3 --tmc_iter 500

python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-5 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-5 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-4 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-4 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-3 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-3 --tmc_iter 500

python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-5 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-5 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-4 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-4 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-3 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-3 --tmc_iter 500
