#!/bin/sh
python task_ntk.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000
# python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500

python task_ntk.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000
# python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
# python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
# python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 5 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 15 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 20 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 25 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_shapley.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
python task_data_selection.py --seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 30 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
