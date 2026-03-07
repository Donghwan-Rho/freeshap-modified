#!/bin/sh
# python task_ntk.py --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000
# python task_ntk.py --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000
# python task_ntk.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-6
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-5
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-4
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-3
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-2
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 1000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-1

CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-6
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-5
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-4
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-3
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-2
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 2000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-1

CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-6
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-5
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-4
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-3
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-2
CUDA_VISIBLE_DEVICES=1 python task_shapley_acc.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --approximate eigen --eigen_rank 10 --lambda_ 1e-1
