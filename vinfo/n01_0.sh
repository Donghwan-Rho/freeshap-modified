#!/bin/sh
python task_ntk.py --dataset_name rte --num_train_dp 1000 --val_sample_num 277
python task_shapley_acc.py --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate inv --lambda_ 1e-5
python task_shapley_acc.py --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate inv --lambda_ 1e-4
python task_shapley_acc.py --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate inv --lambda_ 1e-3
python task_shapley_acc.py --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate inv --lambda_ 1e-2
python task_shapley_acc.py --dataset_name rte --num_train_dp 1000 --val_sample_num 277 --approximate inv --lambda_ 1e-1
