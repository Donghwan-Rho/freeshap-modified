#!/bin/sh
python task_feature_concentration.py --seed 2024 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 
python task_feature_concentration.py --seed 2024 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000
python task_feature_concentration.py --seed 2024 --dataset_name mr --num_train_dp 5000 --val_sample_num 1000
python task_feature_concentration.py --seed 2024 --dataset_name mrpc --num_train_dp 3668 --val_sample_num 408
python task_feature_concentration.py --seed 2024 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000
python task_feature_concentration.py --seed 2024 --dataset_name rte --num_train_dp 2490 --val_sample_num 277
python task_feature_concentration.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872