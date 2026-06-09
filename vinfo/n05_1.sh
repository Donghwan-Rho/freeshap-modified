#!/bin/sh
# sst2 / num_train_dp=1000 / val=872 / seed=2024 / tmc_iter=500
# Nystrom mode, nystrom_d=10 (=10% of num_dp -> 100 landmarks), nystrom_lambda in {1e-3, 1e-2}
# python task_ntk.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872

python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 1 --nystrom_lambda_ 1e-4 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 1 --nystrom_lambda_ 1e-4 --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 1 --nystrom_lambda_ 1e-5 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 1 --nystrom_lambda_ 1e-5 --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 1 --nystrom_lambda_ 1e-6 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 1 --nystrom_lambda_ 1e-6 --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-2 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-2 --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-3 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-3 --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-4 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-4 --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-5 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-5 --inv_lambda_ 1e-6 --tmc_iter 500
python task_shapley.py        --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-6 --inv_lambda_ 1e-6 --tmc_iter 500
python task_data_selection.py --seed 2024 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 872 --approximate nystrom --nystrom_d 5 --nystrom_lambda_ 1e-6 --inv_lambda_ 1e-6 --tmc_iter 500
