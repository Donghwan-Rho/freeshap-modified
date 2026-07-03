#!/bin/sh
# CIFAR-10 vision 실행 스크립트 (n05_0.sh 형식)
# 실행 위치: cd /extdata1/donghwan/freeshap/vinfo  ← 여기서 실행해야 상대경로가 맞음
#   CUDA_VISIBLE_DEVICES=<빈GPU> ./vision/n_vision.sh

# ===== 1) NTK =====
python vision/task_ntk_vision.py --config ntk_vision --seed 2024 --dataset_name cifar10 --num_train_dp 2500 --val_sample_num 1000

# ===== 2) Shapley + Data selection: inv =====
python vision/task_shapley_vision.py        --config ntk_vision --seed 2024 --dataset_name cifar10 --num_train_dp 2500 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
python vision/task_data_selection_vision.py --config ntk_vision --seed 2024 --dataset_name cifar10 --num_train_dp 2500 --val_sample_num 1000 --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500

# ===== 3) Shapley + Data selection: eigen (rank 여러 개) =====
for r in 1 5 10 15 20 25 30; do
  python vision/task_shapley_vision.py        --config ntk_vision --seed 2024 --dataset_name cifar10 --num_train_dp 2500 --val_sample_num 1000 --approximate eigen --eigen_rank $r --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
  python vision/task_data_selection_vision.py --config ntk_vision --seed 2024 --dataset_name cifar10 --num_train_dp 2500 --val_sample_num 1000 --approximate eigen --eigen_rank $r --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
done

# ===== 4) Shapley + Data selection: nystrom (d 여러 개) =====
for d in 1 5 10 15 20 25 30; do
  python vision/task_shapley_vision.py        --config ntk_vision --seed 2024 --dataset_name cifar10 --num_train_dp 2500 --val_sample_num 1000 --approximate nystrom --nystrom_d $d --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --tmc_iter 500
  python vision/task_data_selection_vision.py --config ntk_vision --seed 2024 --dataset_name cifar10 --num_train_dp 2500 --val_sample_num 1000 --approximate nystrom --nystrom_d $d --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --tmc_iter 500
done
