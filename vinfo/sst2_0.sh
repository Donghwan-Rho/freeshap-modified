#!/bin/sh
# CUDA_VISIBLE_DEVICES=0,1 python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate inv
# CUDA_VISIBLE_DEVICES=0,1 python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 1000 --lambda_ 1e-6
CUDA_VIISBLE_DEVICES=0 python prompt_finetune_train.py --num_train_dp 5000 --tmc_iter 500 --approximate inv --lambda_ 1e-6  --sweep_percentages --sweep_start 1 --sweep_end 100 --sweep_step 1
CUDA_VIISBLE_DEVICES=0,1 python prompt_finetune_train.py --num_train_dp 10000 --tmc_iter 500 --approximate inv --lambda_ 1e-6  --sweep_percentages --sweep_start 1 --sweep_end 100 --sweep_step 1