#!/bin/sh
python prompt_finetune_train.py --dataset_name rte --num_train_dp 2000 --approximate inv --lambda_ 1e-6 --sweep_percentages --sweep_mode count --sweep_start 1 --sweep_end 100 --sweep_step 1
python prompt_finetune_train.py --dataset_name rte --num_train_dp 2000 --approximate inv --lambda_ 1e-6 --sweep_percentages --sweep_mode count --sweep_start 1 --sweep_end 100 --sweep_step 1