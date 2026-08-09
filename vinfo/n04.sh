#!/bin/sh

python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2024 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500

python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2025 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500

python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate inv --inv_lambda_ 1e-6 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 1 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 5 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 10 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 15 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 20 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 25 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
python task_wrong_label_detection.py --dataset_name mrpc --config ntk_llama --seed 2026 --num_train_dp 3668 --val_sample_num 408 --approximate eigen --eigen_rank 30 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
