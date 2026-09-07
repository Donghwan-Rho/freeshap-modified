#!/bin/sh
# llama sst2 (n=2000) removal 백필 — SV pkl 재사용 (TMC 없음), inv + eigen rank sweep.

REM="$(seq -s' ' 0 99)"
for S in 2024 2025 2026; do
  python task_data_removal.py --config ntk_llama --seed $S --dataset_name sst2 \
    --num_train_dp 2000 --val_sample_num 872 --approximate inv --inv_lambda_ 1e-6 \
    --tmc_iter 500 --num_train_removed_list $REM
  for R in 1 5 10 15 20 25 30; do
    python task_data_removal.py --config ntk_llama --seed $S --dataset_name sst2 \
      --num_train_dp 2000 --val_sample_num 872 --approximate eigen --eigen_rank $R \
      --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500 \
      --num_train_removed_list $REM
  done
done
