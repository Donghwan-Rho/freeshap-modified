#!/bin/sh
cd /extdata1/donghwan/freeshap/vinfo

# wld
for S in 2024 2025 2026; do

  # 블록1: nystrom nyseps sweep (nystrom_d=20, lam=1e-2)
  for L in 1e-3 1e-2 1e-1; do
    for E in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 1e+1; do
      python task_wrong_label_detection.py --dataset_name mr --seed $S \
        --num_train_dp 2000 --val_sample_num 1000 --approximate nystrom \
        --nystrom_d 20 --nystrom_lambda_ $L --nyseps $E \
        --poison_pct 10 --tmc_iter 500 --out_root ./jitter_exp/res
    done
  done

  # # 블록2: eigen 참조 (eigen_rank=20, eigeps=1e-8, lam=1e-2)
  # for L in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1; do
  #   for E in 1e-8 1e-7 1e-6 1e-5 1e-4; do
  #     python task_wrong_label_detection.py --dataset_name mr --seed $S \
  #       --num_train_dp 2000 --val_sample_num 1000 --approximate eigen \
  #       --eigen_rank 20 --eigen_lambda_ $L --eigeps $E \
  #       --poison_pct 10 --tmc_iter 500 --out_root ./jitter_exp/res
  #   done
  # done

  # # 블록3: inv (exact 참조)
  # python task_wrong_label_detection.py --dataset_name mr --seed $S \
  #   --num_train_dp 2000 --val_sample_num 1000 --approximate inv \
  #   --inv_lambda_ 1e-6 \
  #   --poison_pct 10 --tmc_iter 500 --out_root ./jitter_exp/res

done

# data selection
# for S in 2024 2025 2026; do

#   # 블록1: nyseps sweep (nystrom_lambda_ = 1e-2 고정)
#   for E in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 1e+1 1e+2 1e+3; do
#     python task_shapley.py --config ntk_prompt --seed $S --dataset_name ag_news \
#       --num_train_dp 2000 --val_sample_num 1000 --approximate nystrom --nystrom_d 20 \
#       --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps $E --tmc_iter 500 --out_root ./jitter_exp/res
#     python task_data_selection.py --config ntk_prompt --seed $S --dataset_name ag_news \
#       --num_train_dp 2000 --val_sample_num 1000 --approximate nystrom --nystrom_d 20 \
#       --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps $E --tmc_iter 500 --out_root ./jitter_exp/res
#   done

#   # 블록2: nystrom_lambda_ sweep (nyseps = 1e+1 고정)
#   for L in 1e-6 1e-5 1e-4 1e-3 1e-1 1; do
#     python task_shapley.py --config ntk_prompt --seed $S --dataset_name ag_news \
#       --num_train_dp 2000 --val_sample_num 1000 --approximate nystrom --nystrom_d 20 \
#       --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps 1e+1 --tmc_iter 500 --out_root ./jitter_exp/res
#     python task_data_selection.py --config ntk_prompt --seed $S --dataset_name ag_news \
#       --num_train_dp 2000 --val_sample_num 1000 --approximate nystrom --nystrom_d 20 \
#       --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps 1e+1 --tmc_iter 500 --out_root ./jitter_exp/res
#   done

# done
