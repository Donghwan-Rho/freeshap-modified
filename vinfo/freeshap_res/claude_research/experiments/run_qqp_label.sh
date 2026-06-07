#!/bin/sh
# A1 label-aware Shapley + data selection for QQP (n=2000, val=1000, seed=2026).
set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
SHAPLEY=$SCRIPT_DIR/task_label_shapley.py
SELECT=$SCRIPT_DIR/task_label_data_selection.py

COMMON_ARGS="--seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --tmc_iter 500"

for R in 1 5 10 15 20 25 30; do
    python $SHAPLEY $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

python $SCRIPT_DIR/aggregate_label_summary.py \
    --dataset_name qqp --seed 2026 --num_train_dp 2000 --val_sample_num 1000 \
    --tmc_iter 500 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 \
    --ranks 1 5 10 15 20 25 30

echo "[run_qqp_label.sh] done"
