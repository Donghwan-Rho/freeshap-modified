#!/bin/sh
# A1 label-aware Shapley + data selection for SST-2 (n=2000, val=872, seed=2026).
# Mirrors run_rte_label.sh; skips the inv baseline (already exists upstream).
#
# Invoke from vinfo/:
#   cd /extdata1/donghwan/freeshap/vinfo
#   bash freeshap_res/claude_research/experiments/run_sst2_label.sh

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
SHAPLEY=$SCRIPT_DIR/task_label_shapley.py
SELECT=$SCRIPT_DIR/task_label_data_selection.py

COMMON_ARGS="--seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --tmc_iter 500"

# A1 label-aware Eigen, rank sweep matching the upstream pattern.
for R in 1 5 10 15 20 25 30; do
    python $SHAPLEY $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

# Aggregate into a single 15-line comparison txt (see run_agnews_label.sh).
python $SCRIPT_DIR/aggregate_label_summary.py \
    --dataset_name sst2 --seed 2026 --num_train_dp 2000 --val_sample_num 872 \
    --tmc_iter 500 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 \
    --ranks 1 5 10 15 20 25 30

echo "[run_sst2_label.sh] done"
