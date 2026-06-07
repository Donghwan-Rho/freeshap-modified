#!/bin/sh
# A1 label-aware Shapley + data selection for AG News (n=2000, val=1000, seed=2026).
# Mirrors run_rte_label.sh; skips the inv baseline (already exists upstream).
#
# Invoke from vinfo/:
#   cd /extdata1/donghwan/freeshap/vinfo
#   bash freeshap_res/claude_research/experiments/run_agnews_label.sh

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
SHAPLEY=$SCRIPT_DIR/task_label_shapley.py
SELECT=$SCRIPT_DIR/task_label_data_selection.py

COMMON_ARGS="--seed 2026 --dataset_name ag_news --num_train_dp 2000 --val_sample_num 1000 --tmc_iter 500"

# A1 label-aware Eigen, rank sweep matching the upstream pattern.
for R in 1 5 10 15 20 25 30; do
    python $SHAPLEY $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

# Aggregate into a single 15-line comparison txt:
#   inv_lam1e_6                       = [...]  (FreeShap baseline, from upstream)
#   r{1..30}_eigen_lam_inv1e_6        = [...]  (LRFShap top-r by lambda, from upstream ipynb)
#   r{1..30}_eigen_label_lam_inv1e_6  = [...]  (A1 top-r by s_i, from CR sidecar JSON)
python $SCRIPT_DIR/aggregate_label_summary.py \
    --dataset_name ag_news --seed 2026 --num_train_dp 2000 --val_sample_num 1000 \
    --tmc_iter 500 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 \
    --ranks 1 5 10 15 20 25 30

echo "[run_agnews_label.sh] done"
