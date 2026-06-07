#!/bin/sh
# node01 — AG News cls85_05_05_05 only (offloaded from node04 for parallel).
# 4-class, label 0 majority 85%, labels 1/2/3 each 5%.
# Same grid as node04 sequential: inv + LRFShap (7 ranks) + A1 (7 ranks) = 16 runs.

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

RATIOS="0.85,0.05,0.05,0.05"
TAG="cls85_05_05_05"
COMMON="--seed 2026 --dataset_name ag_news --num_train_dp 2000 --val_sample_num 1000 --tmc_iter 500"

echo "==============================================================="
echo "[node01-ag_news] $TAG (class_ratios=$RATIOS) — computing NTK ..."
echo "==============================================================="
python $NTK $COMMON --class_ratios $RATIOS

for METHOD in lrfshap a1; do
    echo "---------------------------------------------------------------"
    echo "[node01-ag_news] $TAG method=$METHOD inv baseline"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --class_ratios $RATIOS --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6

    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[node01-ag_news] $TAG method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --class_ratios $RATIOS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --class_ratios $RATIOS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_ag_news_cls85_node01.sh] all done"
