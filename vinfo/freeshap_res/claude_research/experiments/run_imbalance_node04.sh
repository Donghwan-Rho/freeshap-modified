#!/bin/sh
# node04 — MNLI (60/20/20, 90/5/5) + AG News (55/15/15/15, 85/5/5/5).
# label 0 = majority for all multi-class settings.
#
# Sequential within this script (~24-32 h estimated wall clock).
# Invoke from vinfo/:
#   cd /extdata1/donghwan/freeshap/vinfo
#   bash freeshap_res/claude_research/experiments/run_imbalance_node04.sh

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

run_one_grid() {
    DS=$1
    RATIOS=$2     # e.g., "0.6,0.2,0.2"
    TAG=$3        # e.g., "cls60_20_20"
    VAL=$4
    COMMON="--seed 2026 --dataset_name $DS --num_train_dp 2000 --val_sample_num $VAL --tmc_iter 500"

    echo "==============================================================="
    echo "[node04] $DS class_ratios=$RATIOS tag=$TAG — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --class_ratios $RATIOS

    for METHOD in lrfshap a1; do
        echo "---------------------------------------------------------------"
        echo "[node04] $DS $TAG method=$METHOD inv baseline"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --class_ratios $RATIOS --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6
        python $SELECT  $COMMON --class_ratios $RATIOS --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6

        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[node04] $DS $TAG method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON --class_ratios $RATIOS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --class_ratios $RATIOS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

# MNLI (val=1000)
run_one_grid mnli "0.6,0.2,0.2"       cls60_20_20      1000
run_one_grid mnli "0.9,0.05,0.05"     cls90_05_05      1000

# AG News (val=1000)
run_one_grid ag_news "0.55,0.15,0.15,0.15"      cls55_15_15_15  1000
# cls85_05_05_05 is offloaded to node01 to parallelize — handled by
# run_imbalance_ag_news_cls85_node01.sh.  Comment out here to avoid race.
# run_one_grid ag_news "0.85,0.05,0.05,0.05"      cls85_05_05_05  1000

echo "[run_imbalance_node04.sh] all done"
