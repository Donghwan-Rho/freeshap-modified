#!/bin/sh
# node01 — QQP imbalance experiment.
#   pos50  -> class_ratios = [0.5, 0.5]   (forced balance vs natural 63/37)
#   pos10  -> class_ratios = [0.9, 0.1]   (label 0 = natural majority, 90%)
# binary, label 0 = majority direction matches QQP's natural distribution.

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON_BASE="--seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --tmc_iter 500"

for RATIO_TAG in "0.5,0.5" "0.9,0.1"; do
    echo "==============================================================="
    echo "[node01-qqp] class_ratios=$RATIO_TAG — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON_BASE --class_ratios $RATIO_TAG

    for METHOD in lrfshap a1; do
        echo "---------------------------------------------------------------"
        echo "[node01-qqp] ratios=$RATIO_TAG method=$METHOD inv baseline"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON_BASE --class_ratios $RATIO_TAG --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6
        python $SELECT  $COMMON_BASE --class_ratios $RATIO_TAG --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6

        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[node01-qqp] ratios=$RATIO_TAG method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON_BASE --class_ratios $RATIO_TAG --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON_BASE --class_ratios $RATIO_TAG --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
done

echo "[run_imbalance_qqp.sh] all done"
