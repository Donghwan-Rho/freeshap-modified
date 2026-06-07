#!/bin/sh
# H100 narnia — MNLI n=5000 valbal, 3 ratios in requested order:
#   cls60_20_20 → cls33_33_33 → cls90_05_05
# Per ratio: NTK (cache hit expected) → INV baseline once (LR == A1 in INV mode)
#            → lrfshap eigen×7 → a1 eigen×7.  (a1 INV intentionally skipped.)
#
# class_ratios 0.333,0.333,0.334 -> tag is cls33_33_33 (ratio_tag rounds each
# component, so the third entry rounds to 33, not 34).

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"

run_grid_multi () {
    RATIOS="$1"
    TAG="$2"
    echo "==============================================================="
    echo "[mnli-n5000-valbal] $TAG (class_ratios=$RATIOS) — NTK"
    echo "==============================================================="
    python $NTK $COMMON --class_ratios $RATIOS

    echo "---------------------------------------------------------------"
    echo "[mnli-n5000-valbal] $TAG INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[mnli-n5000-valbal] $TAG method=$METHOD eigen r=$R%"
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

run_grid_multi "0.6,0.2,0.2"        cls60_20_20
run_grid_multi "0.333,0.333,0.334"  cls33_33_33
run_grid_multi "0.9,0.05,0.05"      cls90_05_05

echo "[run_imbalance_mnli_n5000_valbal_h100.sh] all done"
