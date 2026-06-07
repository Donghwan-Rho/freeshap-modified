#!/bin/sh
# node05 — MNLI then QQP, both n=5000, val=1000 (balanced), 3 ratios each.
#
# MNLI (3-class, label0 = majority direction):
#   balanced  0.333,0.333,0.334  → 1665/1665/1670 (true even baseline)
#   mild      0.6,0.2,0.2         → 3000/1000/1000
#   extreme   0.9,0.05,0.05       → 4500/250/250
#   val balanced: per_class=333, remainder=1 → random class gets +1 (total 1000)
#
# QQP (binary, label0 = natural majority):
#   balanced  pos_ratio=0.5  → 2500/2500
#   mild      pos_ratio=0.3  → label0 3500, label1 1500 (70/30 with label0 maj)
#   extreme   pos_ratio=0.1  → label0 4500, label1 500 (90/10 with label0 maj)
#   val balanced: per_class=500 (total 1000)
#
# INV baseline once per ratio (LR == A1 in inv mode).

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON_MNLI="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"
COMMON_QQP="--seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"

run_grid_multi () {
    COMMON="$1"
    RATIOS="$2"
    TAG="$3"
    DS_LABEL="$4"
    echo "==============================================================="
    echo "[$DS_LABEL] $TAG (class_ratios=$RATIOS) — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --class_ratios $RATIOS

    echo "---------------------------------------------------------------"
    echo "[$DS_LABEL] $TAG INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[$DS_LABEL] $TAG method=$METHOD eigen r=$R%"
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

run_grid_binary () {
    COMMON="$1"
    POS="$2"
    DS_LABEL="$3"
    echo "==============================================================="
    echo "[$DS_LABEL] pos_ratio=$POS — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --pos_ratio $POS

    echo "---------------------------------------------------------------"
    echo "[$DS_LABEL] pos=$POS INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[$DS_LABEL] pos=$POS method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --pos_ratio $POS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

# ===== MNLI =====
run_grid_multi "$COMMON_MNLI" "0.333,0.333,0.334"  cls33_33_34   mnli-n5000-valbal
run_grid_multi "$COMMON_MNLI" "0.6,0.2,0.2"        cls60_20_20   mnli-n5000-valbal
run_grid_multi "$COMMON_MNLI" "0.9,0.05,0.05"      cls90_05_05   mnli-n5000-valbal

# ===== QQP =====
run_grid_binary "$COMMON_QQP" 0.5  qqp-n5000-valbal
run_grid_binary "$COMMON_QQP" 0.3  qqp-n5000-valbal
run_grid_binary "$COMMON_QQP" 0.1  qqp-n5000-valbal

echo "[run_imbalance_mnli_qqp_n5000_valbal_node05.sh] all done"
