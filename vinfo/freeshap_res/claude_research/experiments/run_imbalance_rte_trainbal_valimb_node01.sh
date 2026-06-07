#!/bin/sh
# node01 — RTE train pos50 (n=1300, balanced binary) + val IMBALANCED.
#   val 1: pos70 (label1 maj 70%, n=145)
#   val 2: pos90 (label1 maj 90%, n=145)
# Each val: NTK + INV baseline + (lrfshap, a1) x rank {1,5,10,15,20,25,30}.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON_BASE="--seed 2026 --dataset_name rte --num_train_dp 1300 --val_sample_num 145 --tmc_iter 500 --pos_ratio 0.5"

run_val_setting () {
    VPR=$1   # val_pos_ratio, e.g. 0.7
    TAG=$2   # printable tag, e.g. valimb_pos70
    COMMON="$COMMON_BASE --val_pos_ratio $VPR"

    echo "==============================================================="
    echo "[rte-trainbal-valimb] $TAG (val_pos_ratio=$VPR) — NTK"
    echo "==============================================================="
    python $NTK $COMMON

    echo "[rte-trainbal-valimb] $TAG INV baseline (lrfshap)"
    python $SHAPLEY $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "[rte-trainbal-valimb] $TAG method=$METHOD eigen r=$R%"
            python $SHAPLEY $COMMON --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

run_val_setting 0.7 "valimb_pos70"
run_val_setting 0.9 "valimb_pos90"

echo "[run_imbalance_rte_trainbal_valimb_node01.sh] all done"
