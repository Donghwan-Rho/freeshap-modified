#!/bin/sh
# node05 — AG News train BALANCED 25/25/25/25 (n=5000) + val IMBALANCED.
#   val 1: cls55_15_15_15 (label0 maj 55%)
#   val 2: cls85_05_05_05 (label0 maj 85%)
# Each val setting: NTK + INV baseline + (lrfshap, a1) x rank {1,5,10,15,20,25,30}.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON_BASE="--seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --class_ratios 0.25,0.25,0.25,0.25"

run_val_setting () {
    VCR=$1   # val_class_ratios string, e.g. "0.55,0.15,0.15,0.15"
    TAG=$2   # printable tag, e.g. "valimb_cls55"
    COMMON="$COMMON_BASE --val_class_ratios $VCR"

    echo "==============================================================="
    echo "[agnews-trainbal-valimb] $TAG ($VCR) — NTK"
    echo "==============================================================="
    python $NTK $COMMON

    echo "[agnews-trainbal-valimb] $TAG INV baseline (lrfshap)"
    python $SHAPLEY $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "[agnews-trainbal-valimb] $TAG method=$METHOD eigen r=$R%"
            python $SHAPLEY $COMMON --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

run_val_setting "0.55,0.15,0.15,0.15" "valimb_cls55"
run_val_setting "0.85,0.05,0.05,0.05" "valimb_cls85"

echo "[run_imbalance_agnews_trainbal_valimb_node05.sh] all done"
