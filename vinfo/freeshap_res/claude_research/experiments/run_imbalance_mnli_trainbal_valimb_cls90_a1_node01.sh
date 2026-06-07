#!/bin/sh
# node01 — MNLI train BALANCED cls33_33_34 (n=5000) + val IMBALANCED cls90_05_05 (n=1000).
# This script: A1 eigen x rank {1,5,10,15,20,25,30}. ONLY.
# NTK + INV (lrfshap) + lrfshap eigen are produced by the node03 sibling launcher.
# Launch order: start node03 first, wait for NTK ([dataset_name=mnli] NTK eNTK saved ...),
# then start this one.

SCRIPT_DIR=freeshap_res/claude_research/experiments
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --class_ratios 0.333,0.333,0.334 --val_class_ratios 0.9,0.05,0.05"
TAG="valimb_cls90"

for R in 1 5 10 15 20 25 30; do
    echo "[mnli-trainbal-valimb-cls90 / node01] $TAG method=a1 eigen r=$R%"
    python $SHAPLEY $COMMON --method a1 --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --method a1 --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

echo "[run_imbalance_mnli_trainbal_valimb_cls90_a1_node01.sh] all done"
