#!/bin/sh
# node03 — MNLI train BALANCED cls33_33_34 (n=5000) + val IMBALANCED cls90_05_05 (n=1000).
# This script: NTK + INV baseline (lrfshap) + lrfshap eigen x rank {1,5,10,15,20,25,30}.
# A1 eigen sub-runs are launched separately on node01.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --class_ratios 0.333,0.333,0.334 --val_class_ratios 0.9,0.05,0.05"
TAG="valimb_cls90"

echo "==============================================================="
echo "[mnli-trainbal-valimb-cls90 / node03] $TAG — NTK"
echo "==============================================================="
python $NTK $COMMON

echo "[mnli-trainbal-valimb-cls90 / node03] $TAG INV baseline (lrfshap)"
python $SHAPLEY $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6

for R in 1 5 10 15 20 25 30; do
    echo "[mnli-trainbal-valimb-cls90 / node03] $TAG method=lrfshap eigen r=$R%"
    python $SHAPLEY $COMMON --method lrfshap --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --method lrfshap --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

echo "[run_imbalance_mnli_trainbal_valimb_cls90_lrfshap_inv_node03.sh] all done"
