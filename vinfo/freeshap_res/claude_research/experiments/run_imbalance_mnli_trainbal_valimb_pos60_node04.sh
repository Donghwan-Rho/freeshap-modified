#!/bin/sh
# node04 — MNLI train BALANCED cls33_33_34 (n=5000) + val IMBALANCED cls60_20_20 (n=1000).
# Single val setting; the cls90_05_05 val launcher is separate.
# NTK + INV baseline + (lrfshap, a1) x rank {1,5,10,15,20,25,30}.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --class_ratios 0.333,0.333,0.334 --val_class_ratios 0.6,0.2,0.2"
TAG="valimb_cls60"

echo "==============================================================="
echo "[mnli-trainbal-valimb] $TAG — NTK"
echo "==============================================================="
python $NTK $COMMON

echo "[mnli-trainbal-valimb] $TAG INV baseline (lrfshap)"
python $SHAPLEY $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "[mnli-trainbal-valimb] $TAG method=$METHOD eigen r=$R%"
        python $SHAPLEY $COMMON --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_mnli_trainbal_valimb_pos60_node04.sh] all done"
