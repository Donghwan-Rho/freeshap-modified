#!/bin/sh
# node05 — MNLI train BALANCED cls33_33_34 (n=5000) + val IMBALANCED cls90_05_05 (n=1000).
# lrfshap eigen x rank {1,5,10,15,20,25,30}. ONLY. (no NTK, no INV)
# NTK is produced by the node03 sibling launcher (lrfshap_inv).
# This script duplicates the eigen step from node03's launcher to parallelize:
#   node03 lrfshap_inv: NTK + INV + eigen x 7 (INV alone ~38h)
#   node05 lrfshap_eigen (this): eigen x 7 only, runs concurrently
# The redundant eigen sub-runs from node03 (started ~38h later) will simply
# overwrite the same sidecar JSON with identical contents (seed=2026).

SCRIPT_DIR=freeshap_res/claude_research/experiments
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --class_ratios 0.333,0.333,0.334 --val_class_ratios 0.9,0.05,0.05"
TAG="valimb_cls90"

for R in 1 5 10 15 20 25 30; do
    echo "[mnli-trainbal-valimb-cls90 / node05] $TAG method=lrfshap eigen r=$R%"
    python $SHAPLEY $COMMON --method lrfshap --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --method lrfshap --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

echo "[run_imbalance_mnli_trainbal_valimb_cls90_lrfshap_eigen_node05.sh] all done"
