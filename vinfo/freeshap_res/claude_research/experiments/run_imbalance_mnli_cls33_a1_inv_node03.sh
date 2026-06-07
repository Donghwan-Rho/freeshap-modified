#!/bin/sh
# node03 — MNLI cls33_33_33 (balanced) n=5000 valbal=1000, A1 INV ONLY.
# Note: INV mode is method-independent in our framework (LR == A1 because
# monkey-patch only affects EigenNTKRegression._precompute_eigen_features).
# This produces a sidecar at a1/inv/* numerically identical to lrfshap/inv/*.
# User requested it for analysis completeness / path consistency.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"
RATIOS="0.333,0.333,0.334"

echo "==============================================================="
echo "[mnli-n5000-valbal] cls33_33_33 — NTK (cache hit expected)"
echo "==============================================================="
python $NTK $COMMON --class_ratios $RATIOS

echo "---------------------------------------------------------------"
echo "[mnli-n5000-valbal] cls33_33_33 method=a1 INV (full rank)"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --class_ratios $RATIOS --method a1 \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --class_ratios $RATIOS --method a1 \
    --approximate inv --inv_lambda_ 1e-6

echo "[run_imbalance_mnli_cls33_a1_inv_node03.sh] all done"
