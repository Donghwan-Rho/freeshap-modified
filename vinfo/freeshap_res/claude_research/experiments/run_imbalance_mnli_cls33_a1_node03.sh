#!/bin/sh
# node03 — MNLI cls33_33_33 (balanced 3-class) n=5000 valbal=1000.
# This node handles ONLY a1 eigen × 7.
# lrfshap (INV + eigen) is handled separately on node01.
# NTK already cached.

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

for R in 1 5 10 15 20 25 30; do
    echo "---------------------------------------------------------------"
    echo "[mnli-n5000-valbal] cls33_33_33 method=a1 eigen r=$R%"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method a1 \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --class_ratios $RATIOS --method a1 \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

echo "[run_imbalance_mnli_cls33_a1_node03.sh] all done"
