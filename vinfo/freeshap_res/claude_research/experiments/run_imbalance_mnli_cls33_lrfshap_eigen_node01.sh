#!/bin/sh
# node01 — MNLI cls33_33_33 (balanced) n=5000 valbal=1000, lrfshap eigen × 7 ONLY.
# Skips INV baseline (user request — INV is heavy ~20h, not needed for now).
# A1 eigen × 7 is on node03.

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
    echo "[mnli-n5000-valbal] cls33_33_33 method=lrfshap eigen r=$R%"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

echo "[run_imbalance_mnli_cls33_lrfshap_eigen_node01.sh] all done"
