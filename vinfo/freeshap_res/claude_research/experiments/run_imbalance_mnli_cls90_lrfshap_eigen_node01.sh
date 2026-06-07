#!/bin/sh
# node01 — MNLI cls90_05_05 (label0 90/5/5) n=5000 valbal=1000.
# lrfshap eigen × 7 ONLY (r=1, 5, 10, 15, 20, 25, 30).
# INV is on node05; A1 eigen is on node04.  NTK already cached.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"
RATIOS="0.9,0.05,0.05"

echo "==============================================================="
echo "[mnli-n5000-valbal] cls90_05_05 — NTK (cache hit expected)"
echo "==============================================================="
python $NTK $COMMON --class_ratios $RATIOS

for R in 1 5 10 15 20 25 30; do
    echo "---------------------------------------------------------------"
    echo "[mnli-n5000-valbal] cls90_05_05 method=lrfshap eigen r=$R%"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

echo "[run_imbalance_mnli_cls90_lrfshap_eigen_node01.sh] all done"
